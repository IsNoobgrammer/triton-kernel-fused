"""Fully-fused SwiGLU activation (Triton forward + Triton backward).

Fuses the SwiGLU activation `silu(gate) * up` and its gradient into ONE kernel each.
GEMMs stay on cuBLAS (gate_up projection, down projection) — only the activation is
fused, which is where the HBM round-trips are. This eliminates the intermediate
`silu(gate)` and `silu(gate)*up` tensors (forward) and computes both `grad_gate` and
`grad_up` in a single backward pass.

Input layout: a single concatenated `gate_up` tensor of shape (M, 2*I) — first I cols
are `gate`, last I cols are `up`. Produce it with one fused `gate_up_proj` (Linear ->
2*I) rather than two separate Linears; that is both faster (one GEMM) and what this
kernel expects.

Drop-in:
    from kernels.swiglu import fused_swiglu, FusedSwiGLUMLP
    out = fused_swiglu(gate_up)                 # (M, I)
    mlp = FusedSwiGLUMLP(hidden, inter)         # nn.Module, fused gate_up_proj

Backward is exact vs eager `silu(gate)*up` (fp16 atol ~1e-3, fp32 ~1e-5). Activation
math is done in float32 in-register to avoid GLU-derivative overflow in fp16.
"""
import torch
import torch.nn as nn
import triton
import triton.language as tl

__all__ = ["fused_swiglu", "FusedSwiGLUMLP", "FusedSwiGLU"]


# These kernels ARE torch.compile/inductor's own generated SwiGLU kernels, lifted verbatim from
# `TORCH_LOGS=output_code` and run as plain Triton. We benchmarked our hand-written 2D-tiled+autotune
# version against this on a T4 + an RTX 3050: inductor's 1D-flat pattern is faster on BOTH (fwd ~4%,
# fwd+bwd ~0.5%, every run) and simpler. SwiGLU is a memory-bound streaming op (2 loads, 1 store, zero
# reuse → SRAM/tiling buys nothing), so the only lever is the access pattern, and the compiler already
# nailed it. We adopted its kernel rather than keep losing to it. One generalization: inductor baked
# the static shapes in as literals (so it can magic-number-fold the % and //); we pass `I` as a
# `tl.constexpr` for the same constant-folding at any I. Assumes a contiguous row-major (M,2I) input.


@triton.jit
def _swiglu_fwd_kernel(GateUp_ptr, Out_ptr, n_out, I: tl.constexpr, XBLOCK: tl.constexpr):
    # 1D flat over the (M,I) OUTPUT. gate/up read from the (M,2I) input by stride
    # (col + 2I*row) and (I + col + 2I*row) — no slice materialized. 2 loads, 1 store (the floor).
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < n_out
    x0 = xindex % I
    x1 = xindex // I
    gate = tl.load(GateUp_ptr + (x0 + 2 * I * x1), xmask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + (I + x0 + 2 * I * x1), xmask, other=0.0).to(tl.float32)
    res = gate * tl.sigmoid(gate) * up
    tl.store(Out_ptr + xindex, res.to(Out_ptr.dtype.element_ty), xmask)


@triton.jit
def _swiglu_bwd_kernel(GradOut_ptr, GateUp_ptr, GradGateUp_ptr, n_full, I: tl.constexpr,
                       XBLOCK: tl.constexpr):
    # 1D flat over the FULL (M,2I) grad, ONE where-branched pass, single store. up-half (col>=I):
    # grad_up = grad_out*silu(gate); gate-half (col<I): grad_gate = grad_out*up*sig*(1+gate*(1-sig)).
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    xmask = xindex < n_full
    x0 = xindex % (2 * I)
    x1 = xindex // (2 * I)
    is_up = x0 >= I
    g_up = tl.load(GradOut_ptr + (x0 - I + I * x1), is_up & xmask, other=0.0).to(tl.float32)
    gate_u = tl.load(GateUp_ptr + (xindex - I), is_up & xmask, other=0.0).to(tl.float32)
    grad_up = tl.where(is_up, g_up * gate_u * tl.sigmoid(gate_u), 0.0)
    is_gate = x0 < I
    g_g = tl.load(GradOut_ptr + (x0 + I * x1), is_gate & xmask, other=0.0).to(tl.float32)
    up_g = tl.load(GateUp_ptr + (xindex + I), is_gate & xmask, other=0.0).to(tl.float32)
    gate_g = tl.load(GateUp_ptr + xindex, is_gate & xmask, other=0.0).to(tl.float32)
    sig = tl.sigmoid(gate_g)
    dsilu = sig * (1.0 + gate_g * (1.0 - sig))
    grad_gate = tl.where(is_gate, g_g * up_g * dsilu, 0.0)
    tl.store(GradGateUp_ptr + xindex,
             (grad_up + grad_gate).to(GradGateUp_ptr.dtype.element_ty), xmask)


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up):
        gate_up = gate_up.contiguous()  # 1D-flat kernel assumes row-major (M,2I)
        ctx.save_for_backward(gate_up)
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
        n = M * I
        if n:
            _swiglu_fwd_kernel[(triton.cdiv(n, 1024),)](gate_up, out, n, I, XBLOCK=1024)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gate_up, = ctx.saved_tensors
        M, twoI = gate_up.shape
        I = twoI // 2
        grad_gate_up = torch.empty_like(gate_up)
        n = M * twoI
        if n:
            _swiglu_bwd_kernel[(triton.cdiv(n, 1024),)](grad_output.contiguous(), gate_up,
                                                        grad_gate_up, n, I, XBLOCK=1024)
        return grad_gate_up


def fused_swiglu(gate_up: torch.Tensor) -> torch.Tensor:
    """gate_up (M, 2*I) — concatenated [gate | up] -> silu(gate) * up, shape (M, I)."""
    return FusedSwiGLU.apply(gate_up)


class FusedSwiGLUMLP(nn.Module):
    """SwiGLU MLP with a single fused gate_up projection + the fused activation kernel.
    Drop-in for a Llama/Qwen-style MLP (hidden -> inter -> hidden), one fewer GEMM than
    separate gate_proj/up_proj. Load split weights via `load_from_gate_up(gate_w, up_w, down_w)`."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_up_proj = nn.Linear(hidden_size, 2 * intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def forward(self, x):
        shape = x.shape[:-1]
        gate_up = self.gate_up_proj(x.reshape(-1, self.hidden_size))
        return self.down_proj(fused_swiglu(gate_up)).reshape(*shape, self.hidden_size)

    @torch.no_grad()
    def load_from_gate_up(self, gate_weight, up_weight, down_weight):
        """Copy weights from a model that has separate gate_proj/up_proj/down_proj."""
        self.gate_up_proj.weight.copy_(torch.cat([gate_weight, up_weight], dim=0))
        self.down_proj.weight.copy_(down_weight)
