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


_CONFIGS = [
    triton.Config({"BLOCK_M": 16, "BLOCK_I": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 16, "BLOCK_I": 256}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_I": 128}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 32, "BLOCK_I": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_I": 64}, num_warps=4, num_stages=2),
    triton.Config({"BLOCK_M": 64, "BLOCK_I": 128}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 64, "BLOCK_I": 256}, num_warps=8, num_stages=2),
    triton.Config({"BLOCK_M": 128, "BLOCK_I": 64}, num_warps=4, num_stages=3),
    triton.Config({"BLOCK_M": 128, "BLOCK_I": 128}, num_warps=8, num_stages=2),
]


@triton.autotune(configs=_CONFIGS, key=["M", "I"])
@triton.jit
def _swiglu_fwd_kernel(GateUp_ptr, Out_ptr, M, I,
                       s_gu_m, s_gu_i, s_o_m, s_o_i,
                       BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m < M)[:, None] & (offs_i < I)[None, :]
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i,
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i,
                 mask=mask, other=0.0).to(tl.float32)
    res = (gate * (1.0 / (1.0 + tl.exp(-gate)))) * up
    tl.store(Out_ptr + offs_m[:, None] * s_o_m + offs_i[None, :] * s_o_i,
             res.to(Out_ptr.dtype.element_ty), mask=mask)


@triton.autotune(configs=_CONFIGS, key=["M", "I"])
@triton.jit
def _swiglu_bwd_kernel(GradOut_ptr, GateUp_ptr, GradGateUp_ptr, M, I,
                       s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i,
                       BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask = (offs_m < M)[:, None] & (offs_i < I)[None, :]
    grad_out = tl.load(GradOut_ptr + offs_m[:, None] * s_go_m + offs_i[None, :] * s_go_i,
                       mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i,
                   mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i,
                 mask=mask, other=0.0).to(tl.float32)
    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    grad_up = grad_out * silu
    dsilu = sig * (1.0 + gate * (1.0 - sig))
    grad_gate = grad_out * up * dsilu
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + offs_i[None, :] * s_ggu_i,
             grad_gate, mask=mask)
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + (I + offs_i)[None, :] * s_ggu_i,
             grad_up, mask=mask)


class FusedSwiGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up):
        ctx.save_for_backward(gate_up)
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
        if M == 0:
            return out
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(I, meta["BLOCK_I"]))
        _swiglu_fwd_kernel[grid](gate_up, out, M, I,
                                 gate_up.stride(0), gate_up.stride(1), out.stride(0), out.stride(1))
        return out

    @staticmethod
    def backward(ctx, grad_output):
        gate_up, = ctx.saved_tensors
        M = gate_up.shape[0]
        I = gate_up.shape[1] // 2
        grad_gate_up = torch.empty_like(gate_up)
        if M == 0:
            return grad_gate_up
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), triton.cdiv(I, meta["BLOCK_I"]))
        _swiglu_bwd_kernel[grid](grad_output, gate_up, grad_gate_up, M, I,
                                 grad_output.stride(0), grad_output.stride(1),
                                 gate_up.stride(0), gate_up.stride(1),
                                 grad_gate_up.stride(0), grad_gate_up.stride(1))
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
