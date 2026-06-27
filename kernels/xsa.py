"""Fused XSA — Exclusive Self Attention (arXiv:2603.09078), Triton fwd + bwd.

XSA removes, from each attention output, the component lying along that position's own
value vector (a parameter-free rejection):

    z_i = y_i - (y_i . v_i / ||v_i||^2) v_i

This whole op (normalize V + dot + scale + subtract, with GQA repeat) is fused into ONE
forward and ONE backward kernel:
  - GQA is handled by BROADCASTING V across the query group IN-KERNEL (SDPA enable_gqa
    style) — the (B,H,S,D) repeat_kv copy and the normalized-V are never materialized.
  - reductions in float32 in-register.
  - grad_Y = reject(grad_z, v_hat); grad_V analytic, accumulated over the group.

Drop-in (apply after attention's value aggregation, before o_proj):
    from kernels.xsa import fused_xsa
    attn_out = fused_xsa(attn_out, value_states)   # Y (B,H,S,D), V (B,Hkv,S,D)

Grad-exact vs the eager rejection (fp16 atol ~1e-3).
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_xsa", "FusedXSA"]


@triton.jit
def _xsa_fwd_kernel(Y, V, Z, S, D, H, Hkv, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)                 # over B*Hkv*S
    s = pid % S
    t = pid // S
    kv = t % Hkv
    b = t // Hkv
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v_base = (b * Hkv + kv) * S * D + s * D
    v = tl.load(V + v_base + offs, mask=mask, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=0)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    for j in range(GROUP):
        h = kv * GROUP + j
        y_base = (b * H + h) * S * D + s * D
        y = tl.load(Y + y_base + offs, mask=mask, other=0.0).to(tl.float32)
        coeff = tl.sum(y * v, axis=0) * inv
        z = y - coeff * v
        tl.store(Z + y_base + offs, z.to(Z.dtype.element_ty), mask=mask)


@triton.jit
def _xsa_bwd_kernel(GZ, Y, V, GY, GV, S, D, H, Hkv, GROUP: tl.constexpr, BLOCK_D: tl.constexpr):
    pid = tl.program_id(0)
    s = pid % S
    t = pid // S
    kv = t % Hkv
    b = t // Hkv
    offs = tl.arange(0, BLOCK_D)
    mask = offs < D
    v_base = (b * Hkv + kv) * S * D + s * D
    v = tl.load(V + v_base + offs, mask=mask, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=0)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    gv_acc = tl.zeros((BLOCK_D,), dtype=tl.float32)
    for j in range(GROUP):
        h = kv * GROUP + j
        base = (b * H + h) * S * D + s * D
        y = tl.load(Y + base + offs, mask=mask, other=0.0).to(tl.float32)
        gz = tl.load(GZ + base + offs, mask=mask, other=0.0).to(tl.float32)
        dot = tl.sum(y * v, axis=0)
        gzv = tl.sum(gz * v, axis=0)
        coeff = dot * inv
        gy = gz - gzv * inv * v                                     # grad_Y = reject(gz)
        tl.store(GY + base + offs, gy.to(GY.dtype.element_ty), mask=mask)
        gv_acc += -gzv * inv * y + 2.0 * dot * gzv * inv * inv * v - coeff * gz
    tl.store(GV + v_base + offs, gv_acc.to(GV.dtype.element_ty), mask=mask)


class FusedXSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y, V):
        Y = Y.contiguous()
        V = V.contiguous()
        B, H, S, D = Y.shape
        Hkv = V.shape[1]
        group = H // Hkv
        Z = torch.empty_like(Y)
        BLOCK_D = triton.next_power_of_2(D)
        grid = (B * Hkv * S,)
        _xsa_fwd_kernel[grid](Y, V, Z, S, D, H, Hkv, GROUP=group, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(Y, V)
        ctx.shape = (B, H, S, D, Hkv, group, BLOCK_D)
        return Z

    @staticmethod
    def backward(ctx, gZ):
        Y, V = ctx.saved_tensors
        B, H, S, D, Hkv, group, BLOCK_D = ctx.shape
        gZ = gZ.contiguous()
        GY = torch.empty_like(Y)
        GV = torch.empty_like(V)
        grid = (B * Hkv * S,)
        _xsa_bwd_kernel[grid](gZ, Y, V, GY, GV, S, D, H, Hkv, GROUP=group, BLOCK_D=BLOCK_D)
        return GY, GV


def fused_xsa(attn_output: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    """attn_output Y (B,H,S,D), value_states V (B,Hkv,S,D) — Hkv<=H (GQA broadcast in-kernel).
    Returns XSA-corrected output (B,H,S,D)."""
    return FusedXSA.apply(attn_output, value_states)
