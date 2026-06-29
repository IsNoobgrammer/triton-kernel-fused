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
    from kernels.sm75.xsa import fused_xsa
    attn_out = fused_xsa(attn_out, value_states)   # Y (B,H,S,D), V (B,Hkv,S,D)

Grad-exact vs the eager rejection (fp16 atol ~1e-3).
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_xsa", "FusedXSA"]


def _cfgs():
    # Tile XBLOCK rows per program (the win over the old one-program-per-row launch). XBLOCK + warps
    # are shape-stable (depend only on D/GROUP, not the per-step token count) -> safe to autotune.
    return [triton.Config({"XBLOCK": xb}, num_warps=w)
            for xb in (1, 2, 4, 8, 16, 32) for w in (2, 4, 8)]


@triton.autotune(configs=_cfgs(), key=["S", "D", "H", "Hkv"])
@triton.jit
def _xsa_fwd_kernel(Y, V, Z, n_rows, S, D, H, Hkv, GROUP: tl.constexpr,
                    BLOCK_D: tl.constexpr, XBLOCK: tl.constexpr):
    # One program handles XBLOCK rows of (B*Hkv*S); the D-reduction is vectorized over the inner
    # contiguous axis (axis=1) — saturates HBM, unlike the old per-row launch. V is read ONCE per
    # row and reused across the GQA group in-register (the structural edge: inductor reads V twice).
    rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)        # [XBLOCK]
    rmask = rows < n_rows
    offs_d = tl.arange(0, BLOCK_D)                                  # [BLOCK_D]
    full = rmask[:, None] & (offs_d < D)[None, :]
    s = rows % S
    t = rows // S
    kv = t % Hkv
    b = t // Hkv
    # V is (B,Hkv,S,D) contiguous -> flat row index == `rows`
    v = tl.load(V + rows[:, None] * D + offs_d[None, :], mask=full, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=1)                                      # [XBLOCK]
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    for j in range(GROUP):
        y_row = (b * H + kv * GROUP + j) * S + s                    # row into Y (B*H*S, D)
        yp = Y + y_row[:, None] * D + offs_d[None, :]
        y = tl.load(yp, mask=full, other=0.0).to(tl.float32)
        coeff = tl.sum(y * v, axis=1) * inv                         # [XBLOCK]
        z = y - coeff[:, None] * v
        tl.store(Z + y_row[:, None] * D + offs_d[None, :], z.to(Z.dtype.element_ty), mask=full)


@triton.autotune(configs=_cfgs(), key=["S", "D", "H", "Hkv"])
@triton.jit
def _xsa_bwd_kernel(GZ, Y, V, GY, GV, n_rows, S, D, H, Hkv, GROUP: tl.constexpr,
                    BLOCK_D: tl.constexpr, XBLOCK: tl.constexpr):
    rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    rmask = rows < n_rows
    offs_d = tl.arange(0, BLOCK_D)
    full = rmask[:, None] & (offs_d < D)[None, :]
    s = rows % S
    t = rows // S
    kv = t % Hkv
    b = t // Hkv
    v = tl.load(V + rows[:, None] * D + offs_d[None, :], mask=full, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=1)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)                         # [XBLOCK]
    gv = tl.zeros((XBLOCK, BLOCK_D), dtype=tl.float32)
    for j in range(GROUP):
        row = (b * H + kv * GROUP + j) * S + s
        bp = row[:, None] * D + offs_d[None, :]
        y = tl.load(Y + bp, mask=full, other=0.0).to(tl.float32)
        gz = tl.load(GZ + bp, mask=full, other=0.0).to(tl.float32)
        dot = tl.sum(y * v, axis=1)                                 # [XBLOCK]
        gzv = tl.sum(gz * v, axis=1)
        coeff = dot * inv
        gy = gz - (gzv * inv)[:, None] * v                          # grad_Y = reject(gz)
        tl.store(GY + bp, gy.to(GY.dtype.element_ty), mask=full)
        gv += (-(gzv * inv))[:, None] * y + (2.0 * dot * gzv * inv * inv)[:, None] * v - coeff[:, None] * gz
    tl.store(GV + rows[:, None] * D + offs_d[None, :], gv.to(GV.dtype.element_ty), mask=full)


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
        n_rows = B * Hkv * S
        grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]),)
        _xsa_fwd_kernel[grid](Y, V, Z, n_rows, S, D, H, Hkv, GROUP=group, BLOCK_D=BLOCK_D)
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
        n_rows = B * Hkv * S
        grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]),)
        _xsa_bwd_kernel[grid](gZ, Y, V, GY, GV, n_rows, S, D, H, Hkv, GROUP=group, BLOCK_D=BLOCK_D)
        return GY, GV


def fused_xsa(attn_output: torch.Tensor, value_states: torch.Tensor) -> torch.Tensor:
    """attn_output Y (B,H,S,D), value_states V (B,Hkv,S,D) — Hkv<=H (GQA broadcast in-kernel).
    Returns XSA-corrected output (B,H,S,D)."""
    return FusedXSA.apply(attn_output, value_states)
