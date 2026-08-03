"""Fused Attention-Residual (AR) mix -- Kimi K3 Block AttnRes, one kernel, one read of V.

The reference (K3 `_apply_attn_res`, and BiBo `exp/modeling_bibo.apply_attention_residual`) is

    V      = cat(block_residual, prefix_sum[:, None, :])    # (T, N, H)
    Vf     = V.float()
    var    = Vf.pow(2).mean(-1, keepdim=True)
    K      = Vf * rsqrt(var + eps)
    scores = (K * w).sum(-1)                                # w = norm.weight * proj.weight
    probs  = softmax(scores, -1)
    out    = (probs @ Vf)

which in eager touches V roughly six times in fp32: the `cat`, the `.float()` copy, the squared
copy for the variance, the normalized copy, the `K*w` product, and the final matmul -- and holds
several (T, N, H) fp32 tensors alive for backward, at every residual site of every layer. That is
what costs 41% throughput and OOMs a 95 GB card at block_size=1.

Everything above is a reduction over H followed by a reduction over N, so it all fits in one pass:

  * `cat` is avoided -- the kernel indexes block_residual and prefix_sum in place, selecting the
    last row from prefix_sum with a `where`;
  * the RMS is computed from the SAME registers the dot product reads, so normalizing costs
    nothing extra and no normalized copy is ever built;
  * the softmax runs across N in registers;
  * the weighted sum reuses the already-loaded tile.

V is read once from HBM in its native dtype and the output is written once. Accumulation is fp32
throughout, matching the reference's precision policy exactly.

`sq_sum` can optionally be supplied for the block rows: a committed block representative never
changes, so its squared norm is the same at every downstream site and every layer, and recomputing
it 2L+1 times is pure waste. Pass `block_sq_sum` to skip it.
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_attn_res", "attn_res", "FusedAttnRes", "attn_res_reference"]


@triton.jit
def _attn_res_fwd(
    BR, PS, W, OUT, BSQ,
    T, N, H, eps,
    sbr_t, sbr_n, sbr_h,
    sps_t, sps_h,
    sout_t, sout_h,
    HAS_BSQ: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    t = tl.program_id(0)
    if t >= T:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    mask_n = offs_n < N
    mask_h = offs_h < H
    is_last = offs_n == (N - 1)                       # the prefix_sum row

    # ---- load V once. Rows [0, N-1) come from block_residual, row N-1 from prefix_sum. The
    # `cat` in the reference exists only to put them in one tensor; here a select does it.
    br = tl.load(BR + t * sbr_t + offs_n[:, None] * sbr_n + offs_h[None, :] * sbr_h,
                 mask=(mask_n & (~is_last))[:, None] & mask_h[None, :], other=0.0)
    ps = tl.load(PS + t * sps_t + offs_h[None, :] * sps_h,
                 mask=mask_h[None, :], other=0.0)
    v = tl.where(is_last[:, None], ps.to(tl.float32), br.to(tl.float32))

    w = tl.load(W + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    # ---- scores: RMS and dot product from the SAME registers, one reduction pass each
    dot = tl.sum(v * w[None, :], axis=1)
    if HAS_BSQ:
        # squared norms of committed blocks are constant -- caller cached them
        bsq = tl.load(BSQ + t * N + offs_n, mask=mask_n & (~is_last), other=0.0)
        psq = tl.sum(tl.where(is_last[:, None], v * v, 0.0), axis=1)
        sq = tl.where(is_last, psq, bsq)
    else:
        sq = tl.sum(v * v, axis=1)
    score = dot * tl.rsqrt(sq / H + eps)
    score = tl.where(mask_n, score, float("-inf"))

    # ---- softmax over DEPTH, in registers
    p = tl.exp(score - tl.max(score, axis=0))
    p = p / tl.sum(p, axis=0)

    # ---- weighted sum, reusing the loaded tile
    out = tl.sum(p[:, None] * v, axis=0)
    tl.store(OUT + t * sout_t + offs_h * sout_h, out, mask=mask_h)


def fused_attn_res(block_residual, prefix_sum, score_weight, eps=1e-6, block_sq_sum=None):
    """Depth-attention mix over [block_residual..., prefix_sum].

    block_residual : (T, N-1, H)  committed block representatives
    prefix_sum     : (T, H)       current within-block accumulation
    score_weight   : (H,)         norm.weight * proj.weight, folded by the caller
    block_sq_sum   : (T, N) or None -- cached sum(v^2) for the block rows (last column unused)

    Returns (T, H) in prefix_sum's dtype. Forward only; see FusedAttnRes for autograd.
    """
    assert prefix_sum.ndim == 2 and block_residual.ndim == 3
    T, H = prefix_sum.shape
    N = block_residual.shape[1] + 1
    assert block_residual.shape[0] == T and block_residual.shape[2] == H
    # MATCH torch.cat's type promotion. Inside the model block_residual is fp32 (it is seeded
    # from the fp32 embedding) while prefix_sum is bf16, and the reference does
    # `cat(br, ps).float() ... .to(values.dtype)` -- i.e. it returns the PROMOTED dtype, fp32.
    # Returning empty_like(prefix_sum) instead silently handed the model a bf16 tensor where
    # eager gave fp32, which changed every downstream layernorm and residual add.
    out_dtype = torch.promote_types(block_residual.dtype, prefix_sum.dtype)
    out = torch.empty(prefix_sum.shape, device=prefix_sum.device, dtype=out_dtype)
    # Size the N tile to N, not to a fixed floor: at N=2 a BLOCK_N of 16 wastes 8x the lanes on
    # masked padding, and the profile showed exactly that (315 GB/s at N=2 vs 805 at N=8).
    BLOCK_N = triton.next_power_of_2(N)
    BLOCK_H = triton.next_power_of_2(H)
    _attn_res_fwd[(T,)](
        block_residual, prefix_sum, score_weight.contiguous().float(), out,
        block_sq_sum if block_sq_sum is not None else block_residual,
        T, N, H, eps,
        block_residual.stride(0), block_residual.stride(1), block_residual.stride(2),
        prefix_sum.stride(0), prefix_sum.stride(1),
        out.stride(0), out.stride(1),
        HAS_BSQ=block_sq_sum is not None,
        BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        num_warps=4,
    )
    return out


@triton.jit
def _attn_res_bwd(
    BR, PS, W, DOUT, DBR, DPS, DWP,
    T, N, H, eps,
    sbr_t, sbr_n, sbr_h,
    sps_t, sps_h,
    sdo_t, sdo_h,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
):
    """Recompute the forward from the SAVED INPUTS, then backprop -- one read of V, no fp32
    (T,N,H) tensor is ever stored between forward and backward. That is the whole point: the
    eager path keeps several of them alive per site, which is where AttnRes's memory went."""
    t = tl.program_id(0)
    if t >= T:
        return

    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    mask_n = offs_n < N
    mask_h = offs_h < H
    is_last = offs_n == (N - 1)

    br = tl.load(BR + t * sbr_t + offs_n[:, None] * sbr_n + offs_h[None, :] * sbr_h,
                 mask=(mask_n & (~is_last))[:, None] & mask_h[None, :], other=0.0)
    ps = tl.load(PS + t * sps_t + offs_h[None, :] * sps_h, mask=mask_h[None, :], other=0.0)
    v = tl.where(is_last[:, None], ps.to(tl.float32), br.to(tl.float32))
    w = tl.load(W + offs_h, mask=mask_h, other=0.0).to(tl.float32)
    dout = tl.load(DOUT + t * sdo_t + offs_h * sdo_h, mask=mask_h, other=0.0).to(tl.float32)

    # ---- recompute forward
    sq = tl.sum(v * v, axis=1)
    dot = tl.sum(v * w[None, :], axis=1)
    inv = tl.rsqrt(sq / H + eps)
    score = tl.where(mask_n, dot * inv, float("-inf"))
    p = tl.exp(score - tl.max(score, axis=0))
    p = p / tl.sum(p, axis=0)

    # ---- backward
    dp = tl.sum(dout[None, :] * v, axis=1)                 # d out / d p_i
    ds = p * (dp - tl.sum(p * dp, axis=0))                 # softmax jacobian
    d_dot = ds * inv
    dsq = -0.5 * ds * dot * inv * inv * inv / H            # through rsqrt(sq/H + eps)

    dv = p[:, None] * dout[None, :] + d_dot[:, None] * w[None, :] + 2.0 * v * dsq[:, None]

    tl.store(DBR + t * sbr_t + offs_n[:, None] * sbr_n + offs_h[None, :] * sbr_h, dv,
             mask=(mask_n & (~is_last))[:, None] & mask_h[None, :])
    dps = tl.sum(tl.where(is_last[:, None], dv, 0.0), axis=0)
    tl.store(DPS + t * sps_t + offs_h * sps_h, dps, mask=mask_h)

    # dw's per-token contribution, reduced HERE in fp32 from the same v and d_dot the rest of the
    # pass uses. Doing it outside meant casting the fp32 d_dot down to the input dtype to matmul
    # against block_residual, and in bf16 that made d_w measurably worse than eager (4.4e-3 vs
    # 3.4e-3 against the fp32 reference) while d_block and d_prefix matched exactly.
    tl.store(DWP + t * H + offs_h, tl.sum(d_dot[:, None] * v, axis=0), mask=mask_h)


class FusedAttnRes(torch.autograd.Function):
    """Autograd wrapper. Saves ONLY the inputs -- block_residual (already alive and shared across
    every site of the layer), prefix_sum, and the folded weight -- and recomputes the mix in
    backward. Nothing of shape (T, N, H) is retained, which is the memory the eager path spends."""

    @staticmethod
    def forward(ctx, block_residual, prefix_sum, score_weight, eps):
        out = fused_attn_res(block_residual, prefix_sum, score_weight, eps)
        ctx.save_for_backward(block_residual, prefix_sum, score_weight)
        ctx.eps = eps
        return out

    @staticmethod
    def backward(ctx, dout):
        br, ps, w = ctx.saved_tensors
        T, H = ps.shape
        N = br.shape[1] + 1
        dout = dout.contiguous()
        dbr = torch.empty_like(br)
        dps = torch.empty_like(ps)
        dwp = torch.empty(T, H, device=ps.device, dtype=torch.float32)
        _attn_res_bwd[(T,)](
            br, ps, w.contiguous().float(), dout, dbr, dps, dwp,
            T, N, H, ctx.eps,
            br.stride(0), br.stride(1), br.stride(2),
            ps.stride(0), ps.stride(1),
            dout.stride(0), dout.stride(1),
            BLOCK_N=triton.next_power_of_2(N), BLOCK_H=triton.next_power_of_2(H),
            num_warps=4,
        )
        # Only the cross-token reduction is left, and it is a plain fp32 sum over a (T, H)
        # partial -- no atomics (16384 x 512 of them would dominate the pass) and no downcast.
        return dbr, dps, dwp.sum(0).to(w.dtype), None


def attn_res(block_residual, prefix_sum, score_weight, eps=1e-6):
    """Differentiable fused AR mix. Drop-in for `apply_attention_residual`."""
    return FusedAttnRes.apply(block_residual, prefix_sum, score_weight, eps)


def attn_res_reference(block_residual, prefix_sum, score_weight, eps=1e-6):
    """K3's `_apply_attn_res`, verbatim in shape and precision. Numerics target."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = v.float()
    var = vf.pow(2).mean(-1, keepdim=True)
    k = vf * torch.rsqrt(var + eps)
    scores = (k * score_weight.float()).sum(-1)
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(v.dtype)
