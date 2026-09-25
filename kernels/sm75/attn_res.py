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

V is read once from HBM in its native dtype and the output is written once.

ACCUMULATION IS FP32, and that is a THROUGHPUT decision made against measurement, not a
precision policy. fp64 accumulation here is monotonically more accurate (54/54 vs eager) but
7-13x SLOWER -- 1.48 ms vs 18.71 ms at N=5, T=65536 -- because unlike residual_add this kernel
holds a live (BLOCK_N x BLOCK_H) tile across two reductions and a softmax, and the backward
unrolls TILE=4 tokens on top of that. In fp64 the tile doubles to 32 KB per program and occupancy
collapses. End-to-end that cost 12.7% of training throughput (154.7k -> 135.0k tps).

What fp32 gives up, measured in the PRODUCTION layout (block_residual fp32, prefix_sum bf16):
    N=2  0.85 / 0.77   N=3  0.80 / 0.81   N=4  1.12 / 1.06
    N=5  1.04 / 0.93   N=8  0.84 / 0.96          (kernel/eager mean err, spread 1 / 1e4)
So it is better than eager at most N and up to 12% worse at N=4-5, all at the ~2e-8 fp32 floor.
A hybrid (fp32 tile, fp64 score/softmax) was also measured: 6/54 worse instead of 14/54, but
5.70 ms at N=5 -- 4x the cost for a partial gain. Rejected.

The old contract was ACCURACY, not bit-identity: graded by
parity_check/grade_attn_res.py against fp64 truth, the kernel must be at least as close as eager
in every dtype layout. It used to accumulate in fp32 "matching the reference's precision policy
exactly", and under that policy it was measurably WORSE than eager on 14 of 54 configs (worst
1.46x on mean). Matching a reference's precision is not the same as being correct -- the reference
is fp32 because of autocast, not because fp32 is right, and an fp32 training run has no bf16
rounding for the policy to match. fp64 is affordable because this kernel is memory-bound.

MEASURED (54 configs: every block_residual x prefix_sum dtype pair over {bf16,fp32,fp16}, N in
{2,4,8}, and a 1e4 magnitude spread across candidates to reproduce the real embedding-vs-prefix
range). Relative error against fp64 truth, kernel / eager:
    MEAN  54 better, 0 worse, median ratio 0.1181   (~8x more accurate typically)
    MAX   42 better, 12 tie, 0 worse, median 0.0335
Strictly monotone -- never worse than eager on either statistic, in any layout.
And FASTER, fwd+bwd at T=65536 H=512:  N=4  8.56 ms vs 15.25 ms (1.78x)
                                       N=8 16.63 ms vs 29.17 ms (1.75x)

`sq_sum` can optionally be supplied for the block rows: a committed block representative never
changes, so its squared norm is the same at every downstream site and every layer, and recomputing
it 2L+1 times is pure waste. Pass `block_sq_sum` to skip it.
"""
import os

import torch
import triton
import triton.language as tl

__all__ = ["fused_attn_res", "attn_res", "FusedAttnRes", "attn_res_reference", "BlockStore"]

# Tokens per backward program. >1 shrinks the dw partial from (T,H) to (T/TILE,H) AND amortizes
# the (H,) score-weight load, but the loop is UNROLLED, so a large value blows up registers and
# I-cache. Swept at T=16384 H=512 (ms, and the dwp size it buys):
#   TILE    dwp     N=3     N=5    N=11
#      1   33.6   0.204   0.283   0.525
#      2   16.8   0.189   0.265   0.502   <- best at N=11
#      4    8.4   0.184   0.262   0.524   <- best at N=3,5
#     32    1.0   0.234   0.378   1.023   <- unrolling dominates
# The crossover is the per-token tile size, which grows with N, so pick on N. Override with
# BIBO_AR_BWD_TILE.
_BWD_TILE_ENV = os.environ.get("BIBO_AR_BWD_TILE")

# Launch shape. Overridable so it can be SWEPT rather than guessed -- num_warps was hardcoded at 4
# for every N, and at N=2 that is a 2x512 tile spread over 4 warps. Same reasoning as BLOCK_N being
# sized to N rather than a fixed floor. Not @triton.autotune: this repo has been bitten once by
# autotuning on a grid-size dimension (the `S` eval stall), and N/H are fixed by the model anyway,
# so a swept constant is the honest form. Values below are MEASURED, see the table at each site.
_FWD_WARPS = int(os.environ.get("BIBO_AR_FWD_WARPS", "4"))
_FWD_STAGES = int(os.environ.get("BIBO_AR_FWD_STAGES", "0"))     # 0 = triton default
_BWD_WARPS = int(os.environ.get("BIBO_AR_BWD_WARPS", "4"))
_BWD_STAGES = int(os.environ.get("BIBO_AR_BWD_STAGES", "0"))


def _launch_kw(warps, stages):
    kw = {"num_warps": warps}
    if stages:
        kw["num_stages"] = stages
    return kw


def _bwd_tile(N):
    if _BWD_TILE_ENV is not None:
        return int(_BWD_TILE_ENV)
    return 4 if N <= 8 else 2


def _eff_topk(topk, N):
    """TOPK the kernel should actually compile with. `topk >= N` selects every candidate, i.e. it
    IS the dense path -- compiling the selection anyway would burn TOPK-1 reductions per token to
    produce a mask of all ones. At block_size=1 that is layers 0-4 paying for nothing."""
    return int(topk) if 0 < int(topk) < N else 0


@triton.jit
def _topk_sel(score, is_last, mask_n, TOPK: tl.constexpr):
    """Boolean lane mask: the TOPK highest scores, with the prefix-sum row FORCED IN.

    No sort. BLOCK_N is 16 at the largest N this model reaches, so TOPK-1 rounds of
    max-then-erase is cheaper than any ordering network, and it is a register reduction either
    way. `key` is `score` with the prefix-sum lane raised to +inf, which both guarantees its
    selection and makes it consume one of the TOPK slots -- top-6 means the live stream plus the
    best 5 committed blocks, not 6 blocks alongside it.

    WHY THE LIVE STREAM IS NOT ALLOWED TO LOSE: rows 0..N-2 are frozen block representatives, but
    row N-1 is the prefix sum, the only candidate carrying THIS layer's attention output. Dropping
    it does not down-weight attention, it disconnects it -- the layer contributes nothing to the
    depth read and receives no gradient through this path for that token. That is a
    discontinuity in the loss surface, not a soft preference, and it costs one `where` to remove.

    N <= TOPK falls out for free: every real lane gets erased before the rounds run out, so
    `kth` lands at -inf and the mask degenerates to `mask_n`. Exact score ties erase together,
    which can select slightly MORE than TOPK -- the permissive direction, and correct on a tie.
    """
    key = tl.where(is_last, float("inf"), score)      # masked lanes are already -inf
    cur = key
    for _ in tl.static_range(TOPK - 1):
        m = tl.max(cur, axis=0)
        cur = tl.where(cur == m, float("-inf"), cur)
    kth = tl.max(cur, axis=0)
    return mask_n & (key >= kth)


@triton.jit
def _block_offsets(offs_n, O1, O2, O3, O4, O5, O6, O7):
    """Per-row element offset of block n from block 0 (LIST mode). Scalars, not a device table:
    building a table would be a host->device copy per call."""
    z = offs_n.to(tl.int64) * 0
    return (tl.where(offs_n == 1, z + O1, z) + tl.where(offs_n == 2, z + O2, z)
            + tl.where(offs_n == 3, z + O3, z) + tl.where(offs_n == 4, z + O4, z)
            + tl.where(offs_n == 5, z + O5, z) + tl.where(offs_n == 6, z + O6, z)
            + tl.where(offs_n == 7, z + O7, z))


# The block offsets are allocation-address differences: letting Triton specialize them (on
# divisibility by 16 / == 1) recompiled the kernel on almost every call -- 380 ms per board step.
@triton.jit(do_not_specialize=["O1", "O2", "O3", "O4", "O5", "O6", "O7"])
def _attn_res_fwd(
    BR, PS, W, OUT, BSQ,
    T, N, H, eps,
    sbr_t, sbr_n, sbr_h,
    sps_t, sps_h,
    sout_t, sout_h,
    O1, O2, O3, O4, O5, O6, O7,
    HAS_BSQ: tl.constexpr,
    SCORE_MODE: tl.constexpr,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LIST: tl.constexpr = False,
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
    if LIST:
        # blocks are separate (T, H) tensors: row n lives at BR + O_n (element offset from block 0)
        boff = _block_offsets(offs_n, O1, O2, O3, O4, O5, O6, O7)
        br = tl.load(BR + boff[:, None] + t.to(tl.int64) * sbr_t + offs_h[None, :] * sbr_h,
                     mask=(mask_n & (~is_last))[:, None] & mask_h[None, :], other=0.0)
    else:
        br = tl.load(BR + t * sbr_t + offs_n[:, None] * sbr_n + offs_h[None, :] * sbr_h,
                     mask=(mask_n & (~is_last))[:, None] & mask_h[None, :], other=0.0)
    ps = tl.load(PS + t * sps_t + offs_h[None, :] * sps_h,
                 mask=mask_h[None, :], other=0.0)
    # FP64 for the whole mix. The reference is fp32 (`vf = v.float()`), so this is strictly
    # tighter, and it is the same change that made residual_add monotone at no measured cost --
    # both kernels are memory-bound, so the arithmetic hides behind the loads. It matters more
    # here: candidate RMS spans ~0.04 (raw embedding) to ~426 (prefix sum) in the real model,
    # four orders of magnitude summed inside one softmax weighting.
    v = tl.where(is_last[:, None], ps.to(tl.float32), br.to(tl.float32))

    w = tl.load(W + offs_h, mask=mask_h, other=0.0).to(tl.float32)

    # ---- scores: RMS and dot product from the SAME registers, one reduction pass each
    dot = tl.sum(v * w[None, :], axis=1)
    if HAS_BSQ:
        # squared norms of committed blocks are constant -- caller cached them
        bsq = tl.load(BSQ + t * N + offs_n, mask=mask_n & (~is_last), other=0.0).to(tl.float32)
        psq = tl.sum(tl.where(is_last[:, None], v * v, 0.0), axis=1)
        sq = tl.where(is_last, psq, bsq)
    else:
        sq = tl.sum(v * v, axis=1)
    score = dot * tl.rsqrt(sq / H + eps)
    score = tl.where(mask_n, score, float("-inf"))

    # ---- SPARSE DEPTH: keep only the TOPK candidates, renormalize over those. Dropping the
    # losers to -inf is all it takes: softmax sends exp(-inf) to 0 and sigmoid(-inf) to 0, so the
    # normalizer below rebuilds itself over the survivors with no other change. The weights still
    # sum to 1 -- this is a convex combination on a TOPK-face of the simplex, not off it.
    live = mask_n
    if TOPK > 0:
        live = _topk_sel(score, is_last, mask_n, TOPK)
        score = tl.where(live, score, float("-inf"))

    # ---- weights over DEPTH, in registers
    if SCORE_MODE == 0:
        p = tl.exp(score - tl.max(score, axis=0))
        p = p / tl.sum(p, axis=0)
    else:
        # SIGNORM: sigmoid(x_i) / sum_j sigmoid(x_j). Still a convex combination -- the depth
        # weights sum to 1 exactly as under softmax, which is the point: the residual stream
        # stays a weighted average and cannot grow with N. What changes is the map onto the
        # simplex. softmax is SHIFT-INVARIANT, so N scores carry only N-1 usable degrees of
        # freedom; this is shift-sensitive, so all N are live. It also saturates toward UNIFORM
        # when every score is large and positive, where softmax would still separate them.
        # Masked lanes hold -inf, and sigmoid(-inf) is 0, so they drop out of both the numerator
        # and the sum on their own -- the `where` is belt and braces.
        sp = tl.where(live, tl.sigmoid(score), 0.0)
        p = sp / tl.maximum(tl.sum(sp, axis=0), 1e-30)

    # ---- weighted sum, reusing the loaded tile
    out = tl.sum(p[:, None] * v, axis=0)
    tl.store(OUT + t * sout_t + offs_h * sout_h, out.to(OUT.dtype.element_ty), mask=mask_h)


def fused_attn_res(block_residual, prefix_sum, score_weight, eps=1e-6, block_sq_sum=None,
                   score_mode=0, topk=0):
    """Depth-attention mix over [block_residual..., prefix_sum].

    block_residual : (T, N-1, H)  committed block representatives
    prefix_sum     : (T, H)       current within-block accumulation
    score_weight   : (H,)         norm.weight * proj.weight, folded by the caller
    block_sq_sum   : (T, N) or None -- cached sum(v^2) for the block rows (last column unused)
    score_mode     : 0 = softmax (K3 default, every arm before Aug 5), 1 = sigmoid/sum
    topk           : 0 = dense (mix over all N). k > 0 = mix over the k highest-scoring
                     candidates only, prefix_sum always among them. No-op where k >= N.

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
        0, 0, 0, 0, 0, 0, 0,
        HAS_BSQ=block_sq_sum is not None, SCORE_MODE=score_mode,
        TOPK=_eff_topk(topk, N), BLOCK_N=BLOCK_N, BLOCK_H=BLOCK_H,
        **_launch_kw(_FWD_WARPS, _FWD_STAGES),
    )
    return out


@triton.jit(do_not_specialize=["O1", "O2", "O3", "O4", "O5", "O6", "O7",
                               "A1", "A2", "A3", "A4", "A5", "A6", "A7", "FRESH"])
def _attn_res_bwd(
    BR, PS, W, DOUT, DBR, DPS, DWP,
    T, N, H, eps,
    sbr_t, sbr_n, sbr_h,
    sps_t, sps_h,
    sdo_t, sdo_h,
    O1, O2, O3, O4, O5, O6, O7,
    A1, A2, A3, A4, A5, A6, A7, FRESH,
    SCORE_MODE: tl.constexpr,
    TOPK: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_H: tl.constexpr,
    LIST: tl.constexpr = False,
):
    """Recompute the forward from the SAVED INPUTS, then backprop -- one read of V, no fp32
    (T,N,H) tensor stored between forward and backward.

    Each program walks TILE tokens SEQUENTIALLY rather than owning one. A 3D (TILE, N, H) tile
    would spill: at H=512, N=11 a single token is already 22 KB of fp32 registers. The loop keeps
    the per-token tile identical while buying two things: the (H,) score weight is loaded once per
    TILE instead of once per token, and the dw partial shrinks from (T, H) to (T/TILE, H) -- at
    T=65536 that is 134 MB written AND read back per site, for a gradient 512 wide."""
    pid = tl.program_id(0)
    offs_n = tl.arange(0, BLOCK_N)
    offs_h = tl.arange(0, BLOCK_H)
    mask_n = offs_n < N
    mask_h = offs_h < H
    is_last = offs_n == (N - 1)

    # FP64 throughout, same reasoning as the forward. acc_dw in particular is a reduction over the
    # whole TILE of tokens, so it is the one most exposed to fp32 accumulation drift.
    w = tl.load(W + offs_h, mask=mask_h, other=0.0).to(tl.float32)   # once per TILE, not per token
    acc_dw = tl.zeros([BLOCK_H], dtype=tl.float32)
    if LIST:
        boff = _block_offsets(offs_n, O1, O2, O3, O4, O5, O6, O7)
        aoff = _block_offsets(offs_n, A1, A2, A3, A4, A5, A6, A7)
        # bit n of FRESH: block n's accumulator has no gradient yet -> write, else read-add-write
        fresh = ((FRESH >> offs_n) & 1) == 1

    for k in tl.static_range(TILE):
        t = pid * TILE + k
        if t < T:
            bmask = (mask_n & (~is_last))[:, None] & mask_h[None, :]
            if LIST:
                br = tl.load(BR + boff[:, None] + t.to(tl.int64) * sbr_t + offs_h[None, :] * sbr_h,
                             mask=bmask, other=0.0)
            else:
                br = tl.load(BR + t * sbr_t + offs_n[:, None] * sbr_n + offs_h[None, :] * sbr_h,
                             mask=bmask, other=0.0)
            ps = tl.load(PS + t * sps_t + offs_h[None, :] * sps_h,
                         mask=mask_h[None, :], other=0.0)
            v = tl.where(is_last[:, None], ps.to(tl.float32), br.to(tl.float32))
            dout = tl.load(DOUT + t * sdo_t + offs_h * sdo_h, mask=mask_h, other=0.0).to(tl.float32)

            sq = tl.sum(v * v, axis=1)
            dot = tl.sum(v * w[None, :], axis=1)
            inv = tl.rsqrt(sq / H + eps)
            score = tl.where(mask_n, dot * inv, float("-inf"))
            # Selection is PIECEWISE CONSTANT in the scores, so it contributes no gradient of its
            # own -- the derivative is that of the dense mix restricted to the selected face.
            # Re-deriving it here rather than saving a mask keeps backward's contract with forward:
            # both recompute from the same saved inputs, so they cannot disagree about who won.
            live = mask_n
            if TOPK > 0:
                live = _topk_sel(score, is_last, mask_n, TOPK)
                score = tl.where(live, score, float("-inf"))
            # `pref` is the ONLY thing the score mode changes downstream. For softmax
            # dp_i/dx_k = p_k(delta_ik - p_i); for sigmoid/sum it is (s_k(1-s_k)/S)(delta_ik - p_i).
            # The bracket (dp - sum(p*dp)) is identical, so both modes share every line below.
            # Unselected lanes need no special case either: softmax sends exp(-inf) to p=0, and
            # signorm's `live` where sends sp=0, so pref=0 and ds=0 -- dv is exactly zero there.
            if SCORE_MODE == 0:
                p = tl.exp(score - tl.max(score, axis=0))
                p = p / tl.sum(p, axis=0)
                pref = p
            else:
                sp = tl.where(live, tl.sigmoid(score), 0.0)
                ssum = tl.maximum(tl.sum(sp, axis=0), 1e-30)
                p = sp / ssum
                pref = sp * (1.0 - sp) / ssum

            dp = tl.sum(dout[None, :] * v, axis=1)
            ds = pref * (dp - tl.sum(p * dp, axis=0))
            d_dot = ds * inv
            dsq = -0.5 * ds * dot * inv * inv * inv / H

            dv = p[:, None] * dout[None, :] + d_dot[:, None] * w[None, :] + 2.0 * v * dsq[:, None]

            if LIST:
                ap = DBR + aoff[:, None] + t.to(tl.int64) * sbr_t + offs_h[None, :] * sbr_h
                old = tl.load(ap, mask=bmask & (~fresh)[:, None], other=0.0)
                tl.store(ap, old + dv, mask=bmask)
            else:
                tl.store(DBR + t * sbr_t + offs_n[:, None] * sbr_n + offs_h[None, :] * sbr_h,
                         dv.to(DBR.dtype.element_ty), mask=bmask)
            tl.store(DPS + t * sps_t + offs_h * sps_h,
                     tl.sum(tl.where(is_last[:, None], dv, 0.0), axis=0).to(DPS.dtype.element_ty),
                     mask=mask_h)
            acc_dw += tl.sum(d_dot[:, None] * v, axis=0)

    tl.store(DWP + pid * H + offs_h, acc_dw.to(DWP.dtype.element_ty), mask=mask_h)


class FusedAttnRes(torch.autograd.Function):
    """Autograd wrapper. Saves ONLY the inputs -- block_residual (already alive and shared across
    every site of the layer), prefix_sum, and the folded weight -- and recomputes the mix in
    backward. Nothing of shape (T, N, H) is retained, which is the memory the eager path spends."""

    @staticmethod
    def forward(ctx, block_residual, prefix_sum, score_weight, eps, score_mode=0, topk=0):
        out = fused_attn_res(block_residual, prefix_sum, score_weight, eps,
                             score_mode=score_mode, topk=topk)
        ctx.save_for_backward(block_residual, prefix_sum, score_weight)
        ctx.eps = eps
        ctx.score_mode = score_mode
        ctx.topk = topk
        return out

    @staticmethod
    def backward(ctx, dout):
        br, ps, w = ctx.saved_tensors
        T, H = ps.shape
        N = br.shape[1] + 1
        dout = dout.contiguous()
        dbr = torch.empty_like(br)
        dps = torch.empty_like(ps)
        TILE = _bwd_tile(N)
        n_prog = triton.cdiv(T, TILE)
        dwp = torch.empty(n_prog, H, device=ps.device, dtype=torch.float32)
        _attn_res_bwd[(n_prog,)](
            br, ps, w.contiguous().float(), dout, dbr, dps, dwp,
            T, N, H, ctx.eps,
            br.stride(0), br.stride(1), br.stride(2),
            ps.stride(0), ps.stride(1),
            dout.stride(0), dout.stride(1),
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            SCORE_MODE=ctx.score_mode, TOPK=_eff_topk(ctx.topk, N),
            TILE=TILE, BLOCK_N=triton.next_power_of_2(N),
            BLOCK_H=triton.next_power_of_2(H), **_launch_kw(_BWD_WARPS, _BWD_STAGES),
        )
        # Only the cross-token reduction is left, and it is a plain fp32 sum over a (T, H)
        # partial -- no atomics (16384 x 512 of them would dominate the pass) and no downcast.
        return dbr, dps, dwp.sum(0).to(w.dtype), None, None, None


def attn_res(block_residual, prefix_sum, score_weight, eps=1e-6, score_mode=0, topk=0):
    """Differentiable fused AR mix. Drop-in for `apply_attention_residual`.

    score_mode 0 = softmax, 1 = sigmoid(x_i)/sum_j sigmoid(x_j).
    topk       0 = dense; k > 0 mixes only the k best-scoring candidates (prefix_sum forced in).
    """
    return FusedAttnRes.apply(block_residual, prefix_sum, score_weight, eps, score_mode, topk)


def _topk_mask_reference(scores, topk):
    """-inf everything outside the top `topk`, last column forced in. Mirrors `_topk_sel`,
    including its tie behaviour: `<` keeps every candidate equal to the k-th, so an exact tie
    selects slightly more than k rather than breaking it arbitrarily."""
    n = scores.shape[-1]
    if not topk or topk >= n:
        return scores
    key = scores.clone()
    key[..., -1] = float("inf")
    kth = key.topk(topk, dim=-1).values[..., -1:]
    return scores.masked_fill(key < kth, float("-inf"))


def attn_res_reference(block_residual, prefix_sum, score_weight, eps=1e-6, score_mode=0, topk=0):
    """K3's `_apply_attn_res`, verbatim in shape and precision. Numerics target."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = v.float()
    var = vf.pow(2).mean(-1, keepdim=True)
    k = vf * torch.rsqrt(var + eps)
    scores = (k * score_weight.float()).sum(-1)
    scores = _topk_mask_reference(scores, topk)
    if score_mode == 0:
        probs = scores.softmax(-1)
    else:
        # Same 1e-30 floor as the kernel. Not cosmetic: every candidate saturating sigmoid to 0
        # would make this 0/0, and matching the guard is what keeps reference and kernel
        # comparable at the tail instead of one returning NaN and the other a number.
        sp = torch.sigmoid(scores)
        probs = sp / sp.sum(-1, keepdim=True).clamp_min(1e-30)
    return torch.matmul(probs.unsqueeze(1), vf).squeeze(1).to(v.dtype)


# ───────────────────── LIST mode: blocks as separate tensors, grads accumulated in-kernel ─────────────
# The model used to torch.cat each new block onto a (T, n, H) block_residual (a full copy at every
# boundary) and let autograd (a) slice the cat's gradient back apart and (b) SUM each block's
# gradient over every site that read it -- 18 full (T, H) adds per micro-batch at b3 x 10 layers.
# ~30 ms/board step, measured. Here a block is archived once, the mix kernel reads it in place by
# pointer offset, and each site's backward ADDS its block gradients into one fp32 accumulator per
# block (fixed order: the graph's backward order, so deterministic). A tiny autograd node per block
# hands the accumulator back once every site that read it has run -- autograd orders that for free,
# because the sites consume the node's output even though they return no gradient for it.

_MAXB = 8                                     # blocks per store (O1..O7 kernel scalars)


class _Archive(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, store, k):
        ctx.set_materialize_grads(False)
        ctx.store, ctx.k = store, k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        acc = ctx.store.acc[ctx.k]
        ctx.store.acc[ctx.k] = None
        if acc is None:
            return g, None, None
        acc = acc.to(ctx.store.blocks[ctx.k].dtype)
        return (acc if g is None else g + acc), None, None


class _ListMix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, ps, w, eps, score_mode, topk, store, n, *blocks):
        T, H = ps.shape
        N = n + 1
        b0 = blocks[0]
        es = b0.element_size()
        offs = [(b.data_ptr() - b0.data_ptr()) // es for b in blocks[1:]]
        offs += [0] * (_MAXB - 1 - len(offs))
        out_dtype = torch.promote_types(b0.dtype, ps.dtype)
        out = torch.empty(ps.shape, device=ps.device, dtype=out_dtype)
        _attn_res_fwd[(T,)](
            b0, ps, w.contiguous().float(), out, b0, T, N, H, eps,
            b0.stride(0), 0, b0.stride(1), ps.stride(0), ps.stride(1), out.stride(0), out.stride(1),
            *offs, HAS_BSQ=False, SCORE_MODE=score_mode, TOPK=_eff_topk(topk, N),
            BLOCK_N=triton.next_power_of_2(N), BLOCK_H=triton.next_power_of_2(H), LIST=True,
            **_launch_kw(_FWD_WARPS, _FWD_STAGES))
        ctx.save_for_backward(ps, w, *blocks)
        ctx.eps, ctx.score_mode, ctx.topk, ctx.store, ctx.n = eps, score_mode, topk, store, n
        return out

    @staticmethod
    def backward(ctx, dout):
        ps, w, *blocks = ctx.saved_tensors
        store, n = ctx.store, ctx.n
        T, H = ps.shape
        N = n + 1
        dout = dout.contiguous()
        fresh = 0
        for k in range(n):
            if store.acc[k] is None:
                store.acc[k] = torch.empty(T, H, device=ps.device, dtype=torch.float32)
                fresh |= 1 << k
        a0 = store.acc[0]
        b0 = blocks[0]
        es, ea = b0.element_size(), a0.element_size()
        offs = [(b.data_ptr() - b0.data_ptr()) // es for b in blocks[1:]]
        offs += [0] * (_MAXB - 1 - len(offs))
        aoffs = [(store.acc[k].data_ptr() - a0.data_ptr()) // ea for k in range(1, n)]
        aoffs += [0] * (_MAXB - 1 - len(aoffs))
        dps = torch.empty_like(ps)
        TILE = _bwd_tile(N)
        n_prog = triton.cdiv(T, TILE)
        dwp = torch.empty(n_prog, H, device=ps.device, dtype=torch.float32)
        _attn_res_bwd[(n_prog,)](
            b0, ps, w.contiguous().float(), dout, a0, dps, dwp, T, N, H, ctx.eps,
            b0.stride(0), 0, b0.stride(1), ps.stride(0), ps.stride(1), dout.stride(0), dout.stride(1),
            *offs, *aoffs, fresh,
            SCORE_MODE=ctx.score_mode, TOPK=_eff_topk(ctx.topk, N), TILE=TILE,
            BLOCK_N=triton.next_power_of_2(N), BLOCK_H=triton.next_power_of_2(H), LIST=True,
            **_launch_kw(_BWD_WARPS, _BWD_STAGES))
        return (dps, dwp.sum(0).to(w.dtype), None, None, None, None, None) + (None,) * n


class BlockStore:
    """The committed AttnRes blocks of ONE forward pass. archive() a (T, H) block (it is not copied);
    mix() the depth read over every block archived so far plus the live prefix sum."""

    def __init__(self):
        self.blocks, self.refs, self.acc = [], [], []

    def __len__(self):
        return len(self.blocks)

    def archive(self, x):
        assert x.ndim == 2 and x.stride(1) == 1 and x.stride(0) == x.shape[1], "need a contiguous (T, H) block"
        assert len(self.blocks) < _MAXB, f"at most {_MAXB} blocks"
        if self.blocks:
            assert x.dtype == self.blocks[0].dtype and x.shape == self.blocks[0].shape
        k = len(self.blocks)
        self.blocks.append(x)
        self.acc.append(None)
        self.refs.append(_Archive.apply(x, self, k))
        return self

    def mix(self, prefix_sum, score_weight, eps=1e-6, score_mode=0, topk=0):
        n = len(self.blocks)
        assert n > 0
        return _ListMix.apply(prefix_sum, score_weight, eps, score_mode, topk, self, n, *self.refs[:n])
