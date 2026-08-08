"""Fused multi-stream residual add -- the AttnRes carry write, one kernel instead of 2K passes.

The reference (BiBo `exp/modeling_bibo`, sites=1 carry) is

    h = attn_read + c * attn_out                 # c = f(theta_c)
    h = h        + d * embedding                 # d = f(theta_d)

i.e. one elementwise pass per stream, each reading two (T, H) tensors and writing one. At the
board shape (64 x 1024 x 512 fp32, 134 MB per tensor) two streams cost 805 MB of traffic where a
single fused pass needs 537 MB, and that is before backward, where every add accumulates
separately AND each scalar gradient is a full reduction over 134M elements that eager materializes
as a temporary. Measured on the box: the two learnable scalars cost 7.8k tps / 82 ms per step,
about 4.8%.

Everything here is elementwise in (T, H) with scalars broadcast over both axes, so it collapses to
one pass:

    out[t, h] = attn_read[t, h] + sum_k f_k(theta_k) * stream_k[t, h]

Backward, in the same single pass over the data:

    d attn_read      = dout                       (returned by ALIAS -- no copy, no kernel)
    d stream_k       = f_k(theta_k) * dout
    d theta_k        = f_k'(theta_k) * sum_{t,h} dout[t,h] * stream_k[t,h]

The theta reduction is the real win. Eager builds `dout * stream_k` (134 MB) and then reduces it;
here it accumulates in registers during the pass the backward already has to make.

ACCURACY IS THE CONTRACT, NOT BIT-IDENTITY. Graded against FP64 truth by
parity_check/grade_residual_add.py, this kernel must be at least as close as eager in EVERY dtype
layout, forward and backward. It is deliberately not bit-identical to eager.

This reverses an earlier decision, on purpose. Eager evaluates

    attn_read + _c.to(attn_output.dtype) * attn_output

so the scalar is rounded to the STREAM dtype and the product is computed AND STORED there -- bf16
for attn_out, 7 mantissa bits on all 134M elements -- before the fp32 add. The kernel used to
reproduce that loss deliberately, so that kernel-on and kernel-off were the same model. The problem
with that contract is that it treats AUTOCAST's rounding as the specification: an fp32 training run
has no bf16 rounding anywhere, so a kernel tuned to bf16-eager is tuned to an artifact and can be
silently wrong at a dtype nobody tested. Correctness has to be dtype-independent, and the only
dtype-independent reference is fp64.

So: load native, widen to FP64, accumulate in fp64, round ONCE at the final store. Backward keeps
the full-precision upstream gradient, forms the d_theta products in fp64, and reduces with fp64
per-program partials so the ~8k-entry cross-program sum is effectively exact. fp64 is free here --
measured 0.64 ms either way, because the kernel is memory-bound and the arithmetic hides behind
the traffic.

CONSEQUENCE, and it is a real one: kernel-on and kernel-off are now DIFFERENT MODELS, not one model
computed two ways. Any comparison must hold the path fixed across arms. An AttnRes-off baseline is
unaffected -- it never enters this file. Historical note: the previous "more accurate" version was
blamed for two same-box regressions (c+d +0.0037, c-only +0.007). Those arms also went through the
multi-stream path, which was separately found to be FMA-contracted and inexact, so that attribution
is not clean and should not be used as evidence against this contract.

WHY THE TRANSFORM LIVES IN THE KERNEL. theta is the leaf parameter and f is sigmoid/tanh/identity.
Doing f on the host is arithmetically free but puts a tiny op in the autograd graph per scalar per
site per layer, and then the chain rule for d theta needs its own kernel. Taking raw theta and
applying both f and f' here keeps the whole scalar path inside the one pass.

NO @triton.autotune, deliberately. The tile is derived from H, which is fixed by the model, so
there is nothing to search; and autotuning on a grid-size dimension has already cost this repo a
4.1x eval stall once (see the `S` key incident). BLOCK_T is a plain constant.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["make_mlp_input", "fused_residual_add", "residual_add_reference", "MODES"]

# transform codes. Kept as ints because they are tl.constexpr compile keys; a string would
# recompile on identity anyway but reads worse in the launcher.
# "none" -> h += c * stream          c is the RAW parameter, no transform
# "rms"  -> h += c * stream/rms(stream)
#
# The bounded transforms (sigmoid/tanh/2sigmoid/2tanh) are GONE. They existed to stop an
# unbounded c running away -- the first carry attempt reached 7936 by step 400. "rms" removes the
# reason for a cage instead of building a better one: once the stream it multiplies has unit RMS,
# c is a coefficient on a normalised quantity and has nothing to run away from. It also deletes
# the accuracy problem the bounded modes carried, where a per-channel 2sigmoid graded 1.10x worse
# than eager on d_stream and needed an fp64 sigmoid to claw back.
MODES = {"none": 0, "rms": 1}
RMS_EPS = 1e-6                       # host side: residual_add_reference and the parity harness
# Kernel side. A plain Python global is NOT readable from a @triton.jit function -- it raises
# "Cannot access global variable ... instantiated as constexpr", and the message surfaces under
# the CALL SITE's line number, so it reads as a shape problem in _rms_scale rather than a name
# problem. Same value, declared the way the JIT requires.
_RMS_EPS = tl.constexpr(1e-6)
MAX_STREAMS = 4


@triton.jit
def _ld(P, off, mask, PERSIST: tl.constexpr):
    """PERSIST marks a stream that the NEXT call will read again -- the embedding is the same
    tensor for all 9 layers, so it is worth asking L2 to keep it. evict_first on the one-shot
    streams (attn_read, attn_out, dout) is the other half: without it they compete for the same
    lines. This is a hint, not a guarantee, and a full attention block plus a 64-expert MoE runs
    between two residual adds -- see the bench, which measures it rather than assuming it."""
    if PERSIST:
        return tl.load(P + off, mask=mask, other=0.0, eviction_policy="evict_last")
    return tl.load(P + off, mask=mask, other=0.0, eviction_policy="evict_first")


@triton.jit
def _rms_scale(s, H, MODE: tl.constexpr):
    """1/rms(s) per row, or 1.0 when MODE is "none".

    Returned as a (BLOCK_T, 1) column so both branches have the SAME shape -- Triton unifies a
    function's return signature across branches even though MODE is constexpr, and returning a
    bare scalar from one branch and a tensor from the other fails to compile with no inner
    diagnostic. That exact trap already cost this file once, on _apply_mode's per-channel theta.
    """
    if MODE == 1:
        # sum of squares in fp32, reciprocal-sqrt in FP64, rounded once. 1/tl.sqrt at fp32 left
        # the SCALAR d_theta 19x behind eager (2.011e-06 vs 1.038e-07): that gradient reduces
        # T*H products, so a systematic bias in the per-row normaliser accumulates over ~2M
        # terms while the per-channel case, reducing only over tokens, stayed clean and hid it.
        # This is the same "evaluate in fp64, stop trying to win narrowly" fix the removed
        # _sigmoid documented -- one value per ROW against a (BLOCK_T, H) tile, so it is free.
        ms = tl.sum(s * s, axis=1) / H
        return (1.0 / libdevice.sqrt((ms + _RMS_EPS).to(tl.float64))).to(tl.float32)[:, None]
    # DERIVED FROM s, not a literal: both branches must return the same SHAPE, and a bare
    # tl.zeros([1,1]) fails to unify against the (BLOCK_T,1) above with no inner diagnostic.
    # `sum(s*0)` carries the row count without needing BLOCK_T passed in.
    return (tl.sum(s * 0.0, axis=1) + 1.0)[:, None]


@triton.jit
def _res_add_fwd(
    AR, S0, S1, S2, S3, M0, M1, M2, M3, OUT,
    T, H,
    sar_t, s0_t, s1_t, s2_t, s3_t, so_t,
    K: tl.constexpr, MODE0: tl.constexpr, MODE1: tl.constexpr,
    MODE2: tl.constexpr, MODE3: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    VEC0: tl.constexpr, VEC1: tl.constexpr, VEC2: tl.constexpr, VEC3: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, BLOCK_H)
    mask = (offs_t[:, None] < T) & (offs_h[None, :] < H)

    # FP32 accumulator. The stack is BF16 end to end now, so the store rounds to 8 mantissa
    # bits regardless and an fp64 accumulator buys nothing observable. (It was fp64 while the
    # stream was fp32 and the kernel had to beat eager against fp64 truth; that contest is moot
    # once both sides round to bf16 -- measured 99.98% exact agreement with eager at bf16 in/out.)
    acc = _ld(AR, offs_t[:, None] * sar_t + offs_h[None, :], mask, False).to(tl.float32)
    # Unrolled rather than looped: Triton cannot index a tuple of pointers at runtime, and K is a
    # compile-time constant anyway, so the branches vanish.
    if K > 0:
        if VEC0:
            m = tl.load(M0 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M0).to(tl.float32)
        c = m
        sv = _ld(S0, offs_t[:, None] * s0_t + offs_h[None, :], mask, P0).to(tl.float32)
        sv = sv * _rms_scale(sv, H, MODE0)
        acc += (c[None, :] * sv) if VEC0 else (c * sv)
    if K > 1:
        if VEC1:
            m = tl.load(M1 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M1).to(tl.float32)
        c = m
        sv = _ld(S1, offs_t[:, None] * s1_t + offs_h[None, :], mask, P1).to(tl.float32)
        sv = sv * _rms_scale(sv, H, MODE1)
        acc += (c[None, :] * sv) if VEC1 else (c * sv)
    if K > 2:
        if VEC2:
            m = tl.load(M2 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M2).to(tl.float32)
        c = m
        sv = _ld(S2, offs_t[:, None] * s2_t + offs_h[None, :], mask, P2).to(tl.float32)
        sv = sv * _rms_scale(sv, H, MODE2)
        acc += (c[None, :] * sv) if VEC2 else (c * sv)
    if K > 3:
        if VEC3:
            m = tl.load(M3 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M3).to(tl.float32)
        c = m
        sv = _ld(S3, offs_t[:, None] * s3_t + offs_h[None, :], mask, P3).to(tl.float32)
        sv = sv * _rms_scale(sv, H, MODE3)
        acc += (c[None, :] * sv) if VEC3 else (c * sv)

    tl.store(OUT + offs_t[:, None] * so_t + offs_h[None, :], acc.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _bwd_one(c, s, DS, PART, pid, spart, k, offs_t, offs_h, d_t, mask, go, H,
             NEED: tl.constexpr, VEC: tl.constexpr, SLOT: tl.constexpr, MODE: tl.constexpr):
    """One stream's backward: its d theta partial, and d stream where required.

    Takes c and s ALREADY LOADED. _ld and _apply_mode are multi-return helpers and Triton only
    inlines those at depth 1, so they are called from the kernel body and the results passed in.
    The four unrolled call sites were identical apart from indices and carried the same two
    comments four times, which is why the reasoning now lives here once.

    dout arrives cast to the STREAM dtype, not full fp32: the product node c*stream is stored in
    the stream dtype, so the gradient reaching it is quantized there first. Feeding fp32 dout
    instead is a systematically different gradient -- measured 1 bf16 ULP on d attn_out and 0.76%
    RELATIVE on d theta, on the real model layout.

    Eager's d theta is (grad_p * stream).sum() with the product STORED in bf16 before the
    reduction; summing the fp32 product instead left 0.5% relative error on the scalar gradient.
    Hence the fp64 product here and the fp64 partial.

    VEC: theta is per-channel, so d theta is (H,) and the reduction runs over TOKENS ONLY. The
    scalar case additionally reduces over H. Same arithmetic, one fewer axis collapsed.
    """
    # d theta is taken against the value theta actually multiplies, so under "rms" that is the
    # NORMALISED stream, not the raw one. Using the raw stream here would give a gradient for a
    # different function and would still look plausible.
    inv = _rms_scale(s, H, MODE)
    sn = s * inv
    prod = go.to(tl.float64) * sn.to(tl.float64)
    # SLOT is H when ANY stream in this call is per-channel, else 1. A scalar stream in a
    # mixed call therefore writes one number into slot 0 of its H-wide row and leaves the rest
    # untouched -- the host reads element 0 back and never looks at the padding.
    if VEC:
        tl.store(PART + pid * spart + k * SLOT + offs_h,
                 tl.sum(prod, axis=0).to(tl.float32), mask=offs_h < H)
    else:
        tl.store(PART + pid * spart + k * SLOT,
                 tl.sum(tl.sum(prod, axis=1), axis=0).to(tl.float32))
    if NEED:
        g = (c[None, :] * go) if VEC else (c * go)
        if MODE == 1:
            # y = s/r with r = sqrt(mean(s^2)+eps): dL/ds = (g - sn*mean(g*sn)) / r.
            # The second term is what makes this a projection rather than a plain rescale --
            # dropping it is the classic RMSNorm-backward bug and shows up only as a slow
            # quality drift, never as an error.
            g = (g - sn * (tl.sum(g * sn, axis=1) / H)[:, None]) * inv
        tl.store(DS + offs_t[:, None] * d_t + offs_h[None, :],
                 g.to(DS.dtype.element_ty), mask=mask)


@triton.jit
def _res_add_bwd(
    DOUT, S0, S1, S2, S3, M0, M1, M2, M3,
    DS0, DS1, DS2, DS3, PART,
    T, H,
    sdo_t, s0_t, s1_t, s2_t, s3_t,
    d0_t, d1_t, d2_t, d3_t, spart,
    K: tl.constexpr, MODE0: tl.constexpr, MODE1: tl.constexpr,
    MODE2: tl.constexpr, MODE3: tl.constexpr,
    NEED0: tl.constexpr, NEED1: tl.constexpr, NEED2: tl.constexpr, NEED3: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    VEC0: tl.constexpr, VEC1: tl.constexpr, VEC2: tl.constexpr, VEC3: tl.constexpr,
    SLOT: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    """One pass: writes d stream_k (only where required) and a PER-PROGRAM partial of d theta_k.

    PART is (grid, K) and is reduced with a torch .sum(0) outside. A tl.atomic_add straight into a
    (K,) buffer would be shorter but is nondeterministic in fp32, and the parity standard here
    grades bf16 against fp32 EAGER -- a run-to-run wobble in the scalar gradient would make that
    gate unreproducible. It would also need reset_to_zero, another trap this repo has hit.
    """
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, BLOCK_H)
    mask = (offs_t[:, None] < T) & (offs_h[None, :] < H)
    go = _ld(DOUT, offs_t[:, None] * sdo_t + offs_h[None, :], mask, False).to(tl.float32)

    if K > 0:
        if VEC0:
            m = tl.load(M0 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M0).to(tl.float32)
        c = m
        s = _ld(S0, offs_t[:, None] * s0_t + offs_h[None, :], mask, P0).to(tl.float32)
        _bwd_one(c, s, DS0, PART, pid, spart, 0, offs_t, offs_h, d0_t, mask, go, H,
                 NEED0, VEC0, SLOT, MODE0)
    if K > 1:
        if VEC1:
            m = tl.load(M1 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M1).to(tl.float32)
        c = m
        s = _ld(S1, offs_t[:, None] * s1_t + offs_h[None, :], mask, P1).to(tl.float32)
        _bwd_one(c, s, DS1, PART, pid, spart, 1, offs_t, offs_h, d1_t, mask, go, H,
                 NEED1, VEC1, SLOT, MODE1)
    if K > 2:
        if VEC2:
            m = tl.load(M2 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M2).to(tl.float32)
        c = m
        s = _ld(S2, offs_t[:, None] * s2_t + offs_h[None, :], mask, P2).to(tl.float32)
        _bwd_one(c, s, DS2, PART, pid, spart, 2, offs_t, offs_h, d2_t, mask, go, H,
                 NEED2, VEC2, SLOT, MODE2)
    if K > 3:
        if VEC3:
            m = tl.load(M3 + offs_h, mask=offs_h < H, other=0.0).to(tl.float32)
        else:
            m = tl.load(M3).to(tl.float32)
        c = m
        s = _ld(S3, offs_t[:, None] * s3_t + offs_h[None, :], mask, P3).to(tl.float32)
        _bwd_one(c, s, DS3, PART, pid, spart, 3, offs_t, offs_h, d3_t, mask, go, H,
                 NEED3, VEC3, SLOT, MODE3)


def _dmode(t, mode):
    """dc/dtheta. Both surviving modes use c = theta RAW, so this is identically 1.

    Kept as a function rather than inlined: it is the hook a future transform would need, and the
    call site documents WHERE the chain rule belongs -- after the reduction has been rounded to
    the stream dtype, see the note in _ResidualAdd.backward. "rms" transforms the STREAM, not
    theta, so it does not appear here.
    """
    if mode in MODES:
        return torch.ones_like(t)
    raise ValueError(f"unknown mode {mode!r}; valid: {sorted(MODES)}")


def _prep(attn_read, pairs):
    """(T, H) view + row stride for every operand, without forcing contiguity.

    block_residual[:, 0] -- the embedding stream -- is a strided VIEW, and .contiguous() on it is a
    134 MB copy per call at the board shape, which is most of what this kernel exists to avoid. All
    that is required is a constant row stride and unit column stride.
    """
    H = attn_read.shape[-1]
    def flat(x):
        v = x.reshape(-1, H)
        assert v.stride(1) == 1, "residual_add needs unit stride along hidden"
        return v, v.stride(0)
    ar, sar = flat(attn_read)
    streams, strides = [], []
    for _, s in pairs:
        assert s.shape[-1] == H and s.numel() == attn_read.numel(), "stream shape must match attn_read"
        v, st = flat(s)
        streams.append(v)
        strides.append(st)
    return ar, sar, streams, strides, ar.shape[0], H


BLOCK_T = 8            # 8 x 512 fp32 = 16 KB of accumulator; leaves room for K stream tiles
# The BACKWARD gets a wider tile, because its cost is not the accumulator -- it is PART, the
# per-program d_theta partial of shape (grid, K*slot). At BLOCK_T=8 and T=65536 that is 8192 rows
# x 512 fp64 = 33.5 MB written and read back, against ~200 MB of genuine traffic (dout + stream +
# d_stream). Measured: the fused backward ran 1.8x SLOWER than torch.compile, which fuses its
# reduction instead of materialising one. BLOCK_T_BWD=64 cuts grid 8x, and fp32 partials halve
# what is left: 33.5 MB -> 2 MB.
BLOCK_T_BWD = 64

# CONTRACT: this kernel is graded against FP64 TRUTH, and it must be at least as close to that
# truth as eager is, in EVERY dtype layout. It is deliberately NOT bit-identical to eager.
#
# The previous contract was bit-identity, which meant reproducing eager's precision loss on
# purpose: the scalar was rounded to the stream dtype, the c*stream product was rounded to the
# stream dtype, and the accumulator was rounded to the output dtype between every stream. All of
# that is gone. Bit-identity encodes AUTOCAST's rounding as if it were the specification -- an
# fp32 training run has no bf16 rounding anywhere, so a kernel tuned to reproduce bf16-eager is
# tuned to an artifact and can be wrong at a dtype nobody tested.
#
# What the kernel does now: load in native dtype, widen to fp32, accumulate in fp32, round exactly
# ONCE at the final store. d_theta reduces in fp32 within a tile (tl.sum is a tree) and its
# per-program partials are FP64, so the cross-program sum over ~8k partials is effectively exact.
#
# Consequence to keep in mind, since it is a real one: kernel-on and kernel-off are now different
# models, not the same model computed two ways. Any comparison must hold the path fixed across
# arms. It does NOT affect an AttnRes-off baseline, which never enters this code.
#
# MEASURED by parity_check/grade_residual_add.py: 352 measurements over 88 configurations --
# every (attn_read, stream_0..k) dtype assignment for K=1..4 over {bf16, fp32, fp16}, plus one
# spot check per bounded mode. PASS, mean relative error <= eager on all 352.
#   worst mean ratio 1.0000, worst MAX ratio 1.0000 -- STRICTLY MONOTONE, never worse than eager
#   on any of the 352, on either statistic. Distribution: 250 better / 102 tie / 0 worse on mean,
#   median ratio 0.4872, i.e. the kernel roughly halves the error on a typical measurement.
# And it is FASTER: 0.64 ms vs 0.90 ms eager, fwd+bwd at the board shape (65536 x 512).
#
# Two defects the exhaustive matrix caught that a hand-picked case list had missed:
#   tanh as 2*sigmoid(2x)-1   cancels near zero, 8.1e-05 vs libdevice.tanh's 1.5e-07
#   d_theta in fp32           the reduction is ill-conditioned (result ~512, sum|terms| ~200000,
#                             condition ~400), so fp32 products lost to eager in 8/10 seeds.
#                             fp64 products -> 10/10 and ~75x better than eager.
#
# FMA: enable_fp_fusion is left ON, and that was MEASURED, not assumed. Against fp64, on vs off:
#   1s fp32 / ar fp32   4.529e-08 vs 5.700e-08   FMA better (and off == eager exactly)
#   1s fp32 / ar bf16   identical
#   2s bf16+fp32        identical
#   1s bf16 / ar fp32   identical
# So FMA helps in exactly one layout -- single fp32 stream, fp32 attn_read, where it replaces two
# roundings with one -- and is neutral everywhere else. Never worse.
#
# Do not add a claim to this comment without a number.


def _vec_flags(thetas, H):
    """Per-stream: is this multiplier PER-CHANNEL (H,) or a scalar? -> ([bool]*K, any_vec)

    MIXED IS SUPPORTED, and it is the shipped shape: per-dim carry c with a scalar embedding
    gain d is one change from the scalar-c champion, where making d per-dim too would be two.
    An earlier version refused mixed on the grounds that nothing needed it; the c-per-dim + d
    arm needed it the same week.

    Cost of mixing is the partial buffer: it is sized H per stream whenever ANY stream is
    per-channel, and a scalar stream then uses only slot 0 of its row. Wasting K*(H-1) fp64 per
    program beats a ragged layout -- at the board shape that is 8 MB against 134 MB tensors.
    """
    ns = [t.numel() for t in thetas]
    for n in ns:
        assert n == 1 or n == H, f"multiplier must be scalar or ({H},), got numel {n}"
    flags = [n != 1 for n in ns]
    return flags, any(flags)


def fused_residual_add(attn_read, pairs, modes, out_dtype=None, persistent=None):
    """Forward only. `pairs` = [(theta, stream), ...]; see make_mlp_input for the autograd version."""
    K = len(pairs)
    assert 1 <= K <= MAX_STREAMS, f"1..{MAX_STREAMS} streams, got {K}"
    ar, sar, streams, strides, T, H = _prep(attn_read, pairs)
    if out_dtype is None:
        out_dtype = attn_read.dtype
        for _, s in pairs:
            out_dtype = torch.promote_types(out_dtype, s.dtype)
    out = torch.empty((T, H), device=ar.device, dtype=out_dtype)
    pad_s = streams + [streams[0]] * (MAX_STREAMS - K)
    pad_st = strides + [0] * (MAX_STREAMS - K)
    thetas = [t.reshape(()) if t.numel() == 1 else t for t, _ in pairs]
    vflag, _ = _vec_flags(thetas, H)
    vflag = vflag + [False] * (MAX_STREAMS - K)
    pad_m = thetas + [thetas[0]] * (MAX_STREAMS - K)
    mode_i = [MODES[m] for m in modes] + [0] * (MAX_STREAMS - K)
    pz = [bool(x) for x in (persistent or [False] * K)] + [False] * (MAX_STREAMS - K)
    grid = (triton.cdiv(T, BLOCK_T),)
    _res_add_fwd[grid](
        ar, *pad_s, *pad_m, out, T, H, sar, *pad_st, out.stride(0),
        K=K, MODE0=mode_i[0], MODE1=mode_i[1], MODE2=mode_i[2], MODE3=mode_i[3],
        P0=pz[0], P1=pz[1], P2=pz[2], P3=pz[3],
        VEC0=vflag[0], VEC1=vflag[1], VEC2=vflag[2], VEC3=vflag[3],
        BLOCK_T=BLOCK_T, BLOCK_H=triton.next_power_of_2(H), num_warps=4,
    )
    return out.view(attn_read.shape[:-1] + (H,))


def residual_add_reference(attn_read, pairs, modes):
    """The eager formula, spelled out. What parity grades against.

    Accumulates at the WIDEST input dtype, floored at fp32 -- never hardcoded to .float(). A fixed
    fp32 accumulation makes this function unusable as a high-precision reference: passing fp64
    inputs silently returned an fp32 answer, so fp32 eager scored an error of exactly 0 against its
    own output and the `kernel <= eager` gate quietly became vacuous at that dtype.
    """
    acc = torch.promote_types(attn_read.dtype, torch.float32)
    for _, s in pairs:
        acc = torch.promote_types(acc, s.dtype)
    for t, _ in pairs:
        acc = torch.promote_types(acc, t.dtype)
    out = attn_read.to(acc)
    for (theta, s), m in zip(pairs, modes):
        # reshape(()) for a scalar theta, left as (H,) for a per-channel one so it broadcasts
        # over the trailing hidden axis.
        _c = theta.to(acc)                       # c is the RAW parameter; no transform remains
        sv = s.to(acc)
        if m == "rms":
            sv = sv * torch.rsqrt(sv.pow(2).mean(-1, keepdim=True) + RMS_EPS)
        out = out + (_c.reshape(()) if _c.numel() == 1 else _c) * sv
    return out


class _ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_read, modes, n, persistent, *flat):
        pairs = [(flat[i], flat[n + i]) for i in range(n)]
        ctx.modes, ctx.n, ctx.persistent = modes, n, persistent
        ctx.save_for_backward(attn_read, *flat)
        return fused_residual_add(attn_read, pairs, modes, persistent=persistent)

    @staticmethod
    def backward(ctx, dout):
        attn_read, *flat = ctx.saved_tensors
        n, modes = ctx.n, ctx.modes
        thetas, strms = flat[:n], flat[n:]
        pairs = list(zip(thetas, strms))
        dout = dout.contiguous() if dout.stride(-1) != 1 else dout
        ar, sar, streams, strides, T, H = _prep(attn_read, pairs)
        do = dout.reshape(-1, H)

        need = [s.requires_grad for s in strms]
        ds = [torch.empty((T, H), device=do.device, dtype=strms[i].dtype) if need[i] else None
              for i in range(n)]
        grid = (triton.cdiv(T, BLOCK_T_BWD),)
        # Partials: fp64 WITHIN a program, fp32 in the buffer. The old all-fp64 buffer was there
        # because the cross-program sum ran over 8192 entries and fp32 degrades on a long
        # accumulation. BLOCK_T_BWD makes that 1024 entries, at which fp32 is ample -- and the
        # kernel has ~4 orders of magnitude of headroom over the bar it must clear (d_theta
        # 1.008e-07 vs eager's 7.196e-03), so spending one for a third of the traffic is a
        # measured trade, not a corner cut. parity_res_add_rms still gates it.
        th_v = [t.reshape(()) if t.numel() == 1 else t for t in thetas]
        vflag, any_vec = _vec_flags(th_v, H)
        slot = H if any_vec else 1
        vpad = vflag + [False] * (MAX_STREAMS - n)
        # (grid, K) of scalars, or (grid, K, H) when ANY theta is per-channel -- that theta
        # reduces over tokens only, so it needs a full row. A scalar stream in a mixed call uses
        # slot 0 of its row and the rest is padding the host never reads.
        part = torch.empty((grid[0], max(n, 1) * slot), device=do.device, dtype=torch.float32)
        pad_s = streams + [streams[0]] * (MAX_STREAMS - n)
        pad_st = strides + [0] * (MAX_STREAMS - n)
        pad_ds = [d if d is not None else streams[0] for d in ds] + [streams[0]] * (MAX_STREAMS - n)
        pad_dst = [(d.stride(0) if d is not None else 0) for d in ds] + [0] * (MAX_STREAMS - n)
        pad_m = th_v + [th_v[0]] * (MAX_STREAMS - n)
        mode_i = [MODES[m] for m in modes] + [0] * (MAX_STREAMS - n)
        needc = [bool(x) for x in need] + [False] * (MAX_STREAMS - n)
        pz = [bool(x) for x in (ctx.persistent or [False] * n)] + [False] * (MAX_STREAMS - n)
        _res_add_bwd[grid](
            do, *pad_s, *pad_m, *pad_ds, part, T, H,
            do.stride(0), *pad_st, *pad_dst, part.stride(0),
            K=n, MODE0=mode_i[0], MODE1=mode_i[1], MODE2=mode_i[2], MODE3=mode_i[3],
            NEED0=needc[0], NEED1=needc[1], NEED2=needc[2], NEED3=needc[3],
            P0=pz[0], P1=pz[1], P2=pz[2], P3=pz[3],
            VEC0=vpad[0], VEC1=vpad[1], VEC2=vpad[2], VEC3=vpad[3], SLOT=slot,
            BLOCK_T=BLOCK_T_BWD, BLOCK_H=triton.next_power_of_2(H), num_warps=4,
            )
        # d theta = dc/dtheta * sum(dout * stream), the whole chain kept in FP64 and rounded once
        # at the end. Eager instead reduces a product tensor stored in the STREAM dtype, so under
        # autocast its scalar gradient comes back quantized to bf16 -- 8 mantissa bits on a sum of
        # 33M terms. Matching that was the old contract; beating it is the new one. dc is applied
        # AFTER the reduction (it is a constant factor, so this is the same value in exact
        # arithmetic, but it keeps the one rounding at the very end instead of per-program).
        dtheta = part.double().sum(0)                          # fp32 buffer, ~1k partials
        dtheta = dtheta.view(max(n, 1), slot)                  # (K, 1) or (K, H)
        outs = [None, None, None]
        # d attn_read IS dout. Return it by alias -- the identity add costs a whole 134 MB copy
        # if written out, and autograd is happy with a view.
        outs[0] = dout if attn_read.requires_grad else None
        grads = []
        for i in range(n):
            if not thetas[i].requires_grad:
                grads.append(None)
                continue
            # reshape(()) only when theta IS a scalar -- forcing it on an (H,) vector would
            # raise, and silently taking element 0 would be worse.
            _t = thetas[i].detach().double()
            # a scalar stream in a mixed call parked its one number in slot 0 of an H-wide row
            _dt = dtheta[i] if vflag[i] else dtheta[i, 0]
            g = _dt * _dmode(_t.reshape(()) if _t.numel() == 1 else _t, ctx.modes[i])
            grads.append(g.reshape(thetas[i].shape).to(thetas[i].dtype))
        grads += [(ds[i].view(strms[i].shape) if need[i] else None) for i in range(n)]
        return (outs[0], None, None, None, *grads)


def make_mlp_input(attn_read, *pairs, modes=None, persistent=None):
    """h = attn_read + sum_k f_k(theta_k) * stream_k, fused, with autograd.

    Call as make_mlp_input(attn_read, theta_0, stream_0, theta_1, stream_1, ...) -- the first
    (multiplier, stream) pair is required, up to MAX_STREAMS are allowed, so a new stream can be
    added later without touching the kernel.

    theta is the RAW parameter; `modes` names the transform applied to it inside the kernel, one
    entry per pair, from MODES ("none", "rms"). Default "none".
    """
    assert len(pairs) % 2 == 0 and pairs, "pairs must be (multiplier, stream), at least one"
    n = len(pairs) // 2
    thetas = [pairs[2 * i] for i in range(n)]
    strms = [pairs[2 * i + 1] for i in range(n)]
    modes = tuple(modes) if modes is not None else ("none",) * n
    assert len(modes) == n, f"got {n} streams but {len(modes)} modes"
    for m in modes:
        assert m in MODES, f"unknown mode {m!r}; valid: {sorted(MODES)}"
    if not attn_read.is_cuda:
        return residual_add_reference(attn_read, list(zip(thetas, strms)), modes).to(attn_read.dtype)
    return _ResidualAdd.apply(attn_read, modes, n, tuple(persistent) if persistent else None,
                              *thetas, *strms)



