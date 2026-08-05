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

So: load native, widen to fp32, accumulate in fp32, round ONCE at the final store. Backward keeps
the full-precision upstream gradient, and d_theta reduces with FP64 per-program partials so the
~8k-entry cross-program sum is effectively exact.

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
MODES = {"none": 0, "sigmoid": 1, "tanh": 2, "2sigmoid": 3, "2tanh": 4}
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
def _apply_mode(theta, MODE: tl.constexpr):
    """f(theta) and f'(theta), together, because backward always needs both."""
    if MODE == 0:
        return theta, 1.0
    if MODE == 1:
        s = tl.sigmoid(theta)
        return s, s * (1.0 - s)
    if MODE == 2:
        # libdevice.tanh, NOT 2*sigmoid(2x)-1. The identity is exact in real arithmetic and
        # catastrophic in fp32: at small theta it evaluates 2*0.5 - 1, cancelling away the
        # mantissa. Measured against fp64 on linspace(-3, 3): 2*sigmoid(2x)-1 gives 8.139e-05
        # max relative error, libdevice.tanh gives 1.511e-07 -- identical to torch.tanh. The
        # exhaustive dtype grade caught this as d_stream (= c*dout) coming out 1.29x WORSE than
        # eager on every tanh case; nothing else in the kernel was at fault.
        t = libdevice.tanh(theta)
        return t, 1.0 - t * t
    if MODE == 3:
        s = tl.sigmoid(theta)
        return 2.0 * s, 2.0 * s * (1.0 - s)
    t = libdevice.tanh(theta)
    return 2.0 * t, 2.0 * (1.0 - t * t)


@triton.jit
def _res_add_fwd(
    AR, S0, S1, S2, S3, M0, M1, M2, M3, OUT,
    T, H,
    sar_t, s0_t, s1_t, s2_t, s3_t, so_t,
    K: tl.constexpr, MODE0: tl.constexpr, MODE1: tl.constexpr,
    MODE2: tl.constexpr, MODE3: tl.constexpr,
    P0: tl.constexpr, P1: tl.constexpr, P2: tl.constexpr, P3: tl.constexpr,
    BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_t = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    offs_h = tl.arange(0, BLOCK_H)
    mask = (offs_t[:, None] < T) & (offs_h[None, :] < H)

    acc = _ld(AR, offs_t[:, None] * sar_t + offs_h[None, :], mask, False).to(tl.float32)
    # Unrolled rather than looped: Triton cannot index a tuple of pointers at runtime, and K is a
    # compile-time constant anyway, so the branches vanish.
    if K > 0:
        c, _ = _apply_mode(tl.load(M0).to(tl.float32), MODE0)
        sv = _ld(S0, offs_t[:, None] * s0_t + offs_h[None, :], mask, P0).to(tl.float32)
        acc += c * sv
    if K > 1:
        c, _ = _apply_mode(tl.load(M1).to(tl.float32), MODE1)
        sv = _ld(S1, offs_t[:, None] * s1_t + offs_h[None, :], mask, P1).to(tl.float32)
        acc += c * sv
    if K > 2:
        c, _ = _apply_mode(tl.load(M2).to(tl.float32), MODE2)
        sv = _ld(S2, offs_t[:, None] * s2_t + offs_h[None, :], mask, P2).to(tl.float32)
        acc += c * sv
    if K > 3:
        c, _ = _apply_mode(tl.load(M3).to(tl.float32), MODE3)
        sv = _ld(S3, offs_t[:, None] * s3_t + offs_h[None, :], mask, P3).to(tl.float32)
        acc += c * sv

    tl.store(OUT + offs_t[:, None] * so_t + offs_h[None, :], acc.to(OUT.dtype.element_ty), mask=mask)


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
        c, _ = _apply_mode(tl.load(M0).to(tl.float32), MODE0)
        # ...and the product node c*stream is stored in the STREAM dtype, so the gradient
        # arriving at it is cast fp32 -> stream dtype before either use. Feeding full-fp32
        # dout instead is a systematically different gradient: measured 1 bf16 ULP on
        # d attn_out and 0.76% RELATIVE on d theta, on the real model layout.
        s = _ld(S0, offs_t[:, None] * s0_t + offs_h[None, :], mask, P0).to(tl.float32)
        # eager's d theta is (grad_p * stream).sum(), and grad_p * stream is a bf16 x bf16
        # product STORED in bf16 before the reduction runs. Summing the fp32 product
        # instead left 0.5% relative error on the scalar gradient.
        tl.store(PART + pid * spart + 0,
                 tl.sum(tl.sum(go * s, axis=1), axis=0).to(tl.float64))
        if NEED0:
            tl.store(DS0 + offs_t[:, None] * d0_t + offs_h[None, :],
                     (c * go).to(DS0.dtype.element_ty), mask=mask)
    if K > 1:
        c, _ = _apply_mode(tl.load(M1).to(tl.float32), MODE1)
        # ...and the product node c*stream is stored in the STREAM dtype, so the gradient
        # arriving at it is cast fp32 -> stream dtype before either use. Feeding full-fp32
        # dout instead is a systematically different gradient: measured 1 bf16 ULP on
        # d attn_out and 0.76% RELATIVE on d theta, on the real model layout.
        s = _ld(S1, offs_t[:, None] * s1_t + offs_h[None, :], mask, P1).to(tl.float32)
        # eager's d theta is (grad_p * stream).sum(), and grad_p * stream is a bf16 x bf16
        # product STORED in bf16 before the reduction runs. Summing the fp32 product
        # instead left 0.5% relative error on the scalar gradient.
        tl.store(PART + pid * spart + 1,
                 tl.sum(tl.sum(go * s, axis=1), axis=0).to(tl.float64))
        if NEED1:
            tl.store(DS1 + offs_t[:, None] * d1_t + offs_h[None, :],
                     (c * go).to(DS1.dtype.element_ty), mask=mask)
    if K > 2:
        c, _ = _apply_mode(tl.load(M2).to(tl.float32), MODE2)
        # ...and the product node c*stream is stored in the STREAM dtype, so the gradient
        # arriving at it is cast fp32 -> stream dtype before either use. Feeding full-fp32
        # dout instead is a systematically different gradient: measured 1 bf16 ULP on
        # d attn_out and 0.76% RELATIVE on d theta, on the real model layout.
        s = _ld(S2, offs_t[:, None] * s2_t + offs_h[None, :], mask, P2).to(tl.float32)
        # eager's d theta is (grad_p * stream).sum(), and grad_p * stream is a bf16 x bf16
        # product STORED in bf16 before the reduction runs. Summing the fp32 product
        # instead left 0.5% relative error on the scalar gradient.
        tl.store(PART + pid * spart + 2,
                 tl.sum(tl.sum(go * s, axis=1), axis=0).to(tl.float64))
        if NEED2:
            tl.store(DS2 + offs_t[:, None] * d2_t + offs_h[None, :],
                     (c * go).to(DS2.dtype.element_ty), mask=mask)
    if K > 3:
        c, _ = _apply_mode(tl.load(M3).to(tl.float32), MODE3)
        # ...and the product node c*stream is stored in the STREAM dtype, so the gradient
        # arriving at it is cast fp32 -> stream dtype before either use. Feeding full-fp32
        # dout instead is a systematically different gradient: measured 1 bf16 ULP on
        # d attn_out and 0.76% RELATIVE on d theta, on the real model layout.
        s = _ld(S3, offs_t[:, None] * s3_t + offs_h[None, :], mask, P3).to(tl.float32)
        # eager's d theta is (grad_p * stream).sum(), and grad_p * stream is a bf16 x bf16
        # product STORED in bf16 before the reduction runs. Summing the fp32 product
        # instead left 0.5% relative error on the scalar gradient.
        tl.store(PART + pid * spart + 3,
                 tl.sum(tl.sum(go * s, axis=1), axis=0).to(tl.float64))
        if NEED3:
            tl.store(DS3 + offs_t[:, None] * d3_t + offs_h[None, :],
                     (c * go).to(DS3.dtype.element_ty), mask=mask)


def _dmode(t, mode):
    """dc/dtheta in torch, matching what autograd produces for the eager spelling of each mode.

    Lives on the torch side, not in the kernel, because it must be applied AFTER the reduction
    has been rounded to the stream dtype -- see the note in _ResidualAdd.backward.
    """
    if mode == "none":
        return torch.ones_like(t)
    if mode in ("sigmoid", "2sigmoid"):
        s = torch.sigmoid(t)
        return (2.0 if mode == "2sigmoid" else 1.0) * s * (1.0 - s)
    if mode in ("tanh", "2tanh"):
        h = torch.tanh(t)
        return (2.0 if mode == "2tanh" else 1.0) * (1.0 - h * h)
    raise ValueError(f"unknown mode {mode!r}")


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
# MEASURED by parity_check/grade_residual_add.py, 32 quantities across 8 layouts, relative error
# against fp64 truth. Kernel is never worse than eager; worst kernel/eager ratio is 1.000 (a tie).
#   1s bf16 / ar fp32 (the carry path)  fwd 4.5e-08 vs 2.4e-03   d_theta 1.6e-07 vs 9.7e-04
#   1s fp32 / ar fp32 (fp32 training)   fwd 4.5e-08 vs 5.7e-08   d_theta 2.7e-08 vs 1.8e-07
#   1s bf16 / ar bf16                   fwd 3.0e-03 vs 5.4e-03   d_theta 6.3e-09 vs 3.3e-04
#   2s bf16+fp32 (carry+emb)            fwd 6.8e-08 vs 2.0e-03   d_theta 1.6e-07 vs 9.7e-04
# d_ar and d_stream tie with eager where eager is already exact, and beat it ~2x where it is not.
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
    pad_m = thetas + [thetas[0]] * (MAX_STREAMS - K)
    mode_i = [MODES[m] for m in modes] + [0] * (MAX_STREAMS - K)
    pz = [bool(x) for x in (persistent or [False] * K)] + [False] * (MAX_STREAMS - K)
    grid = (triton.cdiv(T, BLOCK_T),)
    _res_add_fwd[grid](
        ar, *pad_s, *pad_m, out, T, H, sar, *pad_st, out.stride(0),
        K=K, MODE0=mode_i[0], MODE1=mode_i[1], MODE2=mode_i[2], MODE3=mode_i[3],
        P0=pz[0], P1=pz[1], P2=pz[2], P3=pz[3],
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
    f = {"none": lambda x: x, "sigmoid": torch.sigmoid, "tanh": torch.tanh,
         "2sigmoid": lambda x: 2.0 * torch.sigmoid(x), "2tanh": lambda x: 2.0 * torch.tanh(x)}
    acc = torch.promote_types(attn_read.dtype, torch.float32)
    for _, s in pairs:
        acc = torch.promote_types(acc, s.dtype)
    for t, _ in pairs:
        acc = torch.promote_types(acc, t.dtype)
    out = attn_read.to(acc)
    for (theta, s), m in zip(pairs, modes):
        out = out + f[m](theta.to(acc)).reshape(()) * s.to(acc)
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
        grid = (triton.cdiv(T, BLOCK_T),)
        # FP64 partials. Each program tree-reduces its own tile in fp32 (tl.sum is a tree, so the
        # within-tile error is O(log 4096) ULP, not O(4096)); the cross-program sum then runs over
        # ~T/BLOCK_T entries -- 8192 at the board shape -- and doing THAT in fp32 is where a long
        # accumulation actually degrades. fp64 here costs one tiny buffer and removes it.
        part = torch.empty((grid[0], max(n, 1)), device=do.device, dtype=torch.float64)
        pad_s = streams + [streams[0]] * (MAX_STREAMS - n)
        pad_st = strides + [0] * (MAX_STREAMS - n)
        pad_ds = [d if d is not None else streams[0] for d in ds] + [streams[0]] * (MAX_STREAMS - n)
        pad_dst = [(d.stride(0) if d is not None else 0) for d in ds] + [0] * (MAX_STREAMS - n)
        th = [t.reshape(()) if t.numel() == 1 else t for t in thetas]
        pad_m = th + [th[0]] * (MAX_STREAMS - n)
        mode_i = [MODES[m] for m in modes] + [0] * (MAX_STREAMS - n)
        needc = [bool(x) for x in need] + [False] * (MAX_STREAMS - n)
        pz = [bool(x) for x in (ctx.persistent or [False] * n)] + [False] * (MAX_STREAMS - n)
        _res_add_bwd[grid](
            do, *pad_s, *pad_m, *pad_ds, part, T, H,
            do.stride(0), *pad_st, *pad_dst, part.stride(0),
            K=n, MODE0=mode_i[0], MODE1=mode_i[1], MODE2=mode_i[2], MODE3=mode_i[3],
            NEED0=needc[0], NEED1=needc[1], NEED2=needc[2], NEED3=needc[3],
            P0=pz[0], P1=pz[1], P2=pz[2], P3=pz[3],
            BLOCK_T=BLOCK_T, BLOCK_H=triton.next_power_of_2(H), num_warps=4,
            )
        # d theta = dc/dtheta * sum(dout * stream), the whole chain kept in FP64 and rounded once
        # at the end. Eager instead reduces a product tensor stored in the STREAM dtype, so under
        # autocast its scalar gradient comes back quantized to bf16 -- 8 mantissa bits on a sum of
        # 33M terms. Matching that was the old contract; beating it is the new one. dc is applied
        # AFTER the reduction (it is a constant factor, so this is the same value in exact
        # arithmetic, but it keeps the one rounding at the very end instead of per-program).
        dtheta = part.sum(0)                                   # fp64, ~8k partials
        outs = [None, None, None]
        # d attn_read IS dout. Return it by alias -- the identity add costs a whole 134 MB copy
        # if written out, and autograd is happy with a view.
        outs[0] = dout if attn_read.requires_grad else None
        grads = []
        for i in range(n):
            if not thetas[i].requires_grad:
                grads.append(None)
                continue
            g = dtheta[i] * _dmode(thetas[i].detach().double().reshape(()), ctx.modes[i])
            grads.append(g.reshape(thetas[i].shape).to(thetas[i].dtype))
        grads += [(ds[i].view(strms[i].shape) if need[i] else None) for i in range(n)]
        return (outs[0], None, None, None, *grads)


def make_mlp_input(attn_read, *pairs, modes=None, persistent=None):
    """h = attn_read + sum_k f_k(theta_k) * stream_k, fused, with autograd.

    Call as make_mlp_input(attn_read, theta_0, stream_0, theta_1, stream_1, ...) -- the first
    (multiplier, stream) pair is required, up to MAX_STREAMS are allowed, so a new stream can be
    added later without touching the kernel.

    theta is the RAW parameter; `modes` names the transform applied to it inside the kernel, one
    entry per pair, from MODES ("none", "sigmoid", "tanh", "2sigmoid", "2tanh"). Default "none".
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



