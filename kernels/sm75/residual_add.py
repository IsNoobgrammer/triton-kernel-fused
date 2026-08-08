"""Fused AttnRes carry write:  h = attn_read + c * stream            ("none")
                              h = attn_read + c * stream/rms(stream) ("rms")

SINGLE STREAM. The multi-stream machinery (up to 4 pairs, the embedding term, per-stream
persistence hints) is gone: the embedding term was retired, and carrying four unrolled slots cost
register pressure and constexpr specialisations on every call that only ever passes one.

Structure follows what torch.compile's Inductor generates for the same expression, because a
straight measurement said Inductor's code was faster than the previous hand-written version on
BOTH halves (fwd 0.0969 vs 0.1475 ms, bwd 0.1024 vs 0.2940 ms at T=65536, H=512). Three things
were responsible, and all three are adopted here:

  PERSISTENT PER-ROW    H=512 fits one program's registers, so the row reduction never spills to
                        a second pass. The old kernel tiled (BLOCK_T=8, H) and reduced across
                        tiles, which is the wrong decomposition for a 512-wide row.

  SAVED rstd            forward stores 1/rms per row; backward loads it instead of recomputing
                        sum(s^2) over H. Inductor does exactly this (`tl.store(out_ptr1 + x0)`
                        then `tl.load(in_ptr3 + x0)`).

  REGISTER-ACCUMULATED  d_theta accumulates in registers along a grid-stride loop, so the partial
  d_theta               buffer is (NPROG, H) with NPROG ~ SM count. The old kernel materialised
                        (T/8, H) = 8192 rows and reduced it with a separate torch .sum(0), which
                        cost an extra kernel and inflated the main one.

The accuracy contract is unchanged and still gated by parity_check/parity_res_add_rms.py: beat
EAGER against fp64 truth on forward, d_attn_read, d_stream and d_theta.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

__all__ = ["make_mlp_input", "residual_add_reference", "MODES", "RMS_EPS"]

# "none" -> c * stream           c is the RAW parameter; no transform
# "rms"  -> c * stream/rms(stream)
#
# The bounded transforms (sigmoid/tanh/2sigmoid/2tanh) are gone. They existed to stop an unbounded
# c running away -- the first carry attempt reached 7936 by step 400. "rms" removes the REASON for
# a cage instead of building a better one: c multiplies a unit-RMS quantity.
MODES = {"none": 0, "rms": 1}
RMS_EPS = 1e-6
_RMS_EPS = tl.constexpr(1e-6)          # a plain global is unreadable from @triton.jit

_NPROG = 1024                          # backward grid: ~SM count x a few, so PART stays ~2 MB

# Autotuned, not hardcoded. The first rewrite copied XBLOCK=8/num_warps=4 from the old tiled
# kernel and ran 38% behind Inductor on the forward for that reason alone -- Inductor's
# persistent_reduction heuristic picks both per shape.
#
# key=("H",) ONLY. H is a true shape constant; T is the GRID dimension, and keying an autotuner
# on a grid dim has already cost this repo a 4.1x eval stall when a stale cache entry was reused
# across sequence lengths.
_MAX_XBLOCK = 16
_CFGS = [triton.Config({"XBLOCK": b}, num_warps=w, num_stages=st)
         for b in (1, 2, 4, 8, _MAX_XBLOCK) for w in (2, 4, 8) for st in (1, 2)]


def _even(T, H, rb):
    """Can EVERY autotune config run unmasked? XBLOCK is chosen by the autotuner, so this must
    hold for the largest of them, not just the one that happens to win."""
    return H == rb and T % _MAX_XBLOCK == 0


@triton.autotune(configs=_CFGS, key=("H",))
@triton.jit
def _fwd_kernel(AR, S, M, OUT, RSTD, T, H: tl.constexpr, RBLOCK: tl.constexpr,
                VEC: tl.constexpr, MODE: tl.constexpr, EVEN: tl.constexpr,
                XBLOCK: tl.constexpr):
    xs = (tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK))[:, None]
    r = tl.arange(0, RBLOCK)[None, :]
    # EVEN: H == RBLOCK and T divides XBLOCK, so nothing can fall off either edge and every
    # access is unmasked. Inductor emits `tl.load(..., None)` for exactly this reason; carrying
    # predication on the hot path was 38% of the forward gap.
    if EVEN:
        mask = None
        rm = None
    else:
        mask = (xs < T) & (r < H)
        rm = r < H
    s = tl.load(S + xs * H + r, mask=mask, other=0.0).to(tl.float32)
    ar = tl.load(AR + xs * H + r, mask=mask, other=0.0).to(tl.float32)
    if VEC:
        c = tl.load(M + r, mask=rm, other=0.0, eviction_policy="evict_last").to(tl.float32)
    else:
        c = tl.load(M).to(tl.float32)
    if MODE == 1:
        ms = tl.sum(s * s, axis=1)[:, None] / H
        rstd = libdevice.rsqrt(ms + _RMS_EPS)
        # saved: the backward must not recompute this
        tl.store(RSTD + xs, rstd, mask=None if EVEN else (xs < T))
        s = s * rstd
    tl.store(OUT + xs * H + r, (ar + c * s).to(OUT.dtype.element_ty), mask=mask)


@triton.autotune(configs=_CFGS, key=("H",))
@triton.jit
def _bwd_kernel(DO, S, M, RSTD, DS, PART, T, H: tl.constexpr,
                RBLOCK: tl.constexpr, VEC: tl.constexpr, MODE: tl.constexpr,
                NEED_DS: tl.constexpr, NPROG: tl.constexpr, EVEN: tl.constexpr,
                XBLOCK: tl.constexpr):
    """Grid-stride over rows; d_theta lives in registers for the whole walk.

    d_theta is accumulated PER CHANNEL even when theta is a scalar -- the scalar is then one more
    host-side .sum(). Branching the accumulator on VEC instead would double the kernel's shapes
    for a reduction that costs nothing at H=512.
    """
    pid = tl.program_id(0)
    r = tl.arange(0, RBLOCK)[None, :]
    rm = None if EVEN else r < H
    if VEC:
        c = tl.load(M + r, mask=rm, other=0.0, eviction_policy="evict_last").to(tl.float32)
    else:
        c = tl.load(M).to(tl.float32)
    acc = tl.zeros([RBLOCK], tl.float32)[None, :]

    for x0 in tl.range(pid * XBLOCK, T, NPROG * XBLOCK):
        xs = (x0 + tl.arange(0, XBLOCK))[:, None]
        if EVEN:
            xm = None
            mask = None
        else:
            xm = xs < T
            mask = xm & (r < H)
        go = tl.load(DO + xs * H + r, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(S + xs * H + r, mask=mask, other=0.0).to(tl.float32)
        if MODE == 1:
            rstd = tl.load(RSTD + xs, mask=xm, other=0.0)
            sn = s * rstd
        else:
            sn = s
        # d theta is taken against what c actually multiplies -- the NORMALISED stream under "rms".
        # No tl.where when EVEN: out-of-range lanes cannot exist, and the select was on the hot path.
        if EVEN:
            acc += tl.sum(go * sn, axis=0)[None, :]
        else:
            acc += tl.sum(tl.where(mask, go * sn, 0.0), axis=0)[None, :]
        if NEED_DS:
            g = c * go
            if MODE == 1:
                # y = s/r: dL/ds = (g - sn*mean(g*sn))/r. The projection term is not optional --
                # dropping it gives a gradient that is close, never NaN, and surfaces only as a
                # slow quality drift nothing attributes back to this kernel.
                g = (g - sn * (tl.sum(g * sn, axis=1)[:, None] / H)) * rstd
            tl.store(DS + xs * H + r, g.to(DS.dtype.element_ty), mask=mask)

    tl.store(PART + pid * H + r, acc, mask=rm)   # rm is None when EVEN


def residual_add_reference(attn_read, theta, stream, mode="none"):
    """The eager formula, spelled out. What parity grades against.

    Accumulates at the WIDEST input dtype, floored at fp32 -- never hardcoded to .float(), so an
    fp64 call returns fp64 and can serve as ground truth. A fixed fp32 accumulation once made this
    score an error of exactly 0 against its own output and the gate silently became vacuous.
    """
    acc = torch.promote_types(attn_read.dtype, torch.float32)
    acc = torch.promote_types(torch.promote_types(acc, stream.dtype), theta.dtype)
    sv = stream.to(acc)
    if mode == "rms":
        sv = sv * torch.rsqrt(sv.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    c = theta.to(acc)
    return attn_read.to(acc) + (c.reshape(()) if c.numel() == 1 else c) * sv


def _flat(x, H):
    v = x.reshape(-1, H)
    assert v.stride(1) == 1, "residual_add needs unit stride along hidden"
    return v.contiguous() if v.stride(0) != H else v


class _ResidualAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, attn_read, theta, stream, mode):
        H = attn_read.shape[-1]
        ar, sv = _flat(attn_read, H), _flat(stream, H)
        T = ar.shape[0]
        vec = theta.numel() > 1
        out = torch.empty_like(ar)
        rstd = (torch.empty(T, device=ar.device, dtype=torch.float32)
                if mode == "rms" else ar)                      # dummy ptr when unused
        rb = triton.next_power_of_2(H)
        _fwd_kernel[lambda meta: (triton.cdiv(T, meta["XBLOCK"]),)](
            ar, sv, theta, out, rstd, T, H=H, RBLOCK=rb,
            VEC=vec, MODE=MODES[mode], EVEN=_even(T, H, rb))
        ctx.save_for_backward(sv, theta, rstd)
        ctx.mode, ctx.vec, ctx.H, ctx.T = mode, vec, H, T
        ctx.shape = attn_read.shape
        return out.view(attn_read.shape)

    @staticmethod
    def backward(ctx, dout):
        sv, theta, rstd = ctx.saved_tensors
        H, T, mode, vec = ctx.H, ctx.T, ctx.mode, ctx.vec
        do = _flat(dout, H)
        need_ds = ctx.needs_input_grad[2]
        ds = torch.empty_like(sv) if need_ds else sv           # dummy ptr when unused
        part = torch.empty((_NPROG, H), device=do.device, dtype=torch.float32)
        rb = triton.next_power_of_2(H)
        _bwd_kernel[(_NPROG,)](
            do, sv, theta, rstd, ds, part, T, H=H, RBLOCK=rb,
            VEC=vec, MODE=MODES[mode], NEED_DS=need_ds, NPROG=_NPROG,
            EVEN=_even(T, H, rb))
        # (NPROG, H) -> theta's shape. NPROG is ~1k, so fp32 partials are ample here; the old
        # kernel reduced 8192 rows and needed fp64 to stay accurate over that many terms.
        d_theta = None
        if ctx.needs_input_grad[1]:
            col = part.double().sum(0)
            d_theta = (col.sum().reshape(theta.shape) if not vec
                       else col.reshape(theta.shape)).to(theta.dtype)
        # d attn_read IS dout -- returned by alias, since writing the identity out would cost a
        # full 134 MB copy at the board shape for nothing.
        d_ar = dout if ctx.needs_input_grad[0] else None
        return d_ar, d_theta, (ds.view(ctx.shape) if need_ds else None), None


def make_mlp_input(attn_read, *pairs, modes=None, persistent=None):
    """h = attn_read + c * f(stream), fused, with autograd.

    Call as make_mlp_input(attn_read, theta, stream, modes=("rms",)). The *pairs spelling is kept
    so existing call sites do not have to change shape, but ONE pair is the contract now -- the
    embedding stream is retired, and silently accepting a second one would compute a different
    model than the caller wrote.
    """
    assert len(pairs) == 2, (f"single stream only: expected (theta, stream), got {len(pairs)} "
                             f"positional args. The multi-stream/embedding path was removed.")
    theta, stream = pairs
    mode = (modes[0] if modes else "none")
    assert mode in MODES, f"unknown mode {mode!r}; valid: {sorted(MODES)}"
    if not attn_read.is_cuda:
        return residual_add_reference(attn_read, theta, stream, mode).to(attn_read.dtype)
    return _ResidualAdd.apply(attn_read, theta, stream, mode)
