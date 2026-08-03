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
        t = (2.0 * tl.sigmoid(2.0 * theta)) - 1.0      # tanh; tl.math.tanh is not on every arch
        return t, 1.0 - t * t
    if MODE == 3:
        s = tl.sigmoid(theta)
        return 2.0 * s, 2.0 * s * (1.0 - s)
    s = tl.sigmoid(2.0 * theta)
    t = (2.0 * s) - 1.0
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
        acc += c * _ld(S0, offs_t[:, None] * s0_t + offs_h[None, :], mask, P0).to(tl.float32)
    if K > 1:
        c, _ = _apply_mode(tl.load(M1).to(tl.float32), MODE1)
        acc += c * _ld(S1, offs_t[:, None] * s1_t + offs_h[None, :], mask, P1).to(tl.float32)
    if K > 2:
        c, _ = _apply_mode(tl.load(M2).to(tl.float32), MODE2)
        acc += c * _ld(S2, offs_t[:, None] * s2_t + offs_h[None, :], mask, P2).to(tl.float32)
    if K > 3:
        c, _ = _apply_mode(tl.load(M3).to(tl.float32), MODE3)
        acc += c * _ld(S3, offs_t[:, None] * s3_t + offs_h[None, :], mask, P3).to(tl.float32)

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
        c, dc = _apply_mode(tl.load(M0).to(tl.float32), MODE0)
        s = _ld(S0, offs_t[:, None] * s0_t + offs_h[None, :], mask, P0).to(tl.float32)
        tl.store(PART + pid * spart + 0, tl.sum(tl.sum(go * s, axis=1), axis=0) * dc)
        if NEED0:
            tl.store(DS0 + offs_t[:, None] * d0_t + offs_h[None, :],
                     (c * go).to(DS0.dtype.element_ty), mask=mask)
    if K > 1:
        c, dc = _apply_mode(tl.load(M1).to(tl.float32), MODE1)
        s = _ld(S1, offs_t[:, None] * s1_t + offs_h[None, :], mask, P1).to(tl.float32)
        tl.store(PART + pid * spart + 1, tl.sum(tl.sum(go * s, axis=1), axis=0) * dc)
        if NEED1:
            tl.store(DS1 + offs_t[:, None] * d1_t + offs_h[None, :],
                     (c * go).to(DS1.dtype.element_ty), mask=mask)
    if K > 2:
        c, dc = _apply_mode(tl.load(M2).to(tl.float32), MODE2)
        s = _ld(S2, offs_t[:, None] * s2_t + offs_h[None, :], mask, P2).to(tl.float32)
        tl.store(PART + pid * spart + 2, tl.sum(tl.sum(go * s, axis=1), axis=0) * dc)
        if NEED2:
            tl.store(DS2 + offs_t[:, None] * d2_t + offs_h[None, :],
                     (c * go).to(DS2.dtype.element_ty), mask=mask)
    if K > 3:
        c, dc = _apply_mode(tl.load(M3).to(tl.float32), MODE3)
        s = _ld(S3, offs_t[:, None] * s3_t + offs_h[None, :], mask, P3).to(tl.float32)
        tl.store(PART + pid * spart + 3, tl.sum(tl.sum(go * s, axis=1), axis=0) * dc)
        if NEED3:
            tl.store(DS3 + offs_t[:, None] * d3_t + offs_h[None, :],
                     (c * go).to(DS3.dtype.element_ty), mask=mask)


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
    """The eager formula, spelled out. This is what parity grades against, in fp32."""
    f = {"none": lambda x: x, "sigmoid": torch.sigmoid, "tanh": torch.tanh,
         "2sigmoid": lambda x: 2.0 * torch.sigmoid(x), "2tanh": lambda x: 2.0 * torch.tanh(x)}
    out = attn_read.float()
    for (theta, s), m in zip(pairs, modes):
        out = out + f[m](theta.float()).reshape(*([1] * (s.ndim - 1)), -1).squeeze() * s.float()
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
        part = torch.empty((grid[0], max(n, 1)), device=do.device, dtype=torch.float32)
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
        dtheta = part.sum(0)
        outs = [None, None, None]
        # d attn_read IS dout. Return it by alias -- the identity add costs a whole 134 MB copy
        # if written out, and autograd is happy with a view.
        outs[0] = dout if attn_read.requires_grad else None
        grads = [(dtheta[i].reshape(thetas[i].shape) if thetas[i].requires_grad else None)
                 for i in range(n)]
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
