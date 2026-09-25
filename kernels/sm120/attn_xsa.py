"""Fused attention for BiBo: [qk RMSNorm] -> [RoPE] -> causal / sliding-window GQA attention ->
[XSA], one kernel family, deterministic backward (no atomics anywhere).

    z = attn_xsa(q, k, v, scale=..., window=None, xsa=True, alpha=None,
                 q_norm_w=None, k_norm_w=None, q_scale=1.0, k_scale=1.0, eps=1e-6,
                 cos=None, sin=None)

q (B, H, S, D), k / v (B, Hkv, S, D): bf16, ANY strides with a unit last dim -- the (B, S, H, D)
projection views transposed to (B, H, S, D) are read in place, and z / dq / dk / dv come back in
the SAME layout as their input, so no .contiguous() copy happens on either side of attention.
  qk norm   q_norm_w / k_norm_w (D,) -> q' = q_scale * RMSNorm(q) * w_q (same for k). None = no
            norm. q_scale / k_scale are fixed hyperparameters, folded into the softmax scale.
  rope      cos / sin (S, D) or (B, S, D), HF rotate_half convention (tables are cat(f, f), so only
            the first half is read), applied AFTER the norm.
  window    None = causal over the whole sequence; W = key j visible to query i iff i-W < j <= i.
  xsa       z = o - a * (o.v_i / |v_i|^2) v_i with v_i the query position's own value row and
            a = tanh(alpha[h]) (alpha None -> a = 1). xsa=False returns o.

Layout of the work:
  prep   (only with norm and/or RoPE) one pass per Q / K row -> rope(norm(x) * w) in bf16, so every
         attention inner loop is plain bf16 loads + MMA; recomputed in the backward, never saved.
  fwd    one program per (query block, batch, kv head); BOTH query heads of the GQA group stacked
         into one tile, so each K/V tile is loaded once per group. Base-2 online softmax, masks
         only on diagonal / window-edge / tail blocks, XSA in the epilogue (v_i is the V row at the
         query's own position). Without XSA the output buffer doubles as O.
  bwd    pre (XSA backward + delta = dO . O; without XSA dO is dZ itself), dkdv (one program per
         key block), dq (one program per query block): every output written by exactly one
         program, so the backward is bitwise repeatable. Norm / RoPE are differentiated at the
         final store of dq / dk; norm-weight grads are per-program partials summed in fixed order.
Q and K are held as two (rows, D/2) halves, so RoPE's rotate_half is [-x2, x1] in registers and
QK^T is two chained K=D/2 MMAs into one accumulator.
"""
import torch
import triton
import triton.language as tl

__all__ = ["attn_xsa", "AttnXSA", "attn_xsa_reference", "CFG"]

LOG2E = 1.4426950408889634

# Per-kernel tiles by mode and sequence length: [(max_S, cfg), ...], first entry with S <= max_S
# wins. Chosen by SHAPE only (never by timing), so a run's numerics never depend on a benchmark.
# Swept with parity_check/parity_attn_xsa.py --sweep (S=1024) / --sweep_long (S=4096).
CFG = {
    "causal": [
        (1024, {"fwd": dict(BM=64, BN=32, warps=8, stages=3), "pre": dict(BM=64, warps=4),
                "dkdv": dict(BM=32, BN=32, warps=4, stages=3), "dq": dict(BM=32, BN=32, warps=4, stages=2)}),
        (1 << 30, {"fwd": dict(BM=64, BN=32, warps=8, stages=3), "pre": dict(BM=64, warps=4),
                   "dkdv": dict(BM=64, BN=32, warps=8, stages=2), "dq": dict(BM=64, BN=32, warps=8, stages=3)}),
    ],
    "window": [
        (1 << 30, {"fwd": dict(BM=64, BN=32, warps=8, stages=3), "pre": dict(BM=64, warps=4),
                   "dkdv": dict(BM=32, BN=32, warps=4, stages=3), "dq": dict(BM=32, BN=32, warps=4, stages=3)}),
    ],
}


def cfg_for(mode, S):
    return next(c for max_s, c in CFG[mode] if S <= max_s)
PREP_ROWS = 64


@triton.jit
def _qk_in(P, off, msk, pos, WN, COS, SIN, cbase, css, eps,
           D: tl.constexpr, HD: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    """Raw rows at element offsets `off` -> rope(norm(x) * w) halves (fp32), rstd, raw halves."""
    dh = tl.arange(0, HD)
    x1 = tl.load(P + off[:, None] + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(P + off[:, None] + HD + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
    r = tl.sum(x1 * 0.0, axis=1) + 1.0
    if QK_NORM:
        r = 1.0 / tl.sqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D + eps)
        y1 = x1 * r[:, None] * tl.load(WN + dh).to(tl.float32)[None, :]
        y2 = x2 * r[:, None] * tl.load(WN + HD + dh).to(tl.float32)[None, :]
    else:
        y1 = x1
        y2 = x2
    if ROPE:                   # half tables: HF cos/sin are cat(f, f), both halves identical
        co = cbase + pos.to(tl.int64) * css
        c = tl.load(COS + co[:, None] + dh[None, :], mask=msk[:, None], other=1.0).to(tl.float32)
        sn = tl.load(SIN + co[:, None] + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        z1 = y1 * c - y2 * sn
        z2 = y2 * c + y1 * sn
    else:
        z1 = y1
        z2 = y2
    return z1, z2, r, x1, x2


@triton.jit
def _qk_back(g1, g2, x1, x2, r, pos, msk, WN, COS, SIN, cbase, css,
             D: tl.constexpr, HD: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    """Grad w.r.t. rope(norm(x) * w) -> grad w.r.t. raw x, plus the per-column norm-weight terms."""
    dh = tl.arange(0, HD)
    if ROPE:                   # transpose of the rotation
        co = cbase + pos.to(tl.int64) * css
        c = tl.load(COS + co[:, None] + dh[None, :], mask=msk[:, None], other=1.0).to(tl.float32)
        sn = tl.load(SIN + co[:, None] + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        d1 = g1 * c + g2 * sn
        d2 = g2 * c - g1 * sn
    else:
        d1 = g1
        d2 = g2
    if QK_NORM:
        w1 = tl.load(WN + dh).to(tl.float32)
        w2 = tl.load(WN + HD + dh).to(tl.float32)
        xh1 = x1 * r[:, None]
        xh2 = x2 * r[:, None]
        gh1 = d1 * w1[None, :]
        gh2 = d2 * w2[None, :]
        mu = (tl.sum(gh1 * xh1, axis=1) + tl.sum(gh2 * xh2, axis=1)) / D
        dx1 = r[:, None] * (gh1 - xh1 * mu[:, None])
        dx2 = r[:, None] * (gh2 - xh2 * mu[:, None])
        dw1 = tl.sum(d1 * xh1, axis=0)
        dw2 = tl.sum(d2 * xh2, axis=0)
    else:
        dx1 = d1
        dx2 = d2
        dw1 = tl.sum(d1 * 0.0, axis=0)
        dw2 = dw1
    return dx1, dx2, dw1, dw2


@triton.jit
def _prep(X, OUT, WN, COS, SIN, xsb, xsh, xss, csb, css, NR, S, NH, eps,
          D: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr, BR: tl.constexpr):
    """rope(norm(x) * w) for every (b, h, s) row of a (B, NH, S, D) tensor, into a contiguous bf16
    copy. Row-local, so the attention loops never redo it per tile."""
    HD: tl.constexpr = D // 2
    r_ = tl.program_id(0) * BR + tl.arange(0, BR)
    msk = r_ < NR
    s = r_ % S
    bh = r_ // S
    b = bh // NH
    h = bh % NH
    off = b.to(tl.int64) * xsb + h.to(tl.int64) * xsh + s.to(tl.int64) * xss
    z1, z2, _r, _a, _b = _qk_in(X, off, msk, s, WN, COS, SIN, b.to(tl.int64) * csb, css, eps,
                                D, HD, QK_NORM, ROPE)
    dh = tl.arange(0, HD)
    o = r_.to(tl.int64)[:, None] * D + dh[None, :]
    tl.store(OUT + o, z1.to(OUT.dtype.element_ty), mask=msk[:, None])
    tl.store(OUT + o + HD, z2.to(OUT.dtype.element_ty), mask=msk[:, None])


@triton.jit
def _fwd(Q, K, V, O, Z, LSE, A,
         qsb, qsh, qss, ksb, ksh, kss, vsb, vsh, vss, osb, osh, oss, zsb, zsh, zss,
         S, H, HKV, sm_scale, W,
         GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         WINDOW: tl.constexpr, XSA: tl.constexpr, HAS_A: tl.constexpr):
    HD: tl.constexpr = D // 2
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)      # longest-first: last query blocks do the most work
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    m0 = pid_m * BM
    r_ = tl.arange(0, GROUP * BM)
    s = m0 + (r_ % BM)
    h = kvh * GROUP + r_ // BM
    rmask = s < S
    d = tl.arange(0, D)
    dh = tl.arange(0, HD)
    qoff = b.to(tl.int64) * qsb + h.to(tl.int64) * qsh + s.to(tl.int64) * qss
    q1 = tl.load(Q + qoff[:, None] + dh[None, :], mask=rmask[:, None], other=0.0)
    q2 = tl.load(Q + qoff[:, None] + HD + dh[None, :], mask=rmask[:, None], other=0.0)
    kb0 = b.to(tl.int64) * ksb + kvh.to(tl.int64) * ksh
    vb0 = b.to(tl.int64) * vsb + kvh.to(tl.int64) * vsh
    m_i = tl.full((GROUP * BM,), float("-inf"), tl.float32)
    l_i = tl.zeros((GROUP * BM,), tl.float32)
    acc = tl.zeros((GROUP * BM, D), tl.float32)
    lo = 0
    if WINDOW:
        lo = tl.maximum(m0 - W + 1, 0) // BN * BN
    hi = tl.minimum(m0 + BM, S)
    for n0 in range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        nmask = n < S
        koff = kb0 + n.to(tl.int64) * kss
        k1 = tl.load(K + koff[:, None] + dh[None, :], mask=nmask[:, None], other=0.0)
        k2 = tl.load(K + koff[:, None] + HD + dh[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + vb0 + n.to(tl.int64)[:, None] * vss + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q2, tl.trans(k2), tl.dot(q1, tl.trans(k1))) * sm_scale   # base 2: log2e in sm_scale
        edge = (n0 + BN > m0) | (n0 + BN > S)            # diagonal / tail blocks need a mask
        if WINDOW:
            edge = edge | (n0 < m0 + BM - W)              # ... and the window's lower edge
        if edge:
            valid = (n[None, :] <= s[:, None]) & nmask[None, :]
            if WINDOW:
                valid = valid & ((s[:, None] - n[None, :]) < W)
            qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = tl.dot(p.to(tl.bfloat16), v, acc * corr[:, None])
        m_i = m_new
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = acc / l_safe[:, None]
    irow = (b * H + h).to(tl.int64) * S + s
    tl.store(LSE + irow, m_i + tl.log2(l_safe), mask=rmask)                   # base-2 LSE
    ooff = b.to(tl.int64) * osb + h.to(tl.int64) * osh + s.to(tl.int64) * oss
    tl.store(O + ooff[:, None] + d[None, :], o.to(O.dtype.element_ty), mask=rmask[:, None])
    if XSA:
        # XSA on the bf16-ROUNDED o, i.e. exactly the O the backward reads back: the alpha / v
        # gradients are then taken at the same point the forward used (fp32 o here made d_alpha
        # 1.67x noisier than production at S=4096)
        o = o.to(O.dtype.element_ty).to(tl.float32)
        vs = tl.load(V + vb0 + s.to(tl.int64)[:, None] * vss + d[None, :], mask=rmask[:, None],
                     other=0.0).to(tl.float32)
        n2 = tl.sum(vs * vs, axis=1)
        c = tl.sum(o * vs, axis=1) * tl.where(n2 > 0.0, 1.0 / n2, 0.0)
        if HAS_A:
            c = c * tl.load(A + h).to(tl.float32)
        o = o - c[:, None] * vs
        zoff = b.to(tl.int64) * zsb + h.to(tl.int64) * zsh + s.to(tl.int64) * zss
        tl.store(Z + zoff[:, None] + d[None, :], o.to(Z.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _bwd_pre(O, DZ, V, A, DO, DELTA, GA, GVS, zsb, zsh, zss, vsb, vsh, vss, osb, osh, oss, S, H, HKV,
             GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
             XSA: tl.constexpr, HAS_A: tl.constexpr):
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    s = pid_m * BM + tl.arange(0, BM)
    rmask = s < S
    d = tl.arange(0, D)
    v = tl.load(V + b.to(tl.int64) * vsb + kvh.to(tl.int64) * vsh + s.to(tl.int64)[:, None] * vss + d[None, :],
                mask=rmask[:, None], other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=1)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    gv = tl.zeros((BM, D), tl.float32)
    for jj in tl.static_range(GROUP):
        h = kvh * GROUP + jj
        irow = (b * H + h).to(tl.int64) * S + s
        ooff = b.to(tl.int64) * osb + h.to(tl.int64) * osh + s.to(tl.int64) * oss
        o = tl.load(O + ooff[:, None] + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        zoff = b.to(tl.int64) * zsb + h.to(tl.int64) * zsh + s.to(tl.int64) * zss
        gz = tl.load(DZ + zoff[:, None] + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        if XSA:
            dot = tl.sum(o * v, axis=1)
            gzv = tl.sum(gz * v, axis=1)
            coeff = dot * inv
            a = 1.0
            if HAS_A:
                a = tl.load(A + h).to(tl.float32)
                tl.store(GA + irow, -coeff * gzv, mask=rmask)
            gy = gz - (a * gzv * inv)[:, None] * v
            gv += a * ((-(gzv * inv))[:, None] * o + (2.0 * dot * gzv * inv * inv)[:, None] * v
                       - coeff[:, None] * gz)
            gy = gy.to(DO.dtype.element_ty)
            tl.store(DO + irow[:, None] * D + d[None, :], gy, mask=rmask[:, None])
            tl.store(DELTA + irow, tl.sum(gy.to(tl.float32) * o, axis=1), mask=rmask)
        else:                  # no XSA: dO IS dZ (read in place by dkdv / dq), only delta is new
            tl.store(DELTA + irow, tl.sum(gz * o, axis=1), mask=rmask)
    if XSA:
        tl.store(GVS + ((b * HKV + kvh).to(tl.int64) * S + s)[:, None] * D + d[None, :], gv, mask=rmask[:, None])


@triton.jit
def _bwd_dkdv(QN, KN, KR, V, DO, LSE, DELTA, GVS, DK, DV, PWK, WK, COS, SIN,
              qsb, qsh, qss, ksb, ksh, kss, rsb, rsh, rss, vsb, vsh, vss, dsb, dsh, dss, csb, css,
              S, H, HKV, sm_scale, nat_scale, eps, W,
              GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              WINDOW: tl.constexpr, XSA: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    HD: tl.constexpr = D // 2
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    n0 = pid_n * BN
    n = n0 + tl.arange(0, BN)
    nmask = n < S
    d = tl.arange(0, D)
    dh = tl.arange(0, HD)
    koff = b.to(tl.int64) * ksb + kvh.to(tl.int64) * ksh + n.to(tl.int64) * kss
    voff = b.to(tl.int64) * vsb + kvh.to(tl.int64) * vsh + n.to(tl.int64) * vss
    k1 = tl.load(KN + koff[:, None] + dh[None, :], mask=nmask[:, None], other=0.0)
    k2 = tl.load(KN + koff[:, None] + HD + dh[None, :], mask=nmask[:, None], other=0.0)
    v = tl.load(V + voff[:, None] + d[None, :], mask=nmask[:, None], other=0.0)
    dk1 = tl.zeros((BN, HD), tl.float32)
    dk2 = tl.zeros((BN, HD), tl.float32)
    dv = tl.zeros((BN, D), tl.float32)
    lo = n0 // BM * BM
    hi = S
    if WINDOW:
        hi = tl.minimum(n0 + BN + W - 1, S)
    for m0 in range(lo, hi, BM):
        s = m0 + tl.arange(0, BM)
        smask = s < S
        edge = (m0 < n0 + BN) | (m0 + BM > S) | (n0 + BN > S)
        if WINDOW:
            edge = edge | (m0 + BM - 1 - n0 >= W)
        for jj in tl.static_range(GROUP):
            h = kvh * GROUP + jj
            qoff = b.to(tl.int64) * qsb + h.to(tl.int64) * qsh + s.to(tl.int64) * qss
            q1 = tl.load(QN + qoff[:, None] + dh[None, :], mask=smask[:, None], other=0.0)
            q2 = tl.load(QN + qoff[:, None] + HD + dh[None, :], mask=smask[:, None], other=0.0)
            irow = (b * H + h).to(tl.int64) * S + s
            dooff = b.to(tl.int64) * dsb + h.to(tl.int64) * dsh + s.to(tl.int64) * dss
            do = tl.load(DO + dooff[:, None] + d[None, :], mask=smask[:, None], other=0.0)
            lse = tl.load(LSE + irow, mask=smask, other=0.0)
            dl = tl.load(DELTA + irow, mask=smask, other=0.0)
            st = tl.dot(k2, tl.trans(q2), tl.dot(k1, tl.trans(q1))) * sm_scale   # (BN, BM) = S^T
            p = tl.exp2(st - lse[None, :])
            if edge:
                valid = (n[:, None] <= s[None, :]) & smask[None, :] & nmask[:, None]
                if WINDOW:
                    valid = valid & ((s[None, :] - n[:, None]) < W)
                p = tl.where(valid, p, 0.0)
            dv = tl.dot(p.to(tl.bfloat16), do, dv)
            dp = tl.dot(v, tl.trans(do))
            ds = (p * (dp - dl[None, :])).to(tl.bfloat16)
            dk1 = tl.dot(ds, q1, dk1)
            dk2 = tl.dot(ds, q2, dk2)
    dk1 = dk1 * nat_scale
    dk2 = dk2 * nat_scale
    if QK_NORM or ROPE:
        roff = b.to(tl.int64) * rsb + kvh.to(tl.int64) * rsh + n.to(tl.int64) * rss
        cbase = b.to(tl.int64) * csb
        _z1, _z2, rk, kx1, kx2 = _qk_in(KR, roff, nmask, n, WK, COS, SIN, cbase, css, eps, D, HD, QK_NORM, ROPE)
        dk1, dk2, dw1, dw2 = _qk_back(dk1, dk2, kx1, kx2, rk, n, nmask, WK, COS, SIN, cbase, css,
                                      D, HD, QK_NORM, ROPE)
        if QK_NORM:
            pw = (pid.to(tl.int64) * tl.num_programs(0) + pid_n) * D
            tl.store(PWK + pw + dh, dw1)
            tl.store(PWK + pw + HD + dh, dw2)
    else:
        roff = b.to(tl.int64) * rsb + kvh.to(tl.int64) * rsh + n.to(tl.int64) * rss
    if XSA:
        dv += tl.load(GVS + ((b * HKV + kvh).to(tl.int64) * S + n)[:, None] * D + d[None, :],
                      mask=nmask[:, None], other=0.0)
    tl.store(DK + roff[:, None] + dh[None, :], dk1.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DK + roff[:, None] + HD + dh[None, :], dk2.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DV + voff[:, None] + d[None, :], dv.to(DV.dtype.element_ty), mask=nmask[:, None])


@triton.jit
def _bwd_dq(QN, KN, QR, V, DO, LSE, DELTA, DQ, PWQ, WQ, COS, SIN,
            qsb, qsh, qss, ksb, ksh, kss, rsb, rsh, rss, vsb, vsh, vss, dsb, dsh, dss, csb, css,
            S, H, HKV, sm_scale, nat_scale, eps, W,
            GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            WINDOW: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    HD: tl.constexpr = D // 2
    pid_m = tl.num_programs(0) - 1 - tl.program_id(0)      # longest-first (see _fwd)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    m0 = pid_m * BM
    r_ = tl.arange(0, GROUP * BM)
    s = m0 + (r_ % BM)
    h = kvh * GROUP + r_ // BM
    rmask = s < S
    d = tl.arange(0, D)
    dh = tl.arange(0, HD)
    qoff = b.to(tl.int64) * qsb + h.to(tl.int64) * qsh + s.to(tl.int64) * qss
    q1 = tl.load(QN + qoff[:, None] + dh[None, :], mask=rmask[:, None], other=0.0)
    q2 = tl.load(QN + qoff[:, None] + HD + dh[None, :], mask=rmask[:, None], other=0.0)
    irow = (b * H + h).to(tl.int64) * S + s
    dooff = b.to(tl.int64) * dsb + h.to(tl.int64) * dsh + s.to(tl.int64) * dss
    do = tl.load(DO + dooff[:, None] + d[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + irow, mask=rmask, other=0.0)
    dl = tl.load(DELTA + irow, mask=rmask, other=0.0)
    kb0 = b.to(tl.int64) * ksb + kvh.to(tl.int64) * ksh
    vb0 = b.to(tl.int64) * vsb + kvh.to(tl.int64) * vsh
    dq1 = tl.zeros((GROUP * BM, HD), tl.float32)
    dq2 = tl.zeros((GROUP * BM, HD), tl.float32)
    lo = 0
    if WINDOW:
        lo = tl.maximum(m0 - W + 1, 0) // BN * BN
    hi = tl.minimum(m0 + BM, S)
    for n0 in range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        nmask = n < S
        koff = kb0 + n.to(tl.int64) * kss
        k1 = tl.load(KN + koff[:, None] + dh[None, :], mask=nmask[:, None], other=0.0)
        k2 = tl.load(KN + koff[:, None] + HD + dh[None, :], mask=nmask[:, None], other=0.0)
        v = tl.load(V + vb0 + n.to(tl.int64)[:, None] * vss + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q2, tl.trans(k2), tl.dot(q1, tl.trans(k1))) * sm_scale
        p = tl.exp2(qk - lse[:, None])
        edge = (n0 + BN > m0) | (n0 + BN > S)
        if WINDOW:
            edge = edge | (n0 < m0 + BM - W)
        if edge:
            valid = (n[None, :] <= s[:, None]) & nmask[None, :]
            if WINDOW:
                valid = valid & ((s[:, None] - n[None, :]) < W)
            p = tl.where(valid, p, 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - dl[:, None])).to(tl.bfloat16)
        dq1 = tl.dot(ds, k1, dq1)
        dq2 = tl.dot(ds, k2, dq2)
    dq1 = dq1 * nat_scale
    dq2 = dq2 * nat_scale
    roff = b.to(tl.int64) * rsb + h.to(tl.int64) * rsh + s.to(tl.int64) * rss
    if QK_NORM or ROPE:
        cbase = b.to(tl.int64) * csb
        _z1, _z2, rq, qx1, qx2 = _qk_in(QR, roff, rmask, s, WQ, COS, SIN, cbase, css, eps, D, HD, QK_NORM, ROPE)
        dq1, dq2, dw1, dw2 = _qk_back(dq1, dq2, qx1, qx2, rq, s, rmask, WQ, COS, SIN, cbase, css,
                                      D, HD, QK_NORM, ROPE)
        if QK_NORM:
            pw = (pid.to(tl.int64) * tl.num_programs(0) + pid_m) * D
            tl.store(PWQ + pw + dh, dw1)
            tl.store(PWQ + pw + HD + dh, dw2)
    tl.store(DQ + roff[:, None] + dh[None, :], dq1.to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(DQ + roff[:, None] + HD + dh[None, :], dq2.to(DQ.dtype.element_ty), mask=rmask[:, None])


_FIT = {}


def _launch(kern, grid_fn, c, key, *args, **kw):
    """Launch with tile config c; on OutOfResources fall back deterministically (fewer stages, then
    half BN, then half BM) and remember the config that fits for this key. The choice depends only
    on the hardware limit, never on timing, so the numerics stay identical run to run."""
    c = dict(_FIT.get(key, c))
    while True:
        try:
            kern[grid_fn(c)](*args, BM=c["BM"], BN=c["BN"], num_warps=c["warps"], num_stages=c["stages"], **kw)
            _FIT[key] = c
            return c
        except triton.runtime.errors.OutOfResources:
            if c["stages"] > 1:
                c["stages"] -= 1
            elif c["BN"] > 16:
                c["BN"] //= 2
            elif c["BM"] > 16:
                c["BM"] //= 2
            else:
                raise


def _st(t):
    return t.stride(0), t.stride(1), t.stride(2)


def _like(t):
    """Same shape AND strides as t (a transposed projection view stays a transposed view)."""
    return torch.empty_strided(t.shape, t.stride(), device=t.device, dtype=t.dtype)


def _prepped(x, w, cos, sin, csb, css, eps, qk_norm, rope):
    """rope(norm(x) * w) as a contiguous bf16 (B, NH, S, D) tensor, or x itself when neither applies."""
    if not (qk_norm or rope):
        return x
    B, NH, S, D = x.shape
    out = torch.empty(B, NH, S, D, device=x.device, dtype=x.dtype)
    NR = B * NH * S
    _prep[(triton.cdiv(NR, PREP_ROWS),)](x, out, w if qk_norm else x, cos, sin, *_st(x), csb, css, NR, S, NH,
                                         float(eps), D=D, QK_NORM=qk_norm, ROPE=rope, BR=PREP_ROWS, num_warps=4)
    return out


class AttnXSA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, alpha, q_norm_w, k_norm_w, cos, sin, scale, window, xsa, q_scale, k_scale, eps):
        B, H, S, D = q.shape
        HKV = k.shape[1]
        G = H // HKV
        for t in (q, k, v):
            if t.stride(-1) != 1:
                raise ValueError("attn_xsa needs a unit-stride head dim")
        qk_norm = q_norm_w is not None
        rope = cos is not None
        A = torch.tanh(alpha.float()).contiguous() if alpha is not None else None
        wq = q_norm_w.contiguous() if qk_norm else None
        wk = k_norm_w.contiguous() if qk_norm else None
        if rope:
            # HF tables are cat(freqs, freqs): the kernels read ONE half (S, D/2) for both halves
            cos = cos[..., : D // 2].to(q.dtype).contiguous()
            sin = sin[..., : D // 2].to(q.dtype).contiguous()
            if cos.shape[-2] < S or (cos.dim() == 3 and cos.shape[0] not in (1, B)):
                raise ValueError(f"cos/sin {tuple(cos.shape)} do not cover (B={B}, S={S})")
            # a batch dim of 1 (one position table for the whole batch, as the model passes it)
            # broadcasts: stride 0, never index b into it
            csb = cos.stride(0) if (cos.dim() == 3 and cos.shape[0] > 1) else 0
            css = cos.stride(-2)
        else:
            cos = sin = q
            csb = css = 0
        qn = _prepped(q, wq, cos, sin, csb, css, eps, qk_norm, rope)
        kn = _prepped(k, wk, cos, sin, csb, css, eps, qk_norm, rope)
        Z = _like(q)
        O = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype) if xsa else Z   # no XSA: z IS o
        LSE = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        W = int(window) if window is not None else 0
        mode = "window" if window is not None else "causal"
        sm = float(scale) * float(q_scale) * float(k_scale)          # the scalars fold into the logits
        flags = dict(WINDOW=window is not None, XSA=bool(xsa), HAS_A=A is not None)
        _launch(_fwd, lambda c: (triton.cdiv(S, c["BM"]), B * HKV), cfg_for(mode, S)["fwd"],
                ("fwd", mode, S, D, G, q.dtype, tuple(flags.values())),
                qn, kn, v, O, Z, LSE, A if A is not None else q,
                *_st(qn), *_st(kn), *_st(v), *_st(O), *_st(Z), S, H, HKV, sm * LOG2E, W, GROUP=G, D=D, **flags)
        ctx.save_for_backward(q, k, v, O, LSE, A if A is not None else q.new_zeros(0),
                              wq if qk_norm else q.new_zeros(0), wk if qk_norm else q.new_zeros(0), cos, sin)
        ctx.cfg = (sm, window, bool(xsa), float(eps), A is not None, qk_norm, rope, csb, css,
                   alpha.dtype if alpha is not None else None,
                   q_norm_w.dtype if qk_norm else None, k_norm_w.dtype if qk_norm else None)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        q, k, v, O, LSE, A, wq, wk, cos, sin = ctx.saved_tensors
        sm, window, xsa, eps, has_a, qk_norm, rope, csb, css, a_dtype, wq_dtype, wk_dtype = ctx.cfg
        B, H, S, D = q.shape
        HKV = k.shape[1]
        G = H // HKV
        if dZ.stride(-1) != 1:
            dZ = dZ.contiguous()
        W = int(window) if window is not None else 0
        mode = "window" if window is not None else "causal"
        cf = cfg_for(mode, S)
        key = (mode, S if mode == "causal" else 0, D, G, q.dtype, window is not None, xsa, has_a, qk_norm, rope)
        DO = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype) if xsa else dZ   # no XSA: dO = dZ
        DELTA = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        GA = torch.empty(B, H, S, device=q.device, dtype=torch.float32) if (xsa and has_a) else DELTA
        GVS = torch.empty(B, HKV, S, D, device=q.device, dtype=torch.float32) if xsa else DELTA
        c = cf["pre"]
        _bwd_pre[(triton.cdiv(S, c["BM"]), B * HKV)](
            O, dZ, v, A if has_a else q, DO, DELTA, GA, GVS, *_st(dZ), *_st(v), *_st(O), S, H, HKV,
            GROUP=G, D=D, BM=c["BM"], XSA=xsa, HAS_A=has_a, num_warps=c["warps"])
        wq_ = wq if qk_norm else None
        wk_ = wk if qk_norm else None
        qn = _prepped(q, wq_, cos, sin, csb, css, eps, qk_norm, rope)     # recomputed, never saved
        kn = _prepped(k, wk_, cos, sin, csb, css, eps, qk_norm, rope)
        DK, DV = _like(k), _like(v)
        # norm-weight partials are sized by the tile count, so allocate for the smallest tile the
        # OutOfResources fallback can reach and slice to the real count afterwards
        PWK = torch.empty(B * HKV * triton.cdiv(S, 16), D, device=q.device, dtype=torch.float32) if qk_norm else DELTA
        common = (csb, css, S, H, HKV, sm * LOG2E, sm, float(eps), W)
        c = _launch(_bwd_dkdv, lambda c: (triton.cdiv(S, c["BN"]), B * HKV), cf["dkdv"], ("dkdv",) + key,
                    qn, kn, k, v, DO, LSE, DELTA, GVS, DK, DV, PWK, wk_ if qk_norm else q, cos, sin,
                    *_st(qn), *_st(kn), *_st(k), *_st(v), *_st(DO), *common,
                    GROUP=G, D=D, WINDOW=window is not None, XSA=xsa, QK_NORM=qk_norm, ROPE=rope)
        nkb = triton.cdiv(S, c["BN"])
        DQ = _like(q)
        PWQ = torch.empty(B * HKV * triton.cdiv(S, 16), D, device=q.device, dtype=torch.float32) if qk_norm else DELTA
        c = _launch(_bwd_dq, lambda c: (triton.cdiv(S, c["BM"]), B * HKV), cf["dq"], ("dq",) + key,
                    qn, kn, q, v, DO, LSE, DELTA, DQ, PWQ, wq_ if qk_norm else q, cos, sin,
                    *_st(qn), *_st(kn), *_st(q), *_st(v), *_st(DO), *common,
                    GROUP=G, D=D, WINDOW=window is not None, QK_NORM=qk_norm, ROPE=rope)
        nqb = triton.cdiv(S, c["BM"])
        d_alpha = None
        if xsa and has_a:
            d_alpha = (GA.sum(dim=(0, 2)) * (1.0 - A * A)).to(a_dtype)
        dwq = PWQ[: B * HKV * nqb].sum(0).to(wq_dtype) if qk_norm else None
        dwk = PWK[: B * HKV * nkb].sum(0).to(wk_dtype) if qk_norm else None
        return DQ, DK, DV, d_alpha, dwq, dwk, None, None, None, None, None, None, None, None


def attn_xsa(q, k, v, *, scale, window=None, xsa=True, alpha=None, q_norm_w=None, k_norm_w=None,
             q_scale=1.0, k_scale=1.0, eps=1e-6, cos=None, sin=None):
    return AttnXSA.apply(q, k, v, alpha, q_norm_w, k_norm_w, cos, sin, scale, window, xsa,
                         q_scale, k_scale, eps)


def _rotate_half(x):
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def attn_xsa_reference(q, k, v, *, scale, window=None, xsa=True, alpha=None, q_norm_w=None,
                       k_norm_w=None, q_scale=1.0, k_scale=1.0, eps=1e-6, cos=None, sin=None,
                       dtype=torch.float32):
    """Plain PyTorch in `dtype` (fp32 / fp64): the numerics target for parity."""
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if q_norm_w is not None:
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * q_norm_w.to(dtype)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * k_norm_w.to(dtype)
    q, k = q * q_scale, k * k_scale
    if cos is not None:
        c, sn = cos.to(dtype), sin.to(dtype)
        c, sn = (c[:, None], sn[:, None]) if c.dim() == 3 else (c, sn)
        q = q * c + _rotate_half(q) * sn
        k = k * c + _rotate_half(k) * sn
    G = q.shape[1] // k.shape[1]
    kr, vr = k.repeat_interleave(G, 1), v.repeat_interleave(G, 1)
    S = q.shape[2]
    i = torch.arange(S, device=q.device)
    m = i[None, :] <= i[:, None]
    if window is not None:
        m = m & (i[:, None] - i[None, :] < window)
    att = torch.softmax(((q @ kr.transpose(-1, -2)) * scale).masked_fill(~m, float("-inf")), -1)
    o = att @ vr
    if not xsa:
        return o
    a = torch.tanh(alpha.to(dtype))[None, :, None, None] if alpha is not None else 1.0
    return o - a * ((o * vr).sum(-1, keepdim=True) / (vr * vr).sum(-1, keepdim=True).clamp_min(1e-30)) * vr
