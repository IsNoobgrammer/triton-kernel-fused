"""Fused attention for BiBo: [qk RMSNorm] -> [RoPE] -> causal / sliding-window GQA attention ->
[XSA], one kernel family, deterministic backward (no atomics anywhere).

    z = attn_xsa(q, k, v, scale=..., window=None, xsa=True, alpha=None,
                 q_norm_w=None, k_norm_w=None, q_scale=1.0, k_scale=1.0, eps=1e-6,
                 cos=None, sin=None)

q (B, H, S, D), k / v (B, Hkv, S, D): bf16, ANY strides with a unit last dim -- the (B, S, H, D)
projection views transposed to (B, H, S, D) are read in place, and z / dq / dk / dv come back in
the SAME layout as their input, so no .contiguous() copy happens on either side of attention.
  qk norm   q_norm_w / k_norm_w (D,) -> q' = q_scale * RMSNorm(q) * w_q (same for k). None = no
            norm (q_scale / k_scale still multiply). The scalars are fixed hyperparameters.
  rope      cos / sin (S, D) or (B, S, D), HF rotate_half convention, applied AFTER the norm.
  window    None = causal over the whole sequence; W = key j visible to query i iff i-W < j <= i.
  xsa       z = o - a * (o.v_i / |v_i|^2) v_i with v_i the query position's own value row and
            a = tanh(alpha[h]) (alpha None -> a = 1). xsa=False returns o.

Q and K tiles are held as two (rows, D/2) HALVES everywhere, so rotate_half is [-x2, x1] in
registers and QK^T is two K=D/2 dots; V / O stay full width.

Forward: one program per (query block, batch, kv head); BOTH query heads of the GQA group are
stacked into one tile, so each K/V tile is loaded once per group. Online softmax; XSA in the
epilogue (v_i is the V row at the query's own position).
Backward, three passes, each output written by exactly one program (bitwise repeatable):
  pre    XSA backward (dZ -> dO, self-value grad, per-row alpha-grad terms) + delta = dO . O
  dkdv   one program per key block, loops over every query block that sees it (both heads)
  dq     one program per query block, loops over its key blocks
norm and RoPE are recomputed on every load and differentiated through at the store; norm-weight
grads are per-program partial rows summed on the host in a fixed order.
"""
import torch
import triton
import triton.language as tl

__all__ = ["attn_xsa", "AttnXSA", "attn_xsa_reference", "CFG"]

# Per-kernel tiles (swept at the board shape with parity_check/parity_attn_xsa.py --sweep).
CFG = {"fwd": dict(BM=64, BN=64, warps=8, stages=2), "pre": dict(BM=64, warps=4),
       "dkdv": dict(BM=32, BN=32, warps=4, stages=2), "dq": dict(BM=64, BN=32, warps=8, stages=2)}


@triton.jit
def _qk_in(P, off, msk, pos, WN, COS, SIN, csb, css, cbase, scale, eps,
           D: tl.constexpr, HD: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    """Load rows at element offsets `off` -> transformed halves (fp32) + rstd + raw halves."""
    dh = tl.arange(0, HD)
    x1 = tl.load(P + off[:, None] + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
    x2 = tl.load(P + off[:, None] + HD + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
    r = tl.sum(x1 * 0.0, axis=1) + 1.0
    if QK_NORM:
        r = 1.0 / tl.sqrt((tl.sum(x1 * x1, axis=1) + tl.sum(x2 * x2, axis=1)) / D + eps)
        y1 = x1 * r[:, None] * tl.load(WN + dh).to(tl.float32)[None, :] * scale
        y2 = x2 * r[:, None] * tl.load(WN + HD + dh).to(tl.float32)[None, :] * scale
    else:
        y1 = x1 * scale
        y2 = x2 * scale
    if ROPE:
        co = cbase + pos.to(tl.int64) * css
        c1 = tl.load(COS + co[:, None] + dh[None, :], mask=msk[:, None], other=1.0).to(tl.float32)
        c2 = tl.load(COS + co[:, None] + HD + dh[None, :], mask=msk[:, None], other=1.0).to(tl.float32)
        s1 = tl.load(SIN + co[:, None] + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        s2 = tl.load(SIN + co[:, None] + HD + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        z1 = y1 * c1 - y2 * s1
        z2 = y2 * c2 + y1 * s2
    else:
        z1 = y1
        z2 = y2
    return z1, z2, r, x1, x2


@triton.jit
def _qk_back(g1, g2, x1, x2, r, pos, msk, WN, COS, SIN, cbase, css, scale,
             D: tl.constexpr, HD: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    """Grad w.r.t. the transformed halves -> grad w.r.t. the raw halves + norm-weight grad terms."""
    dh = tl.arange(0, HD)
    if ROPE:                   # z1 = y1 c1 - y2 s1, z2 = y2 c2 + y1 s2
        co = cbase + pos.to(tl.int64) * css
        c1 = tl.load(COS + co[:, None] + dh[None, :], mask=msk[:, None], other=1.0).to(tl.float32)
        c2 = tl.load(COS + co[:, None] + HD + dh[None, :], mask=msk[:, None], other=1.0).to(tl.float32)
        s1 = tl.load(SIN + co[:, None] + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        s2 = tl.load(SIN + co[:, None] + HD + dh[None, :], mask=msk[:, None], other=0.0).to(tl.float32)
        d1 = g1 * c1 + g2 * s2
        d2 = g2 * c2 - g1 * s1
    else:
        d1 = g1
        d2 = g2
    if QK_NORM:
        w1 = tl.load(WN + dh).to(tl.float32)
        w2 = tl.load(WN + HD + dh).to(tl.float32)
        xh1 = x1 * r[:, None]
        xh2 = x2 * r[:, None]
        gh1 = d1 * w1[None, :] * scale
        gh2 = d2 * w2[None, :] * scale
        mu = (tl.sum(gh1 * xh1, axis=1) + tl.sum(gh2 * xh2, axis=1)) / D
        dx1 = r[:, None] * (gh1 - xh1 * mu[:, None])
        dx2 = r[:, None] * (gh2 - xh2 * mu[:, None])
        dw1 = tl.sum(d1 * xh1 * scale, axis=0)
        dw2 = tl.sum(d2 * xh2 * scale, axis=0)
    else:
        dx1 = d1 * scale
        dx2 = d2 * scale
        dw1 = tl.sum(d1 * 0.0, axis=0)
        dw2 = dw1
    return dx1, dx2, dw1, dw2


@triton.jit
def _fwd(Q, K, V, O, Z, LSE, A, WQ, WK, COS, SIN,
         qsb, qsh, qss, ksb, ksh, kss, vsb, vsh, vss, csb, css,
         S, H, HKV, sm_scale, q_scale, k_scale, eps, W,
         GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         WINDOW: tl.constexpr, XSA: tl.constexpr, HAS_A: tl.constexpr, QK_NORM: tl.constexpr,
         ROPE: tl.constexpr):
    HD: tl.constexpr = D // 2
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    m0 = pid_m * BM
    r_ = tl.arange(0, GROUP * BM)
    s = m0 + (r_ % BM)
    h = kvh * GROUP + r_ // BM
    rmask = s < S
    d = tl.arange(0, D)
    cbase = b.to(tl.int64) * csb
    qoff = b.to(tl.int64) * qsb + h.to(tl.int64) * qsh + s.to(tl.int64) * qss
    q1, q2, _qr, _qa, _qb = _qk_in(Q, qoff, rmask, s, WQ, COS, SIN, csb, css, cbase, q_scale, eps,
                                   D, HD, QK_NORM, ROPE)
    q1 = q1.to(tl.bfloat16)
    q2 = q2.to(tl.bfloat16)
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
        k1, k2, _kr, _ka, _kb = _qk_in(K, kb0 + n.to(tl.int64) * kss, nmask, n, WK, COS, SIN, csb, css,
                                       cbase, k_scale, eps, D, HD, QK_NORM, ROPE)
        v = tl.load(V + vb0 + n.to(tl.int64)[:, None] * vss + d[None, :], mask=nmask[:, None], other=0.0)
        qk = (tl.dot(q1, tl.trans(k1.to(tl.bfloat16))) + tl.dot(q2, tl.trans(k2.to(tl.bfloat16)))) * sm_scale
        valid = (n[None, :] <= s[:, None]) & nmask[None, :]
        if WINDOW:
            valid = valid & ((s[:, None] - n[None, :]) < W)
        qk = tl.where(valid, qk, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        corr = tl.exp(m_i - m_safe)
        p = tl.exp(qk - m_safe[:, None])
        l_i = l_i * corr + tl.sum(p, axis=1)
        acc = acc * corr[:, None] + tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new
    l_safe = tl.where(l_i > 0.0, l_i, 1.0)
    o = acc / l_safe[:, None]
    irow = (b * H + h).to(tl.int64) * S + s
    tl.store(LSE + irow, m_i + tl.log(l_safe), mask=rmask)
    tl.store(O + irow[:, None] * D + d[None, :], o.to(O.dtype.element_ty), mask=rmask[:, None])
    if XSA:
        vs = tl.load(V + vb0 + s.to(tl.int64)[:, None] * vss + d[None, :], mask=rmask[:, None],
                     other=0.0).to(tl.float32)
        n2 = tl.sum(vs * vs, axis=1)
        c = tl.sum(o * vs, axis=1) * tl.where(n2 > 0.0, 1.0 / n2, 0.0)
        if HAS_A:
            c = c * tl.load(A + h).to(tl.float32)
        o = o - c[:, None] * vs
    tl.store(Z + qoff[:, None] + d[None, :], o.to(Z.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _bwd_pre(O, DZ, V, A, DO, DELTA, GA, GVS, zsb, zsh, zss, vsb, vsh, vss, S, H, HKV,
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
        o = tl.load(O + irow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
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
        else:
            gy = gz
        gy = gy.to(DO.dtype.element_ty)
        tl.store(DO + irow[:, None] * D + d[None, :], gy, mask=rmask[:, None])
        tl.store(DELTA + irow, tl.sum(gy.to(tl.float32) * o, axis=1), mask=rmask)
    if XSA:
        tl.store(GVS + ((b * HKV + kvh).to(tl.int64) * S + s)[:, None] * D + d[None, :], gv, mask=rmask[:, None])


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, GVS, DK, DV, PWK, WQ, WK, COS, SIN,
              qsb, qsh, qss, ksb, ksh, kss, vsb, vsh, vss, csb, css,
              S, H, HKV, sm_scale, q_scale, k_scale, eps, W,
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
    cbase = b.to(tl.int64) * csb
    koff = b.to(tl.int64) * ksb + kvh.to(tl.int64) * ksh + n.to(tl.int64) * kss
    voff = b.to(tl.int64) * vsb + kvh.to(tl.int64) * vsh + n.to(tl.int64) * vss
    k1, k2, rk, kx1, kx2 = _qk_in(K, koff, nmask, n, WK, COS, SIN, csb, css, cbase, k_scale, eps,
                                  D, HD, QK_NORM, ROPE)
    k1b = k1.to(tl.bfloat16)
    k2b = k2.to(tl.bfloat16)
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
        valid = (n[:, None] <= s[None, :]) & smask[None, :] & nmask[:, None]
        if WINDOW:
            valid = valid & ((s[None, :] - n[:, None]) < W)
        for jj in tl.static_range(GROUP):
            h = kvh * GROUP + jj
            qoff = b.to(tl.int64) * qsb + h.to(tl.int64) * qsh + s.to(tl.int64) * qss
            q1, q2, _qr, _qa, _qb = _qk_in(Q, qoff, smask, s, WQ, COS, SIN, csb, css, cbase, q_scale, eps,
                                           D, HD, QK_NORM, ROPE)
            q1b = q1.to(tl.bfloat16)
            q2b = q2.to(tl.bfloat16)
            irow = (b * H + h).to(tl.int64) * S + s
            do = tl.load(DO + irow[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
            lse = tl.load(LSE + irow, mask=smask, other=0.0)
            dl = tl.load(DELTA + irow, mask=smask, other=0.0)
            st = (tl.dot(k1b, tl.trans(q1b)) + tl.dot(k2b, tl.trans(q2b))) * sm_scale   # (BN, BM) = S^T
            p = tl.where(valid, tl.exp(st - lse[None, :]), 0.0)
            dv += tl.dot(p.to(tl.bfloat16), do)
            dp = tl.dot(v, tl.trans(do))
            ds = (p * (dp - dl[None, :])).to(tl.bfloat16)
            dk1 += tl.dot(ds, q1b)
            dk2 += tl.dot(ds, q2b)
    dx1, dx2, dw1, dw2 = _qk_back(dk1 * sm_scale, dk2 * sm_scale, kx1, kx2, rk, n, nmask, WK, COS, SIN,
                                  cbase, css, k_scale, D, HD, QK_NORM, ROPE)
    if QK_NORM:
        pw = (pid.to(tl.int64) * tl.num_programs(0) + pid_n) * D
        tl.store(PWK + pw + dh, dw1)
        tl.store(PWK + pw + HD + dh, dw2)
    if XSA:
        dv += tl.load(GVS + ((b * HKV + kvh).to(tl.int64) * S + n)[:, None] * D + d[None, :],
                      mask=nmask[:, None], other=0.0)
    tl.store(DK + koff[:, None] + dh[None, :], dx1.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DK + koff[:, None] + HD + dh[None, :], dx2.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DV + voff[:, None] + d[None, :], dv.to(DV.dtype.element_ty), mask=nmask[:, None])


@triton.jit
def _bwd_dq(Q, K, V, DO, LSE, DELTA, DQ, PWQ, WQ, WK, COS, SIN,
            qsb, qsh, qss, ksb, ksh, kss, vsb, vsh, vss, csb, css,
            S, H, HKV, sm_scale, q_scale, k_scale, eps, W,
            GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            WINDOW: tl.constexpr, QK_NORM: tl.constexpr, ROPE: tl.constexpr):
    HD: tl.constexpr = D // 2
    pid_m = tl.program_id(0)
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
    cbase = b.to(tl.int64) * csb
    qoff = b.to(tl.int64) * qsb + h.to(tl.int64) * qsh + s.to(tl.int64) * qss
    q1, q2, rq, qx1, qx2 = _qk_in(Q, qoff, rmask, s, WQ, COS, SIN, csb, css, cbase, q_scale, eps,
                                  D, HD, QK_NORM, ROPE)
    q1b = q1.to(tl.bfloat16)
    q2b = q2.to(tl.bfloat16)
    irow = (b * H + h).to(tl.int64) * S + s
    do = tl.load(DO + irow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
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
        k1, k2, _kr, _ka, _kb = _qk_in(K, kb0 + n.to(tl.int64) * kss, nmask, n, WK, COS, SIN, csb, css,
                                       cbase, k_scale, eps, D, HD, QK_NORM, ROPE)
        k1b = k1.to(tl.bfloat16)
        k2b = k2.to(tl.bfloat16)
        v = tl.load(V + vb0 + n.to(tl.int64)[:, None] * vss + d[None, :], mask=nmask[:, None], other=0.0)
        qk = (tl.dot(q1b, tl.trans(k1b)) + tl.dot(q2b, tl.trans(k2b))) * sm_scale
        valid = (n[None, :] <= s[:, None]) & nmask[None, :]
        if WINDOW:
            valid = valid & ((s[:, None] - n[None, :]) < W)
        p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = (p * (dp - dl[:, None])).to(tl.bfloat16)
        dq1 += tl.dot(ds, k1b)
        dq2 += tl.dot(ds, k2b)
    dx1, dx2, dw1, dw2 = _qk_back(dq1 * sm_scale, dq2 * sm_scale, qx1, qx2, rq, s, rmask, WQ, COS, SIN,
                                  cbase, css, q_scale, D, HD, QK_NORM, ROPE)
    if QK_NORM:
        pw = (pid.to(tl.int64) * tl.num_programs(0) + pid_m) * D
        tl.store(PWQ + pw + dh, dw1)
        tl.store(PWQ + pw + HD + dh, dw2)
    tl.store(DQ + qoff[:, None] + dh[None, :], dx1.to(DQ.dtype.element_ty), mask=rmask[:, None])
    tl.store(DQ + qoff[:, None] + HD + dh[None, :], dx2.to(DQ.dtype.element_ty), mask=rmask[:, None])


def _st(t):
    return t.stride(0), t.stride(1), t.stride(2)


def _like(t):
    """Same shape AND strides as t (a transposed projection view stays a transposed view)."""
    return torch.empty_strided(t.shape, t.stride(), device=t.device, dtype=t.dtype)


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
        wq = q_norm_w.contiguous() if qk_norm else q
        wk = k_norm_w.contiguous() if qk_norm else q
        if rope:
            cos, sin = cos.contiguous(), sin.contiguous()
            csb, css = (cos.stride(0), cos.stride(1)) if cos.dim() == 3 else (0, cos.stride(0))
        else:
            cos = sin = q
            csb = css = 0
        O = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype)
        Z = _like(q)
        LSE = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        W = int(window) if window is not None else 0
        c = CFG["fwd"]
        _fwd[(triton.cdiv(S, c["BM"]), B * HKV)](
            q, k, v, O, Z, LSE, A if A is not None else q, wq, wk, cos, sin,
            *_st(q), *_st(k), *_st(v), csb, css, S, H, HKV, float(scale), float(q_scale), float(k_scale),
            float(eps), W, GROUP=G, D=D, BM=c["BM"], BN=c["BN"], WINDOW=window is not None,
            XSA=bool(xsa), HAS_A=A is not None, QK_NORM=qk_norm, ROPE=rope,
            num_warps=c["warps"], num_stages=c["stages"])
        ctx.save_for_backward(q, k, v, O, LSE, A if A is not None else q.new_zeros(0),
                              q_norm_w if qk_norm else q.new_zeros(0),
                              k_norm_w if qk_norm else q.new_zeros(0), cos, sin)
        ctx.cfg = (scale, window, bool(xsa), float(q_scale), float(k_scale), float(eps),
                   A is not None, qk_norm, rope, csb, css, alpha.dtype if alpha is not None else None)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        q, k, v, O, LSE, A, wq, wk, cos, sin = ctx.saved_tensors
        scale, window, xsa, q_scale, k_scale, eps, has_a, qk_norm, rope, csb, css, a_dtype = ctx.cfg
        B, H, S, D = q.shape
        HKV = k.shape[1]
        G = H // HKV
        if dZ.stride(-1) != 1:
            dZ = dZ.contiguous()
        W = int(window) if window is not None else 0
        DO = torch.empty(B, H, S, D, device=q.device, dtype=q.dtype)
        DELTA = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        GA = torch.empty(B, H, S, device=q.device, dtype=torch.float32) if (xsa and has_a) else DELTA
        GVS = torch.empty(B, HKV, S, D, device=q.device, dtype=torch.float32) if xsa else DELTA
        c = CFG["pre"]
        _bwd_pre[(triton.cdiv(S, c["BM"]), B * HKV)](
            O, dZ, v, A if has_a else q, DO, DELTA, GA, GVS, *_st(dZ), *_st(v), S, H, HKV,
            GROUP=G, D=D, BM=c["BM"], XSA=xsa, HAS_A=has_a, num_warps=c["warps"])
        wq_ = wq if qk_norm else q
        wk_ = wk if qk_norm else q
        common = (*_st(q), *_st(k), *_st(v), csb, css, S, H, HKV, float(scale), q_scale, k_scale, eps, W)
        DK, DV = _like(k), _like(v)
        c = CFG["dkdv"]
        nkb = triton.cdiv(S, c["BN"])
        PWK = torch.empty(B * HKV * nkb, D, device=q.device, dtype=torch.float32) if qk_norm else DELTA
        _bwd_dkdv[(nkb, B * HKV)](
            q, k, v, DO, LSE, DELTA, GVS, DK, DV, PWK, wq_, wk_, cos, sin, *common,
            GROUP=G, D=D, BM=c["BM"], BN=c["BN"], WINDOW=window is not None, XSA=xsa,
            QK_NORM=qk_norm, ROPE=rope, num_warps=c["warps"], num_stages=c["stages"])
        DQ = _like(q)
        c = CFG["dq"]
        nqb = triton.cdiv(S, c["BM"])
        PWQ = torch.empty(B * HKV * nqb, D, device=q.device, dtype=torch.float32) if qk_norm else DELTA
        _bwd_dq[(nqb, B * HKV)](
            q, k, v, DO, LSE, DELTA, DQ, PWQ, wq_, wk_, cos, sin, *common,
            GROUP=G, D=D, BM=c["BM"], BN=c["BN"], WINDOW=window is not None, QK_NORM=qk_norm,
            ROPE=rope, num_warps=c["warps"], num_stages=c["stages"])
        d_alpha = None
        if xsa and has_a:
            d_alpha = (GA.sum(dim=(0, 2)) * (1.0 - A * A)).to(a_dtype)
        dwq = PWQ.sum(0).to(wq.dtype) if qk_norm else None
        dwk = PWK.sum(0).to(wk.dtype) if qk_norm else None
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
