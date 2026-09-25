"""Fused attention for BiBo: [qk RMSNorm] -> causal / sliding-window GQA attention -> [XSA], one
kernel family, deterministic backward (no atomics anywhere).

    z = attn_xsa(q, k, v, scale=..., window=None, xsa=True, alpha=None,
                 q_norm_w=None, k_norm_w=None, q_scale=1.0, k_scale=1.0, eps=1e-6)

q (B, H, S, D) and k, v (B, Hkv, S, D), bf16, contiguous, RAW (pre-norm) projections.
  qk norm   q_norm_w / k_norm_w (D,) -> q' = q_scale * RMSNorm(q) * w_q, same for k. None = no norm
            (then q_scale / k_scale still multiply q / k). The scalars are fixed hyperparameters.
  window    None = causal over the whole sequence; W = key j visible to query i iff i-W < j <= i.
  xsa       z = o - a * (o.v_i / |v_i|^2) v_i with v_i the query position's own value row and
            a = tanh(alpha[h]) (alpha None -> a = 1). xsa=False returns o.

Forward: one program per (query block, batch, kv head). BOTH query heads of the GQA group are
stacked into one (GROUP*BM, D) tile, so each K/V tile is loaded once for the group (no repeat_kv).
Online softmax; XSA in the epilogue (v_i is a row of V at the query's own position).

Backward, three passes, each output written by exactly one program:
  pre    XSA backward (dZ -> dO, the self-value grad, per-row alpha-grad terms) and the softmax
         delta D_i = dO_i . O_i
  dkdv   one program per key block, loops over every query block that sees it (both heads)
  dq     one program per query block, loops over its key blocks
qk-norm is recomputed on every load and differentiated through at the store; the norm-weight grads
are per-program partial rows summed on the host in a fixed order.
"""
import torch
import triton
import triton.language as tl

__all__ = ["attn_xsa", "AttnXSA", "attn_xsa_reference"]

# Per-kernel tiles. dkdv/dq at 2 stages with 64x64 tiles need 115-124 KB of shared memory (> 101 KB
# on sm120), so their tiles are set independently. Swept by parity_check/parity_attn_xsa.py --sweep.
CFG = {"fwd": dict(BM=64, BN=64, warps=8, stages=2), "pre": dict(BM=64, warps=4),
       "dkdv": dict(BM=64, BN=64, warps=8, stages=1), "dq": dict(BM=64, BN=64, warps=8, stages=1)}


@triton.jit
def _norm(x, w, scale, eps, D: tl.constexpr, QK_NORM: tl.constexpr):
    """x (R, D) fp32 raw -> (normalized * w * scale, rstd). Without QK_NORM: (x * scale, 1)."""
    if QK_NORM:
        r = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / D + eps)
        return x * r[:, None] * w[None, :] * scale, r
    return x * scale, tl.sum(x * 0.0, axis=1) + 1.0


@triton.jit
def _fwd(Q, K, V, O, Z, LSE, A, WQ, WK, S, H, HKV, sm_scale, q_scale, k_scale, eps, W,
         GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
         WINDOW: tl.constexpr, XSA: tl.constexpr, HAS_A: tl.constexpr, QK_NORM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    m0 = pid_m * BM
    r = tl.arange(0, GROUP * BM)
    j = r // BM
    s = m0 + (r % BM)
    h = kvh * GROUP + j
    rmask = s < S
    d = tl.arange(0, D)
    qrow = (b * H + h).to(tl.int64) * S + s
    kvbase = (b * HKV + kvh).to(tl.int64) * S
    wq = tl.load(WQ + d).to(tl.float32)     # a dummy pointer without QK_NORM; unused then
    wk = tl.load(WK + d).to(tl.float32)
    q = tl.load(Q + qrow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    q, _rq = _norm(q, wq, q_scale, eps, D, QK_NORM)
    q = q.to(tl.bfloat16)
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
        k = tl.load(K + (kvbase + n)[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        k, _rk = _norm(k, wk, k_scale, eps, D, QK_NORM)
        v = tl.load(V + (kvbase + n)[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(q, tl.trans(k.to(tl.bfloat16))) * sm_scale
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
    tl.store(LSE + qrow, m_i + tl.log(l_safe), mask=rmask)
    tl.store(O + qrow[:, None] * D + d[None, :], o.to(O.dtype.element_ty), mask=rmask[:, None])
    if XSA:
        vs = tl.load(V + (kvbase + s)[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        n2 = tl.sum(vs * vs, axis=1)
        inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
        c = tl.sum(o * vs, axis=1) * inv
        if HAS_A:
            c = c * tl.load(A + h).to(tl.float32)
        o = o - c[:, None] * vs
        tl.store(Z + qrow[:, None] * D + d[None, :], o.to(Z.dtype.element_ty), mask=rmask[:, None])


@triton.jit
def _bwd_pre(O, DZ, V, A, DO, DELTA, GA, GVS, S, H, HKV,
             GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr,
             XSA: tl.constexpr, HAS_A: tl.constexpr):
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    s = pid_m * BM + tl.arange(0, BM)
    rmask = s < S
    d = tl.arange(0, D)
    kvrow = (b * HKV + kvh).to(tl.int64) * S + s
    v = tl.load(V + kvrow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=1)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    gv = tl.zeros((BM, D), tl.float32)
    for jj in tl.static_range(GROUP):
        h = kvh * GROUP + jj
        row = (b * H + h).to(tl.int64) * S + s
        o = tl.load(O + row[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        gz = tl.load(DZ + row[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
        if XSA:
            dot = tl.sum(o * v, axis=1)
            gzv = tl.sum(gz * v, axis=1)
            coeff = dot * inv
            a = 1.0
            if HAS_A:
                a = tl.load(A + h).to(tl.float32)
                tl.store(GA + row, -coeff * gzv, mask=rmask)
            gy = gz - (a * gzv * inv)[:, None] * v
            gv += a * ((-(gzv * inv))[:, None] * o + (2.0 * dot * gzv * inv * inv)[:, None] * v
                       - coeff[:, None] * gz)
        else:
            gy = gz
        gy = gy.to(DO.dtype.element_ty)
        tl.store(DO + row[:, None] * D + d[None, :], gy, mask=rmask[:, None])
        tl.store(DELTA + row, tl.sum(gy.to(tl.float32) * o, axis=1), mask=rmask)
    if XSA:
        tl.store(GVS + kvrow[:, None] * D + d[None, :], gv, mask=rmask[:, None])


@triton.jit
def _norm_bwd(g, xr, r, w, scale, D: tl.constexpr):
    """Gradient through y = scale * w * (x * r), r = rsqrt(mean(x^2)+eps): (dx, per-column dw terms)."""
    xh = xr * r[:, None]
    gh = g * w[None, :] * scale
    dx = r[:, None] * (gh - xh * (tl.sum(gh * xh, axis=1) / D)[:, None])
    return dx, tl.sum(g * xh * scale, axis=0)


@triton.jit
def _bwd_dkdv(Q, K, V, DO, LSE, DELTA, GVS, DK, DV, PWK, WQ, WK,
              S, H, HKV, sm_scale, q_scale, k_scale, eps, W,
              GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
              WINDOW: tl.constexpr, XSA: tl.constexpr, QK_NORM: tl.constexpr):
    pid_n = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    n0 = pid_n * BN
    n = n0 + tl.arange(0, BN)
    nmask = n < S
    d = tl.arange(0, D)
    kvrow = (b * HKV + kvh).to(tl.int64) * S + n
    wq = tl.load(WQ + d).to(tl.float32)     # a dummy pointer without QK_NORM; unused then
    wk = tl.load(WK + d).to(tl.float32)
    kraw = tl.load(K + kvrow[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
    kn, rk = _norm(kraw, wk, k_scale, eps, D, QK_NORM)
    kb = kn.to(tl.bfloat16)
    v = tl.load(V + kvrow[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
    dk = tl.zeros((BN, D), tl.float32)
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
            row = (b * H + kvh * GROUP + jj).to(tl.int64) * S + s
            q = tl.load(Q + row[:, None] * D + d[None, :], mask=smask[:, None], other=0.0).to(tl.float32)
            q, _rq = _norm(q, wq, q_scale, eps, D, QK_NORM)
            qb = q.to(tl.bfloat16)
            do = tl.load(DO + row[:, None] * D + d[None, :], mask=smask[:, None], other=0.0)
            lse = tl.load(LSE + row, mask=smask, other=0.0)
            dl = tl.load(DELTA + row, mask=smask, other=0.0)
            st = tl.dot(kb, tl.trans(qb)) * sm_scale                         # (BN, BM) = S^T
            p = tl.where(valid, tl.exp(st - lse[None, :]), 0.0)
            dv += tl.dot(p.to(tl.bfloat16), do)
            dp = tl.dot(v, tl.trans(do))
            ds = p * (dp - dl[None, :])
            dk += tl.dot(ds.to(tl.bfloat16), qb)
    dk = dk * sm_scale
    if QK_NORM:
        dk, dwk = _norm_bwd(dk, kraw, rk, wk, k_scale, D)
        tl.store(PWK + (pid.to(tl.int64) * tl.num_programs(0) + pid_n) * D + d, dwk)
    else:
        dk = dk * k_scale
    if XSA:
        dv += tl.load(GVS + kvrow[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
    tl.store(DK + kvrow[:, None] * D + d[None, :], dk.to(DK.dtype.element_ty), mask=nmask[:, None])
    tl.store(DV + kvrow[:, None] * D + d[None, :], dv.to(DV.dtype.element_ty), mask=nmask[:, None])


@triton.jit
def _bwd_dq(Q, K, DO, LSE, DELTA, V, DQ, PWQ, WQ, WK, S, H, HKV, sm_scale, q_scale, k_scale, eps, W,
            GROUP: tl.constexpr, D: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
            WINDOW: tl.constexpr, QK_NORM: tl.constexpr):
    pid_m = tl.program_id(0)
    pid = tl.program_id(1)
    b = pid // HKV
    kvh = pid % HKV
    m0 = pid_m * BM
    r = tl.arange(0, GROUP * BM)
    s = m0 + (r % BM)
    h = kvh * GROUP + r // BM
    rmask = s < S
    d = tl.arange(0, D)
    qrow = (b * H + h).to(tl.int64) * S + s
    kvbase = (b * HKV + kvh).to(tl.int64) * S
    wq = tl.load(WQ + d).to(tl.float32)     # a dummy pointer without QK_NORM; unused then
    wk = tl.load(WK + d).to(tl.float32)
    qraw = tl.load(Q + qrow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    qn, rq = _norm(qraw, wq, q_scale, eps, D, QK_NORM)
    qb = qn.to(tl.bfloat16)
    do = tl.load(DO + qrow[:, None] * D + d[None, :], mask=rmask[:, None], other=0.0)
    lse = tl.load(LSE + qrow, mask=rmask, other=0.0)
    dl = tl.load(DELTA + qrow, mask=rmask, other=0.0)
    dq = tl.zeros((GROUP * BM, D), tl.float32)
    lo = 0
    if WINDOW:
        lo = tl.maximum(m0 - W + 1, 0) // BN * BN
    hi = tl.minimum(m0 + BM, S)
    for n0 in range(lo, hi, BN):
        n = n0 + tl.arange(0, BN)
        nmask = n < S
        k = tl.load(K + (kvbase + n)[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0).to(tl.float32)
        k, _rk = _norm(k, wk, k_scale, eps, D, QK_NORM)
        kb = k.to(tl.bfloat16)
        v = tl.load(V + (kvbase + n)[:, None] * D + d[None, :], mask=nmask[:, None], other=0.0)
        qk = tl.dot(qb, tl.trans(kb)) * sm_scale
        valid = (n[None, :] <= s[:, None]) & nmask[None, :]
        if WINDOW:
            valid = valid & ((s[:, None] - n[None, :]) < W)
        p = tl.where(valid, tl.exp(qk - lse[:, None]), 0.0)
        dp = tl.dot(do, tl.trans(v))
        ds = p * (dp - dl[:, None])
        dq += tl.dot(ds.to(tl.bfloat16), kb)
    dq = dq * sm_scale
    if QK_NORM:
        dq, dwq = _norm_bwd(dq, qraw, rq, wq, q_scale, D)
        tl.store(PWQ + (pid.to(tl.int64) * tl.num_programs(0) + pid_m) * D + d, dwq)
    else:
        dq = dq * q_scale
    tl.store(DQ + qrow[:, None] * D + d[None, :], dq.to(DQ.dtype.element_ty), mask=rmask[:, None])


class AttnXSA(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, alpha, q_norm_w, k_norm_w, scale, window, xsa, q_scale, k_scale, eps):
        B, H, S, D = q.shape
        HKV = k.shape[1]
        G = H // HKV
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        qk_norm = q_norm_w is not None
        A = torch.tanh(alpha.float()).contiguous() if alpha is not None else None
        wq = q_norm_w.contiguous() if qk_norm else q
        wk = k_norm_w.contiguous() if qk_norm else q
        O = torch.empty_like(q)
        Z = torch.empty_like(q) if xsa else O
        LSE = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        W = int(window) if window is not None else 0
        c = CFG["fwd"]
        _fwd[(triton.cdiv(S, c["BM"]), B * HKV)](
            q, k, v, O, Z, LSE, A if A is not None else q, wq, wk, S, H, HKV, float(scale),
            float(q_scale), float(k_scale), float(eps), W,
            GROUP=G, D=D, BM=c["BM"], BN=c["BN"], WINDOW=window is not None, XSA=bool(xsa),
            HAS_A=A is not None, QK_NORM=qk_norm, num_warps=c["warps"], num_stages=c["stages"])
        ctx.save_for_backward(q, k, v, O, LSE, A if A is not None else q.new_zeros(0),
                              q_norm_w if qk_norm else q.new_zeros(0),
                              k_norm_w if qk_norm else q.new_zeros(0))
        ctx.cfg = (scale, window, bool(xsa), float(q_scale), float(k_scale), float(eps),
                   A is not None, qk_norm, alpha.dtype if alpha is not None else None)
        return Z

    @staticmethod
    def backward(ctx, dZ):
        q, k, v, O, LSE, A, wq, wk = ctx.saved_tensors
        scale, window, xsa, q_scale, k_scale, eps, has_a, qk_norm, a_dtype = ctx.cfg
        B, H, S, D = q.shape
        HKV = k.shape[1]
        G = H // HKV
        dZ = dZ.contiguous()
        W = int(window) if window is not None else 0
        DO = torch.empty_like(q)
        DELTA = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
        GA = torch.empty(B, H, S, device=q.device, dtype=torch.float32) if (xsa and has_a) else DELTA
        GVS = torch.empty(B, HKV, S, D, device=q.device, dtype=torch.float32) if xsa else DELTA
        c = CFG["pre"]
        _bwd_pre[(triton.cdiv(S, c["BM"]), B * HKV)](
            O, dZ, v, A if has_a else q, DO, DELTA, GA, GVS, S, H, HKV,
            GROUP=G, D=D, BM=c["BM"], XSA=xsa, HAS_A=has_a, num_warps=c["warps"])
        wq_ = wq if qk_norm else q
        wk_ = wk if qk_norm else q
        DK, DV = torch.empty_like(k), torch.empty_like(v)
        c = CFG["dkdv"]
        nkb = triton.cdiv(S, c["BN"])
        PWK = torch.empty(B * HKV * nkb, D, device=q.device, dtype=torch.float32) if qk_norm else DELTA
        _bwd_dkdv[(nkb, B * HKV)](
            q, k, v, DO, LSE, DELTA, GVS, DK, DV, PWK, wq_, wk_, S, H, HKV, float(scale),
            q_scale, k_scale, eps, W, GROUP=G, D=D, BM=c["BM"], BN=c["BN"], WINDOW=window is not None,
            XSA=xsa, QK_NORM=qk_norm, num_warps=c["warps"], num_stages=c["stages"])
        DQ = torch.empty_like(q)
        c = CFG["dq"]
        nqb = triton.cdiv(S, c["BM"])
        PWQ = torch.empty(B * HKV * nqb, D, device=q.device, dtype=torch.float32) if qk_norm else DELTA
        _bwd_dq[(nqb, B * HKV)](
            q, k, DO, LSE, DELTA, v, DQ, PWQ, wq_, wk_, S, H, HKV, float(scale), q_scale, k_scale,
            eps, W, GROUP=G, D=D, BM=c["BM"], BN=c["BN"], WINDOW=window is not None, QK_NORM=qk_norm,
            num_warps=c["warps"], num_stages=c["stages"])
        d_alpha = None
        if xsa and has_a:
            d_alpha = (GA.sum(dim=(0, 2)) * (1.0 - A * A)).to(a_dtype)
        dwq = PWQ.sum(0).to(wq.dtype) if qk_norm else None
        dwk = PWK.sum(0).to(wk.dtype) if qk_norm else None
        return DQ, DK, DV, d_alpha, dwq, dwk, None, None, None, None, None, None


def attn_xsa(q, k, v, *, scale, window=None, xsa=True, alpha=None, q_norm_w=None, k_norm_w=None,
             q_scale=1.0, k_scale=1.0, eps=1e-6):
    return AttnXSA.apply(q, k, v, alpha, q_norm_w, k_norm_w, scale, window, xsa, q_scale, k_scale, eps)


def attn_xsa_reference(q, k, v, *, scale, window=None, xsa=True, alpha=None, q_norm_w=None,
                       k_norm_w=None, q_scale=1.0, k_scale=1.0, eps=1e-6, dtype=torch.float32):
    """Plain PyTorch in `dtype` (fp32 / fp64): the numerics target for parity."""
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    if q_norm_w is not None:
        q = q * torch.rsqrt(q.pow(2).mean(-1, keepdim=True) + eps) * q_norm_w.to(dtype)
        k = k * torch.rsqrt(k.pow(2).mean(-1, keepdim=True) + eps) * k_norm_w.to(dtype)
    q, k = q * q_scale, k * k_scale
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
