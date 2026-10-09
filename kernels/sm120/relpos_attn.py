"""Fused banded relative-position attention (sm120): NeMo RelPositionMultiHeadAttention's core, chunked_limited mask.

NeMo (rel_pos, use_pytorch_sdpa off) builds per layer and step:
  matrix_ac = (q+u) k^T                       (B, H, T, T)
  matrix_bd = rel_shift((q+v) p^T)            (B, H, T, 2T-1) -> pad (copy) -> view -> slice; 2T-1 is odd (cuBLAS align1)
  softmax(masked (ac + bd) / sqrt(d)), dropout, @ v
with a mask that keeps, under att_context_size [left, right] (right >= 0): chunk = right + 1, query i sees key j iff
0 <= i//chunk - j//chunk <= left // chunk, and both frames are inside the utterance. With [70, 13] a query sees <= 84
frames, so most of every T x T tile NeMo computes is masked away.

Here, per (b, h, block of BM queries), only the key tiles inside the band are visited (flash-style online softmax,
the T x T scores never exist). The position term needs p[j - i + T - 1] for every (i, j): for a BM x BN tile that is a
band of BM + BN - 1 rows of p, so G = (q + v) @ p_band^T (one tensor-core dot) and bd[a, c] = G[a, c - a + BM - 1]
(tl.gather: the skew).
  forward      : o (B, T, H, D), lse (B, H, T)
  backward     : query-major kernel -> dq (u and v paths: du, dv_bias sums) and every scaled dS tile stored;
                 key-major kernel -> dk, dv; _dp_diag -> d p (the position table) reading the stored dS along its
                 diagonals straight into the dot layout. (Computing d p from a GATHERED tile inside the query kernel
                 was 75% of its time: the gathered layout is converted before every MMA, whatever the precision.)
Deterministic (no atomics, fixed-order sums). Masked scores: NeMo fills -10000 and zeroes fully-masked rows after the
softmax; here masked pairs are skipped and padding query rows give o = 0, grads 0 -- the same numbers.

Precision follows the INPUT dtype, matching what NeMo actually runs:
  bf16 inputs  NeMo under bf16 autocast (avoid_float16_autocast_context only acts on fp16): q/k/v/p are bf16, the
               score and p@v matmuls are bf16 (fp32 accumulate, bf16 outputs), the softmax is fp32. Same here: bf16
               dots, and bf16 rounding where NeMo's tensors are bf16 (ac, bd, their sum, the scaled scores, p before
               @ v, the output; dp, dS in the backward).
  fp32 inputs  dots in 3xTF32 (fp32-accurate; plain TF32 was 4-7x less accurate than cuBLAS TF32 in parity).

    o = relpos_attention(q, k, v, p, pos_bias_u, pos_bias_v, lengths, left=70, right=13, dropout=0.1, seed=s)
q, k, v: (B, T, H, D) (linear(...).view, no transpose); p: (2T-1, H, D); biases (H, D) fp32; lengths (B,) int.
"""
import torch
import triton
import triton.language as tl

__all__ = ["relpos_attention"]

_BM, _BN = 32, 32                      # 64 x 64: the query-major backward needs 164 KB of shared memory


@triton.jit
def _r(x, LOWP: tl.constexpr):
    """round to NeMo's bf16 tensor at this point (identity in fp32 mode)"""
    if LOWP:
        return x.to(tl.bfloat16).to(tl.float32)
    return x


@triton.jit
def _mm(a, b, PREC: tl.constexpr, LOWP: tl.constexpr):
    if LOWP:
        return tl.dot(a.to(tl.bfloat16), b.to(tl.bfloat16))
    return tl.dot(a, b, input_precision=PREC)


@triton.jit
def _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, skb, skt, skh, spr, sph,
            BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr,
            LOWP: tl.constexpr):
    """masked scaled scores (BM, BN) for queries i0.., keys j0..; also returns the k tile."""
    offs_m = i0 + tl.arange(0, BM)
    offs_n = j0 + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    k = tl.load(K + b * skb + offs_n[:, None] * skt + h * skh + offs_d[None, :], mask=offs_n[:, None] < L,
                other=0.0).to(tl.float32)
    ac = _r(_mm(qu, tl.trans(k), PREC, LOWP), LOWP)
    rbase = j0 - i0 - (BM - 1) + T - 1
    pr = rbase + tl.arange(0, BP)
    pband = tl.load(P + pr[:, None] * spr + h * sph + offs_d[None, :],
                    mask=(pr[:, None] >= 0) & (pr[:, None] < 2 * T - 1) & (tl.arange(0, BP)[:, None] < BM + BN - 1),
                    other=0.0).to(tl.float32)
    g = _r(_mm(qv, tl.trans(pband), PREC, LOWP), LOWP)                                # (BM, BP)
    idx = tl.arange(0, BN)[None, :] - tl.arange(0, BM)[:, None] + (BM - 1)            # (BM, BN) in [0, BM+BN-2]
    s = _r(_r(ac + tl.gather(g, idx, axis=1), LOWP) * scale, LOWP)
    d = offs_m[:, None] // CH - offs_n[None, :] // CH
    ok = (d >= 0) & (d <= LC) & (offs_n[None, :] < L) & (offs_m[:, None] < L)
    return tl.where(ok, s, float("-inf")), ok, k


@triton.jit
def _keep(seed, bh, offs_m, offs_n, T, p_drop):
    return tl.rand(seed, ((bh * T + offs_m[:, None]) * T + offs_n[None, :]).to(tl.int32)) >= p_drop


@triton.jit
def _band(i0, L, CH, LC, BM: tl.constexpr):
    ci0 = i0 // CH
    ci1 = (tl.minimum(i0 + BM, L) - 1) // CH
    return tl.maximum(0, (ci0 - LC) * CH), tl.minimum(L, (ci1 + 1) * CH)


@triton.jit
def _qbias(Q, U, VB, qp, offs_m, offs_d, h, T, D: tl.constexpr, LOWP: tl.constexpr):
    """q + u and q + v: NeMo adds the fp32 biases to the bf16 q (-> fp32), autocast casts to bf16 for the matmul"""
    q = tl.load(Q + qp, mask=offs_m[:, None] < T, other=0.0).to(tl.float32)
    qu = _r(q + tl.load(U + h * D + offs_d)[None, :], LOWP)
    qv = _r(q + tl.load(VB + h * D + offs_d)[None, :], LOWP)
    return qu, qv


@triton.jit
def _rpa_fwd(Q, K, V, P, U, VB, O, LSE, LEN, seed, p_drop, T, H, CH, LC, scale,
             sqb, sqt, sqh, spr, sph, sob, sot, soh,
             BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr,
             PREC: tl.constexpr, LOWP: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    L = tl.load(LEN + b)
    i0 = pid_m * BM
    offs_m = i0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    qp = b * sqb + offs_m[:, None] * sqt + h * sqh + offs_d[None, :]
    qu, qv = _qbias(Q, U, VB, qp, offs_m, offs_d, h, T, D, LOWP)
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    if i0 < L:
        jlo, jhi = _band(i0, L, CH, LC, BM)
        for j0 in range(jlo, jhi, BN):
            s, ok, k = _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, sqb, sqt, sqh, spr, sph,
                               BM, BN, BP, D, PREC, LOWP)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            pe = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(pe, 1)
            offs_n = j0 + tl.arange(0, BN)
            if DROP:
                pe = tl.where(_keep(seed, bh, offs_m, offs_n, T, p_drop), pe * (1.0 / (1.0 - p_drop)), 0.0)
            v = tl.load(V + b * sqb + offs_n[:, None] * sqt + h * sqh + offs_d[None, :], mask=offs_n[:, None] < L,
                        other=0.0).to(tl.float32)
            acc = acc * alpha[:, None] + _mm(pe, v, PREC, LOWP)
            m_i = m_new
    live = l_i > 0.0
    o = tl.where(live[:, None], acc / tl.where(live, l_i, 1.0)[:, None], 0.0)
    op = b * sob + offs_m[:, None] * sot + h * soh + offs_d[None, :]
    tl.store(O + op, o.to(O.dtype.element_ty), mask=offs_m[:, None] < T)
    tl.store(LSE + bh * T + offs_m, tl.where(live, m_i + tl.log(tl.where(live, l_i, 1.0)), float("inf")),
             mask=offs_m < T)


@triton.jit
def _ds(s, ok, lse, delta, do, v, seed, bh, offs_m, offs_n, T, p_drop, scale,
        DROP: tl.constexpr, PREC: tl.constexpr, LOWP: tl.constexpr):
    """softmax backward for one tile -> (p, keep-scaled p, scaled dS); dS rounded like NeMo's bf16 score grads"""
    p = tl.exp(s - lse[:, None])                                                    # 0 where masked
    dp = _r(_mm(do, tl.trans(v), PREC, LOWP), LOWP)
    if DROP:
        keep = _keep(seed, bh, offs_m, offs_n, T, p_drop)
        pd = tl.where(keep, p * (1.0 / (1.0 - p_drop)), 0.0)
        dp = tl.where(keep, dp * (1.0 / (1.0 - p_drop)), 0.0)
    else:
        pd = p
    ds = _r(_r(tl.where(ok, p * (dp - delta[:, None]), 0.0), LOWP) * scale, LOWP)
    return pd, ds


@triton.jit
def _bwd_q(Q, K, V, P, U, VB, DO, LSE, DELTA, DQU, DQV, DSB, LEN, seed, p_drop, T, H, CH, LC, scale, NT,
           sqb, sqt, sqh, spr, sph, sob, sot, soh,
           BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr,
           PREC: tl.constexpr, LOWP: tl.constexpr):
    # query-major: dq (u path and v path separately); every scaled dS tile stored (its own slot) for _dp_diag
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    nm = tl.num_programs(0)
    b = bh // H
    h = bh % H
    L = tl.load(LEN + b)
    i0 = pid_m * BM
    offs_m = i0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    qp = b * sqb + offs_m[:, None] * sqt + h * sqh + offs_d[None, :]
    op = b * sob + offs_m[:, None] * sot + h * soh + offs_d[None, :]
    dqu = tl.zeros([BM, D], tl.float32)
    dqv = tl.zeros([BM, D], tl.float32)
    if i0 < L:
        qu, qv = _qbias(Q, U, VB, qp, offs_m, offs_d, h, T, D, LOWP)
        do = tl.load(DO + op, mask=offs_m[:, None] < T, other=0.0).to(tl.float32)
        lse = tl.load(LSE + bh * T + offs_m, mask=offs_m < T, other=float("inf"))
        delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
        jlo, jhi = _band(i0, L, CH, LC, BM)
        sbase = DSB + (bh * nm + pid_m).to(tl.int64) * NT * BM * BN
        for j0 in range(jlo, jhi, BN):
            s, ok, k = _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, sqb, sqt, sqh, spr, sph,
                               BM, BN, BP, D, PREC, LOWP)
            offs_n = j0 + tl.arange(0, BN)
            v = tl.load(V + b * sqb + offs_n[:, None] * sqt + h * sqh + offs_d[None, :], mask=offs_n[:, None] < L,
                        other=0.0).to(tl.float32)
            _, ds = _ds(s, ok, lse, delta, do, v, seed, bh, offs_m, offs_n, T, p_drop, scale, DROP, PREC, LOWP)
            dqu += _mm(ds, k, PREC, LOWP)
            # skew back for the v path: dg[a, m] = ds[a, m - (BM - 1) + a]
            mm = tl.arange(0, BP)[None, :] - (BM - 1) + tl.arange(0, BM)[:, None]
            dg = tl.where((mm >= 0) & (mm < BN), tl.gather(ds, tl.minimum(tl.maximum(mm, 0), BN - 1), axis=1), 0.0)
            pr = j0 - i0 - (BM - 1) + T - 1 + tl.arange(0, BP)
            pband = tl.load(P + pr[:, None] * spr + h * sph + offs_d[None, :],
                            mask=(pr[:, None] >= 0) & (pr[:, None] < 2 * T - 1) & (tl.arange(0, BP)[:, None] < BM + BN - 1),
                            other=0.0).to(tl.float32)
            dqv += _mm(dg, pband, PREC, LOWP)
            t = (j0 - jlo) // BN
            tl.store(sbase + (t * BM + tl.arange(0, BM)[:, None]) * BN + tl.arange(0, BN)[None, :],
                     ds.to(DSB.dtype.element_ty), mask=t < NT)
    tl.store(DQU + op, dqu, mask=offs_m[:, None] < T)
    tl.store(DQV + op, dqv, mask=offs_m[:, None] < T)


@triton.jit
def _bwd_kv(Q, K, V, P, U, VB, DO, LSE, DELTA, DK, DV, LEN, seed, p_drop, T, H, CH, LC, scale,
            sqb, sqt, sqh, spr, sph, sob, sot, soh,
            BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr,
            PREC: tl.constexpr, LOWP: tl.constexpr):
    # key-major: for this block of keys, every query tile whose band reaches it
    pid_n = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    L = tl.load(LEN + b)
    j0 = pid_n * BN
    offs_n = j0 + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    kp = b * sqb + offs_n[:, None] * sqt + h * sqh + offs_d[None, :]
    dk = tl.zeros([BN, D], tl.float32)
    dv = tl.zeros([BN, D], tl.float32)
    if j0 < L:
        v = tl.load(V + kp, mask=offs_n[:, None] < L, other=0.0).to(tl.float32)
        cj0 = j0 // CH
        cj1 = (tl.minimum(j0 + BN, L) - 1) // CH
        ihi = tl.minimum(L, (cj1 + LC + 1) * CH)
        for i0 in range((cj0 * CH // BM) * BM, ihi, BM):                       # query tiles on the forward's grid
            offs_m = i0 + tl.arange(0, BM)
            qp = b * sqb + offs_m[:, None] * sqt + h * sqh + offs_d[None, :]
            qu, qv = _qbias(Q, U, VB, qp, offs_m, offs_d, h, T, D, LOWP)
            s, ok, k = _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, sqb, sqt, sqh, spr, sph,
                               BM, BN, BP, D, PREC, LOWP)
            lse = tl.load(LSE + bh * T + offs_m, mask=offs_m < T, other=float("inf"))
            delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
            do = tl.load(DO + b * sob + offs_m[:, None] * sot + h * soh + offs_d[None, :], mask=offs_m[:, None] < T,
                         other=0.0).to(tl.float32)
            pd, ds = _ds(s, ok, lse, delta, do, v, seed, bh, offs_m, offs_n, T, p_drop, scale, DROP, PREC, LOWP)
            dv += _mm(tl.trans(pd), do, PREC, LOWP)
            dk += _mm(tl.trans(ds), qu, PREC, LOWP)
    kop = b * sob + offs_n[:, None] * sot + h * soh + offs_d[None, :]
    tl.store(DK + kop, dk.to(DK.dtype.element_ty), mask=offs_n[:, None] < T)
    tl.store(DV + kop, dv.to(DV.dtype.element_ty), mask=offs_n[:, None] < T)


@triton.jit
def _dp_diag(DSB, Q, VB, LEN, DPP, T, H, NM, NT, CH, LC, sqb, sqt, sqh,
             BM: tl.constexpr, BN: tl.constexpr, BR: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr,
             LOWP: tl.constexpr):
    # dp[b, r] = sum_i dS[i, j = i + r - (T-1)] (q+v)_i : dS read from its tiles along the diagonal, loaded in the
    # (r, i) orientation the dot wants (no register gather, no layout conversion)
    pid_r = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    bh = b * H + h
    L = tl.load(LEN + b)
    r = pid_r * BR + tl.arange(0, BR)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([BR, D], tl.float32)
    for m in range(0, NM):
        i0 = m * BM
        if i0 < L:
            i = i0 + tl.arange(0, BM)
            jlo, jhi = _band(i0, L, CH, LC, BM)
            nt = (jhi - jlo + BN - 1) // BN                                     # tiles _bwd_q wrote for this block
            j = i[None, :] + r[:, None] - (T - 1)                               # (BR, BM)
            rel = j - jlo
            t = rel // BN
            ok = (rel >= 0) & (t < nt) & (j < L) & (i[None, :] < L) & (r[:, None] < 2 * T - 1)
            ptr = DSB + ((bh * NM + m).to(tl.int64) * NT + t) * (BM * BN) + (i[None, :] - i0) * BN + rel % BN
            dsd = tl.load(ptr, mask=ok, other=0.0).to(tl.float32)
            qv = tl.load(Q + b * sqb + i[:, None] * sqt + h * sqh + offs_d[None, :], mask=(i < L)[:, None],
                         other=0.0).to(tl.float32)
            qv = _r(qv + tl.load(VB + h * D + offs_d)[None, :], LOWP)
            acc += _mm(dsd, qv, PREC, LOWP)
    tl.store(DPP + ((b * (2 * T - 1) + r[:, None]) * H + h).to(tl.int64) * D + offs_d[None, :], acc,
             mask=(r < 2 * T - 1)[:, None])


class _RelPosAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, p, u, vb, lengths, CH, LC, p_drop, seed, prec):
        B, T, H, D = q.shape
        lowp = q.dtype == torch.bfloat16
        dt = torch.bfloat16 if lowp else torch.float32
        q, k, v = (t.to(dt) for t in (q, k, v))
        if not (q.stride() == k.stride() == v.stride() and q.stride(-1) == 1):   # strided views (one QKV GEMM) are
            q, k, v = (t.contiguous() for t in (q, k, v))                        # read in place; outputs contiguous
        p = p.to(dt).contiguous()
        u, vb = u.float().contiguous(), vb.float().contiguous()
        assert k.shape == q.shape and v.shape == q.shape and p.shape == (2 * T - 1, H, D), (q.shape, p.shape)
        lengths = lengths.to(torch.int32).contiguous()
        o = torch.empty(B, T, H, D, device=q.device, dtype=dt)
        lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        BP = triton.next_power_of_2(_BM + _BN - 1)
        st = (q.stride(0), q.stride(1), q.stride(2), p.stride(0), p.stride(1)) + o.stride()[:3]
        _rpa_fwd[(triton.cdiv(T, _BM), B * H)](q, k, v, p, u, vb, o, lse, lengths, seed, p_drop, T, H, CH, LC,
                                               D ** -0.5, *st, BM=_BM, BN=_BN, BP=BP, D=D, DROP=p_drop > 0,
                                               PREC=prec, LOWP=lowp, num_warps=4, num_stages=1)
        ctx.save_for_backward(q, k, v, p, u, vb, o, lse, lengths)
        ctx.cfg = (CH, LC, p_drop, seed, prec, lowp)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, p, u, vb, o, lse, lengths = ctx.saved_tensors
        CH, LC, p_drop, seed, prec, lowp = ctx.cfg
        B, T, H, D = q.shape
        do = do.to(q.dtype).contiguous()
        delta = (do.float() * o.float()).sum(-1).permute(0, 2, 1).contiguous()          # (B, H, T)
        BP = triton.next_power_of_2(_BM + _BN - 1)
        NM = triton.cdiv(T, _BM)
        jspan = (LC + 2) * CH + _BM            # widest key range of one query block (it can straddle BM/CH + 1 chunks)
        NT = min(triton.cdiv(jspan, _BN), triton.cdiv(T, _BN)) + 1    # key tiles per query block (short clips: few)
        dqu = torch.empty(B, T, H, D, device=q.device, dtype=torch.float32)
        dqv = torch.empty_like(dqu)
        dk, dv = torch.empty_like(do), torch.empty_like(do)
        dsb = torch.empty(B * H * NM * NT * _BM * _BN, device=q.device, dtype=q.dtype)   # dS tiles
        st = (q.stride(0), q.stride(1), q.stride(2), p.stride(0), p.stride(1)) + do.stride()[:3]
        common = dict(BM=_BM, BN=_BN, BP=BP, D=D, DROP=p_drop > 0, PREC=prec, LOWP=lowp, num_warps=4, num_stages=1)
        _bwd_q[(NM, B * H)](q, k, v, p, u, vb, do, lse, delta, dqu, dqv, dsb, lengths, seed, p_drop, T, H, CH, LC,
                            D ** -0.5, NT, *st, **common)
        _bwd_kv[(triton.cdiv(T, _BN), B * H)](q, k, v, p, u, vb, do, lse, delta, dk, dv, lengths, seed, p_drop, T,
                                              H, CH, LC, D ** -0.5, *st, **common)
        dpp = torch.empty(B, 2 * T - 1, H, D, device=q.device, dtype=torch.float32)
        _dp_diag[(triton.cdiv(2 * T - 1, 32), H, B)](dsb, q, vb, lengths, dpp, T, H, NM, NT, CH, LC, *st[:3],
                                                     BM=_BM, BN=_BN, BR=32, D=D, PREC=prec, LOWP=lowp, num_warps=4)
        dp = dpp.sum(0).to(p.dtype)                                                     # deterministic sum over b
        return ((dqu + dqv).to(q.dtype), dk, dv, dp, dqu.sum((0, 1)), dqv.sum((0, 1)),
                None, None, None, None, None, None)


def relpos_attention(q, k, v, p, pos_bias_u, pos_bias_v, lengths, left, right, dropout=0.0, seed=None, prec="tf32x3"):
    """softmax(((q+u) k^T + rel_shift((q+v) p^T)) / sqrt(d), chunked_limited mask [left, right]) @ v -> (B, T, H, D),
    in q's dtype. dropout = the caller's attention dropout (0 in eval). prec: fp32 inputs only -- the tl.dot input
    precision ("tf32x3" default, "tf32", "ieee")."""
    assert right >= 0, "chunked_limited with a right context (right == -1 is the plain band: not implemented)"
    CH = right + 1
    LC = left // CH if left >= 0 else 1 << 20
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))
    return _RelPosAttn.apply(q, k, v, p, pos_bias_u, pos_bias_v, lengths, CH, LC, float(dropout), seed, prec)
