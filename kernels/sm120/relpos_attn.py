"""Fused banded relative-position attention (sm120): NeMo RelPositionMultiHeadAttention's core, chunked_limited mask.

NeMo (rel_pos, use_pytorch_sdpa off, autocast off inside: fp32 + TF32 matmuls) builds per layer and step:
  matrix_ac = (q+u) k^T                       (B, H, T, T)
  matrix_bd = rel_shift((q+v) p^T)            (B, H, T, 2T-1) -> pad (copy) -> view -> slice; 2T-1 is odd (cuBLAS align1)
  softmax(masked (ac + bd) / sqrt(d)), dropout, @ v
with a mask that keeps, under att_context_size [left, right] (right >= 0): chunk = right + 1, query i sees key j iff
0 <= i//chunk - j//chunk <= left // chunk, and both frames are inside the utterance. With [70, 13] a query sees <= 84
frames, so most of every T x T tile NeMo computes is masked away.

Here, per (b, h, block of BM queries), only the key tiles inside the band are visited (flash-style online softmax,
the T x T scores never exist). The position term needs p[j - i + T - 1] for every (i, j): for a BM x BN tile that is a
band of BM + BN - 1 rows of p, so G = (q + v) @ p_band^T (one tensor-core dot) and bd[a, c] = G[a, c - a + BM - 1]
(tl.gather: the skew). The backward is the same gather in reverse: dG[a, m] = dS[a, m - BM + 1 + a].
  forward  : o (B, T, H, D) fp32, lse (B, H, T)
  backward : query-major kernel -> dq (split into the u and v paths: du, dv_bias sums) + per-block private
             partials of dp (the position table), key-major kernel -> dk, dv, then a fixed-order torch reduction of dp.
Deterministic (no atomics). Dots run 3xTF32 (as accurate as NeMo's cuBLAS TF32; prec="ieee" for the gradient check).
Masked scores: NeMo fills -10000 and zeroes fully-masked rows after softmax; here masked pairs are skipped and
fully-masked (padding) query rows give o = 0, grads 0 -- the same numbers.

    o = relpos_attention(q, k, v, p, pos_bias_u, pos_bias_v, lengths, left=70, right=13, dropout=0.1, seed=s)
q, k, v: (B, T, H, D) fp32 (linear(...).view, no transpose); p: (2T-1, H, D); biases (H, D); lengths (B,) int.
"""
import torch
import triton
import triton.language as tl

__all__ = ["relpos_attention"]


@triton.jit
def _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, skb, skt, skh, spr, sph,
            BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, PREC: tl.constexpr):
    """masked scaled scores (BM, BN) for queries i0.., keys j0..; also returns the p band and the gather index."""
    offs_m = i0 + tl.arange(0, BM)
    offs_n = j0 + tl.arange(0, BN)
    offs_d = tl.arange(0, D)
    k = tl.load(K + b * skb + offs_n[:, None] * skt + h * skh + offs_d[None, :], mask=offs_n[:, None] < L, other=0.0)
    s = tl.dot(qu, tl.trans(k), input_precision=PREC)
    rbase = j0 - i0 - (BM - 1) + T - 1
    pr = rbase + tl.arange(0, BP)
    pband = tl.load(P + pr[:, None] * spr + h * sph + offs_d[None, :],
                    mask=(pr[:, None] >= 0) & (pr[:, None] < 2 * T - 1) & (tl.arange(0, BP)[:, None] < BM + BN - 1),
                    other=0.0)
    g = tl.dot(qv, tl.trans(pband), input_precision=PREC)                          # (BM, BP)
    idx = tl.arange(0, BN)[None, :] - tl.arange(0, BM)[:, None] + (BM - 1)         # (BM, BN) in [0, BM+BN-2]
    s = (s + tl.gather(g, idx, axis=1)) * scale
    d = offs_m[:, None] // CH - offs_n[None, :] // CH
    ok = (d >= 0) & (d <= LC) & (offs_n[None, :] < L) & (offs_m[:, None] < L)
    return tl.where(ok, s, float("-inf")), ok, k, pband


@triton.jit
def _keep(seed, bh, offs_m, offs_n, T, p_drop):
    return tl.rand(seed, ((bh * T + offs_m[:, None]) * T + offs_n[None, :]).to(tl.int32)) >= p_drop


@triton.jit
def _fwd(Q, K, V, P, U, VB, O, LSE, LEN, seed, p_drop, T, H, CH, LC, scale,
         sqb, sqt, sqh, spr, sph,
         BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr,
         PREC: tl.constexpr, PRECG: tl.constexpr):
    pid_m = tl.program_id(0)
    bh = tl.program_id(1)
    b = bh // H
    h = bh % H
    L = tl.load(LEN + b)
    i0 = pid_m * BM
    offs_m = i0 + tl.arange(0, BM)
    offs_d = tl.arange(0, D)
    qp = b * sqb + offs_m[:, None] * sqt + h * sqh + offs_d[None, :]
    q = tl.load(Q + qp, mask=offs_m[:, None] < T, other=0.0)
    qu = q + tl.load(U + h * D + offs_d)[None, :]
    qv = q + tl.load(VB + h * D + offs_d)[None, :]
    m_i = tl.full([BM], float("-inf"), tl.float32)
    l_i = tl.zeros([BM], tl.float32)
    acc = tl.zeros([BM, D], tl.float32)
    if i0 < L:
        ci0 = i0 // CH
        ci1 = (tl.minimum(i0 + BM, L) - 1) // CH
        jlo = tl.maximum(0, (ci0 - LC) * CH)
        jhi = tl.minimum(L, (ci1 + 1) * CH)
        for j0 in range(jlo, jhi, BN):
            s, ok, k, _ = _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, sqb, sqt, sqh, spr, sph,
                                  BM, BN, BP, D, PREC)
            m_new = tl.maximum(m_i, tl.max(s, 1))
            m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
            alpha = tl.exp(m_i - m_safe)
            pe = tl.exp(s - m_safe[:, None])
            l_i = l_i * alpha + tl.sum(pe, 1)
            offs_n = j0 + tl.arange(0, BN)
            if DROP:
                pe = tl.where(_keep(seed, bh, offs_m, offs_n, T, p_drop), pe * (1.0 / (1.0 - p_drop)), 0.0)
            v = tl.load(V + b * sqb + offs_n[:, None] * sqt + h * sqh + offs_d[None, :], mask=offs_n[:, None] < L,
                        other=0.0)
            acc = acc * alpha[:, None] + tl.dot(pe, v, input_precision=PRECG)
            m_i = m_new
    live = l_i > 0.0
    o = tl.where(live[:, None], acc / tl.where(live, l_i, 1.0)[:, None], 0.0)
    tl.store(O + qp, o, mask=offs_m[:, None] < T)
    tl.store(LSE + bh * T + offs_m, tl.where(live, m_i + tl.log(tl.where(live, l_i, 1.0)), float("inf")),
             mask=offs_m < T)


@triton.jit
def _bwd_q(Q, K, V, P, U, VB, DO, LSE, DELTA, DQU, DQV, DPQ, LEN, seed, p_drop, T, H, CH, LC, scale, NT,
           sqb, sqt, sqh, spr, sph,
           BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr,
           PREC: tl.constexpr, PRECG: tl.constexpr, WIDE: tl.constexpr,
           SKIP: tl.constexpr = 0, PRECP: tl.constexpr = "tf32x3"):
    # query-major: dq (u path and v path separately), and per key tile its partial of d p, each in its own slot
    # (written once: a load-add-store per tile behind a barrier made this kernel 4x the key-major one)
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
    dqu = tl.zeros([BM, D], tl.float32)
    dqv = tl.zeros([BM, D], tl.float32)
    if i0 < L:
        q = tl.load(Q + qp, mask=offs_m[:, None] < T, other=0.0)
        qu = q + tl.load(U + h * D + offs_d)[None, :]
        qv = q + tl.load(VB + h * D + offs_d)[None, :]
        do = tl.load(DO + qp, mask=offs_m[:, None] < T, other=0.0)
        lse = tl.load(LSE + bh * T + offs_m, mask=offs_m < T, other=float("inf"))
        delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
        ci0 = i0 // CH
        ci1 = (tl.minimum(i0 + BM, L) - 1) // CH
        jlo = tl.maximum(0, (ci0 - LC) * CH)
        jhi = tl.minimum(L, (ci1 + 1) * CH)
        dbase = DPQ + (bh * nm + pid_m).to(tl.int64) * NT * BP * D
        for j0 in range(jlo, jhi, BN):
            s, ok, k, pband = _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, sqb, sqt, sqh, spr, sph,
                                      BM, BN, BP, D, PREC)
            p = tl.exp(s - lse[:, None])                                     # 0 where masked
            offs_n = j0 + tl.arange(0, BN)
            v = tl.load(V + b * sqb + offs_n[:, None] * sqt + h * sqh + offs_d[None, :], mask=offs_n[:, None] < L,
                        other=0.0)
            dp = tl.dot(do, tl.trans(v), input_precision=PRECG)
            if DROP:
                dp = tl.where(_keep(seed, bh, offs_m, offs_n, T, p_drop), dp * (1.0 / (1.0 - p_drop)), 0.0)
            ds = tl.where(ok, p * (dp - delta[:, None]), 0.0) * scale
            dqu += tl.dot(ds, k, input_precision=PRECG)
            # skew back: dg[a, m] = ds[a, m - (BM - 1) + a]
            mm = tl.arange(0, BP)[None, :] - (BM - 1) + tl.arange(0, BM)[:, None]
            if WIDE:
                # same-shape gather: ds interleaved with zeros to (BM, 2 BN) = (BM, BP); out-of-range -> a zero slot
                wide = tl.reshape(tl.join(ds, tl.zeros_like(ds)), (BM, 2 * BN))
                dg = tl.gather(wide, tl.where((mm >= 0) & (mm < BN), 2 * mm, 1), axis=1)
            else:
                dg = tl.where((mm >= 0) & (mm < BN), tl.gather(ds, tl.minimum(tl.maximum(mm, 0), BN - 1), axis=1), 0.0)
            if SKIP < 2:
                dqv += tl.dot(dg, pband, input_precision=PRECG)
            if SKIP < 1:
                # dg^T gathered directly from ds^T (axis 0): tl.trans of the gathered dg before the dot was 75% of
                # this kernel; ds^T is a plain register tile
                mt = tl.arange(0, BP)[:, None] - (BM - 1) + tl.arange(0, BM)[None, :]          # (BP, BM)
                dgt = tl.where((mt >= 0) & (mt < BN),
                               tl.gather(tl.trans(ds), tl.minimum(tl.maximum(mt, 0), BN - 1), axis=0), 0.0)
                dpb = tl.dot(dgt, qv, input_precision=PRECP)                 # (BP, D) rows rbase .. rbase+BP-1
                t = (j0 - jlo) // BN
                if SKIP == -1:                                               # debug: the dot without its store
                    dqv += tl.sum(dpb, 0)[None, :] * 1e-30
                else:
                    tl.store(dbase + (t * BP + tl.arange(0, BP)[:, None]) * D + offs_d[None, :], dpb, mask=t < NT)
    tl.store(DQU + qp, dqu, mask=offs_m[:, None] < T)
    tl.store(DQV + qp, dqv, mask=offs_m[:, None] < T)


@triton.jit
def _bwd_kv(Q, K, V, P, U, VB, DO, LSE, DELTA, DK, DV, LEN, seed, p_drop, T, H, CH, LC, scale,
            sqb, sqt, sqh, spr, sph,
            BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr, D: tl.constexpr, DROP: tl.constexpr,
            PREC: tl.constexpr, PRECG: tl.constexpr):
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
        v = tl.load(V + kp, mask=offs_n[:, None] < L, other=0.0)
        cj0 = j0 // CH
        cj1 = (tl.minimum(j0 + BN, L) - 1) // CH
        ilo = cj0 * CH
        ihi = tl.minimum(L, (cj1 + LC + 1) * CH)
        ilo = (ilo // BM) * BM                                               # query tiles on the forward's grid
        for i0 in range(ilo, ihi, BM):
            offs_m = i0 + tl.arange(0, BM)
            qp = b * sqb + offs_m[:, None] * sqt + h * sqh + offs_d[None, :]
            q = tl.load(Q + qp, mask=offs_m[:, None] < T, other=0.0)
            qu = q + tl.load(U + h * D + offs_d)[None, :]
            qv = q + tl.load(VB + h * D + offs_d)[None, :]
            s, ok, k, _ = _scores(qu, qv, K, P, b, h, i0, j0, T, L, CH, LC, scale, sqb, sqt, sqh, spr, sph,
                                  BM, BN, BP, D, PREC)
            lse = tl.load(LSE + bh * T + offs_m, mask=offs_m < T, other=float("inf"))
            delta = tl.load(DELTA + bh * T + offs_m, mask=offs_m < T, other=0.0)
            p = tl.exp(s - lse[:, None])
            do = tl.load(DO + qp, mask=offs_m[:, None] < T, other=0.0)
            dp = tl.dot(do, tl.trans(v), input_precision=PRECG)
            if DROP:
                keep = _keep(seed, bh, offs_m, offs_n, T, p_drop)
                pd = tl.where(keep, p * (1.0 / (1.0 - p_drop)), 0.0)
                dp = tl.where(keep, dp * (1.0 / (1.0 - p_drop)), 0.0)
            else:
                pd = p
            dv += tl.dot(tl.trans(pd), do, input_precision=PRECG)
            ds = tl.where(ok, p * (dp - delta[:, None]), 0.0) * scale
            dk += tl.dot(tl.trans(ds), qu, input_precision=PRECG)
    tl.store(DK + kp, dk, mask=offs_n[:, None] < T)
    tl.store(DV + kp, dv, mask=offs_n[:, None] < T)


@triton.jit
def _dp_reduce(PART, DP, T, H, NM, NT, CH, LC, BM: tl.constexpr, BN: tl.constexpr, BP: tl.constexpr,
               BR: tl.constexpr, D: tl.constexpr):
    # dp[r, h] = sum over (query block m, key tile t) of part[h, m, t, r - off(m) - t BN]; fixed order
    pid_r = tl.program_id(0)
    h = tl.program_id(1)
    r = pid_r * BR + tl.arange(0, BR)
    offs_d = tl.arange(0, D)
    acc = tl.zeros([BR, D], tl.float32)
    for m in range(0, NM):
        i0 = m * BM
        off = tl.maximum(0, (i0 // CH - LC) * CH) - i0 - (BM - 1) + T - 1
        for t in range(0, NT):
            rr = r - off - t * BN
            ok = (rr >= 0) & (rr < BP) & (r < 2 * T - 1)
            ptr = PART + (((h * NM + m) * NT + t) * BP + rr[:, None]).to(tl.int64) * D + offs_d[None, :]
            acc += tl.load(ptr, mask=ok[:, None], other=0.0)
    tl.store(DP + r[:, None] * (H * D) + h * D + offs_d[None, :], acc, mask=(r < 2 * T - 1)[:, None])


_WIDE = __import__("os").environ.get("RPA_WIDE", "1") == "1"
_WQ = int(__import__("os").environ.get("RPA_WQ", 4))
_WKV = int(__import__("os").environ.get("RPA_WKV", 4))
_BM, _BN = int(__import__("os").environ.get("RPA_BM", 32)), int(__import__("os").environ.get("RPA_BN", 32))


class _RelPosAttn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, p, u, vb, lengths, CH, LC, p_drop, seed, prec, precg):
        B, T, H, D = q.shape
        q, k, v = (t.float().contiguous() for t in (q, k, v))
        p, u, vb = p.float().contiguous(), u.float().contiguous(), vb.float().contiguous()
        assert k.shape == q.shape and v.shape == q.shape and p.shape == (2 * T - 1, H, D), (q.shape, p.shape)
        lengths = lengths.to(torch.int32).contiguous()
        o = torch.empty_like(q)
        lse = torch.empty(B, H, T, device=q.device, dtype=torch.float32)
        BP = triton.next_power_of_2(_BM + _BN - 1)
        st = (q.stride(0), q.stride(1), q.stride(2), p.stride(0), p.stride(1))
        _fwd[(triton.cdiv(T, _BM), B * H)](q, k, v, p, u, vb, o, lse, lengths, seed, p_drop, T, H, CH, LC,
                                           D ** -0.5, *st, BM=_BM, BN=_BN, BP=BP, D=D, DROP=p_drop > 0,
                                           PREC=prec, PRECG=precg, num_warps=4, num_stages=1)
        ctx.save_for_backward(q, k, v, p, u, vb, o, lse, lengths)
        ctx.cfg = (CH, LC, p_drop, seed, prec, precg)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, p, u, vb, o, lse, lengths = ctx.saved_tensors
        CH, LC, p_drop, seed, prec, precg = ctx.cfg
        B, T, H, D = q.shape
        do = do.float().contiguous()
        delta = (do * o).sum(-1).permute(0, 2, 1).contiguous()          # (B, H, T)
        BP = triton.next_power_of_2(_BM + _BN - 1)
        NM = triton.cdiv(T, _BM)
        jspan = (LC + 2) * CH + _BM            # widest key range of one query block (it can straddle BM/CH + 1 chunks)
        NT = min(triton.cdiv(jspan, _BN), triton.cdiv(T, _BN)) + 1   # key tiles per query block (short clips: few)
        dqu, dqv, dk, dv = (torch.empty_like(q) for _ in range(4))
        dpq = torch.zeros(B * H * NM * NT * BP * D, device=q.device, dtype=torch.float32)
        st = (q.stride(0), q.stride(1), q.stride(2), p.stride(0), p.stride(1))
        common = dict(BM=_BM, BN=_BN, BP=BP, D=D, DROP=p_drop > 0, PREC=prec, PRECG=precg, num_stages=1)
        _bwd_q[(NM, B * H)](q, k, v, p, u, vb, do, lse, delta, dqu, dqv, dpq, lengths, seed, p_drop, T, H, CH, LC,
                            D ** -0.5, NT, *st, num_warps=_WQ, WIDE=_WIDE and BP == 2 * _BN,
                            SKIP=int(__import__("os").environ.get("RPA_SKIP", 0)),
                            PRECP=__import__("os").environ.get("RPA_PRECP", precg), **common)
        _bwd_kv[(triton.cdiv(T, _BN), B * H)](q, k, v, p, u, vb, do, lse, delta, dk, dv, lengths, seed, p_drop, T,
                                              H, CH, LC, D ** -0.5, *st, num_warps=_WKV, **common)
        part = dpq.view(B, H * NM * NT * BP * D).sum(0)                # deterministic sum over b
        dp = torch.empty_like(p)
        BR = 32
        _dp_reduce[(triton.cdiv(2 * T - 1, BR), H)](part, dp, T, H, NM, NT, CH, LC, BM=_BM, BN=_BN, BP=BP, BR=BR, D=D,
                                                    num_warps=4)
        du = dqu.sum((0, 1))
        dvb = dqv.sum((0, 1))
        return dqu + dqv, dk, dv, dp, du, dvb, None, None, None, None, None, None, None


def relpos_attention(q, k, v, p, pos_bias_u, pos_bias_v, lengths, left, right, dropout=0.0, seed=None, prec="tf32x3", precg=None):
    """softmax(((q+u) k^T + rel_shift((q+v) p^T)) / sqrt(d), chunked_limited mask [left, right]) @ v -> (B, T, H, D).
    dropout = the caller's attention dropout (0 in eval). prec: tl.dot input precision -- "tf32x3" (default: as accurate as
    NeMo's cuBLAS TF32 matmuls; plain "tf32" was 4-7x less accurate in parity), "tf32", or "ieee"; precg: the
    p@v / gradient dots (default = prec). Score dots feed exp(), so they need the precision most."""
    assert right >= 0, "chunked_limited with a right context (right == -1 is the plain band: not implemented)"
    CH = right + 1
    LC = left // CH if left >= 0 else 1 << 20
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))
    return _RelPosAttn.apply(q, k, v, p, pos_bias_u, pos_bias_v, lengths, CH, LC, float(dropout), seed, prec, precg or prec)
