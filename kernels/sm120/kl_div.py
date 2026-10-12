"""Fused per-row KL divergence for logit distillation (sm120): KL(teacher || softmax(student / T)), loss + student grad.

Two teacher forms:
  kl_div_dense(s, t, T)          teacher LOGITS t (N, V), same vocabulary as the student logits s (N, V)
  kl_div_topk(s, idx, logp, T)   teacher as its top-k: token ids idx (N, k) and their log-probs logp (N, k) under the
                                 FULL teacher softmax (at temperature T); the remaining mass r = 1 - sum exp(logp) is
                                 one "rest" bucket matched to the student's mass outside the top-k (a stored teacher
                                 dump never needs densifying)
Per row i (s' = s / T, t' = t / T, q = softmax(s'), p = softmax(t')):
  dense  KL_i = sum_v p_v (log p_v - log q_v) = a / z_t - lse_t + lse_s,   a = sum_v e^(t'_v - m) (t'_v - s'_v)
         dKL_i / ds_v = (q_v - p_v) / T
  top-k  KL_i = sum_k p_k (log p_k - log q_k) + r (log r - log q_rest),     q_rest = 1 - sum_k q_k
         dKL_i / ds_v = (q_v - p_v) / T for v in the top-k,  q_v (1 - r / q_rest) / T otherwise
         (backward: the rest value written for the whole row, then the k entries overwritten after a barrier)
Forward: one pass over the row (online log-sum-exps, rescaled accumulator), per-row loss (N,) fp32 and the student /
teacher lse saved; nothing N x V is stored. Backward: recompute q (and p) from the saved lse, write ds in s's dtype,
scaled by the incoming per-row grad (so masks / weights / sums are the caller's: loss = (kl * w).sum()).
Math in fp32; deterministic (one program per row, no atomics). The Hinton T^2 factor is the caller's.

    kl = kl_div_dense(student_logits, teacher_logits.detach(), T=1.0)          # (N,)
    loss = (kl * valid).sum() / valid.sum()
"""
import torch
import triton
import triton.language as tl

__all__ = ["kl_div_dense", "kl_div_topk"]


def _bv(V):
    return min(triton.next_power_of_2(V), 4096)


@triton.jit
def _dense_fwd(S, Tl, LOSS, LSES, LSET, V, ss, st, inv_T, BV: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    sr, tr = S + i * ss, Tl + i * st
    m_s = tl.full((), float("-inf"), tl.float32)
    z_s = tl.zeros((), tl.float32)
    m_t = tl.full((), float("-inf"), tl.float32)
    z_t = tl.zeros((), tl.float32)
    acc = tl.zeros((), tl.float32)
    for v0 in range(0, V, BV):
        v = v0 + tl.arange(0, BV)
        mk = v < V
        s = tl.load(sr + v, mask=mk, other=0.0).to(tl.float32) * inv_T
        t = tl.load(tr + v, mask=mk, other=0.0).to(tl.float32) * inv_T
        sm = tl.where(mk, s, float("-inf"))
        tm = tl.where(mk, t, float("-inf"))
        nm_s = tl.maximum(m_s, tl.max(sm, 0))
        z_s = z_s * tl.exp(m_s - nm_s) + tl.sum(tl.exp(sm - nm_s), 0)
        m_s = nm_s
        nm_t = tl.maximum(m_t, tl.max(tm, 0))
        c = tl.exp(m_t - nm_t)
        e = tl.exp(tm - nm_t)
        z_t = z_t * c + tl.sum(e, 0)
        acc = acc * c + tl.sum(tl.where(mk, e * (t - s), 0.0), 0)
        m_t = nm_t
    lse_s = m_s + tl.log(z_s)
    lse_t = m_t + tl.log(z_t)
    tl.store(LOSS + i, acc / z_t - lse_t + lse_s)
    tl.store(LSES + i, lse_s)
    tl.store(LSET + i, lse_t)


@triton.jit
def _dense_bwd(S, Tl, LSES, LSET, G, DS, V, ss, st, sd, inv_T, BV: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    lse_s = tl.load(LSES + i)
    lse_t = tl.load(LSET + i)
    g = tl.load(G + i).to(tl.float32) * inv_T
    for v0 in range(0, V, BV):
        v = v0 + tl.arange(0, BV)
        mk = v < V
        s = tl.load(S + i * ss + v, mask=mk, other=0.0).to(tl.float32) * inv_T
        t = tl.load(Tl + i * st + v, mask=mk, other=0.0).to(tl.float32) * inv_T
        d = (tl.exp(s - lse_s) - tl.exp(t - lse_t)) * g
        tl.store(DS + i * sd + v, d.to(DS.dtype.element_ty), mask=mk)


class _Dense(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, t, T):
        assert s.shape == t.shape and s.dim() == 2 and s.stride(1) == 1 and t.stride(1) == 1
        N, V = s.shape
        loss = torch.empty(N, device=s.device, dtype=torch.float32)
        lse_s, lse_t = torch.empty_like(loss), torch.empty_like(loss)
        if N:
            _dense_fwd[(N,)](s, t, loss, lse_s, lse_t, V, s.stride(0), t.stride(0), 1.0 / T, BV=_bv(V), num_warps=8)
        ctx.save_for_backward(s, t, lse_s, lse_t)
        ctx.T = T
        return loss

    @staticmethod
    def backward(ctx, g):
        s, t, lse_s, lse_t = ctx.saved_tensors
        N, V = s.shape
        ds = torch.empty_like(s)
        if N:
            _dense_bwd[(N,)](s, t, lse_s, lse_t, g.contiguous(), ds, V, s.stride(0), t.stride(0), ds.stride(0),
                             1.0 / ctx.T, BV=_bv(V), num_warps=8)
        return ds, None, None


def kl_div_dense(student_logits, teacher_logits, T=1.0):
    """per-row KL(softmax(teacher/T) || softmax(student/T)) -> (N,) fp32; grad flows to the student only"""
    s = student_logits if student_logits.stride(-1) == 1 else student_logits.contiguous()
    t = teacher_logits.detach()
    t = t if t.stride(-1) == 1 else t.contiguous()
    return _Dense.apply(s, t, float(T))


@triton.jit
def _topk_fwd(S, IDX, LP, LOSS, LSES, QREST, V, ss, inv_T, K: tl.constexpr, KP: tl.constexpr, BV: tl.constexpr):
    i = tl.program_id(0).to(tl.int64)
    sr = S + i * ss
    m_s = tl.full((), float("-inf"), tl.float32)
    z_s = tl.zeros((), tl.float32)
    for v0 in range(0, V, BV):
        v = v0 + tl.arange(0, BV)
        mk = v < V
        s = tl.where(mk, tl.load(sr + v, mask=mk, other=0.0).to(tl.float32) * inv_T, float("-inf"))
        nm = tl.maximum(m_s, tl.max(s, 0))
        z_s = z_s * tl.exp(m_s - nm) + tl.sum(tl.exp(s - nm), 0)
        m_s = nm
    lse_s = m_s + tl.log(z_s)
    k = tl.arange(0, KP)
    mkk = k < K
    ix = tl.load(IDX + i * K + k, mask=mkk, other=0)
    lp = tl.load(LP + i * K + k, mask=mkk, other=float("-inf")).to(tl.float32)
    sk = tl.load(sr + ix, mask=mkk, other=0.0).to(tl.float32) * inv_T
    lq = sk - lse_s                                                     # student log q at the teacher's top-k
    p = tl.where(mkk, tl.exp(lp), 0.0)
    q = tl.where(mkk, tl.exp(lq), 0.0)
    r = tl.maximum(1.0 - tl.sum(p, 0), 0.0)
    q_rest = tl.maximum(1.0 - tl.sum(q, 0), 1e-30)
    kl = tl.sum(tl.where(mkk, p * (lp - lq), 0.0), 0)
    kl = kl + tl.where(r > 0, r * (tl.log(tl.maximum(r, 1e-30)) - tl.log(q_rest)), 0.0)
    tl.store(LOSS + i, kl)
    tl.store(LSES + i, lse_s)
    tl.store(QREST + i, q_rest)


@triton.jit
def _topk_bwd(S, IDX, LP, LSES, QREST, G, DS, V, ss, sd, inv_T, K: tl.constexpr, KB: tl.constexpr, BV: tl.constexpr):
    # pass 1: every v gets the rest-bucket grad q_v (1 - r / q_rest); barrier; pass 2: the k teacher entries are
    # overwritten with q_v - p_v (any k; no v x k membership tile)
    i = tl.program_id(0).to(tl.int64)
    lse_s = tl.load(LSES + i)
    q_rest = tl.load(QREST + i)
    g = tl.load(G + i).to(tl.float32) * inv_T
    psum = tl.zeros((), tl.float32)
    for k0 in range(0, K, KB):
        k = k0 + tl.arange(0, KB)
        psum += tl.sum(tl.exp(tl.load(LP + i * K + k, mask=k < K, other=float("-inf")).to(tl.float32)), 0)
    r = tl.maximum(1.0 - psum, 0.0)
    rest = 1.0 - r / q_rest
    for v0 in range(0, V, BV):
        v = v0 + tl.arange(0, BV)
        mk = v < V
        q = tl.exp(tl.load(S + i * ss + v, mask=mk, other=0.0).to(tl.float32) * inv_T - lse_s)
        tl.store(DS + i * sd + v, (q * rest * g).to(DS.dtype.element_ty), mask=mk)
    tl.debug_barrier()
    for k0 in range(0, K, KB):
        k = k0 + tl.arange(0, KB)
        mkk = k < K
        ix = tl.load(IDX + i * K + k, mask=mkk, other=0)
        pv = tl.exp(tl.load(LP + i * K + k, mask=mkk, other=float("-inf")).to(tl.float32))
        q = tl.exp(tl.load(S + i * ss + ix, mask=mkk, other=0.0).to(tl.float32) * inv_T - lse_s)
        tl.store(DS + i * sd + ix, ((q - pv) * g).to(DS.dtype.element_ty), mask=mkk)


class _TopK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, s, idx, logp, T):
        N, V = s.shape
        K = idx.shape[1]
        assert idx.shape == logp.shape == (N, K) and s.stride(1) == 1
        idx, logp = idx.contiguous().to(torch.int32), logp.contiguous()
        loss = torch.empty(N, device=s.device, dtype=torch.float32)
        lse_s, q_rest = torch.empty_like(loss), torch.empty_like(loss)
        KP = triton.next_power_of_2(K)
        if N:
            _topk_fwd[(N,)](s, idx, logp, loss, lse_s, q_rest, V, s.stride(0), 1.0 / T, K=K, KP=KP, BV=_bv(V), num_warps=8)
        ctx.save_for_backward(s, idx, logp, lse_s, q_rest)
        ctx.T, ctx.K, ctx.KP = T, K, KP
        return loss

    @staticmethod
    def backward(ctx, g):
        s, idx, logp, lse_s, q_rest = ctx.saved_tensors
        N, V = s.shape
        ds = torch.empty_like(s)
        if N:
            _topk_bwd[(N,)](s, idx, logp, lse_s, q_rest, g.contiguous(), ds, V, s.stride(0), ds.stride(0), 1.0 / ctx.T,
                            K=ctx.K, KB=min(ctx.KP, 1024), BV=_bv(V), num_warps=8)
        return ds, None, None, None


def kl_div_topk(student_logits, teacher_idx, teacher_logp, T=1.0):
    """per-row KL to a top-k teacher (+ one rest bucket) -> (N,) fp32; teacher_logp are log-probs under the FULL
    teacher softmax at temperature T (sum of exp < 1 leaves the rest mass)"""
    s = student_logits if student_logits.stride(-1) == 1 else student_logits.contiguous()
    return _TopK.apply(s, teacher_idx, teacher_logp.detach(), float(T))
