"""Self-conditioned CTC vocab pass (sm120), fused for TRAINING SPEED (memory is spent, not recomputed).

One pass of a self-conditioned CTC head (BiBo voice/asr/ctc_heads.py SelfCondN): logits z = x W_out^T + b over the
vocabulary, the CTC loss on log_softmax(z), and -- for every pass but the last -- the feedback q = softmax(z) W_fb^T
(W_fb = [W_cur; W_prev], (2d, V): the caller adds q[:, :d] + shift_t(q[:, d:]) to the features of the next pass).

    nll, q = selfcond_ctc_pass(x, w_out, b_out, w_fb, targets, input_lengths, target_lengths, blank)
        x (B, T, d) any float dtype (bf16 math), w_out (V, d), b_out (V,), w_fb (2d, V) or None (last pass)
        -> nll (B,) fp32 (torch ctc_loss reduction='none', zero_infinity) and q (B, T, 2d) bf16 or None

Every GEMM is cuBLAS (bf16, fp32 accumulate). Triton does the V-wide row work:
  _softmax_rows  z (bf16) -> p = softmax(z) (bf16, stored) and lse (fp32) per frame
  _ctc_ab_z      alpha / beta / nll reading log p = z[label] - lse (no V-wide log_softmax tensor)
  _sc_grad       one program per frame: dz = p * (g + a - <dq, q>) - g * occupancy, where g = dL/dnll of the
                 utterance, a = dq W_fb (the feedback gradient, cuBLAS) and occupancy = sum over the CTC states of that
                 token of alpha*beta/p. Blank: one fixed-order sum; labels: first-occurrence writes (as ctc_loss.py).
Then dx = dz W_out, dW_out = dz^T x, db = sum dz, dW_fb = dq^T p (cuBLAS). Deterministic: no atomics anywhere.
Stored per pass: z and p (B*T*V bf16 each), alpha / beta (B*T*S fp32) -- ~125 MB per pass at 15k frames x 2k vocab.
"""
import torch
import triton
import triton.language as tl

__all__ = ["selfcond_ctc_pass"]


@triton.jit
def _lse2(a, b):
    m = tl.maximum(a, b)
    ms = tl.where(m == float("-inf"), 0.0, m)
    return ms + tl.log(tl.exp(a - ms) + tl.exp(b - ms))


@triton.jit
def _softmax_rows(Z, P, LSE, V, BV: tl.constexpr):
    r = tl.program_id(0).to(tl.int64)
    c = tl.arange(0, BV)
    m = c < V
    z = tl.load(Z + r * V + c, mask=m, other=float("-inf")).to(tl.float32)
    mx = tl.max(z, 0)
    e = tl.exp(z - mx)
    s = tl.sum(e, 0)
    tl.store(P + r * V + c, (e / s).to(P.dtype.element_ty), mask=m)
    tl.store(LSE + r, mx + tl.log(s))


@triton.jit
def _ctc_ab_z(Z, LSE, Y, TLEN, ULEN, AL, BE, NLL, sy, T, V, BLANK, BS: tl.constexpr):
    """ctc_loss.py's _ctc_ab with lp = z[lab] - lse[t]; zero_infinity always on."""
    b = tl.program_id(0)
    tb = tl.load(TLEN + b).to(tl.int32)
    ub = tl.load(ULEN + b).to(tl.int32)
    S = 2 * ub + 1
    s = tl.arange(0, BS)
    ms = s < S
    odd = (s % 2) == 1
    lab = tl.where(odd, tl.load(Y + b.to(tl.int64) * sy + s // 2, mask=ms & odd, other=0), BLANK)
    labm2 = tl.load(Y + b.to(tl.int64) * sy + (s - 2) // 2, mask=ms & odd & (s >= 2), other=-1)
    skip = odd & (s >= 2) & (lab != labm2)
    labp2 = tl.load(Y + b.to(tl.int64) * sy + (s + 2) // 2, mask=odd & (s + 2 < S), other=-1)
    skipn = odd & (s + 2 < S) & (labp2 != lab)
    zb = Z + b.to(tl.int64) * T * V
    lb = LSE + b.to(tl.int64) * T
    ab = AL + b.to(tl.int64) * T * BS
    bb = BE + b.to(tl.int64) * T * BS
    a = tl.where(ms & (s < 2), tl.load(zb + lab, mask=ms, other=0.0).to(tl.float32) - tl.load(lb), float("-inf"))
    tl.store(ab + s, a)
    for t in range(1, tb):
        tl.debug_barrier()
        a1 = tl.load(ab + (t - 1) * BS + s - 1, mask=s >= 1, other=float("-inf"))
        a2 = tl.load(ab + (t - 1) * BS + s - 2, mask=skip, other=float("-inf"))
        lp = tl.load(zb + t * V + lab, mask=ms, other=0.0).to(tl.float32) - tl.load(lb + t)
        a = tl.where(ms, _lse2(_lse2(a, a1), a2) + lp, float("-inf"))
        tl.store(ab + t * BS + s, a)
    fin = ms & (s >= S - 2)
    m = tl.max(tl.where(fin, a, float("-inf")), 0)
    msafe = tl.where(m == float("-inf"), 0.0, m)
    nll = -(msafe + tl.log(tl.sum(tl.where(fin, tl.exp(a - msafe), 0.0), 0)))
    nll = tl.where(tb > 0, nll, float("inf"))
    tl.store(NLL + b, tl.where(nll == float("inf"), 0.0, nll))
    tl.debug_barrier()
    bt = tl.where(fin, tl.load(zb + (tb - 1) * V + lab, mask=ms & (tb > 0), other=0.0).to(tl.float32)
                  - tl.load(lb + tb - 1, mask=tb > 0, other=0.0), float("-inf"))
    tl.store(bb + (tb - 1) * BS + s, bt, mask=tb > 0)
    for i in range(1, tb):
        t = tb - 1 - i
        tl.debug_barrier()
        b1 = tl.load(bb + (t + 1) * BS + s + 1, mask=s + 1 < S, other=float("-inf"))
        b2 = tl.load(bb + (t + 1) * BS + s + 2, mask=skipn, other=float("-inf"))
        lp = tl.load(zb + t * V + lab, mask=ms, other=0.0).to(tl.float32) - tl.load(lb + t)
        bt = tl.where(ms, _lse2(_lse2(bt, b1), b2) + lp, float("-inf"))
        tl.store(bb + t * BS + s, bt)


@triton.jit
def _sc_grad(Z, LSE, P, A, QDQ, Y, TLEN, ULEN, AL, BE, NLL, GOUT, DZ, sy, T, V, BLANK,
             BS: tl.constexpr, BV: tl.constexpr, BU: tl.constexpr, CJ: tl.constexpr, HAS_FB: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1)
    row = b.to(tl.int64) * T + t
    tb = tl.load(TLEN + b).to(tl.int32)
    ub = tl.load(ULEN + b).to(tl.int32)
    k = tl.arange(0, BV)
    mk = k < V
    p = tl.load(P + row * V + k, mask=mk, other=0.0).to(tl.float32)
    fb = tl.zeros([BV], tl.float32)
    if HAS_FB:                                         # d/dz of <dq, softmax(z) W_fb^T> = p * (a - <dq, q>)
        fb = tl.load(A + row * V + k, mask=mk, other=0.0).to(tl.float32) - tl.load(QDQ + row)
    s = tl.arange(0, BS)
    S = 2 * ub + 1
    fin = (s < S) & (s >= S - 2)
    reach = tl.max(tl.where(fin, tl.load(AL + (b.to(tl.int64) * T + tb - 1) * BS + s, mask=tb > 0,
                                         other=float("-inf")), float("-inf")), 0) > float("-inf")
    live = (t < tb) & reach
    go = tl.where(live, tl.load(GOUT + b), 0.0)
    nll = tl.load(NLL + b)
    lse = tl.load(LSE + row)
    zr = Z + row * V
    ab = AL + row * BS
    bb = BE + row * BS
    dz = p * (go + fb)
    if live:                                           # blank: every even state (fixed-order sum)
        ev = (s < S) & ((s % 2) == 0)
        gs = tl.load(ab + s, mask=ev, other=float("-inf")) + tl.load(bb + s, mask=ev, other=float("-inf"))
        lpb = tl.load(zr + BLANK).to(tl.float32) - lse
        qb = tl.sum(tl.where(ev, tl.exp(gs + nll - lpb), 0.0), 0)
        dz = tl.where(k == BLANK, dz - qb * go, dz)
    tl.store(DZ + row * V + k, dz.to(DZ.dtype.element_ty), mask=mk)
    if live:                                           # labels: first occurrence writes its token's total
        tl.debug_barrier()
        j = tl.arange(0, BU)
        mj = j < ub
        yj = tl.load(Y + b.to(tl.int64) * sy + j, mask=mj, other=-1)
        q = tl.zeros([BU], tl.float32)
        first = mj
        for j0 in range(0, ub, CJ):
            jj = j0 + tl.arange(0, CJ)
            mjj = jj < ub
            yjj = tl.load(Y + b.to(tl.int64) * sy + jj, mask=mjj, other=-2)
            lpy = tl.load(zr + yjj, mask=mjj, other=0.0).to(tl.float32) - lse
            e = tl.load(ab + 2 * jj + 1, mask=mjj, other=float("-inf")) + tl.load(bb + 2 * jj + 1, mask=mjj,
                                                                                  other=float("-inf"))
            e = tl.where(mjj, tl.exp(e + nll - lpy), 0.0)
            same = (yj[:, None] == yjj[None, :]) & mjj[None, :]
            q += tl.sum(tl.where(same, e[None, :], 0.0), 1)
            first &= tl.sum((same & (jj[None, :] < j[:, None])).to(tl.int32), 1) == 0
        pj = tl.load(P + row * V + yj, mask=mj, other=0.0).to(tl.float32)
        fj = tl.zeros([BU], tl.float32)
        if HAS_FB:
            fj = tl.load(A + row * V + yj, mask=mj, other=0.0).to(tl.float32) - tl.load(QDQ + row)
        tl.store(DZ + row * V + yj, (pj * (go + fj) - q * go).to(DZ.dtype.element_ty), mask=first)


class _Pass(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w_out, b_out, w_fb, y, tlen, ulen, blank):
        B, T, d = x.shape
        V = w_out.shape[0]
        bf = torch.bfloat16
        xb = x.reshape(B * T, d).to(bf)
        wo = w_out.to(bf)
        z = torch.addmm(b_out.to(bf), xb, wo.t())                         # (BT, V) bf16, cuBLAS
        p = torch.empty_like(z)
        lse = torch.empty(B * T, device=x.device, dtype=torch.float32)
        BV = triton.next_power_of_2(V)
        _softmax_rows[(B * T,)](z, p, lse, V, BV=BV, num_warps=8 if BV >= 4096 else 4)
        q = None
        wf = None
        if w_fb is not None:
            wf = w_fb.to(bf)
            q = (p @ wf.t()).reshape(B, T, -1)                           # (B, T, 2d) bf16
        y = y.to(torch.int64).contiguous()
        tlen = tlen.to(torch.int64).contiguous()
        ulen = ulen.to(torch.int64).contiguous()
        BS = triton.next_power_of_2(2 * y.shape[1] + 1)
        al = torch.empty(B, T, BS, device=x.device, dtype=torch.float32)
        be = torch.empty_like(al)
        nll = torch.empty(B, device=x.device, dtype=torch.float32)
        _ctc_ab_z[(B,)](z, lse, y, tlen, ulen, al, be, nll, y.stride(0), T, V, blank, BS=BS,
                        num_warps=4 if BS <= 512 else 8)
        ctx.save_for_backward(xb, wo, wf, z, p, lse, q, y, tlen, ulen, al, be, nll)
        ctx.shape, ctx.blank, ctx.has_fb, ctx.xdtype = (B, T, d, V), blank, w_fb is not None, x.dtype
        return nll, q

    @staticmethod
    def backward(ctx, gnll, gq):
        xb, wo, wf, z, p, lse, q, y, tlen, ulen, al, be, nll = ctx.saved_tensors
        B, T, d, V = ctx.shape
        has_fb = ctx.has_fb and gq is not None
        a = qdq = None
        if has_fb:
            dq = gq.reshape(B * T, -1).to(torch.bfloat16)
            a = dq @ wf                                                   # (BT, V) bf16
            qdq = (q.reshape(B * T, -1).float() * dq.float()).sum(-1)     # (BT,) fp32
        dz = torch.empty_like(z)
        U = y.shape[1]
        gout = (gnll if gnll is not None else torch.zeros(B, device=z.device)).float().contiguous()
        _sc_grad[(T, B)](z, lse, p, a if has_fb else z, qdq if has_fb else lse, y, tlen, ulen, al, be, nll, gout, dz,
                         y.stride(0), T, V, ctx.blank, BS=al.shape[2], BV=triton.next_power_of_2(V),
                         BU=triton.next_power_of_2(max(U, 1)), CJ=32, HAS_FB=has_fb, num_warps=8)
        dx = (dz @ wo).reshape(B, T, d).to(ctx.xdtype)
        dw_out = (dz.t() @ xb).float()
        db = dz.float().sum(0)
        dw_fb = (dq.t() @ p).float() if has_fb else None
        return dx, dw_out, db, dw_fb, None, None, None, None


def selfcond_ctc_pass(x, w_out, b_out, w_fb, targets, input_lengths, target_lengths, blank):
    """One self-conditioned CTC pass; see the module docstring. Returns (nll (B,) fp32, q (B, T, 2d) bf16 or None)."""
    return _Pass.apply(x, w_out, b_out, w_fb, targets, input_lengths, target_lengths, int(blank))
