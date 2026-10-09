"""Deterministic CTC loss (sm120): torch.nn.functional.ctc_loss(reduction='none') semantics, no atomics.

torch's CUDA ctc_loss backward scatters alpha * beta into the vocabulary with atomicAdd (repeated labels and the
blank collide), so a fixed seed still gives different gradient bits run to run (BiBo voice/asr/det_probe.py). Here:

  _ctc_ab    one program per utterance: alpha forward over t, beta backward, the S = 2U + 1 states in registers
             (neighbours s-1, s-2 / s+1, s+2 re-read from the row just written: a barrier per step). nll per sample.
  _ctc_grad  one program per (t, b): grad = (exp(lp) - sum_{s: l'_s = k} exp(alpha + beta + nll - lp)) * g_out,
             torch's formula. Blank: one fixed-order sum over the even states. Labels: per target position j the sum
             over every j' with the same token (fixed order), written by the token's FIRST occurrence only, after the
             row store (barrier) -- every address written once, in a fixed order.

log_probs (B, T, V) any strides, fp32 or bf16 (math in fp32). zero_infinity: an infeasible sample (T too short)
gets loss 0 and grad 0, as torch.

    nll = ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity=True)    # (B,) fp32
"""
import torch
import triton
import triton.language as tl

__all__ = ["ctc_loss"]

_NEG = float("-inf")


@triton.jit
def _lse(a, b):
    m = tl.maximum(a, b)
    ms = tl.where(m == float("-inf"), 0.0, m)
    return ms + tl.log(tl.exp(a - ms) + tl.exp(b - ms))


@triton.jit
def _ctc_ab(LP, Y, TLEN, ULEN, AL, BE, NLL, slb, slt, slv, sy, T, BLANK,
            BS: tl.constexpr, ZERO_INF: tl.constexpr):
    b = tl.program_id(0)
    tb = tl.load(TLEN + b).to(tl.int32)
    ub = tl.load(ULEN + b).to(tl.int32)
    S = 2 * ub + 1
    s = tl.arange(0, BS)
    ms = s < S
    odd = (s % 2) == 1
    lab = tl.where(odd, tl.load(Y + b.to(tl.int64) * sy + s // 2, mask=ms & odd, other=0), BLANK)
    labm2 = tl.load(Y + b.to(tl.int64) * sy + (s - 2) // 2, mask=ms & odd & (s >= 2), other=-1)
    skip = odd & (s >= 2) & (lab != labm2)                                  # s-2 -> s allowed
    labp2 = tl.load(Y + b.to(tl.int64) * sy + (s + 2) // 2, mask=odd & (s + 2 < S), other=-1)
    skipn = odd & (s + 2 < S) & (labp2 != lab)                              # s+2 -> s allowed (beta)
    lpb = LP + b.to(tl.int64) * slb
    ab = AL + b.to(tl.int64) * T * BS
    bb = BE + b.to(tl.int64) * T * BS
    # alpha
    a = tl.where(ms & (s < 2), tl.load(lpb + lab * slv, mask=ms, other=0.0).to(tl.float32), float("-inf"))
    tl.store(ab + s, a)
    for t in range(1, tb):
        tl.debug_barrier()
        a1 = tl.load(ab + (t - 1) * BS + s - 1, mask=s >= 1, other=float("-inf"))
        a2 = tl.load(ab + (t - 1) * BS + s - 2, mask=skip, other=float("-inf"))
        lp = tl.load(lpb + t * slt + lab * slv, mask=ms, other=0.0).to(tl.float32)
        a = tl.where(ms, _lse(_lse(a, a1), a2) + lp, float("-inf"))
        tl.store(ab + t * BS + s, a)
    fin = ms & (s >= S - 2)
    m = tl.max(tl.where(fin, a, float("-inf")), 0)
    msafe = tl.where(m == float("-inf"), 0.0, m)
    nll = -(msafe + tl.log(tl.sum(tl.where(fin, tl.exp(a - msafe), 0.0), 0)))
    nll = tl.where(tb > 0, nll, float("inf"))
    if ZERO_INF:
        nll = tl.where(nll == float("inf"), 0.0, nll)                   # flagged below via the alpha row
    tl.store(NLL + b, nll)
    # beta
    tl.debug_barrier()
    bt = tl.where(fin, tl.load(lpb + (tb - 1) * slt + lab * slv, mask=ms & (tb > 0), other=0.0).to(tl.float32),
                  float("-inf"))
    tl.store(bb + (tb - 1) * BS + s, bt, mask=tb > 0)
    for i in range(1, tb):
        t = tb - 1 - i
        tl.debug_barrier()
        b1 = tl.load(bb + (t + 1) * BS + s + 1, mask=s + 1 < S, other=float("-inf"))
        b2 = tl.load(bb + (t + 1) * BS + s + 2, mask=skipn, other=float("-inf"))
        lp = tl.load(lpb + t * slt + lab * slv, mask=ms, other=0.0).to(tl.float32)
        bt = tl.where(ms, _lse(_lse(bt, b1), b2) + lp, float("-inf"))
        tl.store(bb + t * BS + s, bt)


@triton.jit
def _ctc_grad(LP, Y, TLEN, ULEN, AL, BE, NLL, GOUT, G, slb, slt, slv, sy, sgb, sgt, T, V, BLANK,
              BS: tl.constexpr, BV: tl.constexpr, BU: tl.constexpr, CJ: tl.constexpr):
    t = tl.program_id(0)
    b = tl.program_id(1)
    tb = tl.load(TLEN + b).to(tl.int32)
    ub = tl.load(ULEN + b).to(tl.int32)
    k = tl.arange(0, BV)
    mk = k < V
    gp = G + b.to(tl.int64) * sgb + t.to(tl.int64) * sgt
    nll = tl.load(NLL + b)
    ab = AL + (b.to(tl.int64) * T + t) * BS
    bb = BE + (b.to(tl.int64) * T + t) * BS
    # infeasible sample (alpha never reached the end): zero_infinity -> no gradient, as torch
    s = tl.arange(0, BS)
    S = 2 * ub + 1
    fin = (s < S) & (s >= S - 2)
    reach = tl.max(tl.where(fin, tl.load(AL + (b.to(tl.int64) * T + tb - 1) * BS + s, mask=tb > 0,
                                         other=float("-inf")), float("-inf")), 0) > float("-inf")
    live = (t < tb) & reach
    if not live:
        tl.store(gp + k, tl.zeros([BV], tl.float32).to(G.dtype.element_ty), mask=mk)
        return
    go = tl.load(GOUT + b)
    lpp = LP + b.to(tl.int64) * slb + t.to(tl.int64) * slt
    lp = tl.load(lpp + k * slv, mask=mk, other=0.0).to(tl.float32)
    # blank: every even state
    ev = (s < S) & ((s % 2) == 0)
    gs = tl.load(ab + s, mask=ev, other=float("-inf")) + tl.load(bb + s, mask=ev, other=float("-inf"))
    lpblank = tl.load(lpp + BLANK * slv).to(tl.float32)
    qb = tl.sum(tl.where(ev, tl.exp(gs + nll - lpblank), 0.0), 0)
    g = tl.where(k == BLANK, tl.exp(lp) - qb, tl.exp(lp)) * go
    tl.store(gp + k, g.to(G.dtype.element_ty), mask=mk)
    tl.debug_barrier()
    # labels: q_j = sum over j' with y_j' == y_j (fixed order); the first occurrence writes p_y - q
    j = tl.arange(0, BU)
    mj = j < ub
    yj = tl.load(Y + b.to(tl.int64) * sy + j, mask=mj, other=-1)
    q = tl.zeros([BU], tl.float32)
    first = mj
    for j0 in range(0, ub, CJ):
        jj = j0 + tl.arange(0, CJ)
        mjj = jj < ub
        yjj = tl.load(Y + b.to(tl.int64) * sy + jj, mask=mjj, other=-2)
        lpy = tl.load(lpp + yjj * slv, mask=mjj, other=0.0).to(tl.float32)
        e = tl.load(ab + 2 * jj + 1, mask=mjj, other=float("-inf")) + tl.load(bb + 2 * jj + 1, mask=mjj,
                                                                              other=float("-inf"))
        e = tl.where(mjj, tl.exp(e + nll - lpy), 0.0)
        same = (yj[:, None] == yjj[None, :]) & mjj[None, :]
        q += tl.sum(tl.where(same, e[None, :], 0.0), 1)
        first &= tl.sum((same & (jj[None, :] < j[:, None])).to(tl.int32), 1) == 0
    lpj = tl.load(lpp + yj * slv, mask=mj, other=0.0).to(tl.float32)
    tl.store(gp + yj, ((tl.exp(lpj) - q) * go).to(G.dtype.element_ty), mask=first)


class _CTC(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lp, y, tlen, ulen, blank, zero_inf):
        B, T, V = lp.shape
        y = y.to(torch.int64).contiguous()
        tlen = tlen.to(torch.int64).contiguous()
        ulen = ulen.to(torch.int64).contiguous()
        U = y.shape[1]
        BS = triton.next_power_of_2(2 * U + 1)
        al = torch.empty(B, T, BS, device=lp.device, dtype=torch.float32)
        be = torch.empty_like(al)
        nll = torch.empty(B, device=lp.device, dtype=torch.float32)
        _ctc_ab[(B,)](lp, y, tlen, ulen, al, be, nll, *lp.stride(), y.stride(0), T, blank, BS=BS, ZERO_INF=zero_inf,
                      num_warps=4 if BS <= 512 else 8)
        ctx.save_for_backward(lp, y, tlen, ulen, al, be, nll)
        ctx.blank = blank
        return nll

    @staticmethod
    def backward(ctx, gout):
        lp, y, tlen, ulen, al, be, nll = ctx.saved_tensors
        B, T, V = lp.shape
        U = y.shape[1]
        g = torch.empty(B, T, V, device=lp.device, dtype=lp.dtype)
        _ctc_grad[(T, B)](lp, y, tlen, ulen, al, be, nll, gout.float().contiguous(), g, *lp.stride(), y.stride(0),
                          g.stride(0), g.stride(1), T, V, ctx.blank, BS=al.shape[2], BV=triton.next_power_of_2(V),
                          BU=triton.next_power_of_2(max(U, 1)), CJ=32, num_warps=8)
        return g, None, None, None, None, None


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank, zero_infinity=False):
    """log_probs (B, T, V) log-softmax outputs, targets (B, U) padded, lengths (B,) -> per-sample nll (B,) fp32
    (torch ctc_loss reduction='none'). Deterministic forward and backward."""
    return _CTC.apply(log_probs, targets, input_lengths, target_lengths, int(blank), bool(zero_infinity))
