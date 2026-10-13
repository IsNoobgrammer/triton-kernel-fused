"""Per-position KD (rnnt_joint_node_logp(pos_targets=True) + rnnt_joint_loss(pos=P)) vs fp64.

Teacher target: P_u(v) = sum_t g(t,u) p(v | t, u, not blank), g(t,u) = exp(alpha + lpy + beta(t,u+1) - ll), blank 0.
Student loss: hard * mean(nll) + kd * mean_b sum_u sum_t g_s(t,u) KL(P_u || q(. | t, u, not blank)), g_s detached.
Gates:
  1. teacher P == fp64 within 5e-3 abs, rows sum to 1 (u < U)
  2. 0.3 nll + 0.7 pos-KD: loss / df / dg / dW / dbias no worse than eager bf16 vs fp64 (or rel <= 1e-2)
  3. kd weight 0 == plain loss (bitwise)
  5. a confident-blank, large-logit case (p_blank rounds to 1 on real batches: the first real probe went NaN)
  4. teacher == student (P from the student itself): the KD value equals the student's own entropy term only
     (KL >= 0, ~ Jensen gap) and its gradient is small vs the nll gradient

    python parity_check/parity_rnnt_pos.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import parity_rnnt_joint as pj
import kernels.sm120.rnnt_joint as rj

pj.LAM = 0.0
dev = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def logits64(f, g, W, b, amp=False):
    if amp:
        with AMP:
            lg = torch.nn.functional.linear(torch.relu(f.float()[:, :, None] + g.float()[:, None]), W.float(), b.float())
        return lg.double()
    return torch.relu(f[:, :, None] + g[:, None]) @ W.t() + b


def occupancy(lp, y, tl, yl):
    """per utterance (T, U) g(t, u) from fp64 log-probs lp (B, T, U+1, V), detached"""
    V = lp.shape[-1]
    out = []
    for i in range(lp.shape[0]):
        T, U = int(tl[i]), int(yl[i])
        lpb = lp[i, :T, :U + 1, V - 1].detach().cpu()
        lpy = torch.full((T, U + 1), float("-inf"), dtype=torch.float64)
        if U:
            lpy[:, :U] = lp[i, :T, torch.arange(U), y[i, :U]].detach().cpu()
        a, be = pj.lattice(lpb, lpy, T, U)
        ll = be[0, 0]
        gam = torch.exp(a[:, :U] + lpy[:, :U] + be[:, 1:] - ll) if U else torch.zeros(T, 0, dtype=torch.float64)
        out.append(gam.to(dev))
    return out


def teacher_ref(f, g, W, b, y, tl, yl):
    lp = logits64(f.double(), g.double(), W.bfloat16().double(), b.bfloat16().double()).log_softmax(-1)
    V = lp.shape[-1]
    gams = occupancy(lp, y, tl, yl)
    P = torch.zeros(lp.shape[0], g.shape[1] - 1, V, dtype=torch.float64, device=dev)
    for i, gm in enumerate(gams):
        T, U = gm.shape
        p = lp[i, :T, :U].exp()
        pnb = p / (1 - p[..., V - 1:V])
        pnb[..., V - 1] = 0
        P[i, :U] = (gm[..., None] * pnb).sum(0)
    return P


def pos_ref(f, g, W, b, y, tl, yl, P, amp=False):
    """fp64 (or eager-bf16-logits) KD part: mean_b sum_u sum_t g_s KL(P_u || q_nb), g_s detached; grads"""
    f, g, W, b = (x.double().detach().requires_grad_() for x in (f, g, W, b))
    lg = logits64(f, g, W, b, amp)
    lp = lg.log_softmax(-1)
    V = lp.shape[-1]
    gams = occupancy(lp, y, tl, yl)
    tot = []
    for i, gm in enumerate(gams):
        T, U = gm.shape
        if not U:
            tot.append(lp.sum() * 0)
            continue
        l = lp[i, :T, :U]
        lnb = l[..., :V - 1] - torch.log1p(-l[..., V - 1:].exp())
        Pu = P[i, :U, :V - 1].double()
        kl = (torch.where(Pu > 0, Pu * Pu.clamp_min(1e-300).log(), torch.zeros_like(Pu))[None] - Pu[None] * lnb).sum(-1)
        tot.append((gm * kl).sum())
    loss = torch.stack(tot).mean()
    loss.backward()
    return loss.detach(), f.grad, g.grad, W.grad, b.grad


def ours(f, g, W, b, y, tl, yl, P, kw, hw):
    f, g = (x.detach().clone().requires_grad_() for x in (f, g))
    W, b = (x.detach().clone().requires_grad_() for x in (W, b))
    with AMP:
        out = rj.rnnt_joint_loss(f, g, W, b, y, tl, yl, pos=P, kd_weight=kw, hard_weight=hw)
    out[0].backward()
    return (out[0].detach(), f.grad, g.grad, W.grad, b.grad), out[2]


def plain(f, g, W, b, y, tl, yl):
    f, g = (x.detach().clone().requires_grad_() for x in (f, g))
    W, b = (x.detach().clone().requires_grad_() for x in (W, b))
    with AMP:
        loss, _ = rj.rnnt_joint_loss(f, g, W, b, y, tl, yl)
    loss.backward()
    return loss.detach(), f.grad, g.grad, W.grad, b.grad


def case(name, B, T, U, H, V, blank_bias=0.0, wscale=1.0):
    print(f"{name}: B {B} T {T} U {U} V {V} blank bias {blank_bias} wscale {wscale}", flush=True)
    f, g, W, b, y, tl, yl = pj.data(B, T, U, H, V, wscale=wscale, seed=7)
    ft, gt, Wt, bt, *_ = pj.data(B, T, U, H, V, wscale=wscale, seed=8)
    b[-1] += blank_bias
    bt[-1] += blank_bias
    with AMP:
        *_, P = rj.rnnt_joint_node_logp(ft, gt, Wt, bt, y, tl, yl, pos_targets=True)
    Pr = teacher_ref(ft, gt, Wt, bt, y, tl, yl)
    e1 = (P.double() - Pr).abs().max().item()
    um = torch.arange(U, device=dev)[None, :] < yl[:, None]
    rs = P.sum(-1)[um]
    check("1 teacher per-position targets", e1 < 5e-3 and bool(((rs - 1).abs() < 1e-3).all()),
          f"max abs {e1:.1e}, row sums {rs.min().item():.5f}..{rs.max().item():.5f}")
    hw, kw = 0.3, 0.7
    Wb, bb = W.bfloat16().float(), b.bfloat16().float()
    hard = pj.ground_truth(f, g, Wb, bb, y, tl, yl, None)
    kref = pos_ref(f, g, Wb, bb, y, tl, yl, P)
    gt_ = [hw * h + kw * k for h, k in zip(hard, kref)]
    o, kdu = ours(f, g, W, b, y, tl, yl, P, kw, hw)
    check("0 finite (no NaN / inf)", all(bool(torch.isfinite(x).all()) for x in o) and bool(torch.isfinite(kdu).all()) and bool(torch.isfinite(P).all()), "loss, grads, KD, targets")
    e = [pj.rel(a, r) for a, r in zip(o, gt_)]
    nb = pj.nemo_path(f, g, Wb, bb, y, tl, yl, None, True)
    eb = [pj.rel(hw * h.double() + kw * k.double(), r) for h, k, r in zip(nb, pos_ref(f, g, Wb, bb, y, tl, yl, P, True), gt_)]
    check("2 0.3 nll + 0.7 pos-KD vs fp64, no worse than eager bf16", all(x <= max(1.05 * c, 1e-2) for x, c in zip(e, eb)),
          " ".join(f"{n} {x:.1e} (bf16 {c:.1e})" for n, x, c in zip(pj.NAMES, e, eb))
          + f"  (kd mean {kdu.mean().item():.4f} vs {kref[0].item():.4f})")
    o0, _ = ours(f, g, W, b, y, tl, yl, P, 0.0, 1.0)
    p0 = plain(f, g, W, b, y, tl, yl)
    check("3 kd weight 0 == plain", all(torch.equal(a, c) for a, c in zip(o0[1:], p0[1:])), "bitwise grads")
    with AMP:
        *_, Ps = rj.rnnt_joint_node_logp(f, g, W, b, y, tl, yl, pos_targets=True)
    os_, ks = ours(f, g, W, b, y, tl, yl, Ps, 1.0, 1.0)
    gk = max((a.double() - c.double()).norm().item() / c.double().norm().item() for a, c in zip(os_[1:], p0[1:]))
    check("4 teacher == student: KL >= 0 and small", bool((ks > -1e-3).all()) and gk < 0.5,
          f"kd per utt {ks.min().item():.3f}..{ks.max().item():.3f}, |grad change| / |nll grad| {gk:.2f}")


def main():
    case("small", 4, 30, 10, 64, 129)
    case("wide", 3, 40, 12, 128, 1025)
    case("real-like: confident blank, large logits", 4, 30, 10, 64, 129, blank_bias=20.0, wscale=6.0)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
