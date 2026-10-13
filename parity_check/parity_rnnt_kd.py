"""rnnt_joint_loss(..., kd=(tpb, tpy)) -- collapsed RNN-T distillation fused into the joint backward -- and
rnnt_joint_node_logp (the teacher side) vs fp64.

Reference: fp64 joint from the same bf16 inputs; the hard part is parity_rnnt_joint's ground truth (numba gradient
formula); the KD part is autograd through fp64 log_softmax of
    kd_b = sum over nodes (t < T_b, u <= U_b) of  P_b log(P_b / q_b) + P_y log(P_y / q_y) + P_r log(P_r / q_r)
with P = the teacher's node probabilities (the kernel's own teacher output, so the student side is tested alone),
q = the student's, rest = 1 - blank - label (student rest clamped at 1e-6, as the kernel does).
Gates:
  1. teacher node probabilities (blank, label) == fp64 within 5e-3 abs (bf16 joint GEMMs)
  5. kd only (hard weight 0, no lattice): at least as close to fp64 as eager bf16 (or rel <= 1e-2)
  2. loss = 0.3 nll + 0.7 kd and df / dg / dW / dbias vs fp64: no worse than eager bf16 (NeMo + bf16 KD) or <= 1e-2 (normal and confident-blank logits)
  3. kd_weight 0 with a teacher == the plain loss (loss and grads, rel <= 1e-6)
  4. teacher == student: kd ~ 0 and the KD-only gradient ~ 0 (rel to the hard gradient <= 1e-3)

    python parity_check/parity_rnnt_kd.py
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


def unpack(v, tl, yl):
    """packed (N,) node values -> list over b of (T_b, U_b + 1) fp64 cpu"""
    out, o = [], 0
    for b in range(tl.shape[0]):
        T, U1 = int(tl[b]), int(yl[b]) + 1
        out.append(v[o:o + T * U1].view(T, U1).double().cpu())
        o += T * U1
    return out


def node_lp(f, g, W, b, y, tl, yl, amp=False):
    """fp64 student / teacher node log-probs with autograd: lists of (T, U+1) blank and label log-probs
    (amp: the eager bf16 path -- autocast joint, then fp64 log_softmax)"""
    if amp:
        with AMP:
            lg = torch.nn.functional.linear(torch.relu(f.float()[:, :, None] + g.float()[:, None]), W.float(), b.float())
        lp = lg.double().log_softmax(-1)
    else:
        x = torch.relu(f[:, :, None] + g[:, None])
        lp = (x @ W.t() + b).log_softmax(-1)
    V = W.shape[0]
    lb, ly = [], []
    for i in range(f.shape[0]):
        T, U = int(tl[i]), int(yl[i])
        lb.append(lp[i, :T, :U + 1, V - 1])
        l = torch.full((T, U + 1), float("-inf"), dtype=torch.float64, device=dev)
        if U:
            l = torch.cat([lp[i, :T, torch.arange(U, device=dev), y[i, :U]], l[:, U:]], 1)
        ly.append(l)
    return lb, ly


def kd_ref(f, g, W, b, y, tl, yl, tpb, tpy, amp=False):
    f, g, W, b = (x.double().detach().requires_grad_() for x in (f, g, W, b))
    lb, ly = node_lp(f, g, W, b, y, tl, yl, amp)
    tb, ty = unpack(tpb, tl, yl), unpack(tpy, tl, yl)
    kds = []
    for i in range(len(lb)):
        Pb, Py = tb[i].to(dev).exp(), ty[i].to(dev).exp()
        pb, py = lb[i].exp(), ly[i].exp()
        qr = (1 - pb - py).clamp_min(1e-6)
        Pr = (1 - Pb - Py).clamp_min(0)
        z = torch.zeros_like(pb)
        kl = (torch.where(Pb > 0, Pb * (tb[i].to(dev) - lb[i]), z)
              + torch.where(Py > 0, Py * (ty[i].to(dev) - torch.where(Py > 0, ly[i], z)), z)
              + torch.where(Pr > 0, Pr * (Pr.clamp_min(1e-300).log() - qr.log()), z))
        kds.append(kl.sum() / kl.shape[1])               # kd_per_frame: node sum / (U + 1)
    kd = torch.stack(kds).mean()
    kd.backward()
    return kd.detach(), f.grad, g.grad, W.grad, b.grad


def ours(f, g, W, b, y, tl, yl, kd, kw, hw):
    f, g = (x.detach().clone().requires_grad_() for x in (f, g))
    W, b = (x.detach().clone().requires_grad_() for x in (W, b))
    with AMP:
        out = rj.rnnt_joint_loss(f, g, W, b, y, tl, yl, kd=kd, kd_weight=kw, hard_weight=hw)
    out[0].backward()
    return (out[0].detach(), f.grad, g.grad, W.grad, b.grad), (out[2] if kd is not None else None)


def case(name, B, T, U, H, V, blank_bias=0.0):
    print(f"{name}: B {B} T {T} U {U} H {H} V {V} blank bias {blank_bias}", flush=True)
    f, g, W, b, y, tl, yl = pj.data(B, T, U, H, V, seed=1)
    b[-1] += blank_bias
    ft, gt, Wt, bt, *_ = pj.data(B, T, U, H, V, seed=2)
    bt[-1] += blank_bias
    with AMP:
        tpb, tpy, tnll, _ = rj.rnnt_joint_node_logp(ft, gt, Wt, bt, y, tl, yl)
    with torch.no_grad():
        rb, ry = node_lp(ft.double(), gt.double(), Wt.bfloat16().double(), bt.bfloat16().double(), y, tl, yl)
    eb = max((a.exp() - r.cpu().exp()).abs().max().item() for a, r in zip(unpack(tpb, tl, yl), rb))
    ey = max((a.exp() - r.cpu().exp()).abs().max().item() for a, r in zip(unpack(tpy, tl, yl), ry))
    check("1 teacher node probabilities", max(eb, ey) < 5e-3, f"blank {eb:.1e}  label {ey:.1e}")
    with AMP:                                                   # 1b. teacher blank penalty == blank bias - bp
        qpb, qpy, _, _ = rj.rnnt_joint_node_logp(ft, gt, Wt, bt, y, tl, yl, blank_penalty=1.0)
    btp = bt.bfloat16().double().clone()
    btp[-1] -= 1.0
    with torch.no_grad():
        pb_, py_ = node_lp(ft.double(), gt.double(), Wt.bfloat16().double(), btp, y, tl, yl)
    eb = max((a.exp() - r.cpu().exp()).abs().max().item() for a, r in zip(unpack(qpb, tl, yl), pb_))
    ey = max((a.exp() - r.cpu().exp()).abs().max().item() for a, r in zip(unpack(qpy, tl, yl), py_))
    check("1b teacher blank penalty 1.0 == blank logit - 1, renormalised", max(eb, ey) < 5e-3,
          f"blank {eb:.1e}  label {ey:.1e}")

    hw, kw = 0.3, 0.7
    Wb, bb = W.bfloat16().float(), b.bfloat16().float()          # the kernel's joint weights are bf16
    hard = pj.ground_truth(f, g, Wb, bb, y, tl, yl, None)
    kref = kd_ref(f, g, Wb, bb, y, tl, yl, tpb, tpy)
    gt_ = [hw * h + kw * k for h, k in zip(hard, kref)]
    o, kdu = ours(f, g, W, b, y, tl, yl, (tpb, tpy), kw, hw)
    e = [pj.rel(a, r) for a, r in zip(o, gt_)]
    nb = pj.nemo_path(f, g, Wb, bb, y, tl, yl, None, True)             # eager bf16: NeMo numba on bf16 logits
    eb = [pj.rel(hw * h.double() + kw * k.double(), r) for h, k, r in
          zip(nb, kd_ref(f, g, Wb, bb, y, tl, yl, tpb, tpy, amp=True), gt_)]
    check("2 0.3 nll + 0.7 kd vs fp64, no worse than eager bf16", all(x <= max(1.05 * c, 1e-2) for x, c in zip(e, eb)),
          " ".join(f"{n} {x:.1e} (bf16 {c:.1e})" for n, x, c in zip(pj.NAMES, e, eb))
          + f"  (kd mean {kdu.mean().item():.3f} vs {kref[0].item():.3f})")

    kdo, _ = ours(f, g, W, b, y, tl, yl, (tpb, tpy), 1.0, 0.0)          # hard weight 0: no lattice at all
    e5 = [pj.rel(a, r) for a, r in zip(kdo, kref)]
    eb16 = [pj.rel(a, r) for a, r in zip(kd_ref(f, g, Wb, bb, y, tl, yl, tpb, tpy, amp=True), kref)]
    check("5 kd only (hard 0, lattice skipped) vs fp64, no worse than eager bf16",
          all(a <= max(1.05 * c, 1e-2) for a, c in zip(e5, eb16)),
          " ".join(f"{n} {x:.1e} (bf16 {c:.1e})" for n, x, c in zip(pj.NAMES, e5, eb16)))

    plain, _ = ours(f, g, W, b, y, tl, yl, None, 0.0, 1.0)
    zero, _ = ours(f, g, W, b, y, tl, yl, (tpb, tpy), 0.0, 1.0)
    e3 = [pj.rel(a, r) for a, r in zip(zero, plain)]
    check("3 kd_weight 0 == plain", max(e3) <= 1e-6, " ".join(f"{x:.1e}" for x in e3))

    with AMP:
        spb, spy, _, _ = rj.rnnt_joint_node_logp(f, g, W, b, y, tl, yl)
    self_kd, kself = ours(f, g, W, b, y, tl, yl, (spb, spy), 1.0, 0.0)
    gnorm = max((a.double().norm() / h.double().norm()).item() for a, h in zip(self_kd[1:], plain[1:]))
    check("4 teacher == student: kd ~ 0, grad ~ 0", kself.abs().max().item() < 1e-2 and gnorm <= 1e-3,
          f"max kd {kself.abs().max().item():.1e}  |grad| / |hard grad| {gnorm:.1e}")


def main():
    case("small", 4, 30, 10, 64, 129)
    case("confident blank", 4, 30, 10, 64, 129, blank_bias=8.0)
    case("wide", 3, 40, 12, 128, 1025)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
