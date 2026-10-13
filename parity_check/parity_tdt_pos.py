"""tdt_joint_pos_targets (TDT teacher per-token-position targets) vs an fp64 ground truth.

Ground truth: fp64 joint on the kernel's own bf16 hidden rows (relu(bf16(f + g)) @ W, as the kernel), token and
duration log-softmax (sigma 0), TDT alpha / beta by explicit loops (parity_tdt_joint.tdt_lattice, NeMo's arc rules),
gy(t, u) = posterior mass of the label arc at (t, u) summed over durations, P_u(v) = sum_t gy p(v | t, u, not blank).
Gates:
  1. targets == fp64 (abs <= 2e-3), incl. realistic blank-dominated large-logit rows
  2. every real position sums to 1 (abs 1e-3), blank column 0, finite
  3. nll == fp64 (rel 1e-3)

    python parity_check/parity_tdt_pos.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import kernels.sm120.tdt_joint as tj
import parity_tdt_joint as pt

dev = "cuda"
DUR = [0, 1, 2, 3, 4]
pt.DUR = DUR
ok = True


def gt(f, g, W, b, y, tl, yl):
    ND = len(DUR)
    V = W.shape[0] - ND
    x = torch.relu((f[:, :, None] + g[:, None]).to(torch.bfloat16).float()).to(torch.bfloat16).double()
    logits = (x @ W.to(torch.bfloat16).double().t() + b.to(torch.bfloat16).double()).cpu()
    lt = logits[..., :V].log_softmax(-1)
    ld = logits[..., V:].log_softmax(-1)
    nb = logits[..., :V].clone()
    nb[..., V - 1] = -1e30
    pnb = nb.softmax(-1)
    B, U = y.shape
    P = torch.zeros(B, U, V, dtype=torch.float64)
    nll = []
    for i in range(B):
        T, Ui = int(tl[i]), int(yl[i])
        yy = y[i, :Ui].cpu()
        lpb = lt[i, :T, :Ui + 1, V - 1]
        lpy = torch.full((T, Ui + 1), pt.NEG, dtype=torch.float64)
        if Ui:
            lpy[:, :Ui] = lt[i, :T, torch.arange(Ui), yy]
        lpd = ld[i, :T, :Ui + 1]
        a, be, ll = pt.tdt_lattice(lpb, lpy, lpd, T, Ui)
        nll.append(-ll)
        for t in range(T):
            for u in range(Ui):
                gy = 0.0
                for k, d in enumerate(DUR):
                    if t + d < T:
                        gy = gy + torch.exp(a[t, u] + lpy[t, u] + lpd[t, u, k] + be[t + d, u + 1] - ll)
                P[i, u] += gy * pnb[i, t, u]
    return P, torch.stack(nll)


def case(name, B, T, U, H, V, wscale, blank_bias, seed):
    global ok
    f, g, W, b, y, tl, yl = pt.data(B, T, U, H, V, wscale=wscale, seed=seed)
    b = b.clone()
    b[V - 1] += blank_bias                                         # blank-dominated rows, as a real teacher
    P, nll = tj.tdt_joint_pos_targets(f, g, W, b, y, tl, yl, DUR)
    Pg, nllg = gt(f, g, W, b, y, tl, yl)
    P = P.double().cpu()
    m = torch.zeros(B, U, dtype=torch.bool)
    for i in range(B):
        m[i, :int(yl[i])] = True
    e1 = (P - Pg)[m].abs().max().item() if m.any() else 0.0
    s = P.sum(-1)[m]
    e2 = (s - 1).abs().max().item() if m.any() else 0.0
    fin = bool(torch.isfinite(P).all()) and P[..., V - 1].abs().max().item() == 0.0
    e3 = ((nll.double().cpu() - nllg).abs() / nllg.abs().clamp_min(1)).max().item()
    r = e1 <= 2e-3 and e2 <= 1e-3 and fin and e3 <= 1e-3
    ok &= r
    print(f"  [{'PASS' if r else 'FAIL'}] {name}: targets max abs {e1:.1e}, |sum - 1| {e2:.1e}, finite+blank0 {fin}, "
          f"nll rel {e3:.1e}", flush=True)


def main():
    case("small", 3, 9, 4, 32, 17, 1.0, 0.0, 0)
    case("blank-dominated large logits", 3, 14, 5, 64, 33, 6.0, 20.0, 1)
    case("parakeet-like vocab", 2, 12, 4, 64, 1025, 3.0, 8.0, 2)
    print("ALL PASS" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
