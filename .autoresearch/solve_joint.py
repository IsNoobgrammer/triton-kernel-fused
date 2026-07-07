"""JOINT NS-schedule solver: optimize ALL steps' quintic coeffs at once through the composed map.

Greedy per-step (solve_minimax.py) forces each intermediate interval to contract; jointly only the
FINAL composite must land in [1-delta, 1+delta] on [l0, 1], intermediates just stay in (0, cap].
Objective: maximize softmin of final composite, penalties for final > 1+delta, intermediates
outside (0, cap], and (mild) coeff magnitude for fp16 sanity.

Usage: python solve_joint.py L0 STEPS CAP [DELTA] > sched.json   (log to stderr)
"""
import json
import math
import sys

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"
KJ = (3.4445, -4.775, 2.0315)
PIN = (2.0, -1.5, 0.5)


def composite(P, x):
    """P: (S,3) coeffs. Returns final values and list of intermediates (each (N,))."""
    inter = []
    for k in range(P.shape[0]):
        a, b, c = P[k, 0], P[k, 1], P[k, 2]
        x = a * x + b * x**3 + c * x**5
        inter.append(x)
    return x, inter


def solve(l0, steps, cap, delta=0.12, iters=4000, starts=16, seed=0):
    xg = torch.logspace(math.log10(l0), 0.0, 4001, device=dev, dtype=torch.float64)
    xg = torch.cat([xg, torch.linspace(0.5, 1.0, 1001, device=dev, dtype=torch.float64)])
    torch.manual_seed(seed)
    inits = []
    base = [KJ] * max(0, steps - 2) + [PIN] * min(2, steps)
    inits.append(torch.tensor(base, dtype=torch.float64))
    for _ in range(starts - 1):
        pert = torch.tensor(base, dtype=torch.float64)
        pert *= torch.empty_like(pert).uniform_(0.6, 1.6)
        inits.append(pert)
    best = None
    for si, P0 in enumerate(inits):
        P = P0.clone().to(dev).requires_grad_(True)
        opt = torch.optim.Adam([P], lr=0.02)
        for it in range(iters):
            opt.zero_grad()
            f, inter = composite(P, xg)
            tau = 300.0
            softmin = -torch.logsumexp(-tau * f, 0) / tau
            pen = torch.relu(f - (1.0 + delta)).square().sum() * 1e5
            for z in inter:
                pen = pen + torch.relu(z - cap).square().sum() * 1e5
                pen = pen + torch.relu(1e-9 - z).square().sum() * 1e6
            pen = pen + 1e-6 * P.square().sum()
            (-softmin + pen).backward()
            opt.step()
        with torch.no_grad():
            f, inter = composite(P, xg)
            ok = (f > 0).all() and (f <= 1.0 + delta + 1e-6).all() and all(
                (z > 0).all() and (z <= cap + 1e-6).all() for z in inter)
            lo = f.min().item()
            print(f"  start {si}: floor {lo:.4f} ceil {f.max().item():.4f} feasible={bool(ok)}",
                  file=sys.stderr, flush=True)
            if ok and (best is None or lo > best[0]):
                best = (lo, f.max().item(), P.detach().cpu())
    return best


def main():
    l0 = float(sys.argv[1]); steps = int(sys.argv[2]); cap = float(sys.argv[3])
    delta = float(sys.argv[4]) if len(sys.argv) > 4 else 0.12
    got = solve(l0, steps, cap, delta)
    if got is None:
        print(f"INFEASIBLE l0={l0} steps={steps} cap={cap}", file=sys.stderr)
        print("[]")
        return
    lo, hi, P = got
    print(f"BEST floor {lo:.4f} ceil {hi:.4f} (l0={l0}, {steps} steps, cap {cap})", file=sys.stderr)
    print(json.dumps([[round(v, 8) for v in row] for row in P.tolist()]))


if __name__ == "__main__":
    main()
