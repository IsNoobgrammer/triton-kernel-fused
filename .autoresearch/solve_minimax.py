"""PE-style greedy minimax coefficient solver (offline; not part of the frozen eval).

State: interval [l, u] containing all singular values (u <= UMAX). Per iteration choose
(a, b, c) for p(x) = a x + b x^3 + c x^5 to MAXIMIZE min_{x in [l,u]} p(x), subject to
max_{x in [l,u]} p(x) <= UMAX and p > 0 on [l,u]. Then [l,u] <- [min p, max p].
Greedy per-iteration = PE Algorithm 1's structure (their Remez solution is exactly the
max-min-with-cap program; we solve it numerically on a dense grid with multi-start Adam).

Outputs a schedule of the requested length for a given l0, printed as JSON.
Usage: python solve_minimax.py L0 STEPS [UMAX] > coeffs.json
"""
import json
import math
import sys

import torch

dev = "cuda" if torch.cuda.is_available() else "cpu"


def grid(l, u, n=6001):
    g = torch.logspace(math.log10(l), math.log10(u), n, device=dev, dtype=torch.float64)
    return torch.cat([g, torch.linspace(max(l, 0.5 * u), u, 1001, device=dev, dtype=torch.float64)])


def solve_step(l, u, umax, iters=800, starts=24):
    x = grid(l, u)
    x2, x3, x5 = x * x, None, None
    x3 = x2 * x
    x5 = x3 * x2
    best = None
    torch.manual_seed(0)
    # multi-start around plausible ramps (KJ-like to aggressive)
    inits = []
    for a0 in [2.0, 3.4445, 5.0, 8.0, 12.0, 20.0]:
        inits.append((a0, -4.775 * (a0 / 3.4445), 2.0315 * (a0 / 3.4445)))
    for _ in range(starts - len(inits)):
        a0 = float(torch.empty(1).uniform_(1.5, 25))
        inits.append((a0, float(torch.empty(1).uniform_(-30, 0)), float(torch.empty(1).uniform_(0, 20))))
    for a0, b0, c0 in inits:
        p = torch.tensor([a0, b0, c0], device=dev, dtype=torch.float64, requires_grad=True)
        opt = torch.optim.Adam([p], lr=0.05)
        for it in range(iters):
            opt.zero_grad()
            v = p[0] * x + p[1] * x3 + p[2] * x5
            # soft objective: maximize softmin(v) with hinge penalties on cap and positivity
            tau = 200.0
            softmin = -torch.logsumexp(-tau * v, dim=0) / tau
            pen = torch.relu(v - umax).square().sum() * 1e4 + torch.relu(1e-12 - v).square().sum() * 1e6
            loss = -softmin + pen
            loss.backward()
            opt.step()
        with torch.no_grad():
            v = p[0] * x + p[1] * x3 + p[2] * x5
            if (v <= umax * (1 + 1e-9)).all() and (v > 0).all():
                lo = v.min().item()
                if best is None or lo > best[0]:
                    best = (lo, v.max().item(), [p[0].item(), p[1].item(), p[2].item()])
    return best


def main():
    l0 = float(sys.argv[1]); steps = int(sys.argv[2])
    umax = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    l, u = l0, 1.0
    sched = []
    for k in range(steps):
        # normalize interval to [l/u, 1]; fold scale into coefficients afterwards
        ln = l / u
        cap = umax if ln < 0.5 else 1.05          # ramp phase balloons; endgame contracts to ~1
        got = solve_step(ln, 1.0, cap)
        if got is None:
            print(f"  step {k+1}: INFEASIBLE at [{ln:.3e},1]", file=sys.stderr, flush=True)
            break
        lo, hi, q = got
        coef = [q[0] / u, q[1] / u**3, q[2] / u**5]
        sched.append([round(c, 8) for c in coef])
        print(f"  step {k+1}: [{ln:.3e},1] -> [{lo:.3e},{hi:.4f}] (cap {cap})  "
              f"q=({q[0]:.3f},{q[1]:.3f},{q[2]:.3f})", file=sys.stderr, flush=True)
        l, u = lo, hi
        if l / u > 0.995:  # converged: pinned polish for any remaining steps
            for _ in range(k + 1, steps):
                sched.append([2.0, -1.5, 0.5])
            break
    print(json.dumps(sched))


if __name__ == "__main__":
    main()
