"""A7 — direct mechanistic check of the Manas alignment force. Also the A1 motion tracker.

Trains the eval model with ManasOptimizer and, every MEASURE_EVERY steps, measures at theta
(probe removed):
  1. F_align = -grad_theta sum_{i<j} CosSim(g_i, g_j) over the 3 sources' fixed batches
     (double backward, the TRUE common-minima force the design claims to inject) [A7]
  2. dg = g(theta+d) - g(theta) on a fixed mixed batch (the REALIZED probe perturbation)
  3. cos(dg, F_align)  — the mechanism check: positive = force is being injected
     cos(g_theta, F_align) — baseline: does plain SGD already follow the alignment force?
  4. ||theta_t - theta_{t-k}|| / ||d||  for k in 1..8 — the A1 quasi-static ratio: how far
     the trajectory moves over the probe's memory window vs the probe displacement itself.

Run: ../../BiBo/.venv/Scripts/python.exe a7_mechanism.py [rho]
Out: a7_results.json (curves) + stdout summary. Diagnostic only — not the frozen eval.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
import torch
import torch.nn.functional as F

import eval_manas as E
from kernels.sm75.manas import ManasOptimizer

RHO = float(sys.argv[1]) if len(sys.argv) > 1 else 0.85
GAMMA = 0.005
STEPS, MEASURE_EVERY, SEED = 900, 30, 0
DEV = "cuda"


def flat_grads(model, mats, loss, create_graph=False):
    gs = torch.autograd.grad(loss, mats, create_graph=create_graph)
    return torch.cat([g.reshape(-1) for g in gs])


def alignment_force(model, mats, batches):
    """-grad of sum_{i<j} CosSim(g_i, g_j) w.r.t. mats (double backward)."""
    gs = [flat_grads(model, mats, F.cross_entropy(model(x), y), create_graph=True)
          for x, y in batches]
    sim = sum(F.cosine_similarity(gs[i], gs[j], dim=0)
              for i in range(len(gs)) for j in range(i + 1, len(gs)))
    force = torch.autograd.grad(sim, mats)
    return torch.cat([f.reshape(-1) for f in force])          # +direction INCREASES alignment


def main():
    torch.manual_seed(SEED)
    model = E.Net1D().to(DEV)
    mats, rest = E.split_params(model)
    opt = ManasOptimizer(mats, lr=E.LR, probe_gamma=GAMMA, probe_rho=RHO, weight_decay=E.WD)
    aux = torch.optim.AdamW(rest, lr=E.LR, weight_decay=E.WD)

    tr = {s: E.make_source(s, SEED)[0] for s in ("s0", "s1", "s2")}
    fixed = [(tr[s][0][:256], tr[s][1][:256]) for s in ("s0", "s1", "s2")]
    mix_x = torch.cat([b[0] for b in fixed]); mix_y = torch.cat([b[1] for b in fixed])
    order = ("s0", "s1", "s2")

    hist = []                                                  # ring buffer of theta snapshots
    out = {"step": [], "cos_dg_force": [], "cos_g_force": [], "dnorm": [],
           "motion_over_d": {str(k): [] for k in (1, 2, 4, 8)}}
    g = torch.Generator(device="cpu").manual_seed(SEED)
    for t in range(STEPS):
        s = order[int(torch.randint(0, 3, (1,), generator=g))]
        x, y = tr[s]
        idx = torch.randint(0, x.shape[0], (E.BS,), generator=g).to(DEV)
        with opt.probe():
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
            loss.backward()
        opt.step(); aux.step()

        theta = torch.cat([p.detach().reshape(-1) for p in mats]).clone()
        hist.append(theta)
        if len(hist) > 9:
            hist.pop(0)

        if t % MEASURE_EVERY == 0 and t > 0:
            model.zero_grad(set_to_none=True)
            force = alignment_force(model, mats, fixed)
            g_theta = flat_grads(model, mats, F.cross_entropy(model(mix_x), mix_y))
            with opt.probe():
                g_probe = flat_grads(model, mats, F.cross_entropy(model(mix_x), mix_y))
            dg = g_probe - g_theta
            d_flat = torch.cat([opt._d_of(p).reshape(-1) for p in mats])
            out["step"].append(t)
            # the injected force is -dg's effect on descent: descent step -lr*g_probe differs
            # from -lr*g_theta by -lr*dg; it aids alignment iff cos(-dg, force) > 0
            out["cos_dg_force"].append(F.cosine_similarity(-dg, force, dim=0).item())
            out["cos_g_force"].append(F.cosine_similarity(-g_theta, force, dim=0).item())
            out["dnorm"].append(d_flat.norm().item())
            for k in (1, 2, 4, 8):
                if len(hist) > k:
                    motion = (hist[-1] - hist[-1 - k]).norm().item()
                    out["motion_over_d"][str(k)].append(motion / max(d_flat.norm().item(), 1e-12))
            if t % 150 == 0:
                print(f"step {t}: cos(-dg, F_align) {out['cos_dg_force'][-1]:+.3f}  "
                      f"cos(-g, F_align) {out['cos_g_force'][-1]:+.3f}  "
                      f"motion/||d|| k=8 {out['motion_over_d']['8'][-1]:.1f}", flush=True)

    import numpy as np
    c = np.array(out["cos_dg_force"])
    print(f"\nrho={RHO} gamma={GAMMA}: cos(-dg, F_align) mean {c.mean():+.4f} "
          f"(first-third {c[:len(c)//3].mean():+.4f}, last-third {c[-len(c)//3:].mean():+.4f})")
    for k in (1, 2, 4, 8):
        m = np.array(out["motion_over_d"][str(k)])
        print(f"  motion(k={k})/||d||: median {np.median(m):.1f}")
    with open(os.path.join(HERE, f"a7_rho{RHO}.json"), "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
