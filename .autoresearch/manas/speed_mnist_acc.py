# Acc-evolution companion to speed_mnist.py, all arms on a COSINE LR schedule (lr0 1e-3 ->
# 1e-4): muon vs OLD manas (gamma 0.01 fixed) vs NEW-FIXED (law gamma at lr0, constant -
# the endgame-bias demo) vs NEW-TRACKED (gamma recomputed from the CURRENT lr each step via
# the law - gamma decays as sqrt(lr), gamma_i tracks live). Out: speed_mnist_acc.png/json.
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np
import torch
import torch.nn.functional as F

from speed_mnist import DATA, MODEL, GA, MICRO, STEPS, SEEDS, DEV, law_gamma
from kernels.sm75.manas import ManasOptimizer

EVAL_EVERY = 10
LR0, LR_END = 1e-3, 1e-4


def lr_at(t, steps):
    return LR_END + (LR0 - LR_END) * 0.5 * (1 + math.cos(math.pi * t / steps))


def run(ds, seed, mode, x, y, xt, yt):
    """mode: 'muon' | 'old' (gamma 0.01 fixed) | 'newfix' (law gamma at lr0, fixed) |
    'newtrk' (gamma follows the law at the CURRENT lr, every step)."""
    torch.manual_seed(seed)
    model = MODEL[ds]().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    g0 = {"muon": 0.0, "old": 0.01, "newfix": law_gamma(LR0), "newtrk": law_gamma(LR0)}[mode]
    kw = (dict(micro_vote=True, probe_rho=1.0, probe_rho_step=0.96,
               probe_gamma=g0, probe_rank=8) if mode != "muon" else dict(probe_gamma=0.0))
    opt = ManasOptimizer(mats, lr=LR0, weight_decay=0.01, **kw)
    aux = torch.optim.AdamW(rest, lr=LR0, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    steps, accs, evsteps = STEPS[ds], [], []
    for t in range(steps):
        lr_t = lr_at(t, steps)
        for grp in list(opt.param_groups) + list(aux.param_groups):
            grp["lr"] = lr_t
        if mode == "newtrk":
            opt.probe_gamma = law_gamma(lr_t)     # gamma follows the law; gamma_i tracks live
        opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
        for m in range(GA):
            idx = torch.randint(0, x.shape[0], (MICRO,), generator=g).to(DEV)
            with opt.probe():
                (F.cross_entropy(model(x[idx]), y[idx]) / GA).backward()
            opt.vote()
        opt.step(); aux.step()
        if t % EVAL_EVERY == 0 or t == steps - 1:
            with torch.no_grad():
                accs.append((model(xt).argmax(-1) == yt).float().mean().item())
            evsteps.append(t)
    return evsteps, accs


if __name__ == "__main__":
    ARMS = (("muon", "#888888"), ("old", "#9fc2e0"),
            ("newfix", "#d08a3e"), ("newtrk", "#1f5fa8"))
    out = {}
    for ds in ("mnist1d", "mnist"):
        print(f"=== {ds} ===", flush=True)
        x, y, xt, yt = [t.to(DEV) for t in DATA[ds]()]
        out[ds] = {a: [] for a, _ in ARMS}
        for sd in SEEDS:
            for name, _ in ARMS:
                ev, accs = run(ds, sd, name, x, y, xt, yt)
                out[ds][name].append(accs)
                print(f"  seed {sd} {name}: final acc {accs[-1]:.3f}", flush=True)
        out[ds]["steps"] = ev
    with open(os.path.join(HERE, "speed_mnist_acc.json"), "w") as f:
        json.dump(out, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    LABELS = {"muon": "muon (cosine LR)", "old": "manas old (g0.01 fixed)",
              "newfix": f"manas new, g={law_gamma(LR0):.3f} FIXED (no anneal)",
              "newtrk": "manas new, g = law(lr_t) TRACKED"}
    for ax, ds in zip(axes, ("mnist1d", "mnist")):
        ev = out[ds]["steps"]
        for name, color in ARMS:
            a = np.array(out[ds][name])
            for row in a:
                ax.plot(ev, row, color=color, alpha=0.20, lw=1)
            ax.plot(ev, a.mean(0), color=color, lw=2.4, label=LABELS[name])
        ax.set_title(f"{ds}: test acc vs step (cosine lr {LR0:.0e}->{LR_END:.0e}, 3-seed mean)")
        ax.set_xlabel("step"); ax.set_ylabel("test acc")
        ax.legend(loc="lower right", fontsize=8); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "speed_mnist_acc.png"), dpi=130)
    print("wrote speed_mnist_acc.png/json", flush=True)
