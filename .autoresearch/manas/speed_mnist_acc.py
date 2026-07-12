# Acc-evolution companion to speed_mnist.py: same paired runs at the tuned LR (1e-3),
# test accuracy logged every 10 steps. Out: speed_mnist_acc.png/json (this dir).
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np
import torch
import torch.nn.functional as F

from speed_mnist import DATA, MODEL, GA, MICRO, STEPS, SEEDS, DEV
from kernels.sm75.manas import ManasOptimizer

EVAL_EVERY = 10


def run(ds, seed, lr, manas, x, y, xt, yt):
    torch.manual_seed(seed)
    model = MODEL[ds]().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    kw = (dict(micro_vote=True, probe_rho=1.0, probe_rho_step=0.96,
               probe_gamma=0.01, probe_rank=8) if manas else dict(probe_gamma=0.0))
    opt = ManasOptimizer(mats, lr=lr, weight_decay=0.01, **kw)
    aux = torch.optim.AdamW(rest, lr=lr, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    steps, accs, evsteps = STEPS[ds], [], []
    for t in range(steps):
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
    out = {}
    for ds in ("mnist1d", "mnist"):
        print(f"=== {ds} ===", flush=True)
        x, y, xt, yt = [t.to(DEV) for t in DATA[ds]()]
        out[ds] = {"muon": [], "manas": []}
        for sd in SEEDS:
            for name, mn in (("muon", False), ("manas", True)):
                ev, accs = run(ds, sd, 1e-3, mn, x, y, xt, yt)
                out[ds][name].append(accs)
                print(f"  seed {sd} {name}: final acc {accs[-1]:.3f}", flush=True)
        out[ds]["steps"] = ev
    with open(os.path.join(HERE, "speed_mnist_acc.json"), "w") as f:
        json.dump(out, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ds in zip(axes, ("mnist1d", "mnist")):
        ev = out[ds]["steps"]
        for name, color in (("muon", "#888888"), ("manas", "#2563a8")):
            a = np.array(out[ds][name])                     # (seeds, evals)
            for row in a:
                ax.plot(ev, row, color=color, alpha=0.25, lw=1)
            ax.plot(ev, a.mean(0), color=color, lw=2.4, label=f"{name} (3-seed mean)")
        ax.set_title(f"{ds}: test accuracy vs step (muon LR-tuned 1e-3)")
        ax.set_xlabel("step"); ax.set_ylabel("test acc")
        ax.legend(loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "speed_mnist_acc.png"), dpi=130)
    print("wrote speed_mnist_acc.png/json", flush=True)
