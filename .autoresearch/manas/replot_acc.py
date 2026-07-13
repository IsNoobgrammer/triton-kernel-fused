# Replot from speed_mnist_acc.json: main fig = muon vs manas (final recipe) only;
# ablation fig = all four arms. No reruns.
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
out = json.load(open(os.path.join(HERE, "speed_mnist_acc.json")))


def panel(ax, ds, arms, labels):
    ev = out[ds]["steps"]
    for name, color in arms:
        a = np.array(out[ds][name])
        for row in a:
            ax.plot(ev, row, color=color, alpha=0.20, lw=1)
        ax.plot(ev, a.mean(0), color=color, lw=2.6, label=labels[name])
    ax.set_title(f"{ds}: test accuracy vs step (cosine lr 1e-3 -> 1e-4, 3-seed mean)")
    ax.set_xlabel("step"); ax.set_ylabel("test acc")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.25)


# main: two lines
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
for ax, ds in zip(axes, ("mnist1d", "mnist")):
    panel(ax, ds, (("muon", "#888888"), ("newtrk", "#1f5fa8")),
          {"muon": "Muon (LR-tuned)", "newtrk": "Manas"})
fig.tight_layout()
fig.savefig(os.path.join(HERE, "speed_mnist_acc.png"), dpi=130)

# ablation: all four
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
LBL = {"muon": "muon", "old": "manas old (g0.01 fixed)",
       "newfix": "manas new, gamma FIXED (no anneal)",
       "newtrk": "manas new, gamma = law(lr_t)"}
for ax, ds in zip(axes, ("mnist1d", "mnist")):
    panel(ax, ds, (("muon", "#888888"), ("old", "#9fc2e0"),
                   ("newfix", "#d08a3e"), ("newtrk", "#1f5fa8")), LBL)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "speed_mnist_acc_ablation.png"), dpi=130)
print("wrote speed_mnist_acc.png (2-line) + speed_mnist_acc_ablation.png (4-arm)")
