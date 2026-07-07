"""ManasOptimizer — CONCLUSIVE test on MNIST-1D (non-saturating, overlapping).

The MNIST toy went blind: it saturated and cross-source cosine at convergence measures
distance-from-optimum, not basin geometry. This fixes BOTH:
  * MNIST-1D is deliberately hard/overlapping -> small models plateau ~60-75%, never saturate,
    so mid/late training stays informative.
  * The diagnostic is the PAPER'S ACTUAL CLAIM, not cosine: train to a given TRAIN loss, measure
    loss on a HELD-OUT source (OOD regime) the model NEVER trained on. We plot OOD-loss vs
    TRAIN-loss parametrically over training; if common-minima helps, Manas's curve sits BELOW
    base-Muon's at MATCHED train loss (Nexus Fig-1: same pretraining loss, better downstream).
    This does not go blind at saturation.

Sources = same MNIST-1D templates (digits), DIFFERENT corruption regimes (different per-source
minima, shared common minimum): 3 training sources (varied noise/translation/shear), 1 OOD source
with an unseen regime. Model: Conv1d stem (AdamW; convs are never for Muon) -> GLU trunk (Muon).

Arms: base Muon, Manas full-d, Manas rank-32, Manas rank-frac 0.25, + a gamma sweep. Multi-seed
on the base-vs-manas pair (the comparison that must be conclusive).

Run: ../BiBo/.venv/Scripts/python.exe .autoresearch/manas_mnist1d.py
Out: manas_mnist1d_results.json + manas_mnist1d.png (this dir).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args

from kernels.sm75.manas import ManasOptimizer

DEV = "cuda"
STEPS, BS, EVAL_EVERY = 1500, 128, 30
HERE = os.path.dirname(os.path.abspath(__file__))

# corruption regimes: same digit templates, different generating noise -> different minima.
# 3 in-distribution training sources + 1 held-out OOD regime (unseen at train).
REGIMES = {
    "s0":  dict(corr_noise_scale=0.20, max_translation=36, shear_scale=0.5),
    "s1":  dict(corr_noise_scale=0.30, max_translation=60, shear_scale=1.0),
    "s2":  dict(corr_noise_scale=0.25, max_translation=48, shear_scale=0.75, scale_coeff=0.6),
    "ood": dict(corr_noise_scale=0.45, max_translation=24, shear_scale=1.4, iid_noise_scale=0.08),
}


def make_source(reg, seed):
    a = get_dataset_args()
    a.num_samples = 5000
    a.seed = seed
    for k, v in reg.items():
        setattr(a, k, v)
    d = make_dataset(a)
    x = torch.tensor(d["x"], dtype=torch.float32, device=DEV).unsqueeze(1)   # (N,1,40)
    y = torch.tensor(d["y"], dtype=torch.long, device=DEV)
    xt = torch.tensor(d["x_test"], dtype=torch.float32, device=DEV).unsqueeze(1)
    yt = torch.tensor(d["y_test"], dtype=torch.long, device=DEV)
    return (x, y), (xt, yt)


class GLUBlock(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return x + self.down(F.silu(self.gate(x)) * self.up(x))


class Net1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(1, 16, 5, stride=2)              # 40 -> 18; AdamW
        self.proj = nn.Linear(16 * 18, 192, bias=False)
        self.blocks = nn.Sequential(GLUBlock(192, 384), GLUBlock(192, 384), GLUBlock(192, 384))
        self.head = nn.Linear(192, 10, bias=False)

    def forward(self, x):
        z = F.silu(self.stem(x)).flatten(1)
        return self.head(self.blocks(self.proj(z)))


def run(name, seed, gamma=0.0, rho=0.85, rank=None):
    torch.manual_seed(seed)
    model = Net1D().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    opt = ManasOptimizer(mats, lr=2e-3, probe_gamma=gamma, probe_rho=rho, probe_rank=rank,
                         weight_decay=0.01)
    aux = torch.optim.AdamW(rest, lr=2e-3, weight_decay=0.01)

    tr = {s: SRC[(s, seed)][0] for s in ("s0", "s1", "s2")}    # 3 training sources
    ood_x, ood_y = SRC[("ood", seed)][1]                       # held-out regime, TEST split
    idtest = [SRC[(s, seed)][1] for s in ("s0", "s1", "s2")]   # in-distribution test
    order = ["s0", "s1", "s2"]

    log = {"step": [], "train": [], "id": [], "ood": [], "ood_acc": []}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(STEPS):
        s = order[t % 3]
        x, y = tr[s]
        idx = torch.randint(0, x.shape[0], (BS,), generator=g).to(DEV)
        with opt.probe():
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
            loss.backward()
        opt.step(); aux.step()

        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            with torch.no_grad():
                # train loss = mean over the 3 training sources' TRAIN split (the x-axis)
                trl = np.mean([F.cross_entropy(model(tr[s][0][:1000]), tr[s][1][:1000]).item()
                               for s in order])
                idl = np.mean([F.cross_entropy(model(xt), yt).item() for xt, yt in idtest])
                oodlogits = model(ood_x)
                oodl = F.cross_entropy(oodlogits, ood_y).item()
                ooda = (oodlogits.argmax(-1) == ood_y).float().mean().item()
            log["step"].append(t); log["train"].append(float(trl))
            log["id"].append(float(idl)); log["ood"].append(oodl); log["ood_acc"].append(ooda)
    print(f"  {name:<20} train {log['train'][-1]:.3f}  id {log['id'][-1]:.3f}  "
          f"ood {log['ood'][-1]:.3f}  ood_acc {log['ood_acc'][-1]:.3f}")
    return {"name": name, "seed": seed, "gamma": gamma, "rank": str(rank), "log": log}


def ood_at_matched_train(res, grid=None):
    """Interpolate each run's OOD loss onto a common TRAIN-loss grid (the conclusive comparison:
    OOD at MATCHED train loss). Train loss is monotone-decreasing, so invert it per run."""
    if grid is None:
        grid = np.linspace(0.2, 1.6, 15)
    tr = np.array(res["log"]["train"]); ood = np.array(res["log"]["ood"])
    o = np.argsort(tr)                                         # ASCENDING (np.interp needs xp increasing)
    return np.interp(grid, tr[o], ood[o], left=np.nan, right=np.nan)


if __name__ == "__main__":
    SEEDS = (0, 1, 2)
    print("building MNIST-1D sources (3 train regimes + 1 OOD) x seeds ...")
    SRC = {}
    for s in REGIMES:
        for sd in SEEDS:
            SRC[(s, sd)] = make_source(REGIMES[s], seed=100 * sd + 7)
    print(f"arms x {STEPS} steps; conclusive metric = OOD loss at MATCHED train loss\n")

    R = []
    for sd in SEEDS:                                           # the conclusive pair, multi-seed
        R.append(run(f"base_s{sd}", sd, gamma=0.0))
        R.append(run(f"manas_s{sd}", sd, gamma=1.5e-3))
    R.append(run("manas_g3e-3", 0, gamma=3e-3))                # gamma sweep (seed 0)
    R.append(run("manas_g5e-4", 0, gamma=5e-4))
    R.append(run("manas_r32", 0, gamma=1.5e-3, rank=32))       # rank variants
    R.append(run("manas_rfrac.25", 0, gamma=1.5e-3, rank=0.25))

    # ---- conclusive comparison: OOD @ matched train loss, seed-averaged base vs manas ----
    pairs = [r for r in R if r["name"].startswith(("base_s", "manas_s"))]
    lo = max(min(r["log"]["train"]) for r in pairs)            # grid inside every run's visited range
    hi = min(max(r["log"]["train"]) for r in pairs)
    grid = np.linspace(lo * 1.05, hi * 0.95, 15)
    base = np.nanmean([ood_at_matched_train(r, grid) for r in R if r["name"].startswith("base_")], 0)
    manas = np.nanmean([ood_at_matched_train(r, grid) for r in R if r["name"].startswith("manas_s")], 0)
    delta = base - manas                                       # >0 => manas better OOD at matched train
    print("\n== CONCLUSIVE: OOD loss at matched TRAIN loss (base - manas, seed-avg; >0 = manas wins) ==")
    for tl, b, m, d in zip(grid, base, manas, delta):
        if b == b and m == m:
            print(f"  train={tl:.2f}:  base_ood {b:.4f}  manas_ood {m:.4f}  delta {d:+.4f}")
    valid = delta[~np.isnan(delta)]
    print(f"\n  mean OOD-gap across matched train-loss grid: {np.nanmean(delta):+.4f} "
          f"({(valid > 0).mean()*100:.0f}% of grid points favor manas)")

    with open(os.path.join(HERE, "manas_mnist1d_results.json"), "w") as f:
        json.dump({"runs": R, "grid": grid.tolist(),
                   "base_ood": base.tolist(), "manas_ood": manas.tolist()}, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    for r in R:
        if r["name"].startswith(("base_s", "manas_s")):
            c = "tab:blue" if r["name"].startswith("base") else "tab:red"
            ax[0].plot(r["log"]["step"], r["log"]["ood"], c, alpha=0.4)
            ax[1].plot(r["log"]["train"], r["log"]["ood"], c, alpha=0.4,
                       label=r["name"].rstrip("012") if r["seed"] == 0 else None)
    ax[0].set_title("OOD loss vs step (blue=base, red=manas)"); ax[0].set_xlabel("step")
    ax[1].set_title("OOD vs TRAIN loss (parametric) — LOWER-LEFT better"); ax[1].set_xlabel("train loss")
    ax[1].set_ylabel("ood loss"); ax[1].legend()
    ax[2].plot(grid, base, "b-o", label="base", ms=3)
    ax[2].plot(grid, manas, "r-o", label="manas", ms=3)
    ax[2].set_title("OOD @ matched train loss (seed-avg)"); ax[2].set_xlabel("train loss")
    ax[2].legend()
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "manas_mnist1d.png"), dpi=120)
    print("saved manas_mnist1d_results.json + manas_mnist1d.png")
