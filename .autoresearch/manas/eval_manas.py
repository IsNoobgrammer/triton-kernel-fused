"""FROZEN EVAL — Manas round (run-0 freeze). Do not edit; see scope.md.

MNIST-1D heterogeneous-sources protocol with the C1-C5 fixes:
  * 3 train regimes + 1 held-out OOD regime; source SHUFFLED per step (seeded) [C5]
  * paired seeds: seed fixes init + data + batch/source order identically across arms [C4]
  * primary metric: OOD accuracy at matched train loss along the CUMMIN frontier [C2/C3]
  * secondary: best OOD acc over training, final train loss (parity slice), final OOD CE
  * score(arm, base) = per-seed paired deltas of the frontier metric + noise floor

The protocol is frozen; ARMS are not part of this file. A candidate is any `trainer_factory`:
    factory(model, seed) -> trainer with .train_step(model, x, y) -> float loss
run_seed() drives it through the fixed schedule and measures. Nothing here may be tuned.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args

DEV = "cuda"
STEPS, BS, EVAL_EVERY = 1500, 128, 30
OPT_SEEDS = (0, 1, 2, 3, 4)
HELDOUT_SEEDS = (5, 6, 7)
LR, WD = 2e-3, 0.01

REGIMES = {
    "s0":  dict(corr_noise_scale=0.20, max_translation=36, shear_scale=0.5),
    "s1":  dict(corr_noise_scale=0.30, max_translation=60, shear_scale=1.0),
    "s2":  dict(corr_noise_scale=0.25, max_translation=48, shear_scale=0.75, scale_coeff=0.6),
    "ood": dict(corr_noise_scale=0.45, max_translation=24, shear_scale=1.4, iid_noise_scale=0.08),
}

_SRC_CACHE = {}


def make_source(name, seed):
    key = (name, seed)
    if key not in _SRC_CACHE:
        a = get_dataset_args()
        a.num_samples = 5000
        a.seed = 100 * seed + 7
        for k, v in REGIMES[name].items():
            setattr(a, k, v)
        d = make_dataset(a)
        _SRC_CACHE[key] = (
            (torch.tensor(d["x"], dtype=torch.float32, device=DEV).unsqueeze(1),
             torch.tensor(d["y"], dtype=torch.long, device=DEV)),
            (torch.tensor(d["x_test"], dtype=torch.float32, device=DEV).unsqueeze(1),
             torch.tensor(d["y_test"], dtype=torch.long, device=DEV)),
        )
    return _SRC_CACHE[key]


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
        self.stem = nn.Conv1d(1, 16, 5, stride=2)
        self.proj = nn.Linear(16 * 18, 192, bias=False)
        self.blocks = nn.Sequential(GLUBlock(192, 384), GLUBlock(192, 384), GLUBlock(192, 384))
        self.head = nn.Linear(192, 10, bias=False)

    def forward(self, x):
        z = F.silu(self.stem(x)).flatten(1)
        return self.head(self.blocks(self.proj(z)))


def split_params(model):
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    return mats, rest


def run_seed(trainer_factory, seed):
    """Drive one arm through the fixed schedule. Returns the eval log for metric computation."""
    torch.manual_seed(seed)
    model = Net1D().to(DEV)
    trainer = trainer_factory(model, seed)

    tr = {s: make_source(s, seed)[0] for s in ("s0", "s1", "s2")}
    ood_x, ood_y = make_source("ood", seed)[1]
    order = ("s0", "s1", "s2")

    log = {"step": [], "train": [], "ood_acc": [], "ood_ce": []}
    g = torch.Generator(device="cpu").manual_seed(seed)          # data AND source order [C4]
    for t in range(STEPS):
        s = order[int(torch.randint(0, 3, (1,), generator=g))]   # shuffled source [C5]
        x, y = tr[s]
        idx = torch.randint(0, x.shape[0], (BS,), generator=g).to(DEV)
        trainer.train_step(model, x[idx], y[idx])

        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            probe_on = getattr(getattr(trainer, "opt", None), "_probe_on", False)
            assert not probe_on, "eval must run at theta, probe off [C12]"
            with torch.no_grad():
                trl = float(np.mean([F.cross_entropy(model(tr[s][0][:1000]), tr[s][1][:1000]).item()
                                     for s in order]))
                logits = model(ood_x)
                log["ood_ce"].append(F.cross_entropy(logits, ood_y).item())
                log["ood_acc"].append((logits.argmax(-1) == ood_y).float().mean().item())
            log["step"].append(t)
            log["train"].append(trl)
    return log


def frontier(log):
    """Cummin frontier of train loss [C2]: checkpoints where train loss reaches a new minimum."""
    tr = np.array(log["train"]); acc = np.array(log["ood_acc"])
    keep = tr <= np.minimum.accumulate(tr)
    return tr[keep], acc[keep]


def paired_metrics(cand_log, base_log):
    """One seed's paired comparison. Frontier metric on the common matched grid [C2/C3]."""
    tb, ab = frontier(base_log)
    tc, ac = frontier(cand_log)
    lo = max(tb.min(), tc.min()); hi = min(tb.max(), tc.max())
    grid = np.linspace(lo + 0.02 * (hi - lo), hi - 0.02 * (hi - lo), 25)
    # frontier train loss is strictly decreasing over time -> ascending sort for interp
    fb = np.interp(grid, tb[::-1], ab[::-1])
    fc = np.interp(grid, tc[::-1], ac[::-1])
    return {
        "delta_frontier": float(np.mean(fc - fb)),               # PRIMARY
        "delta_best_acc": float(max(cand_log["ood_acc"]) - max(base_log["ood_acc"])),
        "delta_final_train": float(min(cand_log["train"]) - min(base_log["train"])),  # parity slice
        "delta_final_ce": float(cand_log["ood_ce"][-1] - base_log["ood_ce"][-1]),
    }


def score(cand_logs, base_logs, seeds):
    """Aggregate paired deltas across seeds; noise floor = SEM of the primary deltas."""
    per_seed = {sd: paired_metrics(cand_logs[sd], base_logs[sd]) for sd in seeds}
    d = np.array([per_seed[sd]["delta_frontier"] for sd in seeds])
    return {
        "per_seed": per_seed,
        "delta_frontier_mean": float(d.mean()),
        "noise_sem": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan"),
        "delta_best_acc_mean": float(np.mean([per_seed[sd]["delta_best_acc"] for sd in seeds])),
        "delta_final_train_mean": float(np.mean([per_seed[sd]["delta_final_train"] for sd in seeds])),
    }
