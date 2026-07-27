# MNIST-1D at ~90%: bigger conv stem + wide GLU trunk, Muon vs Manas (tracked gamma).
# MNIST-1D benchmarks (Greydanus): CNN ~94%, GRU ~91%, MLP ~68% - convs crack it, so the
# stem gets 3 conv layers (AdamW); Muon/Manas work the ~4M-param GLU trunk. 64x4 slicing,
# cosine LR, manas gamma = min(law(lr_t), 0.073) tracked per step.
# Run: ../../BiBo/.venv/Scripts/python.exe -u big_mnist1d.py
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.sm75.manas import ManasOptimizer

DEV = "cuda" if torch.cuda.is_available() else "cpu"
GA, MICRO = 4, 64
STEPS = int(os.environ.get("MANAS_STEPS", 1500))
SEEDS = tuple(range(int(os.environ.get('MANAS_NSEEDS', 3))))
EVAL_EVERY = 25
NSAMP = int(os.environ.get("MANAS_SAMPLES", 12000))
TAG = os.environ.get("MANAS_TAG", "")
LR0, LR_END = 1e-3, 1e-4
GCAP = 0.073


def law_gamma(lr):
    return min(0.08 * (lr / 3e-4) ** 0.5 * GA / MICRO ** 0.5, GCAP)


def lr_at(t):
    return LR_END + (LR0 - LR_END) * 0.5 * (1 + math.cos(math.pi * t / STEPS))


class GLUBlock(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)
        self.norm = nn.LayerNorm(h)

    def forward(self, x):
        z = self.norm(x)
        return x + self.down(F.silu(self.gate(z)) * self.up(z))


class Big1D(nn.Module):
    def __init__(self, h=384, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(                      # convs: AdamW (never Muon)
            nn.Conv1d(1, 32, 5, stride=2, padding=2), nn.GELU(),   # 40 -> 20
            nn.Conv1d(32, 64, 3, stride=2, padding=1), nn.GELU(),  # 20 -> 10
            nn.Conv1d(64, 96, 3, stride=1, padding=1), nn.GELU(),  # 10 -> 10
        )
        self.proj = nn.Linear(96 * 10, h, bias=False)
        self.blocks = nn.Sequential(*[GLUBlock(h, 2 * h) for _ in range(blocks)])
        self.norm = nn.LayerNorm(h)
        self.head = nn.Linear(h, 10, bias=False)

    def forward(self, x):
        z = self.stem(x).flatten(1)
        return self.head(self.norm(self.blocks(self.proj(z))))


def data():
    from mnist1d.data import make_dataset, get_dataset_args
    a = get_dataset_args()
    a.num_samples = NSAMP
    a.seed = 42
    d = make_dataset(a)
    return (torch.tensor(d["x"], dtype=torch.float32).unsqueeze(1).to(DEV),
            torch.tensor(d["y"], dtype=torch.long).to(DEV),
            torch.tensor(d["x_test"], dtype=torch.float32).unsqueeze(1).to(DEV),
            torch.tensor(d["y_test"], dtype=torch.long).to(DEV))


def run(seed, manas, x, y, xt, yt):
    torch.manual_seed(seed)
    model = Big1D().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    kw = (dict(micro_vote=True, probe_rho=1.0, probe_rho_step=0.96,
               probe_gamma=law_gamma(LR0), probe_rank=8)
          if manas else dict(probe_gamma=0.0))
    opt = ManasOptimizer(mats, lr=LR0, weight_decay=0.01, **kw)
    aux = torch.optim.AdamW(rest, lr=LR0, weight_decay=0.01)
    g = torch.Generator().manual_seed(seed)
    accs, evs, losses = [], [], []
    for t in range(STEPS):
        lr_t = lr_at(t)
        for grp in list(opt.param_groups) + list(aux.param_groups):
            grp["lr"] = lr_t
        if manas:
            opt.probe_gamma = law_gamma(lr_t)
        opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
        tot = 0.0
        for m in range(GA):
            idx = torch.randint(0, x.shape[0], (MICRO,), generator=g).to(DEV)
            with opt.probe():
                loss = F.cross_entropy(model(x[idx]), y[idx]) / GA
                loss.backward()
            opt.vote()
            tot += loss.item() * GA
        opt.step(); aux.step()
        losses.append(tot / GA)
        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            model.eval()
            with torch.no_grad():
                accs.append((model(xt).argmax(-1) == yt).float().mean().item())
            model.train()
            evs.append(t)
    tail = float(np.mean(losses[int(0.9 * STEPS):]))
    m20 = float(np.mean(losses[int(0.18 * STEPS):int(0.22 * STEPS)]))
    m50 = float(np.mean(losses[int(0.48 * STEPS):int(0.52 * STEPS)]))
    print(f"    loss milestones: 20% {m20:.4f}  50% {m50:.4f}  tail {tail:.4f}"
          + ("  [SATURATED EARLY]" if m20 < 0.05 else ""), flush=True)
    return evs, accs, tail


if __name__ == "__main__":
    x, y, xt, yt = data()
    n = sum(p.numel() for p in Big1D().parameters())
    print(f"device {DEV}, params {n/1e6:.2f}M, gamma0 {law_gamma(LR0):.4f}", flush=True)
    out = {"muon": [], "manas": []}
    for sd in SEEDS:
        for name, mn in (("muon", False), ("manas", True)):
            evs, accs, tail = run(sd, mn, x, y, xt, yt)
            out[name].append(accs)
            print(f"  seed {sd} {name}: final acc {accs[-1]:.3f}  peak {max(accs):.3f}  "
                  f"tail loss {tail:.4f}", flush=True)
    out["steps"] = evs
    with open(os.path.join(HERE, f"big_mnist1d{TAG}.json"), "w") as f:
        json.dump(out, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for name, color, lbl in (("muon", "#888888", "Muon (LR-tuned)"),
                             ("manas", "#1f5fa8", "Manas")):
        a = np.array(out[name])
        for row in a:
            ax.plot(out["steps"], row, color=color, alpha=0.20, lw=1)
        ax.plot(out["steps"], a.mean(0), color=color, lw=2.6,
                label=f"{lbl} (final {a[:, -1].mean():.3f})")
    ax.axhline(0.90, color="#2e7d4f", lw=1, ls="--", alpha=0.6)
    ax.annotate("90%", (0, 0.902), fontsize=9, color="#2e7d4f")
    ax.set_title(f"MNIST-1D ({NSAMP} samples), 5.7M-param conv+GLU (64x4, cosine lr, 3-seed mean)")
    ax.set_xlabel("step"); ax.set_ylabel("test acc")
    ax.legend(loc="lower right"); ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, f"big_mnist1d{TAG}.png"), dpi=130)
    print(f"wrote big_mnist1d{TAG}.png/json", flush=True)
