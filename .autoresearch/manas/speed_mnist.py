"""Does the champion Manas recipe train FASTER than LR-tuned Muon on MNIST + MNIST-1D? (CPU)

Protocol (speed, not OOD): global batch 256 = 64 x ga4 (4 micro-votes/step).
  Phase 1: tune Muon LR per dataset ({1,2,4,8}e-3, seed 0, tail train loss picks).
  Phase 2: at the tuned LR, muon vs manas champion (micro_vote, rho=1.0, rho_step=0.96,
           gamma=gamma_intra=0.01, rank 8), 3 paired seeds.
Metrics: tail train loss (mean over last 20% of steps), steps-to-reach the muon arm's
final loss (speed number), test acc. Run:
  ../../BiBo/.venv/Scripts/python.exe .autoresearch/manas/speed_mnist.py
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.sm75.manas import ManasOptimizer

torch.set_num_threads(os.cpu_count() or 8)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
GA, MICRO = 4, 64                       # global batch 256 = 64 x 4 votes
STEPS = {"mnist1d": 600, "mnist": 350}
SEEDS = (0, 1, 2)


# ---------------- models (from the validated toy harnesses) ----------------
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
        return self.head(self.blocks(self.proj(F.silu(self.stem(x)).flatten(1))))


class NetMNIST(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, stride=2)
        self.proj = nn.Linear(8 * 13 * 13, 256, bias=False)
        self.blocks = nn.Sequential(GLUBlock(256, 512), GLUBlock(256, 512))
        self.head = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        return self.head(self.blocks(self.proj(F.silu(self.stem(x)).flatten(1))))


# ---------------- data ----------------
def data_mnist1d():
    from mnist1d.data import make_dataset, get_dataset_args
    a = get_dataset_args()
    a.num_samples = 8000
    a.seed = 42
    d = make_dataset(a)
    x = torch.tensor(d["x"], dtype=torch.float32).unsqueeze(1)
    y = torch.tensor(d["y"], dtype=torch.long)
    xt = torch.tensor(d["x_test"], dtype=torch.float32).unsqueeze(1)
    yt = torch.tensor(d["y_test"], dtype=torch.long)
    return x, y, xt, yt


def data_mnist():
    from torchvision import datasets, transforms
    root = os.path.join(HERE, "..", "data")
    tr = datasets.MNIST(root, train=True, download=True, transform=transforms.ToTensor())
    x = torch.stack([tr[i][0] for i in range(14000)])
    y = torch.tensor([tr[i][1] for i in range(14000)])
    return x[:12000], y[:12000], x[12000:], y[12000:]


DATA = {"mnist1d": data_mnist1d, "mnist": data_mnist}
MODEL = {"mnist1d": Net1D, "mnist": NetMNIST}


def law_gamma(lr):
    """The dose law at this config: gamma = 0.08 * sqrt(lr/3e-4) * k/sqrt(m)."""
    return 0.08 * (lr / 3e-4) ** 0.5 * GA / MICRO ** 0.5


def run(ds, seed, lr, gamma, x, y, xt, yt):
    """gamma=None -> muon; else manas at that probe_gamma (sketch window + lazy shift +
    ga1 self-gate all on current defaults; ga=4 engages the full stack)."""
    torch.manual_seed(seed)
    model = MODEL[ds]().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    kw = (dict(micro_vote=True, probe_rho=1.0, probe_rho_step=0.96,
               probe_gamma=gamma, probe_rank=8)                     # gamma_intra unset = gamma
          if gamma is not None else dict(probe_gamma=0.0))
    opt = ManasOptimizer(mats, lr=lr, weight_decay=0.01, **kw)
    aux = torch.optim.AdamW(rest, lr=lr, weight_decay=0.01)

    g = torch.Generator().manual_seed(seed)
    losses = []
    steps = STEPS[ds]
    for t in range(steps):
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
    with torch.no_grad():
        acc = (model(xt).argmax(-1) == yt).float().mean().item()
    tail = float(np.mean(losses[int(0.8 * steps):]))
    return {"losses": losses, "tail": tail, "acc": acc}


def steps_to(losses, target, w=25):
    """First step whose trailing-w running mean reaches target (running mean kills batch noise)."""
    rm = np.convolve(losses, np.ones(w) / w, mode="valid")
    hit = np.nonzero(rm <= target)[0]
    return int(hit[0]) + w if len(hit) else None


if __name__ == "__main__":
    RESFILE = os.path.join(HERE, "speed_mnist_results.json")

    def save(out):                                  # incremental: never lose finished arms
        with open(RESFILE, "w") as f:
            json.dump(out, f, indent=1)

    out = {"device": DEV}
    print(f"device: {DEV}", flush=True)
    for ds in ("mnist1d", "mnist"):
        print(f"\n=== {ds} (global 256 = {MICRO}x{GA}, {STEPS[ds]} steps) ===", flush=True)
        x, y, xt, yt = [t.to(DEV) for t in DATA[ds]()]
        t0 = time.time()
        out[ds] = {"tune": {}, "rows": []}
        # Phase 1: muon LR tune
        tune = {}
        for lr in (1e-3, 2e-3, 4e-3, 8e-3):
            r = run(ds, 0, lr, None, x, y, xt, yt)
            tune[lr] = r["tail"]
            out[ds]["tune"][f"{lr:.0e}"] = {"tail": r["tail"], "acc": r["acc"]}
            save(out)
            print(f"  tune muon lr={lr:.0e}: tail {r['tail']:.4f}  acc {r['acc']:.3f}", flush=True)
        lr_star = min(tune, key=tune.get)
        out[ds]["lr_star"] = lr_star
        print(f"  -> tuned muon lr = {lr_star:.0e}", flush=True)
        # Phase 2: paired seeds - muon vs OLD recipe (gamma 0.01, pre-sketch era value) vs
        # NEW recipe (dose-law gamma, sketch window + lazy shift on current defaults)
        g_law = law_gamma(lr_star)
        print(f"  law gamma at lr {lr_star:.0e}: {g_law:.4f}", flush=True)
        rows = []
        for sd in SEEDS:
            rm = run(ds, sd, lr_star, None, x, y, xt, yt)
            ro = run(ds, sd, lr_star, 0.01, x, y, xt, yt)
            rn = run(ds, sd, lr_star, g_law, x, y, xt, yt)
            s2 = steps_to(rn["losses"], rm["tail"])
            print(f"  seed {sd}: muon {rm['tail']:.4f}/{rm['acc']:.3f} | "
                  f"old(g0.01) {ro['tail']:.4f}/{ro['acc']:.3f} ({ro['tail'] - rm['tail']:+.4f}) | "
                  f"NEW(g{g_law:.3f}) {rn['tail']:.4f}/{rn['acc']:.3f} "
                  f"({rn['tail'] - rm['tail']:+.4f}) | new hits muon-final at "
                  f"{s2 if s2 else 'never'}/{STEPS[ds]}", flush=True)
            rows.append({"seed": sd, "muon": rm, "old": ro, "new": rn, "steps_to_muon_final": s2})
            out[ds]["rows"] = [{**r, **{a: {k: v for k, v in r[a].items() if k != "losses"}
                                        for a in ("muon", "old", "new")}} for r in rows]
            save(out)
        dn = np.mean([r["new"]["tail"] - r["muon"]["tail"] for r in rows])
        do = np.mean([r["old"]["tail"] - r["muon"]["tail"] for r in rows])
        da = np.mean([r["new"]["acc"] - r["muon"]["acc"] for r in rows])
        print(f"  == {ds}: NEW mean tail delta {dn:+.4f} (old {do:+.4f}), "
              f"NEW mean acc delta {da:+.4f}, {time.time() - t0:.0f}s ==", flush=True)
        out[ds]["mean_tail_delta_new"] = float(dn)
        out[ds]["mean_tail_delta_old"] = float(do)
        out[ds]["mean_acc_delta_new"] = float(da)
        save(out)
    print("\nwrote speed_mnist_results.json", flush=True)
