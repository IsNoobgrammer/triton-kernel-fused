"""Reach-95, Linear-heavy: MLP on standard 2D MNIST, where every layer is a matrix the
matrix-optimizer actually drives (no convs sidelining Muon/Manas), and MLPs reach ~98%.
Answers "can AdamW / Muon / Manas reach ~95%?" with the optimizer in the driver's seat.
Reports peak test accuracy + steps to cross 95%.

Run: ../../BiBo/.venv/Scripts/python.exe reach95_mnist.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from kernels.sm75.manas import ManasOptimizer

DEV = "cuda"
STEPS, BS, EVAL_EVERY = 3000, 256, 100
SEEDS = (0, 1, 2)
HERE = os.path.dirname(os.path.abspath(__file__))


def load():
    tr = datasets.MNIST(os.path.join(HERE, "data"), train=True, download=True,
                        transform=transforms.ToTensor())
    te = datasets.MNIST(os.path.join(HERE, "data"), train=False, download=True,
                        transform=transforms.ToTensor())
    tx = tr.data.float().view(-1, 784).to(DEV) / 255.0
    ty = tr.targets.to(DEV)
    vx = te.data.float().view(-1, 784).to(DEV) / 255.0
    vy = te.targets.to(DEV)
    return tx, ty, vx, vy


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(784, 512, bias=False), nn.GELU(),
                                 nn.Linear(512, 512, bias=False), nn.GELU(),
                                 nn.Linear(512, 10, bias=False))

    def forward(self, x):
        return self.net(x)


def make_opt(kind, model):
    mats = [p for p in model.parameters() if p.ndim == 2]      # every layer here is 2D
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.01), None
    if kind == "muon":
        return ManasOptimizer(mats, lr=2e-2, probe_gamma=0.0, weight_decay=0.01), None
    return ManasOptimizer(mats, lr=2e-2, probe_gamma=0.08, probe_rho=0.98,
                          probe_rank=8, weight_decay=0.01), None


def run(kind, seed):
    torch.manual_seed(seed)
    model = MLP().to(DEV)
    opt, _ = make_opt(kind, model)
    probe = opt.probe if kind == "manas" else None
    g = torch.Generator(device="cpu").manual_seed(seed)
    peak, hit95 = 0.0, None
    for t in range(STEPS):
        idx = torch.randint(0, TX.shape[0], (BS,), generator=g).to(DEV)
        xb, yb = TX[idx], TY[idx]
        def fb():
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
        if probe is not None:
            with probe(): fb()
        else:
            fb()
        opt.step()
        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            with torch.no_grad():
                acc = (model(VX).argmax(-1) == VY).float().mean().item()
            if acc > peak: peak = acc
            if hit95 is None and acc >= 0.95: hit95 = t
    return peak, hit95


if __name__ == "__main__":
    TX, TY, VX, VY = load()
    print(f"MLP (784-512-512-10, all matrices) on 2D MNIST, {STEPS} steps x {SEEDS}\n")
    res = {}
    for kind in ("adamw", "muon", "manas"):
        peaks, hits = [], []
        for s in SEEDS:
            p, h = run(kind, s); peaks.append(p); hits.append(h)
            print(f"  {kind:<7} seed {s}: peak {p*100:.2f}%  hit95 @ {h if h is not None else '-'}", flush=True)
        res[kind] = (np.mean(peaks), np.std(peaks), [h for h in hits if h is not None])
    print("\n== PEAK 2D-MNIST TEST ACCURACY ==")
    for kind in ("adamw", "muon", "manas"):
        m, s, hits = res[kind]
        a95 = f"{int(np.mean(hits))}" if hits else "never"
        print(f"  {kind:<7} {m*100:5.2f}% +/- {s*100:.2f}   mean steps to 95%: {a95}")
