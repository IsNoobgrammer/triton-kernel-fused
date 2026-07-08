"""Reach-for-95: peak IN-DISTRIBUTION test accuracy on standard (solvable) MNIST-1D.

The frozen research eval uses a deliberately-unseen OOD regime that caps ~45% on purpose
(so it discriminates optimizers). This is a DIFFERENT, legitimate question: on a task that
is actually winnable, how high does each optimizer climb and how fast? Standard MNIST-1D,
train + in-distribution test split, longer run. Reports peak test acc + steps to reach 90%.

Arms: AdamW (all params) | Muon (mats) + AdamW (rest) | Manas (mats) + AdamW (rest).
Run: ../../BiBo/.venv/Scripts/python.exe reach95.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
import torch
import torch.nn.functional as F
from mnist1d.data import make_dataset, get_dataset_args

import eval_manas as E
from kernels.sm75.manas import ManasOptimizer

DEV = "cuda"
STEPS, BS, EVAL_EVERY = 5000, 128, 100
SEEDS = (0, 1, 2)


def data(seed):
    a = get_dataset_args()                        # canonical MNIST-1D (CNN reaches ~95%)
    a.num_samples = 6000
    a.seed = seed
    d = make_dataset(a)
    tx = torch.tensor(d["x"], dtype=torch.float32, device=DEV).unsqueeze(1)
    ty = torch.tensor(d["y"], dtype=torch.long, device=DEV)
    vx = torch.tensor(d["x_test"], dtype=torch.float32, device=DEV).unsqueeze(1)
    vy = torch.tensor(d["y_test"], dtype=torch.long, device=DEV)
    return tx, ty, vx, vy


def make_opt(kind, model):
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    if kind == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.01), None
    if kind == "muon":
        return (ManasOptimizer(mats, lr=2e-3, probe_gamma=0.0, weight_decay=0.01),
                torch.optim.AdamW(rest, lr=3e-3, weight_decay=0.01))
    # manas: rank-8 default, gamma/rho from the round
    return (ManasOptimizer(mats, lr=2e-3, probe_gamma=0.08, probe_rho=0.98,
                           probe_rank=8, weight_decay=0.01),
            torch.optim.AdamW(rest, lr=3e-3, weight_decay=0.01))


def run(kind, seed):
    torch.manual_seed(seed)
    model = E.Net1D().to(DEV)
    opt, aux = make_opt(kind, model)
    probe = opt.probe if (kind == "manas") else None
    tx, ty, vx, vy = DAT[seed]
    g = torch.Generator(device="cpu").manual_seed(seed)
    peak, hit90 = 0.0, None
    for t in range(STEPS):
        idx = torch.randint(0, tx.shape[0], (BS,), generator=g).to(DEV)
        xb, yb = tx[idx], ty[idx]
        def fb():
            loss = F.cross_entropy(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            if aux: aux.zero_grad(set_to_none=True)
            loss.backward()
        if probe is not None:
            with probe(): fb()
        else:
            fb()
        opt.step()
        if aux: aux.step()
        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            with torch.no_grad():
                acc = (model(vx).argmax(-1) == vy).float().mean().item()
            if acc > peak: peak = acc
            if hit90 is None and acc >= 0.90: hit90 = t
    return peak, hit90


if __name__ == "__main__":
    print("building standard MNIST-1D (in-distribution) x seeds ...")
    DAT = {s: data(s) for s in SEEDS}
    print(f"arms x {STEPS} steps, in-distribution TEST accuracy (peak):\n")
    res = {}
    for kind in ("adamw", "muon", "manas"):
        peaks, hits = [], []
        for s in SEEDS:
            p, h = run(kind, s)
            peaks.append(p); hits.append(h)
            print(f"  {kind:<7} seed {s}: peak {p*100:.1f}%  hit90 @ {h if h is not None else '-'}", flush=True)
        res[kind] = (np.mean(peaks), np.std(peaks), [h for h in hits if h is not None])
    print("\n== PEAK IN-DISTRIBUTION TEST ACCURACY ==")
    for kind in ("adamw", "muon", "manas"):
        m, s, hits = res[kind]
        avg90 = f"{int(np.mean(hits))}" if hits else "never"
        print(f"  {kind:<7} {m*100:5.1f}% +/- {s*100:.1f}   mean steps to 90%: {avg90}")
