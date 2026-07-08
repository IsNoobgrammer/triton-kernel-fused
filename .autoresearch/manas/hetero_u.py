"""Task #1 follow-up: does the low-rank U BUFFER help under HETEROGENEITY?

Prediction (our own analysis): u (rho-decayed memory of applied updates, low-rank, comp=+1)
rode along without separating on homogeneous tasks because near-identical batches carry no
differential update-history information. Heterogeneous sources are exactly where it should
start to matter. So the u-minus-no-u benefit should be ~0 at alpha=0 and GROW with alpha.

Paired per seed (same init+data+source order): Manas rank-8 (no u) vs Manas rank-8 + rank-8
u (comp=1.0), across the same alpha grid. Metric: mean per-source test accuracy + loss
spread; report paired delta (u minus no-u) vs alpha.

Run: ../../BiBo/.venv/Scripts/python.exe hetero_u.py
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
STEPS, BS = 1400, 128
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BASE = dict(corr_noise_scale=0.25, max_translation=42, shear_scale=0.75)
OFF = [dict(corr_noise_scale=+0.18, max_translation=-18, shear_scale=-0.35),
       dict(corr_noise_scale=0.0, max_translation=0, shear_scale=0.0),
       dict(corr_noise_scale=-0.10, max_translation=+22, shear_scale=+0.65)]


def source(i, alpha, seed):
    a = get_dataset_args(); a.num_samples = 4000; a.seed = 100 * seed + 7 + i
    for k in BASE:
        v = max(0.01, BASE[k] + alpha * OFF[i][k])
        setattr(a, k, int(round(v)) if k == "max_translation" else v)
    d = make_dataset(a)
    to = lambda z, dt: torch.tensor(z, dtype=dt, device=DEV)
    return ((to(d["x"], torch.float32).unsqueeze(1), to(d["y"], torch.long)),
            (to(d["x_test"], torch.float32).unsqueeze(1), to(d["y_test"], torch.long)))


def run(use_u, alpha, seed):
    torch.manual_seed(seed)
    model = E.Net1D().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    opt = ManasOptimizer(mats, lr=2e-3, probe_gamma=0.08, probe_rho=0.98, probe_rank=8,
                         comp=(1.0 if use_u else None), weight_decay=0.01)
    aux = torch.optim.AdamW(rest, lr=2e-3, weight_decay=0.01)
    src = SRC[(alpha, seed)]; tr = [src[i][0] for i in range(3)]; te = [src[i][1] for i in range(3)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(STEPS):
        s = int(torch.randint(0, 3, (1,), generator=g))
        x, y = tr[s]; idx = torch.randint(0, x.shape[0], (BS,), generator=g).to(DEV)
        with opt.probe():
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True); loss.backward()
        opt.step(); aux.step()
    with torch.no_grad():
        losses = [F.cross_entropy(model(xt), yt).item() for xt, yt in te]
        accs = [(model(xt).argmax(-1) == yt).float().mean().item() for xt, yt in te]
    return {"spread": float(np.std(losses)), "acc": float(np.mean(accs))}


if __name__ == "__main__":
    print("building sources across alpha x seeds ...")
    SRC = {(a, sd): [source(i, a, sd) for i in range(3)] for a in ALPHAS for sd in SEEDS}
    print(f"U-buffer under heterogeneity, {STEPS} steps, {len(SEEDS)} paired seeds")
    print("delta = (Manas+u) minus (Manas no-u), paired per seed\n")
    print(f"{'alpha':>6} | {'acc no-u':>9} | {'acc +u':>8} | {'delta acc':>16} | {'spread no-u':>11} | {'spread +u':>9}")
    print("-" * 82)
    rows = []
    for a in ALPHAS:
        da, sn, su, an, au = [], [], [], [], []
        for sd in SEEDS:
            r0 = run(False, a, sd); r1 = run(True, a, sd)
            da.append(r1["acc"] - r0["acc"]); an.append(r0["acc"]); au.append(r1["acc"])
            sn.append(r0["spread"]); su.append(r1["spread"])
        sem = np.std(da, ddof=1) / np.sqrt(len(da))
        row = dict(alpha=a, acc_nou=float(np.mean(an)), acc_u=float(np.mean(au)),
                   dacc=float(np.mean(da)), dacc_sem=float(sem),
                   spread_nou=float(np.mean(sn)), spread_u=float(np.mean(su)))
        rows.append(row)
        print(f"{a:>6.2f} | {row['acc_nou']*100:>8.2f}% | {row['acc_u']*100:>7.2f}% | "
              f"{row['dacc']*100:>+8.2f}%+/-{sem*100:.2f} | {row['spread_nou']:>11.4f} | {row['spread_u']:>9.4f}",
              flush=True)
    print("\nPrediction: delta acc ~0 at alpha=0, GROWING with alpha if u earns its keep under heterogeneity.")
    import json
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "hetero_u.json"), "w"), default=float)
