"""Task #1 — make Manas visibly diverge from Muon via a HETEROGENEITY DOSE-RESPONSE.

Hypothesis (the mechanism): Manas differs from Muon only when batches disagree. So the
Manas-vs-Muon difference should GROW with how different the training sources are, and
vanish at zero heterogeneity. If true, this is the cleanest possible proof the two are
genuinely different optimizers, not the same path shifted.

Knob: alpha in [0,1] spreads 3 MNIST-1D sources apart from a shared base regime.
  alpha=0 -> all three identical (homogeneous; Manas MUST equal Muon)
  alpha=1 -> widely separated regimes (strong disagreement; Manas should diverge)

Paired per seed (same init + data + source order): run base Muon (gamma=0) and Manas
(rank-8 default) and measure, at each alpha:
  * TRAJECTORY divergence:  ||W_manas - W_muon|| / ||W_muon||   (matrices, end of train)
  * per-source test-loss SPREAD (std across sources) — does Manas equalize the tasks?
  * mean per-source test accuracy
Reports each metric vs alpha. Expect divergence ~0 at alpha=0, rising with alpha.

Run: ../../BiBo/.venv/Scripts/python.exe hetero_dose.py
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
STEPS, BS, EVAL_EVERY = 1400, 128, 100
SEEDS = (0, 1, 2, 3, 4)
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
BASE = dict(corr_noise_scale=0.25, max_translation=42, shear_scale=0.75)
# per-source offset directions; scaled by alpha (source 1 stays at base)
OFF = [dict(corr_noise_scale=+0.18, max_translation=-18, shear_scale=-0.35),
       dict(corr_noise_scale=0.0,   max_translation=0,   shear_scale=0.0),
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


def run(kind, alpha, seed, init_seed=None):
    torch.manual_seed(seed if init_seed is None else init_seed)   # init separable from data
    model = E.Net1D().to(DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    if kind == "muon":
        opt = ManasOptimizer(mats, lr=2e-3, probe_gamma=0.0, weight_decay=0.01)
        probe = None
    else:
        opt = ManasOptimizer(mats, lr=2e-3, probe_gamma=0.08, probe_rho=0.98,
                             probe_rank=8, weight_decay=0.01)
        probe = opt.probe
    aux = torch.optim.AdamW(rest, lr=2e-3, weight_decay=0.01)
    src = SRC[(alpha, seed)]
    tr = [src[i][0] for i in range(3)]
    te = [src[i][1] for i in range(3)]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(STEPS):
        s = int(torch.randint(0, 3, (1,), generator=g))
        x, y = tr[s]
        idx = torch.randint(0, x.shape[0], (BS,), generator=g).to(DEV)
        def fb():
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True); loss.backward()
        if probe is not None:
            with probe(): fb()
        else:
            fb()
        opt.step(); aux.step()
    with torch.no_grad():
        losses = [F.cross_entropy(model(xt), yt).item() for xt, yt in te]
        accs = [(model(xt).argmax(-1) == yt).float().mean().item() for xt, yt in te]
        preds = torch.cat([model(xt).argmax(-1) for xt, yt in te])       # for functional divergence
    return {"spread": float(np.std(losses)), "acc": float(np.mean(accs)),
            "losses": losses, "preds": preds}


if __name__ == "__main__":
    print("building sources across alpha x seeds ...")
    SRC = {}
    for a in ALPHAS:
        for sd in SEEDS:
            SRC[(a, sd)] = [source(i, a, sd) for i in range(3)]
    print(f"heterogeneity dose-response, {STEPS} steps, {len(SEEDS)} paired seeds\n")
    print("func.diverge = %% of test inputs where two models predict different labels")
    print(f"{'alpha':>6} | {'Muon<->Manas':>13} | {'Muon<->Muon(seed)':>18} | {'excess (optim)':>15} | {'acc Muon':>9} | {'acc Manas':>9}")
    print("-" * 92)
    rows = []
    for a in ALPHAS:
        d_opt, d_seed, acm, aca = [], [], [], []
        for sd in SEEDS:
            rmu = run("muon", a, sd)                              # muon, init=data=sd
            rma = run("manas", a, sd)                             # manas, SAME init+data -> pure optimizer
            rmu2 = run("muon", a, sd, init_seed=sd + 50)          # muon, SAME data, different init -> seed floor
            d_opt.append((rmu["preds"] != rma["preds"]).float().mean().item())
            d_seed.append((rmu["preds"] != rmu2["preds"]).float().mean().item())
            acm.append(rmu["acc"]); aca.append(rma["acc"])
        ex = np.array(d_opt) - np.array(d_seed)
        row = dict(alpha=a, d_opt=float(np.mean(d_opt)), d_seed=float(np.mean(d_seed)),
                   excess=float(ex.mean()), excess_sem=float(ex.std(ddof=1) / np.sqrt(len(ex))),
                   acm=float(np.mean(acm)), aca=float(np.mean(aca)))
        rows.append(row)
        print(f"{a:>6.2f} | {row['d_opt']*100:>12.1f}% | {row['d_seed']*100:>17.1f}% | "
              f"{row['excess']*100:>+9.1f}%+/-{row['excess_sem']*100:.1f} | "
              f"{row['acm']*100:>8.2f}% | {row['aca']*100:>8.2f}%", flush=True)
    import json
    json.dump(rows, open(os.path.join(os.path.dirname(__file__), "hetero_dose.json"), "w"), default=float)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        al = [r["alpha"] for r in rows]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(al, [r["d_opt"]*100 for r in rows], "o-", label="Muon<->Manas (optimizer)", color="#c8892a")
        ax[0].plot(al, [r["d_seed"]*100 for r in rows], "o-", label="Muon<->Muon (seed floor)", color="#1f9e86")
        ax[0].set_title("functional divergence vs heterogeneity"); ax[0].set_xlabel("heterogeneity alpha")
        ax[0].set_ylabel("% test predictions differing"); ax[0].legend()
        ax[1].errorbar(al, [r["excess"]*100 for r in rows], yerr=[r["excess_sem"]*100 for r in rows], marker="o", color="#c8892a")
        ax[1].axhline(0, color="gray", lw=.8)
        ax[1].set_title("excess divergence (optimizer beyond seed noise)"); ax[1].set_xlabel("heterogeneity alpha")
        ax[1].set_ylabel("Muon<->Manas minus Muon<->Muon (%)")
        fig.tight_layout(); fig.savefig(os.path.join(os.path.dirname(__file__), "hetero_dose.png"), dpi=120)
        print("\nsaved hetero_dose.png + hetero_dose.json")
    except Exception as e:
        print("plot skipped:", e)
