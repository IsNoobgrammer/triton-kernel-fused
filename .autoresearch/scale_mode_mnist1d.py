"""FusedMuon scale_mode A/B on MNIST-1D: aurora (default) vs normuon vs normuon-EMA (v1/v2).

WHY THIS SCRIPT EXISTS: manas_mnist1d.py drives ManasOptimizer, so it has no scale_mode axis.
This reuses its data + Net1D and swaps in FusedMuon so the modes can be compared directly.

WHAT IT CAN AND CANNOT TEST
  CAN : the RECTANGULAR-matrix case. Per muon_scaling.py, "square matrices have uniform rows
        anyway, so all three coincide there; the modes only differ on rectangular (tall) weights."
        Net1D's gate/up (i,h), down (h,i), proj (192,288) and head (10,192) are all rectangular,
        so the neuron-death effect normuon targets is genuinely exercised.
  CANNOT: PER-EXPERT orthogonalization. Net1D has no 3D stacked params, so the batched (M,rows,cols)
        path never sees M>1. If the question is "is normuon better on the MoE expert stacks", this
        script cannot answer it -- that needs BiBo's 3D experts.gate_up_proj / down_proj.

All modes target update RMS = 0.2 (Moonlight/DeepSeek-V4), so lr and wd carry over unchanged and
the comparison is LR-fair by construction (see muon_scaling.apply_perrow).

    python .autoresearch/scale_mode_mnist1d.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn.functional as F

import manas_mnist1d as M
from kernels.sm75.muon import FusedMuon
from mnist1d.data import make_dataset, get_dataset_args

# 5000/source (manas_mnist1d's default) drives train loss to ~0.10 by step 1500 -- that is
# memorization, and it makes the id/ood comparison mostly a readout of how confidently each mode
# overfits. 15000/source (3x) keeps the same 1500 steps but cuts epochs ~3x, so the generalization
# gap is measured in a regime where the model still has something left to learn.
N_SAMPLES = 15000


def make_source(reg, seed):
    a = get_dataset_args()
    a.num_samples = N_SAMPLES
    a.seed = seed
    for k, v in reg.items():
        setattr(a, k, v)
    d = make_dataset(a)
    import torch as _t
    f = lambda z, dt: _t.tensor(z, dtype=dt, device=M.DEV)
    return ((f(d["x"], _t.float32).unsqueeze(1), f(d["y"], _t.long)),
            (f(d["x_test"], _t.float32).unsqueeze(1), f(d["y_test"], _t.long)))

_KJ, _PIN = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)
NS8 = (_KJ,) * 6 + (_PIN,) * 2          # 6 Keller-Jordan quintic + 2 Pin, as used in BiBo
MODES = ("aurora", "normuon", "aurora_ema", "aurora_ema_v2")
SEEDS = (0, 1, 2)


def run(mode, seed, tr, idtest, ood_x, ood_y):
    torch.manual_seed(seed)
    model = M.Net1D().to(M.DEV)
    mats = [p for p in model.parameters() if p.ndim == 2]
    rest = [p for p in model.parameters() if p.ndim != 2]
    opt = FusedMuon([{"params": mats}], lr=2e-3, momentum=0.95, weight_decay=0.01,
                    coeffs=NS8, ns_dtype=torch.bfloat16, scale_mode=mode)
    aux = torch.optim.AdamW(rest, lr=2e-3, weight_decay=0.01)

    order = ["s0", "s1", "s2"]
    log = {"step": [], "train": [], "id": [], "ood": [], "ood_acc": []}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(M.STEPS):
        x, y = tr[order[t % 3]]
        idx = torch.randint(0, x.shape[0], (M.BS,), generator=g).to(M.DEV)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); aux.step()
        if t % M.EVAL_EVERY == 0 or t == M.STEPS - 1:
            with torch.no_grad():
                trl = np.mean([F.cross_entropy(model(tr[s][0][:1000]), tr[s][1][:1000]).item()
                               for s in order])
                idl = np.mean([F.cross_entropy(model(xt), yt).item() for xt, yt in idtest])
                ol = model(ood_x)
                oodl = F.cross_entropy(ol, ood_y).item()
                ooda = (ol.argmax(-1) == ood_y).float().mean().item()
            for k, v in (("step", t), ("train", float(trl)), ("id", float(idl)),
                         ("ood", oodl), ("ood_acc", ooda)):
                log[k].append(v)
    print(f"  {mode:<14} s{seed}  train {log['train'][-1]:.4f}  id {log['id'][-1]:.4f}  "
          f"ood {log['ood'][-1]:.4f}  ood_acc {log['ood_acc'][-1]:.4f}", flush=True)
    return {"mode": mode, "seed": seed, "log": log}


def main():
    print("building MNIST-1D sources ...", flush=True)
    SRC = {(s, sd): make_source(M.REGIMES[s], seed=100 * sd + 7)
           for s in M.REGIMES for sd in SEEDS}
    n_tr = SRC[("s0", SEEDS[0])][0][0].shape[0]
    print(f"  {N_SAMPLES}/source -> {n_tr} train ex/source, {3 * n_tr} total; "
          f"{M.STEPS} steps x BS {M.BS} = {M.STEPS * M.BS / (3 * n_tr):.1f} epochs", flush=True)
    R = []
    for sd in SEEDS:
        tr = {s: SRC[(s, sd)][0] for s in ["s0", "s1", "s2"]}
        idtest = [SRC[(s, sd)][1] for s in ["s0", "s1", "s2"]]
        ood_x, ood_y = SRC[("ood", sd)][1]
        for mode in MODES:
            R.append(run(mode, sd, tr, idtest, ood_x, ood_y))

    print(f"\n{'mode':<14} {'train':>18} {'id':>18} {'ood':>18} {'ood_acc':>18}")
    base = {}
    for mode in MODES:
        rs = [r for r in R if r["mode"] == mode]
        cells = []
        for k in ("train", "id", "ood", "ood_acc"):
            v = np.array([r["log"][k][-1] for r in rs])
            cells.append(v)
            base.setdefault(k, {})[mode] = v
        print(f"{mode:<14} " + " ".join(f"{c.mean():>11.4f}+-{c.std(ddof=1):.4f}" for c in cells))

    print("\nvs aurora (mean delta / paired-seed sigma; negative = better for train/id/ood):")
    for k in ("train", "id", "ood", "ood_acc"):
        a = base[k]["aurora"]
        for mode in MODES[1:]:
            d = base[k][mode] - a
            sd = d.std(ddof=1) if len(d) > 1 else float("nan")
            print(f"  {k:<8} {mode:<14} {d.mean():+.4f}  (paired sd {sd:.4f}, n={len(d)})")
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "scale_mode_mnist1d.json"), "w"))


if __name__ == "__main__":
    main()
