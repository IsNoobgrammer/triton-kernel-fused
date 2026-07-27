"""FusedMuon scale_mode A/B on MNIST-1D with REAL 3D GLU EXPERT STACKS (E=5, top-2 routed).

WHY A SECOND SCRIPT: scale_mode_mnist1d.py uses Net1D, whose params are all 2D, so Muon's
Newton-Schulz never batches (M=1) and PER-EXPERT orthogonalization is never exercised. That is the
case NorMuon's argument is actually about -- when the rows being leverage-balanced ARE experts.
Here gate_up_proj is (E, 2I, H) and down_proj is (E, H, I), so NS batches over E=5 exactly as it
does on BiBo's experts.gate_up_proj / down_proj.

MODES (all target update RMS 0.2, so lr/wd carry over unchanged -- see muon_scaling.py):
  polar         BASE MUON: plain orthogonalized update, RMS-scaled, nothing else. The baseline the
                other three are supposed to improve on ("neuron death" on tall matrices).
  aurora        current default: row-normalize BEFORE the polar, re-orthogonalize (K passes).
  normuon       per-row EMA 2nd-moment normalize AFTER the polar.
  aurora_ema    aurora + normuon EMA, pre-polar (stays orthogonal).
  aurora_ema_v2 aurora THEN normuon post-hoc (NorMuon-faithful; breaks orthogonality).

    python .autoresearch/scale_mode_moe_mnist1d.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import manas_mnist1d as M
from scale_mode_mnist1d import make_source, N_SAMPLES, NS8
from kernels.sm75.muon import FusedMuon

MODES = ("polar", "aurora", "normuon", "aurora_ema", "aurora_ema_v2")
SEEDS = (0, 1, 2)
E, TOP_K = 5, 2


class MoEGLUBlock(nn.Module):
    """top-2 of E GLU experts, weights stacked 3D so Muon batches NS over the expert axis.
    Dense compute + masked combine: at E=5 the sorted-dispatch machinery buys nothing and would
    only add a confound; what matters here is that the PARAMETER is (E, out, in)."""

    def __init__(self, h, i, e=E):
        super().__init__()
        self.router = nn.Linear(h, e, bias=False)                     # 2D -> Muon, like BiBo's gate_proj
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h))    # 3D -> batched NS (the point)
        self.down_proj = nn.Parameter(torch.empty(e, h, i))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)

    def forward(self, x):
        s = torch.sigmoid(self.router(x))                             # (B,E) DeepSeek-style sigmoid gate
        tw, ti = s.topk(TOP_K, dim=-1)
        w = torch.zeros_like(s).scatter_(-1, ti, tw / tw.sum(-1, keepdim=True))
        gu = torch.einsum("bh,ekh->bek", x, self.gate_up_proj)
        g, u = gu.chunk(2, dim=-1)
        eo = torch.einsum("bei,ehi->beh", F.silu(g) * u, self.down_proj)
        return x + (eo * w.unsqueeze(-1)).sum(1)


class MoENet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv1d(1, 16, 5, stride=2)                     # AdamW (convs never go to Muon)
        self.proj = nn.Linear(16 * 18, 192, bias=False)
        self.blocks = nn.Sequential(*[MoEGLUBlock(192, 384) for _ in range(3)])
        self.head = nn.Linear(192, 10, bias=False)

    def forward(self, x):
        return self.head(self.blocks(self.proj(F.silu(self.stem(x)).flatten(1))))


def run(mode, seed, tr, idtest, ood_x, ood_y):
    torch.manual_seed(seed)
    model = MoENet1D().to(M.DEV)
    # EXCLUDE THE CONV STEM. Its weight is (16,1,5) -- 3D, so a bare `ndim in (2,3)` filter sweeps
    # it into Muon, where NS would batch over 16 slices of a 1x5 "matrix". Meaningless, and
    # manas_mnist1d is explicit that convs never go to Muon. Only the 6 MoE stacks (3 blocks x
    # gate_up/down) plus the 2D linears belong here.
    muon_p = [q for n, q in model.named_parameters() if q.ndim in (2, 3) and "stem" not in n]
    rest = [q for n, q in model.named_parameters() if q.ndim not in (2, 3) or "stem" in n]
    n3 = sum(1 for q in muon_p if q.ndim == 3)
    assert n3 == 6, f"expected 6 3D expert stacks (3 blocks x 2), got {n3}"
    opt = FusedMuon([{"params": muon_p}], lr=2e-3, momentum=0.95, weight_decay=0.01,
                    coeffs=NS8, ns_dtype=torch.bfloat16, scale_mode=mode)
    aux = torch.optim.AdamW(rest, lr=2e-3, weight_decay=0.01)

    order = ["s0", "s1", "s2"]
    log = {"train": [], "id": [], "ood": [], "ood_acc": []}
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
                log["train"].append(float(trl)); log["id"].append(float(idl))
                log["ood"].append(F.cross_entropy(ol, ood_y).item())
                log["ood_acc"].append((ol.argmax(-1) == ood_y).float().mean().item())
    print(f"  {mode:<14} s{seed} [3D stacks: {n3}]  train {log['train'][-1]:.4f}  "
          f"id {log['id'][-1]:.4f}  ood {log['ood'][-1]:.4f}  ood_acc {log['ood_acc'][-1]:.4f}", flush=True)
    return {"mode": mode, "seed": seed, "log": log}


def main():
    print(f"MoE MNIST-1D: E={E} experts, top-{TOP_K}, 3 blocks, {N_SAMPLES}/source", flush=True)
    SRC = {(s, sd): make_source(M.REGIMES[s], seed=100 * sd + 7) for s in M.REGIMES for sd in SEEDS}
    R = []
    for sd in SEEDS:
        tr = {s: SRC[(s, sd)][0] for s in ["s0", "s1", "s2"]}
        idtest = [SRC[(s, sd)][1] for s in ["s0", "s1", "s2"]]
        ox, oy = SRC[("ood", sd)][1]
        for mode in MODES:
            R.append(run(mode, sd, tr, idtest, ox, oy))

    print(f"\n{'mode':<14} {'train':>18} {'id':>18} {'ood':>18} {'ood_acc':>18}")
    got = {}
    for mode in MODES:
        rs = [r for r in R if r["mode"] == mode]
        cells = [np.array([r["log"][k][-1] for r in rs]) for k in ("train", "id", "ood", "ood_acc")]
        for k, c in zip(("train", "id", "ood", "ood_acc"), cells):
            got.setdefault(k, {})[mode] = c
        print(f"{mode:<14} " + " ".join(f"{c.mean():>11.4f}+-{c.std(ddof=1):.4f}" for c in cells))

    for ref in ("polar", "aurora"):
        print(f"\nvs {ref} (paired; negative = better for train/id/ood, positive = better for acc):")
        for k in ("train", "id", "ood", "ood_acc"):
            for mode in MODES:
                if mode == ref:
                    continue
                d = got[k][mode] - got[k][ref]
                t = d.mean() / (d.std(ddof=1) / len(d) ** 0.5) if d.std(ddof=1) > 0 else float("nan")
                print(f"  {k:<8} {mode:<14} {d.mean():+.4f}  (paired sd {d.std(ddof=1):.4f}, t={t:+.2f})")
    json.dump(R, open(os.path.join(os.path.dirname(__file__), "scale_mode_moe_mnist1d.json"), "w"))


if __name__ == "__main__":
    main()
