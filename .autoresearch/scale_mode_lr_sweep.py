"""Per-mode LR sweep then multi-seed confirmation, on 3D GLU expert stacks.

WHY: scale_mode_moe_mnist1d.py compared all five modes at ONE lr (2e-3). If normuon / aurora_ema
want a different LR -- which is the standard claim for EMA-style updates -- that comparison is
unfair to whichever mode is mistuned, and "4-way tie" could just mean "all mistuned equally".
It also ran at ~45% ood_acc, low enough that the modes may not be in their operating range.

So: (1) sweep LR per mode, (2) re-run every mode AT ITS OWN BEST LR across many seeds.
Bigger net than the E=5 probe so accuracy lands in a regime worth measuring.

  python .autoresearch/scale_mode_lr_sweep.py --phase lr     # stage 1: LR x mode, few seeds
  python .autoresearch/scale_mode_lr_sweep.py --phase seeds --lrs-json best_lr.json
"""
import argparse
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
from scale_mode_mnist1d import make_source, NS8
from kernels.sm75.muon import FusedMuon

MODES = ("polar", "aurora", "normuon", "aurora_ema", "aurora_ema_v2")
HERE = os.path.dirname(os.path.abspath(__file__))


class MoEGLUBlock(nn.Module):
    def __init__(self, h, i, e):
        super().__init__()
        self.router = nn.Linear(h, e, bias=False)
        self.gate_up_proj = nn.Parameter(torch.empty(e, 2 * i, h))
        self.down_proj = nn.Parameter(torch.empty(e, h, i))
        nn.init.normal_(self.gate_up_proj, std=0.02)
        nn.init.normal_(self.down_proj, std=0.02)
        self.k = 2

    def forward(self, x):
        s = torch.sigmoid(self.router(x))
        tw, ti = s.topk(self.k, dim=-1)
        w = torch.zeros_like(s).scatter_(-1, ti, tw / tw.sum(-1, keepdim=True))
        g, u = torch.einsum("bh,ekh->bek", x, self.gate_up_proj).chunk(2, dim=-1)
        eo = torch.einsum("bei,ehi->beh", F.silu(g) * u, self.down_proj)
        return x + (eo * w.unsqueeze(-1)).sum(1)


class MoENet(nn.Module):
    def __init__(self, h, i, e, blocks):
        super().__init__()
        self.stem = nn.Conv1d(1, 32, 5, stride=2)
        self.proj = nn.Linear(32 * 18, h, bias=False)
        self.blocks = nn.Sequential(*[MoEGLUBlock(h, i, e) for _ in range(blocks)])
        self.head = nn.Linear(h, 10, bias=False)

    def forward(self, x):
        return self.head(self.blocks(self.proj(F.silu(self.stem(x)).flatten(1))))


def run(mode, seed, lr, S, steps, h, i, e, blocks):
    torch.manual_seed(seed)
    model = MoENet(h, i, e, blocks).to(M.DEV)
    # conv stem NEVER to Muon: its (32,1,5) weight is 3D and NS would batch over 32 slices of a 1x5
    # "matrix". Only the MoE stacks (3D) and the plain linears (2D) belong in the Muon group.
    mp = [q for n, q in model.named_parameters() if q.ndim in (2, 3) and "stem" not in n]
    rest = [q for n, q in model.named_parameters() if q.ndim not in (2, 3) or "stem" in n]
    assert sum(1 for q in mp if q.ndim == 3) == 2 * blocks
    opt = FusedMuon([{"params": mp}], lr=lr, momentum=0.95, weight_decay=0.01,
                    coeffs=NS8, ns_dtype=torch.bfloat16, scale_mode=mode)
    aux = torch.optim.AdamW(rest, lr=lr, weight_decay=0.01)

    tr = {s: S[(s, seed)][0] for s in ("s0", "s1", "s2")}
    idt = [S[(s, seed)][1] for s in ("s0", "s1", "s2")]
    ox, oy = S[("ood", seed)][1]
    order = ["s0", "s1", "s2"]
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(steps):
        x, y = tr[order[t % 3]]
        idx = torch.randint(0, x.shape[0], (M.BS,), generator=g).to(M.DEV)
        loss = F.cross_entropy(model(x[idx]), y[idx])
        opt.zero_grad(set_to_none=True); aux.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); aux.step()
    with torch.no_grad():
        trl = float(np.mean([F.cross_entropy(model(tr[s][0][:2000]), tr[s][1][:2000]).item() for s in order]))
        idl = float(np.mean([F.cross_entropy(model(a), b).item() for a, b in idt]))
        ida = float(np.mean([(model(a).argmax(-1) == b).float().mean().item() for a, b in idt]))
        ol = model(ox)
        oodl = float(F.cross_entropy(ol, oy).item())
        ooda = float((ol.argmax(-1) == oy).float().mean().item())
    if not np.isfinite(trl):
        trl = idl = oodl = float("inf"); ida = ooda = 0.0     # diverged: rank it last, don't crash
    return dict(mode=mode, seed=seed, lr=lr, train=trl, id=idl, id_acc=ida, ood=oodl, ood_acc=ooda)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["lr", "seeds"], default="lr")
    ap.add_argument("--lrs", default="5e-4,1e-3,2e-3,4e-3,8e-3")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--h", type=int, default=384)
    ap.add_argument("--i", type=int, default=768)
    ap.add_argument("--e", type=int, default=5)
    ap.add_argument("--blocks", type=int, default=3)
    ap.add_argument("--best-lr", default="")
    a = ap.parse_args()
    seeds = tuple(range(a.seeds))
    lrs = [float(x) for x in a.lrs.split(",")]

    print(f"MoE h={a.h} i={a.i} E={a.e} blocks={a.blocks} steps={a.steps} seeds={seeds}", flush=True)
    S = {(s, sd): make_source(M.REGIMES[s], seed=100 * sd + 7) for s in M.REGIMES for sd in seeds}

    R = []
    if a.phase == "lr":
        for mode in MODES:
            for lr in lrs:
                rs = [run(mode, sd, lr, S, a.steps, a.h, a.i, a.e, a.blocks) for sd in seeds]
                R += rs
                m = lambda k: np.mean([r[k] for r in rs])
                print(f"  {mode:<14} lr {lr:<8.1e} id_acc {m('id_acc'):.4f}  ood_acc {m('ood_acc'):.4f}  "
                      f"id {m('id'):.4f}  train {m('train'):.4f}", flush=True)
        print("\nBEST LR PER MODE (by id_acc):")
        best = {}
        for mode in MODES:
            cand = {lr: np.mean([r["id_acc"] for r in R if r["mode"] == mode and r["lr"] == lr]) for lr in lrs}
            best[mode] = max(cand, key=cand.get)
            print(f"  {mode:<14} {best[mode]:.1e}  (id_acc {cand[best[mode]]:.4f})  all: "
                  + " ".join(f"{l:.0e}:{v:.3f}" for l, v in cand.items()))
        json.dump(best, open(os.path.join(HERE, "scale_mode_best_lr.json"), "w"))
        print("\n-> wrote scale_mode_best_lr.json; now run --phase seeds")
    else:
        best = json.load(open(os.path.join(HERE, "scale_mode_best_lr.json")))
        print(f"per-mode LR: {best}\n", flush=True)
        for mode in MODES:
            for sd in seeds:
                R.append(run(mode, sd, float(best[mode]), S, a.steps, a.h, a.i, a.e, a.blocks))
            rs = [r for r in R if r["mode"] == mode]
            m = lambda k: np.array([r[k] for r in rs])
            print(f"  {mode:<14} lr {best[mode]:.1e}  id_acc {m('id_acc').mean():.4f}+-{m('id_acc').std(ddof=1):.4f}  "
                  f"ood_acc {m('ood_acc').mean():.4f}+-{m('ood_acc').std(ddof=1):.4f}", flush=True)
        print(f"\n{'mode':<14} {'id_acc':>18} {'ood_acc':>18} {'id':>18} {'train':>18}")
        got = {}
        for mode in MODES:
            rs = [r for r in R if r["mode"] == mode]
            cells = [np.array([r[k] for r in rs]) for k in ("id_acc", "ood_acc", "id", "train")]
            for k, c in zip(("id_acc", "ood_acc", "id", "train"), cells):
                got.setdefault(k, {})[mode] = c
            print(f"{mode:<14} " + " ".join(f"{c.mean():>11.4f}+-{c.std(ddof=1):.4f}" for c in cells))
        for ref in ("polar", "aurora"):
            print(f"\nvs {ref} (paired over seeds; +acc = better, -loss = better):")
            for k in ("id_acc", "ood_acc", "id"):
                for mode in MODES:
                    if mode == ref:
                        continue
                    d = got[k][mode] - got[k][ref]
                    sd_ = d.std(ddof=1)
                    t = d.mean() / (sd_ / len(d) ** 0.5) if sd_ > 0 else float("nan")
                    print(f"  {k:<8} {mode:<14} {d.mean():+.4f}  (sd {sd_:.4f}, t={t:+.2f}, n={len(d)})")
    json.dump(R, open(os.path.join(HERE, f"scale_mode_lr_{a.phase}.json"), "w"))


if __name__ == "__main__":
    main()
