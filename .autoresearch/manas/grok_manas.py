"""Task #1 candidate: does the Manas probe change GROKKING?

Grokking = the model memorizes (train acc ~100%) long before it generalizes (held-out acc
jumps late). It is literally the sharp-memorize -> broad-generalize transition, and Manas is
built to steer toward broad basins. Hypothesis: the probe shifts WHEN grokking happens (or
whether). A shift in grok-step is a big, VISIBLE difference, unlike a 45%-OOD delta.

Clean isolation: baseline = ManasOptimizer with gamma=0 (== Muon, ns8 coeffs); treatment =
same optimizer with the probe ON. Only the look-ahead differs. AdamW on embeddings/head/norm
(grokking needs strong wd there). rho set from the batch via the memory law (window*batch ~
const): batch 512 -> rho ~0.9. Metric: grok-step (held-out acc >= 0.9), curves, plot.

Run: ../../BiBo/.venv/Scripts/python.exe grok_manas.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import numpy as np
import torch
import torch.nn.functional as F

from train_grok import GrokNet
from kernels.sm75.manas import ManasOptimizer

DEV = "cuda"
P, FRAC, STEPS, BATCH, EVAL_EVERY = 97, 0.5, 6000, 512, 100   # HETEROGENEOUS: add + mul (2 sources)
SEEDS = (0, 1, 2, 3, 4, 5, 6, 7)
GAMMA, RHO, RANK = 0.08, 0.9, 8          # rho ~ 1 - batch/N_mem (batch 512, N_mem~6k -> ~0.9)
OPS = ("add", "mul")                     # two structurally different ops = the heterogeneity


def build_ops(p, frac, device):
    """Multi-op exhaustive tables (a op b mod p), each op a distinct token. vocab = p+len+1.
    Heterogeneous: the batches mix genuinely different operations -> the probe's consensus
    has real cross-source disagreement to average, unlike single-op."""
    a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
    a, b = a.flatten(), b.flatten()
    tbl = {"add": (a + b) % p, "sub": (a - b) % p, "mul": (a * b) % p}
    eq = p + len(OPS)
    g = torch.Generator().manual_seed(1234)
    xtr, ytr, xte, yte = [], [], [], []
    for oi, op in enumerate(OPS):
        c = tbl[op]
        x = torch.stack([a, torch.full_like(a, p + oi), b, torch.full_like(a, eq)], dim=1)
        idx = torch.randperm(len(c), generator=g)
        k = int(frac * len(c))
        xtr.append(x[idx[:k]]); ytr.append(c[idx[:k]]); xte.append(x[idx[k:]]); yte.append(c[idx[k:]])
    return (torch.cat(xtr).to(device), torch.cat(ytr).to(device),
            torch.cat(xte).to(device), torch.cat(yte).to(device))


def run(use_probe, seed):
    torch.manual_seed(seed)
    xtr, ytr, xte, yte = build_ops(P, FRAC, DEV)
    model = GrokNet(P + len(OPS) + 1, d=256, layers=3, heads=4).to(DEV)
    hidden = [q for n, q in model.named_parameters() if q.ndim == 2 and "blocks" in n]
    rest = [q for n, q in model.named_parameters() if not (q.ndim == 2 and "blocks" in n)]
    opt = ManasOptimizer(hidden, lr=1e-3, weight_decay=0.1,
                         probe_gamma=(GAMMA if use_probe else 0.0), probe_rho=RHO, probe_rank=RANK)
    adamw = torch.optim.AdamW(rest, lr=1e-3, weight_decay=1.0, betas=(0.9, 0.98))
    probe = opt.probe if use_probe else None
    g = torch.Generator(device=DEV).manual_seed(seed)
    grok_step, best, curve = None, 0.0, []
    for step in range(1, STEPS + 1):
        idx = torch.randint(0, len(ytr), (BATCH,), device=DEV, generator=g)
        def fb():
            loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
            loss.backward()
            return loss
        if probe is not None:
            with probe():
                loss = fb()
        else:
            loss = fb()
        opt.step(); adamw.step(); opt.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        if step % EVAL_EVERY == 0 or step == STEPS:
            model.eval()
            with torch.no_grad():
                pred = torch.cat([model(xte[i:i + 8192]).argmax(-1) for i in range(0, len(yte), 8192)])
                acc = (pred == yte).float().mean().item()
            model.train()
            best = max(best, acc); curve.append([step, round(acc, 4)])
            if grok_step is None and acc >= 0.90:
                grok_step = step
    return {"grok_step": grok_step, "best": best, "curve": curve}


if __name__ == "__main__":
    print(f"GROKKING: Muon (probe off) vs Manas (probe on), p={P} frac={FRAC} batch={BATCH} "
          f"steps={STEPS}, gamma={GAMMA} rho={RHO} rank={RANK}\n")
    res = {"muon": [], "manas": []}
    for seed in SEEDS:
        for kind, use in (("muon", False), ("manas", True)):
            r = run(use, seed)
            res[kind].append(r)
            gs = r["grok_step"] if r["grok_step"] is not None else "never"
            print(f"  {kind:<6} seed {seed}: grok_step {gs}  best {r['best']:.4f}", flush=True)
    print("\n== GROK-STEP (held-out acc >= 0.90) ==")
    for kind in ("muon", "manas"):
        gs = [r["grok_step"] for r in res[kind] if r["grok_step"] is not None]
        nev = sum(1 for r in res[kind] if r["grok_step"] is None)
        m = f"{int(np.mean(gs))} +/- {int(np.std(gs))}" if gs else "never"
        print(f"  {kind:<6} grok_step {m}   (never-grokked: {nev}/{len(SEEDS)})  "
              f"best {np.mean([r['best'] for r in res[kind]]):.4f}")
    # paired stats (same seed = same init/data/order; both grokked)
    gsm = [res["muon"][i]["grok_step"] for i in range(len(SEEDS))]
    gsa = [res["manas"][i]["grok_step"] for i in range(len(SEEDS))]
    bm = [res["muon"][i]["best"] for i in range(len(SEEDS))]
    ba = [res["manas"][i]["best"] for i in range(len(SEEDS))]
    if all(g is not None for g in gsm + gsa):
        dstep = np.array(gsa) - np.array(gsm); dbest = np.array(ba) - np.array(bm)
        print(f"\n  PAIRED (Manas - Muon): grok_step {dstep.mean():+.0f} +/- "
              f"{dstep.std(ddof=1)/np.sqrt(len(dstep)):.0f}  ({(dstep<0).sum()}/{len(SEEDS)} earlier)")
        print(f"  PAIRED best_acc {dbest.mean()*100:+.2f}% +/- "
              f"{dbest.std(ddof=1)/np.sqrt(len(dbest))*100:.2f}  ({(dbest>0).sum()}/{len(SEEDS)} higher)")
    json.dump(res, open(os.path.join(os.path.dirname(__file__), "grok_manas.json"), "w"))
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 5))
        for kind, col in (("muon", "#1f9e86"), ("manas", "#c8892a")):
            for i, r in enumerate(res[kind]):
                c = np.array(r["curve"])
                plt.plot(c[:, 0], c[:, 1], color=col, alpha=0.6,
                         label=kind if i == 0 else None)
        plt.axhline(0.9, color="gray", lw=.8, ls="--")
        plt.xlabel("step"); plt.ylabel("held-out accuracy"); plt.title("Grokking: Muon vs Manas")
        plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(__file__), "grok_manas.png"), dpi=120)
        print("saved grok_manas.png + grok_manas.json")
    except Exception as e:
        print("plot skipped:", e)
