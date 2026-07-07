"""Grokking screen: multi-op modular arithmetic (a op b mod p -> c), tiny transformer.

Frozen-eval rules: data split seed is FIXED (1234) so all arms/seeds see the same train/held-out
tables; only the model/init/batch-order seed varies (--seed). Held-out = every (a,b) pair not in
train, evaluated exactly. Prints one eval line per --eval_every steps (flush) and a final JSON.

Usage: python -u train_grok.py --arm default --seed 0 --frac 0.4 --steps 6000
Needs PYTHONPATH=<triton-kernel-fused root> for kernels.sm75.muon.
"""
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.sm75.muon import FusedMuon, _DSV4_COEFFS

KJ = (3.4445, -4.775, 2.0315)
PIN = (2.0, -1.5, 0.5)
# joint 6-step solve at l0=2e-3 cap 1.6 (solve_joint.py), last step rescaled by 1.12/0.8352
# so the composite band is ~[0.96, 1.12] instead of [0.71, 0.84].
_S = 1.12 / 0.8352
JNS6 = ((3.38826144, -4.83320729, 1.93262457), (3.38647393, -4.8332009, 1.93335609),
        (3.38632321, -4.83319057, 1.93336145), (3.33353763, -4.88596261, 1.92042664),
        (1.88903763, -1.61096239, 0.38903749),
        (1.88903765 * _S, -1.61096238 * _S, 0.38903761 * _S))
ARMS = {
    "default": dict(coeffs=_DSV4_COEFFS),
    "b12":     dict(coeffs=(KJ,) * 10 + (PIN,) * 2),
    "k2":      dict(coeffs=_DSV4_COEFFS, aurora_k=2),
    "ns6":     dict(coeffs=(KJ,) * 4 + (PIN,) * 2),
    "ns8":     dict(coeffs=(KJ,) * 6 + (PIN,) * 2),
    "normuon": dict(coeffs=_DSV4_COEFFS, scale_mode="normuon"),
    "jns6":    dict(coeffs=JNS6),
    "adamw":   None,                                        # control: no Muon at all
}
OPS = ("add", "sub", "mul", "div")


def build_data(p, frac, device):
    """Per-op exhaustive tables, split with the FROZEN seed. Returns train/test (x, y)."""
    rows = []
    for oi, op in enumerate(OPS):
        if op == "div":
            b, c = torch.meshgrid(torch.arange(1, p), torch.arange(p), indexing="ij")
            b, c = b.flatten(), c.flatten()
            a = (b * c) % p
        else:
            a, b = torch.meshgrid(torch.arange(p), torch.arange(p), indexing="ij")
            a, b = a.flatten(), b.flatten()
            c = {"add": (a + b) % p, "sub": (a - b) % p, "mul": (a * b) % p}[op]
        x = torch.stack([a, torch.full_like(a, p + oi), b,
                         torch.full_like(a, p + 4)], dim=1)                 # [a, op, b, =]
        rows.append((x, c))
    g = torch.Generator().manual_seed(1234)                  # FROZEN split seed
    xtr, ytr, xte, yte = [], [], [], []
    for x, y in rows:
        idx = torch.randperm(len(y), generator=g)
        k = int(frac * len(y))
        xtr.append(x[idx[:k]]); ytr.append(y[idx[:k]])
        xte.append(x[idx[k:]]); yte.append(y[idx[k:]])
    op_id_te = torch.cat([torch.full((len(t),), i) for i, t in enumerate(yte)])
    return (torch.cat(xtr).to(device), torch.cat(ytr).to(device),
            torch.cat(xte).to(device), torch.cat(yte).to(device), op_id_te.to(device))


class Block(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.up = nn.Linear(d, 4 * d, bias=False)
        self.down = nn.Linear(4 * d, d, bias=False)

    def forward(self, x, mask):
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.down(F.gelu(self.up(self.ln2(x))))


class GrokNet(nn.Module):
    def __init__(self, vocab, d=256, layers=3, heads=4, seq=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        mask = torch.triu(torch.ones(seq, seq, dtype=torch.bool), diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, x):
        h = self.tok(x) + self.pos.weight[None, : x.shape[1]]
        for b in self.blocks:
            h = b(h, self.mask)
        return self.head(self.lnf(h[:, -1]))                 # predict c at '='


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="default", choices=sorted(ARMS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.4)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)        # AdamW side (emb/head/norm)
    ap.add_argument("--muon_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--adamw_wd", type=float, default=1.0)   # grokking needs strong wd on emb/head
    ap.add_argument("--eval_every", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)

    xtr, ytr, xte, yte, op_te = build_data(args.p, args.frac, dev)
    model = GrokNet(args.p + 5, args.d, args.layers, args.heads).to(dev)
    n_par = sum(q.numel() for q in model.parameters())

    hidden = [q for n, q in model.named_parameters() if q.ndim == 2 and "blocks" in n]
    rest = [q for n, q in model.named_parameters() if not (q.ndim == 2 and "blocks" in n)]
    adamw = torch.optim.AdamW(rest, lr=args.lr, weight_decay=args.adamw_wd, betas=(0.9, 0.98))
    opts = [adamw]
    if ARMS[args.arm] is None:
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.adamw_wd,
                                  betas=(0.9, 0.98))]
    else:
        spec = ARMS[args.arm]
        kw = dict(coeffs=spec["coeffs"], ns_dtype=torch.float16, aurora_k=spec.get("aurora_k"))
        if "scale_mode" in spec:
            kw["scale_mode"] = spec["scale_mode"]
        opts.append(FusedMuon(hidden, lr=args.muon_lr, weight_decay=args.wd, **kw))
    print(f"[grok] arm={args.arm} seed={args.seed} frac={args.frac} p={args.p} "
          f"params={n_par/1e6:.2f}M train={len(ytr)} heldout={len(yte)}", flush=True)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    grok_step, best, curve = None, 0.0, []
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, len(ytr), (args.batch,), device=dev, generator=g)
        loss = F.cross_entropy(model(xtr[idx]), ytr[idx])
        loss.backward()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                accs = []
                pred = torch.cat([model(xte[i:i + 8192]).argmax(-1)
                                  for i in range(0, len(yte), 8192)])
                hit = (pred == yte).float()
                acc = hit.mean().item()
                per_op = [hit[op_te == i].mean().item() for i in range(len(OPS))]
            model.train()
            best = max(best, acc)
            curve.append([step, round(acc, 5)])
            if grok_step is None and acc >= 0.90:
                grok_step = step
            ep = step * args.batch / len(ytr)
            print(f"[grok] step {step:5d} ep {ep:7.1f} loss {loss.item():.4f} acc {acc:.4f} "
                  + " ".join(f"{o}={a:.3f}" for o, a in zip(OPS, per_op)), flush=True)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "frac": args.frac,
                      "acc": round(acc, 5), "best_acc": round(best, 5),
                      "grok_step": grok_step, "final_loss": round(loss.item(), 5),
                      "curve": curve}), flush=True)


if __name__ == "__main__":
    main()
