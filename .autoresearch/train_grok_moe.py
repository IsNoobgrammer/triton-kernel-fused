"""Grok-MoE screen: multi-op modular arithmetic with a BiBo-style MoE MLP in each block.

Question: does uniform-load balancing (DSv3 selection-bias, token-counter updates) fight
functional specialization under a SKEWED op mix, and can optimizer-level diversity pressure
(expert weight repulsion) buy it back?

MoE mirrors BiBo semantics: sigmoid scores; bias added to SELECTION only (never combine
weights); bias += factor * sign(mean_load - load) every --bias_tokens tokens (heuristic,
outside autograd). Experts are stacked (E, din, dout) parameters -> FusedMuon batches them
natively (ndim==3). Router/emb/head/norms on AdamW.

Metrics: held-out acc (+ per-op), grok_step, expert-op mutual information (bits, per layer,
top-1 expert on held-out), load entropy. Frozen split seed 1234; skewed TRAIN op mix 40/30/20/10
(held-out evaluated on ALL ops exhaustively, as before).

Usage: python -u train_grok_moe.py --arm default --seed 0 --frac 0.45 --wd 2.0 [--repulse 1e-3]
POTATO PRESET (RTX 3050-class): --p 61 --batch 1024 --steps 4000  (tables shrink ~2.5x with p,
dense-MoE compute ~5x cheaper overall; grokking still occurs at p=61 with wd ~2).
"""
import argparse
import json
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.sm75.muon import FusedMuon, _DSV4_COEFFS

OPS = ("add", "sub", "mul", "div")
OP_MIX = (0.4, 0.3, 0.2, 0.1)                                # skewed train sampling


def build_data(p, frac, device):
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
        x = torch.stack([a, torch.full_like(a, p + oi), b, torch.full_like(a, p + 4)], dim=1)
        rows.append((x, c))
    g = torch.Generator().manual_seed(1234)                  # FROZEN split
    tr, te = [], []
    for x, y in rows:
        idx = torch.randperm(len(y), generator=g)
        k = int(frac * len(y))
        tr.append((x[idx[:k]].to(device), y[idx[:k]].to(device)))
        te.append((x[idx[k:]].to(device), y[idx[k:]].to(device)))
    xte = torch.cat([x for x, _ in te]); yte = torch.cat([y for _, y in te])
    op_te = torch.cat([torch.full((len(y),), i, device=device) for i, (_, y) in enumerate(te)])
    return tr, xte, yte, op_te


class MoE(nn.Module):
    """Dense-compute MoE (tiny scale): all experts on all tokens, top-k combine.
    BiBo router semantics: sigmoid scores, selection-only bias, unbiased combine weights."""

    def __init__(self, d, E=8, top_k=2, mult=4):
        super().__init__()
        self.E, self.top_k = E, top_k
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, d, mult * d) * (d ** -0.5))
        self.w2 = nn.Parameter(torch.randn(E, mult * d, d) * ((mult * d) ** -0.5))
        self.register_buffer("bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))          # tokens since last bias update

    def forward(self, x):                                     # x: (N, d)
        scores = torch.sigmoid(self.router(x).float())        # (N, E)
        sel = scores + self.bias
        _, idx = torch.topk(sel, self.top_k, dim=-1, sorted=False)
        w = scores.gather(-1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20)             # (N, k)
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=self.E).float()
        h = torch.einsum("nd,edh->neh", x, self.w1)
        h = torch.einsum("neh,ehd->ned", F.gelu(h), self.w2)  # (N, E, d)
        hk = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]).long())
        self._last_top1 = idx[:, 0]                           # for MI metric
        return (hk * w.unsqueeze(-1).to(h.dtype)).sum(1)

    @torch.no_grad()
    def update_bias(self, factor):
        dev = self.load.mean() - self.load
        self.bias += factor * dev.sign()
        self.load.zero_()


class Block(nn.Module):
    def __init__(self, d, h, E, top_k):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, h, batch_first=True)
        self.moe = MoE(d, E, top_k)

    def forward(self, x, mask):
        hh = self.ln1(x)
        a, _ = self.attn(hh, hh, hh, attn_mask=mask, need_weights=False)
        x = x + a
        n, s, d = x.shape
        return x + self.moe(self.ln2(x).reshape(n * s, d)).reshape(n, s, d)


class GrokMoENet(nn.Module):
    def __init__(self, vocab, d=128, layers=3, heads=4, E=8, top_k=2, seq=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        self.blocks = nn.ModuleList(Block(d, heads, E, top_k) for _ in range(layers))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.register_buffer("mask", torch.triu(torch.ones(seq, seq, dtype=torch.bool), 1))

    def forward(self, x):
        h = self.tok(x) + self.pos.weight[None, : x.shape[1]]
        for b in self.blocks:
            h = b(h, self.mask)
        return self.head(self.lnf(h[:, -1]))


def mutual_info(top1, op, E, n_ops):
    """MI(expert, op) in bits from top-1 assignments on held-out."""
    joint = torch.zeros(E, n_ops, device=top1.device)
    joint.index_put_((top1, op), torch.ones_like(top1, dtype=torch.float), accumulate=True)
    joint = joint / joint.sum()
    pe, po = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    nz = joint > 0
    return (joint[nz] * (joint[nz] / (pe @ po)[nz]).log2()).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="default", choices=["default", "adamw"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--frac", type=float, default=0.45)
    ap.add_argument("--p", type=int, default=97)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--top_k", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--muon_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=2.0)          # grok-optimal from prior round
    ap.add_argument("--adamw_wd", type=float, default=1.0)
    ap.add_argument("--bias_tokens", type=int, default=300_000)
    ap.add_argument("--bias_factor", type=float, default=0.01)
    ap.add_argument("--repulse", type=float, default=0.0)     # swarm: W_e += beta*(W_e - mean_E W)
    ap.add_argument("--eval_every", type=int, default=200)
    args = ap.parse_args()
    dev = "cuda"
    torch.manual_seed(args.seed)

    tr, xte, yte, op_te = build_data(args.p, args.frac, dev)
    model = GrokMoENet(args.p + 5, args.d, args.layers, 4, args.experts, args.top_k).to(dev)
    n_par = sum(q.numel() for q in model.parameters())

    def is_hidden(n, q):
        return ("blocks" in n) and ("router" not in n) and q.ndim in (2, 3) and "ln" not in n
    hidden = [q for n, q in model.named_parameters() if is_hidden(n, q)]
    rest = [q for n, q in model.named_parameters() if not is_hidden(n, q)]
    adamw = torch.optim.AdamW(rest, lr=args.lr, weight_decay=args.adamw_wd, betas=(0.9, 0.98))
    opts = [adamw]
    if args.arm == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.adamw_wd, betas=(0.9, 0.98))]
    else:
        opts.append(FusedMuon(hidden, lr=args.muon_lr, weight_decay=args.wd,
                              coeffs=_DSV4_COEFFS, ns_dtype=torch.float16))
    expert_ws = [b.moe.w1 for b in model.blocks] + [b.moe.w2 for b in model.blocks]
    print(f"[grokmoe] arm={args.arm} seed={args.seed} frac={args.frac} E={args.experts} "
          f"topk={args.top_k} repulse={args.repulse} wd={args.wd} params={n_par/1e6:.2f}M "
          f"train={sum(len(y) for _, y in tr)} heldout={len(yte)} opmix={OP_MIX}", flush=True)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    mix = torch.tensor(OP_MIX, device=dev)
    tok_count, grok_step, best, curve = 0, None, 0.0, []
    for step in range(1, args.steps + 1):
        ops = torch.multinomial(mix, args.batch, replacement=True, generator=g)
        xb, yb = [], []
        for oi in range(len(OPS)):                            # skewed per-op sampling
            n = int((ops == oi).sum())
            if n == 0:
                continue
            xo, yo = tr[oi]
            ii = torch.randint(0, len(yo), (n,), device=dev, generator=g)
            xb.append(xo[ii]); yb.append(yo[ii])
        xb, yb = torch.cat(xb), torch.cat(yb)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        if args.repulse > 0:                                  # swarm repulsion on expert stacks
            with torch.no_grad():
                for w in expert_ws:
                    w.add_(w - w.mean(0, keepdim=True), alpha=args.repulse)
        tok_count += args.batch
        if tok_count >= args.bias_tokens:
            for b in model.blocks:
                b.moe.update_bias(args.bias_factor)
            tok_count = 0
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            with torch.no_grad():
                preds, top1s = [], [[] for _ in model.blocks]
                for i in range(0, len(yte), 8192):
                    preds.append(model(xte[i:i + 8192]).argmax(-1))
                    for li, b in enumerate(model.blocks):
                        top1s[li].append(b.moe._last_top1.reshape(-1, 4)[:, -1])  # '=' position
                pred = torch.cat(preds)
                hit = (pred == yte).float()
                acc = hit.mean().item()
                per_op = [hit[op_te == i].mean().item() for i in range(len(OPS))]
                mis = [mutual_info(torch.cat(t), op_te, args.experts, len(OPS)) for t in top1s]
                loads = [b.moe.load / b.moe.load.sum().clamp_min(1) for b in model.blocks]
                lmin = min(l.min().item() for l in loads)
            model.train()
            best = max(best, acc)
            curve.append([step, round(acc, 5), [round(m, 3) for m in mis]])
            if grok_step is None and acc >= 0.90:
                grok_step = step
            print(f"[grokmoe] step {step:5d} loss {loss.item():.4f} acc {acc:.4f} "
                  + " ".join(f"{o}={a:.3f}" for o, a in zip(OPS, per_op))
                  + f" | MI {' '.join(f'{m:.2f}' for m in mis)} | minload {lmin:.3f}", flush=True)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "repulse": args.repulse,
                      "wd": args.wd, "acc": round(acc, 5), "best_acc": round(best, 5),
                      "grok_step": grok_step, "per_op": [round(a, 4) for a in per_op],
                      "mi_final": [round(m, 3) for m in mis], "curve": curve}), flush=True)


if __name__ == "__main__":
    main()
