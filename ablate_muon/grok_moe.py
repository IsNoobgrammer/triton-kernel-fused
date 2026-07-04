"""Grok-MoE ablation harness (self-contained, importable).

Task: multi-op modular arithmetic (a op b mod p) with a BiBo-style MoE MLP in each block.
Question: under a SKEWED op mix, does uniform-load balancing fight functional specialization,
and can optimizer-level diversity pressure (expert weight repulsion) buy it back?

MoE mirrors BiBo semantics: sigmoid scores; bias added to SELECTION only (never combine
weights); bias += factor*sign(mean_load - load) every `bias_tokens` tokens (heuristic, outside
autograd, SLOW global correction). Experts stacked (E, din, dout) -> FusedMuon batches ndim==3.
Router/emb/head/norms on AdamW. Hidden (attn + expert) matrices on Muon.

Metrics: held-out acc (+per-op), grok_step, MI(top-1 expert, op) per layer (bits), min load.
Frozen split seed 1234; skewed TRAIN op mix (default 40/30/20/10); held-out over ALL ops.

`run(cfg: dict) -> dict` is the entry point (see run_ablation.py). No argparse.
"""
import json
import os
import subprocess
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── bootstrap: use the repo's real sm75 FusedMuon (T4 IS the sm75 target) ──────────
try:
    from kernels.sm75.muon import FusedMuon, _DSV4_COEFFS
except ImportError:
    _here = os.path.dirname(os.path.abspath(__file__))
    _repo = os.path.dirname(_here)                                 # ablate_muon/ lives in the repo
    if os.path.exists(os.path.join(_repo, "kernels", "sm75", "muon.py")):
        sys.path.insert(0, _repo)
    else:                                                          # folder copied out alone -> clone
        _dst = os.path.join(os.getcwd(), "triton-kernel-fused")
        if not os.path.exists(_dst):
            subprocess.run(["git", "clone", "--depth", "1",
                            "https://github.com/IsNoobgrammer/triton-kernel-fused", _dst], check=True)
        sys.path.insert(0, _dst)
    from kernels.sm75.muon import FusedMuon, _DSV4_COEFFS

OPS = ("add", "sub", "mul", "div")


def build_data(p, frac, op_mix, device):
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
    g = torch.Generator().manual_seed(1234)                       # FROZEN split
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
    """Dense-compute MoE (tiny scale): all experts on all tokens, top-k combine."""

    def __init__(self, d, E=8, top_k=2, mult=4):
        super().__init__()
        self.E, self.top_k = E, top_k
        self.router = nn.Linear(d, E, bias=False)
        self.w1 = nn.Parameter(torch.randn(E, d, mult * d) * (d ** -0.5))
        self.w2 = nn.Parameter(torch.randn(E, mult * d, d) * ((mult * d) ** -0.5))
        self.register_buffer("bias", torch.zeros(E))
        self.register_buffer("load", torch.zeros(E))

    def forward(self, x):                                          # x: (N, d)
        scores = torch.sigmoid(self.router(x).float())
        sel = scores + self.bias
        _, idx = torch.topk(sel, self.top_k, dim=-1, sorted=False)
        w = scores.gather(-1, idx)
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
        if self.training:
            self.load += torch.bincount(idx.flatten(), minlength=self.E).float()
        h = torch.einsum("nd,edh->neh", x, self.w1)
        h = torch.einsum("neh,ehd->ned", F.gelu(h), self.w2)
        hk = h.gather(1, idx.unsqueeze(-1).expand(-1, -1, h.shape[-1]).long())
        self._last_top1 = idx[:, 0]
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


def _mi(top1, op, E, n_ops):
    joint = torch.zeros(E, n_ops, device=top1.device)
    joint.index_put_((top1, op), torch.ones_like(top1, dtype=torch.float), accumulate=True)
    joint = joint / joint.sum()
    pe, po = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    nz = joint > 0
    return (joint[nz] * (joint[nz] / (pe @ po)[nz]).log2()).sum().item()


_DEF = dict(arm="default", seed=0, frac=0.45, p=97, steps=3000, batch=768, d=128, layers=3,
            experts=8, top_k=2, lr=1e-3, muon_lr=1e-3, wd=2.0, adamw_wd=1.0, expert_wd=None,
            bias_tokens=300_000, bias_factor=0.01, repulse=0.0, decor=0.0, eval_every=200,
            op_mix=(0.4, 0.3, 0.2, 0.1))


def make_tag(c):
    t = f"{c['arm']}_s{c['seed']}"
    if c.get("repulse"):
        t += f"_rep{c['repulse']}"
    if c["arm"] == "adamw":
        t += f"_awd{c['adamw_wd']}"
    if c.get("expert_wd") is not None:
        t += f"_ewd{c['expert_wd']}"
    if c.get("decor"):
        t += f"_dec{c['decor']}"
    if c["steps"] != 3000:
        t += f"_{c['steps']}st"
    return t


def run(cfg):
    """Train one arm. cfg overrides _DEF. Returns a result dict; prints live (flush)."""
    c = {**_DEF, **cfg}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(c["seed"])
    tr, xte, yte, op_te = build_data(c["p"], c["frac"], c["op_mix"], dev)
    model = GrokMoENet(c["p"] + 5, c["d"], c["layers"], 4, c["experts"], c["top_k"]).to(dev)
    n_par = sum(q.numel() for q in model.parameters())

    def is_hidden(n, q):
        return ("blocks" in n) and ("router" not in n) and q.ndim in (2, 3) and "ln" not in n
    hidden = [q for n, q in model.named_parameters() if is_hidden(n, q)]
    rest = [q for n, q in model.named_parameters() if not is_hidden(n, q)]
    if c["arm"] == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=c["lr"],
                                  weight_decay=c["adamw_wd"], betas=(0.9, 0.98))]
    else:
        ewd = c["wd"] if c["expert_wd"] is None else c["expert_wd"]
        opts = [torch.optim.AdamW(rest, lr=c["lr"], weight_decay=c["adamw_wd"], betas=(0.9, 0.98)),
                FusedMuon([q for q in hidden if q.ndim == 2], lr=c["muon_lr"],
                          weight_decay=c["wd"], coeffs=_DSV4_COEFFS, ns_dtype=torch.float16),
                FusedMuon([q for q in hidden if q.ndim == 3], lr=c["muon_lr"],
                          weight_decay=ewd, coeffs=_DSV4_COEFFS, ns_dtype=torch.float16)]
    expert_ws = [b.moe.w1 for b in model.blocks] + [b.moe.w2 for b in model.blocks]
    tag = make_tag(c)
    print(f"[{tag}] E={c['experts']} topk={c['top_k']} repulse={c['repulse']} wd={c['wd']} "
          f"params={n_par/1e6:.2f}M train={sum(len(y) for _, y in tr)} heldout={len(yte)} "
          f"opmix={c['op_mix']}", flush=True)

    g = torch.Generator(device=dev).manual_seed(c["seed"])
    mix = torch.tensor(c["op_mix"], device=dev)
    tok_count, grok_step, best, curve = 0, None, 0.0, []
    for step in range(1, c["steps"] + 1):
        ops = torch.multinomial(mix, c["batch"], replacement=True, generator=g)
        xb, yb = [], []
        for oi in range(len(OPS)):
            n = int((ops == oi).sum())
            if n == 0:
                continue
            xo, yo = tr[oi]
            ii = torch.randint(0, len(yo), (n,), device=dev, generator=g)
            xb.append(xo[ii]); yb.append(yo[ii])
        xb, yb = torch.cat(xb), torch.cat(yb)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        if c["decor"] > 0:                                         # cross-expert update decorrelation:
            with torch.no_grad():                                  # shrink the shared component of the
                for w in expert_ws:                                # expert-stack grads along the E axis
                    w.grad -= c["decor"] * w.grad.mean(0, keepdim=True)
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        if c["repulse"] > 0:
            with torch.no_grad():
                for w in expert_ws:
                    w.add_(w - w.mean(0, keepdim=True), alpha=c["repulse"])
        tok_count += c["batch"]
        if tok_count >= c["bias_tokens"]:
            for b in model.blocks:
                b.moe.update_bias(c["bias_factor"])
            tok_count = 0
        if step % c["eval_every"] == 0 or step == c["steps"]:
            model.eval()
            with torch.no_grad():
                preds, top1s = [], [[] for _ in model.blocks]
                for i in range(0, len(yte), 8192):
                    preds.append(model(xte[i:i + 8192]).argmax(-1))
                    for li, b in enumerate(model.blocks):
                        top1s[li].append(b.moe._last_top1.reshape(-1, 4)[:, -1])
                pred = torch.cat(preds)
                hit = (pred == yte).float()
                acc = hit.mean().item()
                per_op = [hit[op_te == i].mean().item() for i in range(len(OPS))]
                mis = [_mi(torch.cat(t), op_te, c["experts"], len(OPS)) for t in top1s]
                loads = [b.moe.load / b.moe.load.sum().clamp_min(1) for b in model.blocks]
                lmin = min(l.min().item() for l in loads)
            model.train()
            best = max(best, acc)
            curve.append([step, round(acc, 5), [round(m, 3) for m in mis]])
            if grok_step is None and acc >= 0.90:
                grok_step = step
            print(f"[{tag}] step {step:5d} loss {loss.item():.4f} acc {acc:.4f} "
                  + " ".join(f"{o}={a:.3f}" for o, a in zip(OPS, per_op))
                  + f" | MI {' '.join(f'{m:.2f}' for m in mis)} | minload {lmin:.3f}", flush=True)
    return dict(arm=c["arm"], seed=c["seed"], repulse=c["repulse"], wd=c["wd"],
                adamw_wd=c["adamw_wd"], expert_wd=c["expert_wd"], decor=c["decor"],
                steps=c["steps"],
                acc=round(acc, 5), best_acc=round(best, 5), grok_step=grok_step,
                per_op=[round(a, 4) for a in per_op], mi_final=[round(m, 3) for m in mis],
                curve=curve)


if __name__ == "__main__":                                        # single-arm smoke: python grok_moe.py
    print(json.dumps(run(dict(steps=400))))
