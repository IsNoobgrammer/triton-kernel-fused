"""Synthetic-task kappa A/B: does r=1 orthogonalization quality move training?

Task: SORT — input [x_1..x_L, SEP, y_1..y_L] where y = sorted(x); CE loss on the y half only.
Vocab 64 + SEP. No downloads, no tokenizer, data generated on-GPU per step (seeded, identical
across arms -> paired comparison).

Model: tiny pre-LN GPT with SEPARATE q/k/v/o d x d projections (square, r=1 — the matrices under
test) + MLP with configurable ratio: ratio=1 -> ALL Muon matrices square (max kappa exposure);
ratio=4 -> standard rect-dominant control (arms should collapse together if the effect is r=1).

Arms (identical LR/schedule/data; only the Muon polar pipeline differs):
  adamw   : AdamW everything (reference scale)
  default : FusedMuon _DSV4_COEFFS (10 it) + aurora_k1   -> r=1 kappa ~40-450 seed lottery
  b12     : KJ x10 + pin x2 (12 it) + aurora_k1          -> r=1 kappa ~1.3-11
  champ   : b12 + signed-perm dither eps=0.05 on square  -> r=1 kappa 1.00 always

Usage: python train_synth.py ARM MLP_RATIO SEED [STEPS] >> synth_results.jsonl
"""
import json
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, r"C:\Users\shaur\OneDrive\Documents\triton-kernel-fused")
from kernels.sm75.muon import FusedMuon, newton_schulz, _DSV4_COEFFS  # noqa: E402

dev = "cuda"
V, L = 64, 32                    # data vocab, list length
SEP = V
SEQ = 2 * L + 1
KJ = (3.4445, -4.775, 2.0315); PIN = (2.0, -1.5, 0.5)
B12 = (KJ,) * 10 + (PIN,) * 2


class KappaMuon(FusedMuon):
    def __init__(self, params, arm, **kw):
        self.arm = arm
        coeffs = _DSV4_COEFFS if arm == "default" else B12
        super().__init__(params, coeffs=coeffs, ns_dtype=torch.float16, **kw)

    def _polar(self, u):
        if self.arm == "champ" and u.shape[-2] == u.shape[-1]:
            n = u.shape[-1]
            g = torch.Generator(device=u.device).manual_seed(999)
            r = torch.randperm(n, device=u.device, generator=g)
            c = torch.randperm(n, device=u.device, generator=g)
            s = (torch.randint(0, 2, (n,), device=u.device, generator=g) * 2 - 1).to(u.dtype)
            E = torch.zeros(n, n, device=u.device, dtype=u.dtype)
            E[r, c] = s
            smax = 2.0 * u.float().flatten(-2).norm(dim=-1) / (n ** 0.5)
            u = u + 0.05 * smax.view(-1, 1, 1).to(u.dtype) * E
        return newton_schulz(u, self.coeffs, self.ns_dtype)


class Block(nn.Module):
    def __init__(self, d, heads, ratio):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.q, self.k, self.v, self.o = (nn.Linear(d, d, bias=False) for _ in range(4))
        self.fc1 = nn.Linear(d, ratio * d, bias=False)
        self.fc2 = nn.Linear(ratio * d, d, bias=False)
        self.h = heads

    def forward(self, x):
        B, T, d = x.shape
        y = self.ln1(x)
        q, k, v = (p(y).view(B, T, self.h, d // self.h).transpose(1, 2) for p in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, T, d))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))


class TinyGPT(nn.Module):
    def __init__(self, d=256, heads=4, depth=4, ratio=1):
        super().__init__()
        self.emb = nn.Embedding(V + 1, d)
        self.pos = nn.Parameter(torch.zeros(SEQ, d))
        self.blocks = nn.ModuleList(Block(d, heads, ratio) for _ in range(depth))
        self.lnf = nn.LayerNorm(d)
        self.head = nn.Linear(d, V + 1, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.lnf(x))


def batch(bs, gen):
    x = torch.randint(0, V, (bs, L), device=dev, generator=gen)
    y = x.sort(dim=1).values
    sep = torch.full((bs, 1), SEP, device=dev, dtype=torch.long)
    return torch.cat([x, sep, y], dim=1)


def loss_fn(model, seq):
    logits = model(seq[:, :-1])
    tgt = seq[:, 1:]
    lo = logits[:, L:, :]                      # positions predicting y_1..y_L
    ta = tgt[:, L:]
    loss = F.cross_entropy(lo.reshape(-1, V + 1), ta.reshape(-1))
    acc = (lo.argmax(-1) == ta).float().mean()
    return loss, acc


def main():
    arm, ratio, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    steps = int(sys.argv[4]) if len(sys.argv) > 4 else 1500
    torch.manual_seed(seed)
    model = TinyGPT(ratio=ratio).to(dev)
    mats, rest = [], []
    for n, p in model.named_parameters():
        (mats if (p.ndim == 2 and "emb" not in n and "head" not in n) else rest).append(p)
    aw = torch.optim.AdamW(rest, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    if arm == "adamw":
        mu = torch.optim.AdamW(mats, lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    else:
        mu = KappaMuon(mats, arm, lr=1e-3, weight_decay=0.01)
    sq = sum(1 for p in mats if p.shape[0] == p.shape[1])
    gen = torch.Generator(device=dev).manual_seed(1000 + seed)     # SAME data stream for all arms
    vgen = torch.Generator(device=dev).manual_seed(77)
    vbatch = batch(512, vgen)
    warm = 100
    for step in range(1, steps + 1):
        lr_mul = min(1.0, step / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, step / steps)))
        for opt in (mu, aw):
            for grp in opt.param_groups:
                grp["lr"] = grp.get("base_lr", grp.setdefault("base_lr", grp["lr"])) * lr_mul
        loss, _ = loss_fn(model, batch(64, gen))
        loss.backward()
        mu.step(); aw.step()
        mu.zero_grad(set_to_none=True); aw.zero_grad(set_to_none=True)
        if step % 50 == 0 or step == steps:
            with torch.no_grad():
                vl, va = loss_fn(model, vbatch)
            print(json.dumps({"arm": arm, "ratio": ratio, "seed": seed, "step": step,
                              "val_loss": round(vl.item(), 5), "val_acc": round(va.item(), 5)}), flush=True)
    print(json.dumps({"arm": arm, "ratio": ratio, "seed": seed, "final": True, "sq_mats": sq,
                      "val_loss": round(vl.item(), 5), "val_acc": round(va.item(), 5)}), flush=True)


if __name__ == "__main__":
    main()
