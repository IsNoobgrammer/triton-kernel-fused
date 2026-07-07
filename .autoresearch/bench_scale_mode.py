"""A/B: FusedMuon scale_mode 'polarexpress' (Jordan aspect-ratio) vs 'moonlight' (consistent-RMS).

Tiny GPT on a sorting task (predict the sorted copy of a random token sequence).
Same init + same data per seed across arms; only the Muon scale_mode (+ its native LR band)
differs. AdamW handles embed/head/norms/biases at a fixed lr so the comparison isolates Muon.

Phase 1: LR sweep per mode (1 seed) -> pick best LR by final loss.
Phase 2: best LR x 3 seeds -> mean +- std.

Run from the repo root:  python .autoresearch/bench_scale_mode.py
"""
import math
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")
from kernels.sm120.muon import FusedMuon  # noqa: E402

DEV = "cuda"
VOCAB, SEQ_HALF, DIM, LAYERS, HEADS, FFN = 64, 32, 256, 4, 4, 1024
BATCH, STEPS, TAIL = 64, 400, 50
ADAMW_LR = 1e-3


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(DIM), nn.LayerNorm(DIM)
        self.q, self.k, self.v, self.o = (nn.Linear(DIM, DIM, bias=False) for _ in range(4))
        self.up, self.down = nn.Linear(DIM, FFN, bias=False), nn.Linear(FFN, DIM, bias=False)

    def forward(self, x):
        h = self.n1(x)
        B, S, _ = h.shape
        shp = (B, S, HEADS, DIM // HEADS)
        q, k, v = (m(h).view(shp).transpose(1, 2) for m in (self.q, self.k, self.v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.o(a.transpose(1, 2).reshape(B, S, DIM))
        return x + self.down(F.gelu(self.up(self.n2(x))))


class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, DIM)
        self.pos = nn.Embedding(2 * SEQ_HALF, DIM)
        self.blocks = nn.ModuleList(Block() for _ in range(LAYERS))
        self.norm = nn.LayerNorm(DIM)
        self.head = nn.Linear(DIM, VOCAB, bias=False)

    def forward(self, idx):
        x = self.emb(idx) + self.pos.weight[: idx.shape[1]]
        for b in self.blocks:
            x = b(x)
        return self.head(self.norm(x))


def batch(gen):
    src = torch.randint(0, VOCAB, (BATCH, SEQ_HALF), generator=gen, device=DEV)
    tgt = src.sort(dim=1).values
    return torch.cat([src, tgt], 1)


def run(scale_mode, muon_lr, seed):
    torch.manual_seed(seed)
    model = TinyGPT().to(DEV)
    muon_p = [p for n, p in model.named_parameters() if p.ndim == 2 and "emb" not in n
              and "pos" not in n and "head" not in n]
    other_p = [p for n, p in model.named_parameters() if not any(p is q for q in muon_p)]
    muon = FusedMuon(muon_p, lr=muon_lr, weight_decay=0.0, scale_mode=scale_mode)
    adamw = torch.optim.AdamW(other_p, lr=ADAMW_LR, weight_decay=0.0)
    gen = torch.Generator(device=DEV).manual_seed(10_000 + seed)
    tail = []
    for step in range(STEPS):
        seq = batch(gen)
        logits = model(seq[:, :-1])
        # loss only on the sorted half (positions SEQ_HALF-1 .. end predict tgt tokens)
        loss = F.cross_entropy(
            logits[:, SEQ_HALF - 1:].reshape(-1, VOCAB),
            seq[:, SEQ_HALF:].reshape(-1))
        if not math.isfinite(loss.item()):
            return float("inf")
        loss.backward()
        muon.step(); adamw.step()
        muon.zero_grad(set_to_none=True); adamw.zero_grad(set_to_none=True)
        if step >= STEPS - TAIL:
            tail.append(loss.item())
    return sum(tail) / len(tail)


SWEEP = {
    "polarexpress": [5e-3, 1e-2, 2e-2, 4e-2],
    "moonlight": [1e-4, 3e-4, 1e-3, 3e-3],
}

if __name__ == "__main__":
    print(f"torch {torch.__version__} | {torch.cuda.get_device_name(0)}")
    best = {}
    for mode, lrs in SWEEP.items():
        for lr in lrs:
            f = run(mode, lr, seed=0)
            print(f"[sweep] {mode:13s} lr={lr:<8g} final(last{TAIL})={f:.4f}", flush=True)
            if f < best.get(mode, (float("inf"),))[0]:
                best[mode] = (f, lr)
    print()
    for mode, (_, lr) in best.items():
        finals = [run(mode, lr, seed=s) for s in (0, 1, 2)]
        m = sum(finals) / 3
        sd = (sum((x - m) ** 2 for x in finals) / 2) ** 0.5
        print(f"[final] {mode:13s} lr={lr:<8g} loss={m:.4f} +- {sd:.4f}  (seeds: "
              + ", ".join(f"{x:.4f}" for x in finals) + ")", flush=True)
