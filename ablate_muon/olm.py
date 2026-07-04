"""Online LM-emulator: single-epoch compositional modular arithmetic (no memorization).

WHY (user directive 2026-07-04): grokking = memorize-then-generalize, the WRONG regime for
language modeling. LM is compute-bound: every sample is seen ONCE, so all progress is
compression/generalization. This harness emulates that: fresh samples every step, one epoch
forever, held-out val excluded from the stream by key.

COMPRESSION CALIBRATION: LM init CE = ln(vocab) ~ ln(81920) = 11.3 nats; strong LMs land
~1.0 nat/token => ~9% of initial entropy remains at budget. Here init CE = ln(97) = 4.575
nats; the depth mix is tuned so the budget lands near the matched target ~0.4-0.5 nats
(frac ~0.09-0.11). `frac` in the output = val CE / ln(97) -- compare to LM's ~0.09.

TASK: left-fold chains of modular ops, depth d in 1..4 (Zipf-ish mix 0.4/0.3/0.2/0.1):
v0 op1 v1 op2 v2 ... = ? (mod 97), evaluated left-to-right. Depth = composition depth
(shallow learns fast like bigrams, deep learns slowly like long-range structure).
Sample space >> stream size (depth-3 alone ~5.6e9 vs 4.6M samples at 6000x768).

Metrics: val CE (nats), frac-of-initial-entropy, acc, per-depth acc, MI(top1 expert, depth)
per MoE layer, minload. `run(cfg) -> dict`. No argparse.
"""
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grok_moe import FusedMuon, _DSV4_COEFFS, GrokMoENet, _mi          # noqa: E402

P, NUM_OPS = 97, 4
EQ, PAD = P + NUM_OPS, P + NUM_OPS + 1                                # token ids
VOCAB = P + NUM_OPS + 2
MAX_DEPTH = 4
MAXSEQ = 2 * MAX_DEPTH + 2


def _sample(depth, n, g, dev, inv):
    """n random depth-`depth` chains -> (tokens (n, 2*depth+2), result, unique int key)."""
    v = torch.randint(0, P, (n, depth + 1), device=dev, generator=g)
    o = torch.randint(0, NUM_OPS, (n, depth), device=dev, generator=g)
    fix = (o == 3) & (v[:, 1:] == 0)                                  # div needs nonzero rhs
    v[:, 1:][fix] = torch.randint(1, P, (int(fix.sum()),), device=dev, generator=g)
    r = v[:, 0]
    for i in range(depth):
        b = v[:, i + 1]
        cand = torch.stack([(r + b) % P, (r - b) % P, (r * b) % P, (r * inv[b]) % P], 1)
        r = cand.gather(1, o[:, i:i + 1]).squeeze(1)
    key = v[:, 0].clone()
    for i in range(depth):
        key = key * P + v[:, i + 1]
        key = key * NUM_OPS + o[:, i]
    tok = torch.empty(n, 2 * depth + 2, dtype=torch.long, device=dev)
    tok[:, 0] = v[:, 0]
    for i in range(depth):
        tok[:, 2 * i + 1] = P + o[:, i]
        tok[:, 2 * i + 2] = v[:, i + 1]
    tok[:, 2 * depth + 1] = EQ
    return tok, r, key


def _pad_left(tok, dev):
    out = torch.full((tok.shape[0], MAXSEQ), PAD, dtype=torch.long, device=dev)
    out[:, MAXSEQ - tok.shape[1]:] = tok
    return out


_DEF = dict(arm="default", seed=0, steps=6000, batch=768, d=128, layers=4, heads=4,
            experts=8, top_k=2, lr=1e-3, muon_lr=1e-3, wd=0.1, adamw_wd=0.1,
            dense_first=0, bias_tokens=300_000, bias_factor=0.01, nval=4096,
            eval_every=250, depth_mix=(0.4, 0.3, 0.2, 0.1))


def make_tag(c):
    t = f"olm_{c['arm']}_s{c['seed']}_wd{c['wd'] if c['arm'] != 'adamw' else c['adamw_wd']}"
    if c["dense_first"]:
        t += f"_df{c['dense_first']}"
    if c["steps"] != 6000:
        t += f"_{c['steps']}st"
    return t


def run(cfg):
    """Train one arm in the ONLINE (one-epoch) regime. Returns result dict; prints live."""
    c = {**_DEF, **cfg}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(c["seed"])
    inv = torch.tensor([0] + [pow(b, P - 2, P) for b in range(1, P)], device=dev)

    gval = torch.Generator(device=dev).manual_seed(1234)              # FROZEN val stream
    vx, vy, vd, vkeys = [], [], [], {}
    for dep in range(1, MAX_DEPTH + 1):
        tokd, rd, keyd = _sample(dep, c["nval"] * 2, gval, dev, inv)  # oversample, keep first
        seen, pick = set(), []                                        # nval unique keys
        for i, k in enumerate(keyd.tolist()):
            if k not in seen:
                seen.add(k)
                pick.append(i)
            if len(pick) >= c["nval"]:
                break
        pick = torch.tensor(pick, device=dev)
        vx.append(_pad_left(tokd[pick], dev)); vy.append(rd[pick])
        vd.append(torch.full((len(pick),), dep - 1, device=dev))
        vkeys[dep] = keyd[pick].sort().values
    vx, vy, vd = torch.cat(vx), torch.cat(vy), torch.cat(vd)

    model = GrokMoENet(VOCAB, c["d"], c["layers"], c["heads"], c["experts"], c["top_k"],
                       seq=MAXSEQ, dense_layers=tuple(range(c["dense_first"]))).to(dev)
    mblocks = [b for b in model.blocks if b.moe is not None]
    n_par = sum(q.numel() for q in model.parameters())

    def is_hidden(n, q):
        return ("blocks" in n) and ("router" not in n) and q.ndim in (2, 3) and "ln" not in n
    hidden = [q for n, q in model.named_parameters() if is_hidden(n, q)]
    rest = [q for n, q in model.named_parameters() if not is_hidden(n, q)]
    if c["arm"] == "adamw":
        opts = [torch.optim.AdamW(model.parameters(), lr=c["lr"],
                                  weight_decay=c["adamw_wd"], betas=(0.9, 0.98))]
    else:
        opts = [torch.optim.AdamW(rest, lr=c["lr"], weight_decay=c["adamw_wd"], betas=(0.9, 0.98)),
                FusedMuon([q for q in hidden if q.ndim == 2], lr=c["muon_lr"],
                          weight_decay=c["wd"], coeffs=_DSV4_COEFFS, ns_dtype=torch.float16),
                FusedMuon([q for q in hidden if q.ndim == 3], lr=c["muon_lr"],
                          weight_decay=c["wd"], coeffs=_DSV4_COEFFS, ns_dtype=torch.float16)]
    tag = make_tag(c)
    lnP = math.log(P)
    print(f"[{tag}] layers={c['layers']} df={c['dense_first']} E={c['experts']} "
          f"params={n_par/1e6:.2f}M val={len(vy)} init_CE={lnP:.3f} nats "
          f"target~{0.09*lnP:.2f} (LM-matched frac 0.09)", flush=True)

    g = torch.Generator(device=dev).manual_seed(c["seed"] + 7)
    mix = torch.tensor(c["depth_mix"], device=dev)
    tok_count, best, curve = 0, 0.0, []
    for step in range(1, c["steps"] + 1):
        deps = torch.multinomial(mix, c["batch"], replacement=True, generator=g)
        xb, yb = [], []
        for dep in range(1, MAX_DEPTH + 1):
            n = int((deps == dep - 1).sum())
            if n == 0:
                continue
            tok, r, key = _sample(dep, n, g, dev, inv)
            keep = ~torch.isin(key, vkeys[dep])                       # ONLINE: val never trains
            xb.append(_pad_left(tok[keep], dev)); yb.append(r[keep])
        xb, yb = torch.cat(xb), torch.cat(yb)
        loss = F.cross_entropy(model(xb), yb)
        loss.backward()
        for o in opts:
            o.step(); o.zero_grad(set_to_none=True)
        tok_count += c["batch"]
        if tok_count >= c["bias_tokens"]:
            for b in mblocks:
                b.moe.update_bias(c["bias_factor"])
            tok_count = 0
        if step % c["eval_every"] == 0 or step == c["steps"]:
            model.eval()
            with torch.no_grad():
                losses, preds, top1s = [], [], [[] for _ in mblocks]
                for i in range(0, len(vy), 8192):
                    lg = model(vx[i:i + 8192])
                    losses.append(F.cross_entropy(lg, vy[i:i + 8192], reduction="sum"))
                    preds.append(lg.argmax(-1))
                    for li, b in enumerate(mblocks):
                        top1s[li].append(b.moe._last_top1.reshape(-1, MAXSEQ)[:, -1])
                vloss = (torch.stack(losses).sum() / len(vy)).item()
                pred = torch.cat(preds)
                hit = (pred == vy).float()
                acc = hit.mean().item()
                per_d = [hit[vd == i].mean().item() for i in range(MAX_DEPTH)]
                mis = [_mi(torch.cat(t), vd, c["experts"], MAX_DEPTH) for t in top1s]
                lmin = min((b.moe.load / b.moe.load.sum().clamp_min(1)).min().item()
                           for b in mblocks) if mblocks else 0.0
            model.train()
            best = max(best, acc)
            curve.append([step, round(vloss, 4), round(acc, 5)])
            print(f"[{tag}] step {step:5d} val_CE {vloss:.4f} frac {vloss/lnP:.3f} "
                  f"acc {acc:.4f} d1-4 " + " ".join(f"{a:.3f}" for a in per_d)
                  + f" | MI {' '.join(f'{m:.2f}' for m in mis)} | minload {lmin:.3f}", flush=True)
    return dict(arm=c["arm"], seed=c["seed"], wd=c["wd"], adamw_wd=c["adamw_wd"],
                dense_first=c["dense_first"], steps=c["steps"], loss=round(vloss, 4),
                frac=round(vloss / lnP, 4), acc=round(acc, 5), best_acc=round(best, 5),
                per_depth=[round(a, 4) for a in per_d],
                mi_final=[round(m, 3) for m in mis], curve=curve)


if __name__ == "__main__":                                            # smoke: python olm.py
    print(json.dumps(run(dict(steps=20, batch=128, nval=256, eval_every=10))))
