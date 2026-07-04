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
import mech                                                            # noqa: E402
import metrics                                                         # noqa: E402
from grok_moe import FusedMuon, _DSV4_COEFFS, GrokMoENet              # noqa: E402
from kernels.sm75.muon import _PE_COEFFS                              # noqa: E402  (Polar-Express PE-8)

_KJ, _PIN = (3.4445, -4.7750, 2.0315), (2.0, -1.5, 0.5)               # KJ quintic + pinned tail


def _coeffs(ns_kj):
    """ns_kj KJ iterations + 2 pinned = the Muon NS schedule (default 8 = dsv4_10; 6 = ns8)."""
    return (_KJ,) * ns_kj + (_PIN,) * 2

P, NUM_OPS = 97, 4
EQ, PAD = P + NUM_OPS, P + NUM_OPS + 1                                # token ids
VOCAB = P + NUM_OPS + 2


def _sample(depth, n, g, dev, inv, nops=NUM_OPS):
    """n random depth-`depth` chains -> (tokens (n, 2*depth+2), result, unique int key)."""
    v = torch.randint(0, P, (n, depth + 1), device=dev, generator=g)
    o = torch.randint(0, nops, (n, depth), device=dev, generator=g)
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


def _pad_left(tok, dev, maxseq):
    out = torch.full((tok.shape[0], maxseq), PAD, dtype=torch.long, device=dev)
    out[:, maxseq - tok.shape[1]:] = tok
    return out


_DEF = dict(arm="default", seed=0, steps=6000, batch=768, d=128, layers=4, heads=4,
            experts=8, top_k=2, lr=1e-3, muon_lr=1e-3, wd=0.1, adamw_wd=0.1,
            dense_first=1, bias_every=10, bias_factor=0.01, mult=4, nval=4096,
            eval_every=250, max_depth=6, noise=0.05, div_deep=0,
            warmup=500, decay_frac=0.2, min_lr_frac=0.1,               # WSD, same for both opts
            scale_mode="aurora", aurora_k=1, ns_kj=6, coeffs="kj",      # DEFAULT = ns8 (6 KJ) aurora_k1
            #  coeffs: "kj" -> KJ*ns_kj + 2 pin;  "pe" -> Polar-Express PE-8 (8 iters)
            repulse=0.0, decor=0.0, grad_rep=0.0, xorth=0, niche=0.0,   # mechanism knobs (mech.py)
            scap=0.0, cautious=0.0, grokfast=0.0, gf_alpha=0.98, lookahead=0, la_beta=0.5,
            depth_mix=(0.45, 0.25, 0.15, 0.08, 0.045, 0.025))


def _floor(eps):
    """Irreducible val CE under eps label noise: y = truth w.p. 1-eps, uniform w.p. eps."""
    pc = 1 - eps + eps / P
    po = eps / P
    return -(pc * math.log(pc) + (P - 1) * po * math.log(po)) if eps > 0 else 0.0


def make_tag(c):
    t = f"olm_{c['arm']}_s{c['seed']}_wd{c['wd'] if c['arm'] != 'adamw' else c['adamw_wd']}"
    if c["dense_first"]:
        t += f"_df{c['dense_first']}"
    if c.get("warmup", 500) != 500:
        t += f"_wu{c['warmup']}"
    if c["arm"] == "default":
        if c["scale_mode"] != "aurora":
            t += f"_{c['scale_mode']}"
        if c["aurora_k"] != 1:
            t += f"_k{c['aurora_k']}"
        if c["coeffs"] == "pe":
            t += "_pe8"                                           # Polar-Express PE-8 schedule
        elif c["ns_kj"] != 6:                                     # default = 8 iters (6 KJ + 2 pin)
            t += f"_it{c['ns_kj'] + 2}"                           # TOTAL NS iters (avoid ns_kj confusion)
    if c["mult"] != 4:
        t += f"_m{c['mult']}"
    for key, pre in (("repulse", "rep"), ("decor", "dec"), ("grad_rep", "gr"),
                     ("niche", "ni"), ("scap", "sc"), ("cautious", "cw"),
                     ("grokfast", "gf"), ("lookahead", "la")):
        if c.get(key):
            t += f"_{pre}{c[key]}"
    if c.get("xorth"):
        t += "_xo"
    if c["steps"] != 6000:
        t += f"_{c['steps']}st"
    return t


def run(cfg):
    """Train one arm in the ONLINE (one-epoch) regime. Returns result dict; prints live."""
    c = {**_DEF, **cfg}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(c["seed"])
    inv = torch.tensor([0] + [pow(b, P - 2, P) for b in range(1, P)], device=dev)
    maxd, maxseq = c["max_depth"], 2 * c["max_depth"] + 2

    def _nops(dep):                                                   # v3: div is a depth-1 skill;
        return NUM_OPS if (dep == 1 or c["div_deep"]) else 3          # deep chains use +,-,* only

    gval = torch.Generator(device=dev).manual_seed(1234)              # FROZEN val stream
    vx, vy, vd, vkeys = [], [], [], {}
    for dep in range(1, maxd + 1):
        tokd, rd, keyd = _sample(dep, c["nval"] * 2, gval, dev, inv, _nops(dep))
        seen, pick = set(), []                                        # nval unique keys
        for i, k in enumerate(keyd.tolist()):
            if k not in seen:
                seen.add(k)
                pick.append(i)
            if len(pick) >= c["nval"]:
                break
        pick = torch.tensor(pick, device=dev)
        vx.append(_pad_left(tokd[pick], dev, maxseq)); vy.append(rd[pick])
        vd.append(torch.full((len(pick),), dep - 1, device=dev))
        vkeys[dep] = keyd[pick].sort().values
    vx, vy, vd = torch.cat(vx), torch.cat(vy), torch.cat(vd)
    if c["noise"] > 0:                                                # irreducible entropy, val too
        flip = torch.rand(len(vy), device=dev, generator=gval) < c["noise"]
        vy[flip] = torch.randint(0, P, (int(flip.sum()),), device=dev, generator=gval)

    model = GrokMoENet(VOCAB, c["d"], c["layers"], c["heads"], c["experts"], c["top_k"],
                       seq=maxseq, dense_layers=tuple(range(c["dense_first"])),
                       mult=c["mult"]).to(dev)
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
        hwd = 0.0 if c["cautious"] > 0 else c["wd"]                # cautious does manual masked decay
        nsc = _PE_COEFFS if c["coeffs"] == "pe" else _coeffs(c["ns_kj"])
        mkw = dict(lr=c["muon_lr"], weight_decay=hwd, coeffs=nsc,
                   ns_dtype=torch.float16, scale_mode=c["scale_mode"], aurora_k=c["aurora_k"])
        opts = [torch.optim.AdamW(rest, lr=c["lr"], weight_decay=c["adamw_wd"], betas=(0.9, 0.98)),
                FusedMuon([q for q in hidden if q.ndim == 2], **mkw),
                FusedMuon([q for q in hidden if q.ndim == 3], **mkw)]
    all_params = list(model.parameters())
    expert_ws = ([b.moe.w1 for b in mblocks] + [b.moe.w2 for b in mblocks])
    mstate = {}                                                    # per-run mechanism buffers
    tag = make_tag(c)
    lnP = math.log(P)
    floor = _floor(c["noise"])
    print(f"[{tag}] layers={c['layers']} df={c['dense_first']} E={c['experts']} maxd={maxd} "
          f"noise={c['noise']} params={n_par/1e6:.2f}M val={len(vy)} init_CE={lnP:.3f} "
          f"floor={floor:.3f} (frac {floor/lnP:.3f}; LM residual ~0.09)", flush=True)

    g = torch.Generator(device=dev).manual_seed(c["seed"] + 7)
    mix = torch.tensor(c["depth_mix"], device=dev)
    E = c["experts"]
    best, curve = 0.0, []
    for step in range(1, c["steps"] + 1):
        deps = torch.multinomial(mix, c["batch"], replacement=True, generator=g)
        xb, yb = [], []
        for dep in range(1, maxd + 1):
            n = int((deps == dep - 1).sum())
            if n == 0:
                continue
            tok, r, key = _sample(dep, n, g, dev, inv, _nops(dep))
            keep = ~torch.isin(key, vkeys[dep])                       # ONLINE: val never trains
            xb.append(_pad_left(tok[keep], dev, maxseq)); yb.append(r[keep])
        xb, yb = torch.cat(xb), torch.cat(yb)
        if c["noise"] > 0:                                            # same noise in the stream
            flip = torch.rand(len(yb), device=dev, generator=g) < c["noise"]
            yb[flip] = torch.randint(0, P, (int(flip.sum()),), device=dev, generator=g)
        loss = F.cross_entropy(model(xb, pad_mask=(xb != PAD)), yb)   # load counts real tokens only
        loss.backward()
        mech.pre_step(c, all_params, expert_ws, mblocks, mstate)      # grad-space mechanisms
        # WSD schedule (LM-standard), identical for AdamW and Muon: linear warmup ->
        # stable -> cosine decay to min_lr_frac over the final decay_frac of steps.
        if c["warmup"] and step <= c["warmup"]:
            scale = step / c["warmup"]
        else:
            t0 = c["steps"] * (1 - c["decay_frac"])
            if step > t0:
                prog = (step - t0) / max(c["steps"] - t0, 1)
                scale = c["min_lr_frac"] + (1 - c["min_lr_frac"]) * 0.5 * (1 + math.cos(math.pi * prog))
            else:
                scale = 1.0
        for o in opts:
            for gp in o.param_groups:
                gp.setdefault("base_lr", gp["lr"])
                gp["lr"] = scale * gp["base_lr"]
            o.step(); o.zero_grad(set_to_none=True)
        mech.post_step(c, all_params, hidden, expert_ws, mblocks, mstate, dev, c["muon_lr"], step)
        if mblocks and step % c["bias_every"] == 0:                   # DSv3 balancer every N steps
            for b in mblocks:
                b.moe.update_bias(c["bias_factor"])
        if step % c["eval_every"] == 0 or step == c["steps"]:
            model.eval()
            with torch.no_grad():
                celoss, preds = [], []
                lab_idx = [[] for _ in mblocks]                       # answer-position top-k experts
                lab_w = [[] for _ in mblocks]                         # ...and combine weights
                evload = [torch.zeros(E, device=dev) for _ in mblocks]  # utilization over real tokens
                for i in range(0, len(vy), 8192):
                    xb_ = vx[i:i + 8192]
                    pm = (xb_ != PAD)
                    lg = model(xb_, pad_mask=pm)
                    celoss.append(F.cross_entropy(lg, vy[i:i + 8192], reduction="none"))
                    preds.append(lg.argmax(-1))
                    rm = pm.reshape(-1)
                    for li, b in enumerate(mblocks):
                        k = b.moe.top_k
                        lab_idx[li].append(b.moe._last_idx.reshape(-1, maxseq, k)[:, -1, :])
                        lab_w[li].append(b.moe._last_w.reshape(-1, maxseq, k)[:, -1, :])
                        fi = b.moe._last_idx[rm].reshape(-1)          # real-token routing
                        evload[li] += torch.bincount(fi, minlength=E).float()
                ce = torch.cat(celoss)
                pred = torch.cat(preds)
                hit = (pred == vy).float()
                per_d = [hit[vd == i].mean().item() for i in range(maxd)]
                per_dce = [ce[vd == i].mean().item() for i in range(maxd)]
                mixw = (torch.tensor(c["depth_mix"], device=dev))     # eval on TRAIN distribution
                mixw = (mixw / mixw.sum()).tolist()                   # (LM-faithful; unpins the tail)
                vloss = sum(w * l for w, l in zip(mixw, per_dce))
                acc = sum(w * a for w, a in zip(mixw, per_d))
                mi, spec, eff = [], [], []                            # soft top-2 MI + utilization
                for li in range(len(mblocks)):
                    m, s = metrics.soft_mi(torch.cat(lab_idx[li]), torch.cat(lab_w[li]),
                                           vd, E, maxd)
                    mi.append(m); spec.append(s)
                    eff.append(metrics.load_stats(evload[li])["eff"])
                mineff = min(eff) if eff else E
            model.train()
            best = max(best, acc)                                    # per_depth in curve -> d2/d3 plots
            curve.append([step, round(vloss, 4), round(acc, 5), [round(a, 4) for a in per_d]])
            print(f"[{tag}] step {step:5d} val_CE {vloss:.4f} gap {vloss-floor:+.4f} "
                  f"frac {vloss/lnP:.3f} acc {acc:.4f} d " + " ".join(f"{a:.3f}" for a in per_d)
                  + f" | spec {' '.join(f'{s:.2f}' for s in spec)}"
                  + f" | eff/{E} {' '.join(f'{e:.1f}' for e in eff)}", flush=True)
    return dict(arm=c["arm"], seed=c["seed"], wd=c["wd"], adamw_wd=c["adamw_wd"],
                scale_mode=c["scale_mode"], aurora_k=c["aurora_k"], ns_kj=c["ns_kj"],
                coeffs=c["coeffs"],
                dense_first=c["dense_first"], warmup=c["warmup"], noise=c["noise"], max_depth=maxd,
                mult=c["mult"], experts=E, steps=c["steps"], loss=round(vloss, 4),
                gap=round(vloss - floor, 4), frac=round(vloss / lnP, 4),
                acc=round(acc, 5), best_acc=round(best, 5),
                per_depth=[round(a, 4) for a in per_d],
                per_depth_ce=[round(l, 4) for l in per_dce],
                scap_smax=round(mstate.get("scap_smax", 0.0), 3),
                mi_soft=[round(m, 3) for m in mi], spec_frac=[round(s, 3) for s in spec],
                eff_experts=[round(e, 2) for e in eff], min_eff=round(mineff, 2),
                curve=curve)


if __name__ == "__main__":                                            # smoke: python olm.py
    print(json.dumps(run(dict(steps=20, batch=128, nval=256, eval_every=10))))
