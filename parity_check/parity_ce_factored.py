"""Factored-softmax CE (kernels/sm120/ce_factored.py) vs fp32 ground truth and vs the chunked kernel.

Gates, at a multi-chunk N with ignored labels, bf16 autocast, fp32 master weight:
  1. loss, d_hidden, d_weight at least as close to fp32 as the chunked kernel (x1.05 slack)
  2. bitwise repeatable
  3. the out-of-window pass, FORCED onto every row, still meets gate 1
  4. an odd V and H (masked tails) meet gate 1
  5. two MTP heads in one call == the weighted sum of two separate calls (to fp32 rounding)
  6. the no-grad (val) path returns the same loss as the grad path (to fp32 rounding)

    python parity_check/parity_ce_factored.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import kernels.sm120.ce_factored as cf
from kernels.sm120.cross_entropy import fused_linear_cross_entropy as chunked

dev = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
ok = True


def data(N, H, V, scale, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    h = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16)
    w = torch.randn(V, H, device=dev, generator=g) * scale * H ** -0.5
    y = torch.randint(0, V, (N,), device=dev, generator=g)
    y[::7] = -100
    return h, w, y


def grads(ce, h0, w0, y):
    h, w = h0.clone().requires_grad_(), w0.clone().requires_grad_()
    with AMP:
        loss = ce(h, w, y, bwd_logits_budget=64 << 20)          # small budget -> many chunks
    loss.backward()
    return loss.detach().float(), h.grad.float(), w.grad.float()


def truth(h, w, y):
    h, w = h.float().requires_grad_(), w.to(torch.bfloat16).float().requires_grad_()
    loss = torch.nn.functional.cross_entropy(h @ w.t(), y, ignore_index=-100)
    loss.backward()
    return loss.detach(), h.grad, w.grad


def err(a, r):
    return [abs(a[0] - r[0]).item()] + [((x - z).norm() / z.norm()).item() for x, z in zip(a[1:], r[1:])]


def check(name, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"{'ok  ' if cond else 'FAIL'} {name}: {msg}")


for tag, (N, H, V, sc) in {"H512 V81920": (8192, 512, 81920, 2.0), "odd H328 V50257": (6000, 328, 50257, 2.0),
                           "large logits": (8192, 512, 81920, 12.0)}.items():
    h, w, y = data(N, H, V, sc)
    gt = truth(h, w, y)
    ec = err(grads(chunked, h, w, y), gt)
    a = grads(cf.fused_linear_cross_entropy, h, w, y)
    ef = err(a, gt)
    b = grads(cf.fused_linear_cross_entropy, h, w, y)
    check(f"{tag} vs fp32", all(f <= c * 1.05 + 1e-7 for f, c in zip(ef, ec)),
          f"factored {['%.2e' % e for e in ef]}  chunked {['%.2e' % e for e in ec]}")
    check(f"{tag} bitwise", all(torch.equal(x, z) for x, z in zip(a, b)), "rerun identical")
    lo, hi = cf._LO, cf._HI
    cf._LO, cf._HI = float("inf"), float("-inf")
    try:
        ex = err(grads(cf.fused_linear_cross_entropy, h, w, y), gt)
    finally:
        cf._LO, cf._HI = lo, hi
    check(f"{tag} forced out-of-window pass", all(f <= c * 1.05 + 1e-7 for f, c in zip(ex, ec)),
          f"{['%.2e' % e for e in ex]}")
    with torch.no_grad(), AMP:
        lv = cf.fused_linear_cross_entropy(h, w, y)
    check(f"{tag} val path", abs(lv.item() - a[0].item()) < 1e-5, f"val {lv.item():.7f} vs train {a[0].item():.7f}")

# MTP: two heads in one call vs two separate calls
h1, w, y1 = data(8192, 512, 81920, 2.0, 1)
h2, _, y2 = data(8192, 512, 81920, 2.0, 2)
outs = []
for fused in (True, False):
    a1, a2, ww = h1.clone().requires_grad_(), h2.clone().requires_grad_(), w.clone().requires_grad_()
    with AMP:
        if fused:
            loss, _ = cf.fused_linear_cross_entropy_heads([a1, a2], ww, [y1, y2], [1.0, 0.3])
        else:
            loss = cf.fused_linear_cross_entropy(a1, ww, y1) + 0.3 * cf.fused_linear_cross_entropy(a2, ww, y2)
    loss.backward()
    outs.append((loss.detach(), a1.grad.float(), a2.grad.float(), ww.grad))
d = [((x.float() - z.float()).norm() / z.float().norm().clamp(min=1e-30)).item() for x, z in zip(*outs)]
# cuBLAS may pick a different kernel for the concatenated M, so bf16 gh can move by an ulp
check("MTP heads in one call", d[0] < 1e-5 and d[1] < 5e-3 and d[2] < 5e-3 and d[3] < 1e-4,
      f"rel diffs loss/gh1/gh2/gw {['%.1e' % e for e in d]}")

print("ce_factored parity PASS" if ok else "ce_factored parity FAIL")
sys.exit(0 if ok else 1)
