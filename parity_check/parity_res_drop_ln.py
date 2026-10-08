"""res_dropout_layernorm (kernels/sm120/res_drop_ln.py) vs eager NeMo ConformerLayer glue, and vs fp64.

Eager = the exact ops NeMo runs: r = residual + dropout(x) * factor; y = layer_norm(r), with the SAME dropout mask
(ours, read back by running the kernel on residual=0, x=1). Two precision modes:
  autocast     residual fp32 (LayerNorm under autocast outputs fp32), x bf16, LN params fp32
  bf16         everything bf16 (train_asr --bf16_master)
Gates per mode, dropout 0 and 0.1, factor 0.5 and 1, a chain of 4 boundaries (as in one layer):
  1. r, y, d_residual, d_x, d_w, d_b: ours at least as close to fp64 as eager (x1.05, or rel < 1e-6 / bf16 eps)
  2. bitwise repeatable
  3. dropout: the kept fraction is 1 - p (+-0.5%) and the mask is the same in forward and backward

    python parity_check/parity_res_drop_ln.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from kernels.sm120.res_drop_ln import res_dropout_layernorm

dev = "cuda"
ok = True
SEED = 77


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def chain(impl, res, xs, lns, p, factors, masks, dt):
    """4 boundaries: r_k = r_{k-1} + drop(x_k) * f_k, y_k = LN_k(r_k); loss uses every y and the last r."""
    out = 0.0
    r = res
    for k, (x, ln, f) in enumerate(zip(xs, lns, factors)):
        if impl == "ours":
            r, y = res_dropout_layernorm(r, x, ln[0], ln[1], 1e-5, p, f, SEED + k)
        else:
            if dt == torch.float64:
                xd = x.double() * masks[k].double() / (1 - p) if p > 0 else x.double()
                r = r + xd * f
            else:
                xd = (x.float() * (masks[k].float() / (1 - p))).to(x.dtype) if p > 0 else x   # torch dropout's rounding
                r = r + (xd * f if f != 1.0 else xd)
            y = F.layer_norm(r, r.shape[-1:], ln[0].to(r.dtype) if dt == torch.float64 else ln[0],
                             ln[1].to(r.dtype) if dt == torch.float64 else ln[1], 1e-5)
        out = out + (y.double() * torch.linspace(-1, 1, y.shape[-1], device=dev, dtype=torch.float64)).sum()
    return out + r.double().sum() * 0.3


def run(impl, mode, p, factors, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    B, T, C = 6, 37, 512
    rdt = torch.float32 if mode == "autocast" else torch.bfloat16
    pdt = torch.float32 if mode == "autocast" else torch.bfloat16
    res = (torch.randn(B, T, C, device=dev, generator=g) * 2).to(rdt)
    xs = [torch.randn(B, T, C, device=dev, generator=g).to(torch.bfloat16) for _ in range(4)]
    lns = [((1 + 0.1 * torch.randn(C, device=dev, generator=g)).to(pdt), (0.1 * torch.randn(C, device=dev, generator=g)).to(pdt))
           for _ in range(4)]
    masks = [None] * 4
    if p > 0:                                   # each boundary has its own seed
        masks = []
        for k in range(4):
            one = torch.ones(B, T, C, device=dev, dtype=torch.bfloat16)
            masks.append(res_dropout_layernorm(torch.zeros(B, T, C, device=dev), one, torch.ones(C, device=dev),
                                               torch.zeros(C, device=dev), 1e-5, p, 1.0, SEED + k)[0] != 0)
    dt = torch.float64 if impl == "gt" else rdt
    leaves = [res] + xs + [t for ln in lns for t in ln]
    leaves = [t.detach().clone().double().requires_grad_() if impl == "gt" else t.detach().clone().requires_grad_()
              for t in leaves]
    r0, xs_, lns_ = leaves[0], leaves[1:5], [(leaves[5 + 2 * k], leaves[6 + 2 * k]) for k in range(4)]
    loss = chain(impl, r0, xs_, lns_, p, factors, masks, dt)
    loss.backward()
    return [loss.detach()] + [t.grad for t in leaves], masks


def rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp(min=1e-300)).item()


def main():
    names = ["loss", "d_res"] + [f"d_x{k}" for k in range(4)] + [f"d_{n}{k}" for k in range(4) for n in ("w", "b")]
    for mode in ("autocast", "bf16"):
        for p in (0.0, 0.1):
            factors = (0.5, 1.0, 1.0, 0.5)
            print(f"\n== {mode}, dropout {p}, factors {factors}", flush=True)
            gt, _ = run("gt", mode, p, factors)
            ea, _ = run("eager", mode, p, factors)
            ou, masks = run("ours", mode, p, factors)
            floor = 1e-6 if mode == "autocast" else 4e-3
            bad = []
            for n, a, e, g in zip(names, ou, ea, gt):
                eo, ee = rel(a, g), rel(e, g)
                if eo > max(ee * 1.05, floor):
                    bad.append(f"{n} ours {eo:.2e} eager {ee:.2e}")
            worst = max(zip(names, ou, ea, gt), key=lambda t: rel(t[1], t[3]))
            check("1 vs fp64 <= eager", not bad, "; ".join(bad) if bad else
                  f"worst {worst[0]}: ours {rel(worst[1], worst[3]):.2e} eager {rel(worst[2], worst[3]):.2e}")
            ou2, _ = run("ours", mode, p, factors)
            check("2 repeatable", all(torch.equal(a, b) for a, b in zip(ou, ou2)), "bitwise")
            if p > 0:
                kept = (masks[0] != 0).float().mean().item()
                check("3 dropout rate", abs(kept - (1 - p)) < 0.005, f"kept {kept:.4f}")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
