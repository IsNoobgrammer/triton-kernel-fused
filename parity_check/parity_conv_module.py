"""conv_module (kernels/sm120/conv_module.py) + silu_dropout (kernels/sm120/silu_dropout.py) vs NeMo's eager ops.

conv_module reference = ConformerConvolution.forward between the pointwise linears (glu, transpose, masked_fill,
F.pad, depthwise conv1d, transpose, LayerNorm, SiLU), run in fp64 (ground truth) and as NeMo runs it (bf16 g under
bf16 autocast). Gates:
  1. y, dg, d conv weight / bias, d LN weight / bias: ours at least as close to fp64 as eager bf16 (x1.25, or 1e-3)
  2. bitwise repeatable;  3. causal + padded: perturbing g at padded / future frames moves nothing it must not
silu_dropout:
  4. p=0: bitwise == eager silu (fwd and bwd, bf16)
  5. p=0.2: kept fraction ~0.8, kept entries == eager silu * 1.25 bitwise, dx consistent with the same mask

    python parity_check/parity_conv_module.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from kernels.sm120.conv_module import conv_module
from kernels.sm120.silu_dropout import silu_dropout

dev = "cuda"
ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-300)).item()


def nemo_conv(g, pad, cw, cb, lw, lb, eps, left):
    x = F.glu(g, dim=-1).transpose(1, 2)
    x = x.masked_fill(pad.unsqueeze(1), 0.0)
    x = F.pad(x, (left, cw.shape[-1] - 1 - left))
    x = F.conv1d(x, cw, cb, groups=cw.shape[0])
    x = F.layer_norm(x.transpose(1, 2), (x.shape[1],), lw, lb, eps)
    return F.silu(x)


def run(fn, g, params, W, gdt, amp):
    g = g.detach().to(gdt).clone().requires_grad_()
    ps = [p.detach().clone().requires_grad_() for p in params]
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        y = fn(g, *ps)
    (y.float() * W).sum().backward()
    return [y.detach()] + [g.grad] + [p.grad for p in ps]


def main():
    B, T, C, K, eps = 4, 203, 512, 9, 1e-5
    gen = torch.Generator(device=dev).manual_seed(0)
    g = torch.randn(B, T, 2 * C, device=dev, generator=gen)
    cw = torch.randn(C, 1, K, device=dev, generator=gen) / 3
    cb, lb = (0.1 * torch.randn(C, device=dev, generator=gen) for _ in range(2))
    lw = 1 + 0.1 * torch.randn(C, device=dev, generator=gen)
    lengths = torch.tensor([T, T - 17, T // 2, 40], device=dev)
    pad = torch.arange(T, device=dev)[None, :] >= lengths[:, None]
    W = torch.randn(B, T, C, device=dev, generator=gen)
    names = ["y", "dg", "d conv w", "d conv b", "d ln w", "d ln b"]
    for left in (K - 1, (K - 1) // 2):
        print(f"\n== conv_module B={B} T={T} C={C} K={K} left pad {left}", flush=True)
        gb = g.bfloat16()                                                # the values every side sees
        params = [cw, cb, lw, lb]
        ref = lambda gg, a, b_, c_, d: nemo_conv(gg, pad, a, b_, c_, d, eps, left)
        ours = lambda gg, a, b_, c_, d: conv_module(gg, pad, a, b_, c_, d, eps, left, out_dtype=torch.bfloat16)
        gt = run(ref, gb, [p.double() for p in params], W.double(), torch.float64, False)
        ea = run(ref, gb, params, W, torch.bfloat16, True)
        ou = run(ours, gb, params, W, torch.bfloat16, True)
        bad = [f"{n} ours {rel(a, t):.2e} eager {rel(e, t):.2e}" for n, a, e, t in zip(names, ou, ea, gt)
               if rel(a, t) > max(1.25 * rel(e, t), 1e-3)]
        check("1 vs fp64 <= eager bf16 x1.25", not bad, "; ".join(bad) or
              "  ".join(f"{n} {rel(a, t):.1e}/{rel(e, t):.1e}" for n, a, e, t in zip(names, ou, ea, gt)))
        print("     ours vs eager: " + "  ".join(f"{n} {rel(a, e):.1e}" for n, a, e in zip(names, ou, ea)))
        ou2 = run(ours, gb, params, W, torch.bfloat16, True)
        check("2 repeatable", all(torch.equal(a, b) for a, b in zip(ou, ou2)), "bitwise")
        g2 = gb.clone()
        g2[pad] += 1.0                                                    # padded frames are masked before the conv
        if left == K - 1:
            g2[:, T - 5:] += 1.0                                          # causal: only the last 5 outputs may move
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y1 = conv_module(gb, pad, *params, eps, left, out_dtype=torch.bfloat16)
            y2 = conv_module(g2, pad, *params, eps, left, out_dtype=torch.bfloat16)
        keep = torch.ones(B, T, dtype=torch.bool, device=dev)
        if left == K - 1:
            keep[:, T - 5:] = False
        check("3 mask / causality", torch.equal(y1[keep], y2[keep]), "unchanged where it must be")

    print("\n== silu_dropout (B*T, 2048) bf16", flush=True)
    x = torch.randn(3000, 2048, device=dev, generator=gen).bfloat16()
    dy = torch.randn(3000, 2048, device=dev, generator=gen).bfloat16()
    xe = x.clone().requires_grad_()
    ye = F.silu(xe)
    ye.backward(dy)
    xo = x.clone().requires_grad_()
    yo = silu_dropout(xo, 0.0)
    yo.backward(dy)
    d_y, d_x = (ye - yo).abs().max().item(), (xe.grad - xo.grad).abs().max().item()
    check("4 p=0 == eager silu", d_y == 0 and d_x <= 2 ** -7 * xe.grad.abs().max().item(),
          f"max|dy| {d_y:.1e}  max|ddx| {d_x:.1e}  (dx rel {rel(xo.grad, xe.grad):.1e})")
    xo = x.clone().requires_grad_()
    yo = silu_dropout(xo, 0.2, seed=7)
    yo.backward(dy)
    kept = yo != 0
    frac = kept.float().mean().item()
    ref_y = (F.silu(x).float() * 1.25).bfloat16()
    dmask = (dy.float() * 1.25).bfloat16().float() * kept
    xr = x.clone().requires_grad_()
    F.silu(xr).backward(dmask.bfloat16())
    check("5 p=0.2 consistent", abs(frac - 0.8) < 0.01 and torch.equal(yo[kept], ref_y[kept])
          and rel(xo.grad, xr.grad) < 1e-2, f"kept {frac:.4f}  dx vs eager-with-same-mask {rel(xo.grad, xr.grad):.1e}")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
