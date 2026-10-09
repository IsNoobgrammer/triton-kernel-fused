"""ctc_loss (kernels/sm120/ctc_loss.py) vs torch.nn.functional.ctc_loss(reduction='none', zero_infinity=True).

Ground truth = torch in fp64; "current" = torch CUDA fp32 (what NeMo trains with). Small V forces repeated labels
(the s-2 skip rule and colliding gradient writes), plus an empty target, an infeasible sample (T < U + repeats) and
padded time steps. Gates:
  1. loss and grad (w.r.t. log_probs, per-sample weights) at least as close to fp64 as torch fp32 (x1.5, or 1e-6)
  2. bitwise repeatable (torch fp32 is not: its backward uses atomics)
  3. infeasible sample: loss 0 and grad 0 (zero_infinity)
Then a timing at the ASR shape (B 100, T 375, U 150, V 4097, fp32 log_probs).

    python parity_check/parity_ctc_loss.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import triton

from kernels.sm120.ctc_loss import ctc_loss

dev = "cuda"
ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-300)).item()


def torch_ctc(lp, y, tl_, ul, blank):
    return F.ctc_loss(lp.transpose(0, 1), y, tl_, ul, blank=blank, reduction="none", zero_infinity=True)


def run(fn, lp, y, tl_, ul, blank, w, dtype):
    x = lp.detach().to(dtype).clone().requires_grad_()
    nll = fn(x, y, tl_, ul, blank)
    (nll.double() * w).sum().backward()
    return nll.detach(), x.grad


def data(B, T, U, V, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    lp = torch.log_softmax(torch.randn(B, T, V, device=dev, generator=g) * 2, -1)
    y = torch.randint(0, V - 1, (B, U), device=dev, generator=g)
    ul = torch.randint(1, U + 1, (B,), device=dev, generator=g)
    tl_ = torch.randint(T // 2, T + 1, (B,), device=dev, generator=g)
    ul[0], tl_[0] = U, T
    ul[1] = 0                                                       # empty target
    ul[2], tl_[2] = U, max(1, U // 3)                               # infeasible
    w = torch.rand(B, device=dev, generator=g, dtype=torch.float64) + 0.5
    return lp, y, tl_, ul, w


def main():
    for (B, T, U, V) in ((6, 120, 30, 7), (5, 90, 25, 40), (4, 200, 60, 300)):
        blank = V - 1
        lp, y, tl_, ul, w = data(B, T, U, V, seed=B * T)
        print(f"\n== B={B} T={T} U={U} V={V} (blank {blank}) lengths T {tl_.tolist()} U {ul.tolist()}", flush=True)
        ours = lambda *a: ctc_loss(*a, zero_infinity=True)
        gt = run(torch_ctc, lp, y, tl_, ul, blank, w, torch.float64)
        tf = run(torch_ctc, lp, y, tl_, ul, blank, w, torch.float32)
        ou = run(ours, lp, y, tl_, ul, blank, w, torch.float32)
        bad = [f"{n} ours {rel(a, g):.2e} torch {rel(c, g):.2e}" for n, a, c, g in zip(("loss", "grad"), ou, tf, gt)
               if rel(a, g) > max(1.5 * rel(c, g), 1e-6)]
        check("1 vs fp64 <= torch fp32", not bad, "; ".join(bad) or
              "  ".join(f"{n} ours {rel(a, g):.1e} torch {rel(c, g):.1e}" for n, a, c, g in zip(("loss", "grad"), ou, tf, gt)))
        ou2 = run(ours, lp, y, tl_, ul, blank, w, torch.float32)
        tf2 = run(torch_ctc, lp, y, tl_, ul, blank, w, torch.float32)
        check("2 repeatable", torch.equal(ou[0], ou2[0]) and torch.equal(ou[1], ou2[1]),
              f"bitwise (torch fp32 grad repeatable: {torch.equal(tf[1], tf2[1])})")
        check("3 infeasible -> 0", ou[0][2].item() == 0 and bool((ou[1][2] == 0).all()) and gt[0][2].item() == 0,
              f"loss {ou[0][2].item()}, grad max {ou[1][2].abs().max().item()}")

    print("\n== timing at the ASR shape: B=100 T=375 U=150 V=4097 fp32", flush=True)
    B, T, U, V = 100, 375, 150, 4097
    lp, y, tl_, ul, w = data(B, T, U, V, seed=1)
    ul.clamp_(min=1)
    tl_[2] = T
    w = w.float()
    x = lp.clone().requires_grad_()

    def fb(fn):
        def go():
            x.grad = None
            (fn(x, y, tl_, ul, V - 1) * w).sum().backward()
        return go
    tt = triton.testing.do_bench(fb(torch_ctc))
    to = triton.testing.do_bench(fb(lambda *a: ctc_loss(*a, zero_infinity=True)))
    tf_ = triton.testing.do_bench(lambda: torch_ctc(x, y, tl_, ul, V - 1))
    of_ = triton.testing.do_bench(lambda: ctc_loss(x, y, tl_, ul, V - 1, True))
    print(f"  torch fwd {tf_:.3f} ms  fwd+bwd {tt:.3f} ms | ours fwd {of_:.3f} ms  fwd+bwd {to:.3f} ms "
          f"({tt / to:.2f}x)", flush=True)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
