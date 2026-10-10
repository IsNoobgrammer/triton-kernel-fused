"""selfcond_ctc_pass (kernels/sm120/selfcond_ctc.py) vs an eager PyTorch reference of the same pass.

Reference: z = x W_out^T + b (bf16 GEMM, as autocast), log_softmax / softmax in fp32, torch ctc_loss
(reduction='none', zero_infinity), q = softmax(z).bf16 @ W_fb^T. Loss = sum w_b nll_b + <R, q> (R random, so the
feedback path gets a gradient). Ground truth = the same reference in fp64 from the bf16-rounded inputs. Gates:
  1. nll and every gradient (x, W_out, b, W_fb) at least as close to fp64 as the eager bf16/fp32 reference (x1.5)
  2. bitwise repeatable (two runs)
  3. infeasible sample (T too short): nll 0, no CTC gradient (zero_infinity)
Cases: small V with repeated labels + padding + an infeasible sample; with and without feedback (last pass).
Then a timing at the ASR training shape (B 150, T 101, d 512, U 30, V 2049 / 4097): fused vs eager, fwd+bwd.

    python parity_check/parity_selfcond_ctc.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

from kernels.sm120.selfcond_ctc import selfcond_ctc_pass

dev = "cuda"
ok = True


def eager(x, w_out, b, w_fb, y, tl_, ul, blank, dt):
    """The pass in plain torch; dt = compute dtype of the softmax / ctc (fp32 = what training does, fp64 = truth)."""
    B, T, d = x.shape
    if dt == torch.float64:
        z = (x.double() @ w_out.double().t() + b.double())
    else:
        z = (x.to(torch.bfloat16) @ w_out.to(torch.bfloat16).t() + b.to(torch.bfloat16)).float()
    lp = z.log_softmax(-1)
    nll = F.ctc_loss(lp.transpose(0, 1), y, tl_, ul, blank=blank, reduction="none", zero_infinity=True)
    q = None
    if w_fb is not None:
        pz = z.softmax(-1)
        q = pz.double() @ w_fb.double().t() if dt == torch.float64 else pz.to(torch.bfloat16) @ w_fb.to(torch.bfloat16).t()
    return nll, q


def run(fn, x, w_out, b, w_fb, y, tl_, ul, blank, wts, R, dt=None):
    leaves = [t.detach().clone().requires_grad_(True) for t in (x, w_out, b)] + \
        ([w_fb.detach().clone().requires_grad_(True)] if w_fb is not None else [])
    xx, wo, bb = leaves[:3]
    wf = leaves[3] if w_fb is not None else None
    nll, q = fn(xx, wo, bb, wf, y, tl_, ul, blank) if dt is None else fn(xx, wo, bb, wf, y, tl_, ul, blank, dt)
    loss = (nll.double() * wts).sum() + ((q.double() * R).sum() if q is not None else 0.0)
    loss.backward()
    return [nll.detach().double()] + [l.grad.double() for l in leaves]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


def case(name, B, T, d, V, U, feedback, seed):
    global ok
    g = torch.Generator(device=dev).manual_seed(seed)
    rnd = lambda *s, sc=1.0: (torch.randn(*s, device=dev, generator=g) * sc)  # noqa: E731
    x = rnd(B, T, d).to(torch.bfloat16).float()
    w_out = (rnd(V, d, sc=d ** -0.5) * 3).to(torch.bfloat16).float()
    b = rnd(V, sc=0.1).to(torch.bfloat16).float()
    w_fb = rnd(2 * d, V, sc=V ** -0.5).to(torch.bfloat16).float() if feedback else None
    blank = V - 1
    y = torch.randint(0, min(V - 1, 5), (B, U), device=dev, generator=g)             # few tokens: repeats
    ul = torch.randint(U // 2, U + 1, (B,), device=dev, generator=g)
    ul[0] = U
    tl_ = torch.randint(T // 2, T + 1, (B,), device=dev, generator=g)
    tl_[0] = T
    tl_[1] = 2                                                                         # infeasible (U >= 2)
    ul[1] = max(U, 3)
    wts = torch.rand(B, device=dev, generator=g, dtype=torch.float64)
    R = torch.randn(B, T, 2 * d, device=dev, generator=g, dtype=torch.float64) if feedback else None
    truth = run(eager, x, w_out, b, w_fb, y, tl_, ul, blank, wts, R, torch.float64)
    cur = run(eager, x, w_out, b, w_fb, y, tl_, ul, blank, wts, R, torch.float32)
    k1 = run(selfcond_ctc_pass, x, w_out, b, w_fb, y, tl_, ul, blank, wts, R)
    k2 = run(selfcond_ctc_pass, x, w_out, b, w_fb, y, tl_, ul, blank, wts, R)
    names = ["nll", "dx", "dW_out", "db"] + (["dW_fb"] if feedback else [])
    print(f"\n{name}: B {B} T {T} d {d} V {V} U {U} feedback {feedback}")
    for n, t_, c_, k_ in zip(names, truth, cur, k1):
        ek, ec = rel(k_, t_), rel(c_, t_)
        good = ek <= max(1.5 * ec, 1e-6)
        ok &= good
        print(f"  {n:7s} kernel vs fp64 {ek:.2e} | eager vs fp64 {ec:.2e}  {'ok' if good else 'FAIL'}")
    rep = all(torch.equal(a, c) for a, c in zip(k1, k2))
    ok &= rep
    print(f"  bitwise repeatable: {rep}")
    print(f"  infeasible sample: nll {k1[0][1].item():.3g} (expect 0)")
    ok &= k1[0][1].item() == 0.0


case("small V, repeats, padding, infeasible", B=6, T=40, d=64, V=9, U=10, feedback=True, seed=0)
case("small V, last pass (no feedback)", B=6, T=40, d=64, V=9, U=10, feedback=False, seed=1)
case("mid", B=8, T=101, d=512, V=2049, U=30, feedback=True, seed=2)


def bench(V, feedback):
    B, T, d, U = 150, 101, 512, 30
    x = torch.randn(B, T, d, device=dev, dtype=torch.bfloat16, requires_grad=True)
    w_out = (torch.randn(V, d, device=dev) * d ** -0.5).requires_grad_(True)
    b = torch.zeros(V, device=dev, requires_grad=True)
    w_fb = (torch.randn(2 * d, V, device=dev) * V ** -0.5).requires_grad_(True) if feedback else None
    y = torch.randint(0, V - 1, (B, U), device=dev)
    tl_ = torch.full((B,), T, device=dev)
    ul = torch.full((B,), U, device=dev)
    R = torch.randn(B, T, 2 * d, device=dev, dtype=torch.bfloat16) if feedback else None

    def step(fn):
        nll, q = fn(x, w_out, b, w_fb, y, tl_, ul, V - 1)
        loss = nll.sum() + ((q.float() * R).sum() if q is not None else 0.0)
        loss.backward()

    def eag(x, w_out, b, w_fb, y, tl_, ul, blank):
        return eager(x, w_out, b, w_fb, y, tl_, ul, blank, torch.float32)

    out = {}
    for name, fn in (("eager", eag), ("fused", selfcond_ctc_pass)):
        for _ in range(3):
            step(fn)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        for _ in range(10):
            step(fn)
        torch.cuda.synchronize()
        out[name] = (1000 * (time.perf_counter() - t0) / 10, torch.cuda.max_memory_allocated() / 2 ** 20)
    print(f"  V {V} feedback {feedback}: eager {out['eager'][0]:6.2f} ms ({out['eager'][1]:.0f} MB peak) | "
          f"fused {out['fused'][0]:6.2f} ms ({out['fused'][1]:.0f} MB peak) | {out['eager'][0] / out['fused'][0]:.2f}x")


print("\ntiming, one pass fwd+bwd at B 150 x T 101 (1,200 s of audio), d 512, U 30:")
for V in (2049, 4097):
    for fb in (True, False):
        bench(V, fb)
print("\nPARITY_OK" if ok else "\nPARITY_FAIL")
