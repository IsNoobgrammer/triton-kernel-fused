"""Fused TDT joint + loss (kernels/sm120/tdt_joint.py) vs an fp64 ground truth and vs NeMo's current path.

Ground truth: fp64 joint (relu -> dropout -> linear, V token + D duration outputs), token log-softmax - sigma,
duration log-softmax, TDT alpha / beta by explicit loops with NeMo's arc rules (blank: duration >= 1, the last arc a
blank landing exactly on T; label: duration >= 0, landing < T), and the analytic gradient
    token dlogit = occ p_v - gb [blank] - gy [label],   duration dlogit = occ p_d - gd,   occ = gb + gy
chained into f, g, W, bias by autograd. "NeMo bf16" = what training runs without the kernel: autocast joint +
RNNTLoss(loss_name="tdt") (numba) on the fp32-cast logits.

Gates (dropout 0 and 0.2, durations 0-4, sigma 0.05, one empty transcript in the batch):
  0. GT == NeMo TDT numba in fp32 (loss and every grad, rel < 1e-3): the reference is what we train with
  1. ours: loss, df, dg, dW, dbias as close to GT as NeMo bf16 per case (x1.3: both round through bf16 at
     different points, one draw can go either way; or rel < 2e-3), and
  1b. over 8 seeds x 2 shapes, ours' MEAN error <= NeMo bf16's for every quantity (measured Oct 12 2026: ours <= NeMo
     in df 16/16, dg 16/16, dW 15/16, dbias 14/16 cases; worst single-case ratio 1.26)
  (out-of-window: gate 0 is informational -- NeMo's own fp32 TDT gradient is 0.39 off fp64 at saturated logits
   while the loss agrees to 3e-8; ours stays at 2.5e-2, gate 1 still applies)
  2. bitwise repeatable
  3. E recomputed in the backward (e_budget=0) still meets gate 1
  4. no-grad loss == grad-path loss (dropout 0)
  5. omega = 1 (NeMo's sampled RNN-T step) == rnnt_joint_loss on the token rows, duration rows' grads exactly 0

    python parity_check/parity_tdt_joint.py [--big]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as Fn

import kernels.sm120.rnnt_joint as rj
import kernels.sm120.tdt_joint as tj

dev = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
DUR = [0, 1, 2, 3, 4]
SIG = 0.05
ok = True
NEG = float("-inf")


def data(B, T, U, H, V, wscale=1.0, seed=0):
    ND = len(DUR)
    gen = torch.Generator(device=dev).manual_seed(seed)
    tl = torch.randint(max(1, T // 2), T + 1, (B,), device=dev, generator=gen)
    tl[0] = T
    yl = torch.randint(max(0, U // 2), U + 1, (B,), device=dev, generator=gen)
    yl[0] = U
    yl[-1] = 0                                                     # an empty transcript
    y = torch.randint(0, V - 1, (B, U), device=dev, generator=gen)
    f = torch.randn(B, T, H, device=dev, generator=gen).to(torch.bfloat16)
    g = torch.randn(B, U + 1, H, device=dev, generator=gen).to(torch.bfloat16)
    W = torch.randn(V + ND, H, device=dev, generator=gen) * H ** -0.5 * wscale
    b = torch.randn(V + ND, device=dev, generator=gen) * 0.1
    return f, g, W, b, y, tl, yl


def keep_mask(f, g, y, tl, yl, p, seed):
    B, T, H = f.shape
    U1 = g.shape[1]
    X, _, off, _ = rj._pack(torch.ones_like(f), torch.zeros_like(g), y, tl.long(), yl.long(), p, seed)
    m = torch.zeros(B, T, U1, H, device=dev, dtype=torch.bfloat16)
    for i in range(B):
        t_, u1 = int(tl[i]), int(yl[i]) + 1
        m[i, :t_, :u1] = X[int(off[i]):int(off[i]) + t_ * u1, :H].view(t_, u1, H)
    return m


def lse(terms):
    return torch.logsumexp(torch.stack(terms), 0) if terms else torch.tensor(NEG, dtype=torch.float64)


def tdt_lattice(lpb, lpy, lpd, T, U):
    """fp64 alpha / beta / ll by explicit loops (cpu), NeMo's arc rules."""
    a = torch.full((T, U + 1), NEG, dtype=torch.float64)
    be = torch.full((T, U + 1), NEG, dtype=torch.float64)
    for t in range(T):
        for u in range(U + 1):
            if t == 0 and u == 0:
                a[0, 0] = 0.0
                continue
            x = []
            for i, d in enumerate(DUR):
                if t >= d:
                    if d > 0:
                        x.append(a[t - d, u] + lpb[t - d, u] + lpd[t - d, u, i])
                    if u > 0:
                        x.append(a[t - d, u - 1] + lpy[t - d, u - 1] + lpd[t - d, u - 1, i])
            a[t, u] = lse(x)
    ll = lse([a[T - d, U] + lpb[T - d, U] + lpd[T - d, U, i] for i, d in enumerate(DUR) if d > 0 and T >= d])
    for t in range(T - 1, -1, -1):
        for u in range(U, -1, -1):
            x = []
            for i, d in enumerate(DUR):
                if d > 0:
                    if t + d < T:
                        x.append(be[t + d, u] + lpb[t, u] + lpd[t, u, i])
                    elif t + d == T and u == U:
                        x.append(lpb[t, u] + lpd[t, u, i])
                if u < U and t + d < T:
                    x.append(be[t + d, u + 1] + lpy[t, u] + lpd[t, u, i])
            be[t, u] = lse(x)
    return a, be, ll


def ground_truth(f, g, W, b, y, tl, yl, keep):
    ND = len(DUR)
    f, g, W, b = (x.double().detach().requires_grad_() for x in (f, g, W, b))
    x = torch.relu(f[:, :, None] + g[:, None])
    if keep is not None:
        x = x * keep.double()
    logits = x @ W.t() + b
    V = W.shape[0] - ND
    blank = V - 1
    lt = logits.detach()[..., :V].log_softmax(-1).cpu()
    ld = logits.detach()[..., V:].log_softmax(-1).cpu()
    B = f.shape[0]
    dlog = torch.zeros(logits.shape, dtype=torch.float64)
    lls = []
    for i in range(B):
        T, U = int(tl[i]), int(yl[i])
        yy = y[i, :U].cpu()
        lpb = lt[i, :T, :U + 1, blank] - SIG
        lpy = torch.full((T, U + 1), NEG, dtype=torch.float64)
        if U:
            lpy[:, :U] = lt[i, :T, torch.arange(U), yy] - SIG
        lpd = ld[i, :T, :U + 1]
        a, be, ll = tdt_lattice(lpb, lpy, lpd, T, U)
        gb = torch.zeros(T, U + 1, dtype=torch.float64)
        gy = torch.zeros(T, U + 1, dtype=torch.float64)
        gd = torch.zeros(T, U + 1, len(DUR), dtype=torch.float64)
        for t in range(T):
            for u in range(U + 1):
                for k, d in enumerate(DUR):
                    pb = py = 0.0
                    if d > 0 and t + d < T:
                        pb = torch.exp(a[t, u] + lpb[t, u] + lpd[t, u, k] + be[t + d, u] - ll)
                    if d > 0 and t + d == T and u == U:
                        pb = torch.exp(a[t, u] + lpb[t, u] + lpd[t, u, k] - ll)
                    if u < U and t + d < T:
                        py = torch.exp(a[t, u] + lpy[t, u] + lpd[t, u, k] + be[t + d, u + 1] - ll)
                    gb[t, u] += pb
                    gy[t, u] += py
                    gd[t, u, k] = pb + py
        occ = gb + gy
        dt = lt[i, :T, :U + 1].exp() * occ[..., None]
        dt[..., blank] -= gb
        if U:
            dt[:, torch.arange(U), yy] -= gy[:, :U]
        dd = ld[i, :T, :U + 1].exp() * occ[..., None] - gd
        dlog[i, :T, :U + 1, :V] = dt / B
        dlog[i, :T, :U + 1, V:] = dd / B
        lls.append(ll)
    logits.backward(dlog.to(dev))
    return -torch.stack(lls).mean(), f.grad, g.grad, W.grad, b.grad


def nemo_path(f, g, W, b, y, tl, yl, keep, amp):
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    ND = len(DUR)
    loss_fn = RNNTLoss(num_classes=W.shape[0] - ND - 1, reduction="mean_batch", loss_name="tdt",
                       loss_kwargs=dict(durations=list(DUR), sigma=SIG, omega=0.0, fastemit_lambda=0.0, clamp=-1.0))
    dt = torch.float32 if not amp else f.dtype
    f, g = (x.to(dt).detach().requires_grad_() for x in (f, g))
    W, b = (x.detach().clone().requires_grad_() for x in (W, b))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        x = torch.relu(f[:, :, None] + g[:, None])
        if keep is not None:
            x = x * keep.to(x.dtype)
        logits = Fn.linear(x, W, b)
    loss = loss_fn(log_probs=logits.float(), targets=y, input_lengths=tl, target_lengths=yl)
    loss.backward()
    return loss.detach(), f.grad, g.grad, W.grad, b.grad


def ours(f, g, W, b, y, tl, yl, p, seed, e_budget=rj._E_BUDGET, grad=True, omega=0.0):
    f, g = (x.detach().clone().requires_grad_(grad) for x in (f, g))
    W, b = (x.detach().clone().requires_grad_(grad) for x in (W, b))
    with AMP, torch.set_grad_enabled(grad):
        loss, _ = tj.tdt_joint_loss(f, g, W, b, y, tl, yl, durations=DUR, sigma=SIG, omega=omega, dropout=p,
                                    seed=seed, e_budget=e_budget)
    if not grad:
        return (loss.detach(),)
    loss.backward()
    return loss.detach(), f.grad, g.grad, W.grad, b.grad


def rel(a, ref):
    a, ref = a.double(), ref.double()
    return ((a - ref).norm() / ref.norm().clamp(min=1e-300)).item()


NAMES = ("loss", "df", "dg", "dW", "dbias")


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


SLACK = 1.3


def case(name, B, T, U, H, V, p, wscale=1.0):
    print(f"\n== {name}: B={B} T={T} U={U} H={H} V={V}+{len(DUR)} dropout={p} wscale={wscale}", flush=True)
    f, g, W, b, y, tl, yl = data(B, T, U, H, V, wscale)
    seed = 1234
    keep = keep_mask(f, g, y, tl, yl, p, seed) if p > 0 else None
    gt = ground_truth(f, g, W, b, y, tl, yl, keep)
    try:
        nf32 = nemo_path(f, g, W, b, y, tl, yl, keep, amp=False)
        e0 = [rel(a, r) for a, r in zip(nf32, gt)]
        if wscale > 10:
            print("  [info] 0 GT vs NeMo TDT fp32 (saturated logits): " + " ".join(f"{n} {e:.1e}" for n, e in zip(NAMES, e0)))
        else:
            check("0 GT == NeMo TDT fp32", max(e0) < 1e-3, " ".join(f"{n} {e:.1e}" for n, e in zip(NAMES, e0)))
        nb = nemo_path(f, g, W, b, y, tl, yl, keep, amp=True)
        en = [rel(a, r) for a, r in zip(nb, gt)]
    except ImportError:
        print("  (no NeMo: gate 0 skipped, gate 1 uses rel < 2e-3 only)")
        en = [0.0] * 5
    o = ours(f, g, W, b, y, tl, yl, p, seed)
    eo = [rel(a, r) for a, r in zip(o, gt)]
    for n, a, c in zip(NAMES, eo, en):
        check(f"1 {n}", a <= max(c * SLACK, 2e-3), f"ours {a:.2e}  nemo-bf16 {c:.2e}")
    o2 = ours(f, g, W, b, y, tl, yl, p, seed)
    check("2 repeatable", all(torch.equal(a, c) for a, c in zip(o, o2)), "bitwise")
    o3 = ours(f, g, W, b, y, tl, yl, p, seed, e_budget=0)
    e3 = [rel(a, r) for a, r in zip(o3, gt)]
    check("3 recompute meets gate 1", all(a <= max(c * SLACK, 2e-3) for a, c in zip(e3, en)),
          " ".join(f"{n} {e:.1e}" for n, e in zip(NAMES, e3)))
    if p == 0:
        o4 = ours(f, g, W, b, y, tl, yl, 0.0, seed, grad=False)
        check("4 no-grad loss", rel(o4[0], o[0]) < 1e-6, f"{o4[0].item():.7f} vs {o[0].item():.7f}")
    # 5. omega = 1: NeMo's sampled plain RNN-T step on the token rows
    o5 = ours(f, g, W, b, y, tl, yl, p, seed, omega=1.0)
    Vt = W.shape[0] - len(DUR)
    fr, gr = (x.detach().clone().requires_grad_() for x in (f, g))
    Wr, br = (x[:Vt].detach().clone().requires_grad_() for x in (W, b))
    with AMP:
        lr_, _ = rj.rnnt_joint_loss(fr, gr, Wr, br, y, tl, yl, 0.0, p, seed)
    lr_.backward()
    same = (torch.equal(o5[0], lr_.detach()) and torch.equal(o5[1], fr.grad) and torch.equal(o5[2], gr.grad)
            and torch.equal(o5[3][:Vt], Wr.grad) and torch.equal(o5[4][:Vt], br.grad))
    zero = bool((o5[3][Vt:] == 0).all() and (o5[4][Vt:] == 0).all())
    check("5 omega=1 == RNN-T on token rows, durations 0", same and zero, f"equal {same}, duration grads zero {zero}")


def seeds_mean():
    """gate 1b: mean error over 8 seeds x 2 shapes, ours vs NeMo bf16 (dropout 0)."""
    print("\n== 1b: mean over 8 seeds x 2 shapes", flush=True)
    acc = {n: [0.0, 0.0] for n in NAMES}
    k = 0
    for seed in range(8):
        for (B, T, U, H, V) in ((4, 9, 5, 64, 33), (3, 13, 7, 100, 61)):
            f, g, W, b, y, tl, yl = data(B, T, U, H, V, seed=seed)
            gt = ground_truth(f, g, W, b, y, tl, yl, None)
            nb = nemo_path(f, g, W, b, y, tl, yl, None, amp=True)
            o = ours(f, g, W, b, y, tl, yl, 0.0, 1234)
            for n, a, c, r in zip(NAMES, o, nb, gt):
                acc[n][0] += rel(a, r)
                acc[n][1] += rel(c, r)
            k += 1
    for n, (a, c) in acc.items():
        check(f"1b mean {n}", a <= c * 1.0 or a / k < 2e-4, f"ours {a / k:.2e}  nemo-bf16 {c / k:.2e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="add the real joint size (H 640, V 2048 + 5)")
    a = ap.parse_args()
    for p in (0.0, 0.2):
        case("small", 4, 9, 5, 64, 33, p)
        case("odd", 3, 13, 7, 100, 61, p)
    case("out-of-window", 3, 9, 5, 64, 33, 0.0, wscale=200.0)
    try:
        seeds_mean()
    except ImportError:
        print("  (no NeMo: gate 1b skipped)")
    if a.big:
        for p in (0.0, 0.2):
            case("real", 3, 30, 12, 640, 2048, p)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
