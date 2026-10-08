"""Fused RNN-T joint + loss (kernels/sm120/rnnt_joint.py) vs an fp64 ground truth and vs NeMo's current path.

Ground truth: fp64 joint (relu -> dropout -> linear), log-softmax, alpha/beta by explicit loops, and NeMo's numba
gradient formula (FastEmit included) chained into f, g, W, bias by autograd.
"NeMo bf16" = what training runs today: autocast joint + RNNTLoss(warprnnt_numba) on the fp32-cast logits.

Gates (each at dropout 0 and 0.2, FastEmit 0.005, one empty transcript in the batch):
  0. GT == NeMo numba in fp32 (loss and every grad, rel < 1e-5): the reference is what we train with
  1. ours: loss, df, dg, dW, dbias at least as close to GT as NeMo bf16 (x1.05 slack, or rel < 2e-3)
  2. bitwise repeatable
  3. E recomputed in the backward (e_budget=0) == E stored, bitwise
  4. no-grad loss == grad-path loss (to fp32 rounding: the no-grad pass max-shifts its sums)
  5. out-of-window logits (weights x200, forces the FIX pass) still meet gate 1

    python parity_check/parity_rnnt_joint.py [--big]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as Fn

import kernels.sm120.rnnt_joint as rj

dev = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)
LAM = 0.005
ok = True


def data(B, T, U, H, V, wscale=1.0, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    tl = torch.randint(max(1, T // 2), T + 1, (B,), device=dev, generator=gen)
    tl[0] = T
    yl = torch.randint(max(0, U // 2), U + 1, (B,), device=dev, generator=gen)
    yl[0] = U
    yl[-1] = 0                                                     # an empty transcript
    y = torch.randint(0, V - 1, (B, U), device=dev, generator=gen)
    f = torch.randn(B, T, H, device=dev, generator=gen).to(torch.bfloat16)
    g = torch.randn(B, U + 1, H, device=dev, generator=gen).to(torch.bfloat16)
    W = torch.randn(V, H, device=dev, generator=gen) * H ** -0.5 * wscale
    b = torch.randn(V, device=dev, generator=gen) * 0.1
    return f, g, W, b, y, tl, yl


def keep_mask(f, g, y, tl, yl, p, seed):
    """Our kernel's dropout multiplier (0 or 1/(1-p)) unpacked to (B, T, U+1, H): pack f=1, g=0."""
    B, T, H = f.shape
    U1 = g.shape[1]
    one, zero = torch.ones_like(f), torch.zeros_like(g)
    X, _, off, _ = rj._pack(one, zero, y, tl.long(), yl.long(), p, seed)
    m = torch.zeros(B, T, U1, H, device=dev, dtype=torch.bfloat16)
    for b in range(B):
        t_, u1 = int(tl[b]), int(yl[b]) + 1
        m[b, :t_, :u1] = X[int(off[b]):int(off[b]) + t_ * u1, :H].view(t_, u1, H)
    return m


def lattice(lpb, lpy, T, U):
    """fp64 alpha / beta by explicit loops (cpu)."""
    a = torch.full((T, U + 1), float("-inf"), dtype=torch.float64)
    be = torch.full((T, U + 1), float("-inf"), dtype=torch.float64)
    a[0, 0] = 0.0
    for t in range(T):
        for u in range(U + 1):
            if t == 0 and u == 0:
                continue
            x = []
            if t > 0:
                x.append(a[t - 1, u] + lpb[t - 1, u])
            if u > 0:
                x.append(a[t, u - 1] + lpy[t, u - 1])
            a[t, u] = torch.logsumexp(torch.stack(x), 0)
    be[T - 1, U] = lpb[T - 1, U]
    for t in range(T - 1, -1, -1):
        for u in range(U, -1, -1):
            if t == T - 1 and u == U:
                continue
            x = []
            if t < T - 1:
                x.append(be[t + 1, u] + lpb[t, u])
            if u < U:
                x.append(be[t, u + 1] + lpy[t, u])
            be[t, u] = torch.logsumexp(torch.stack(x), 0)
    return a, be


def ground_truth(f, g, W, b, y, tl, yl, keep):
    f, g, W, b = (x.double().detach().requires_grad_() for x in (f, g, W, b))
    x = torch.relu(f[:, :, None] + g[:, None])
    if keep is not None:
        x = x * keep.double()
    logits = x @ W.t() + b
    lp = logits.detach().log_softmax(-1).cpu()
    B, V = f.shape[0], W.shape[0]
    blank = V - 1
    dlog = torch.zeros_like(lp)
    lls = []
    for i in range(B):
        T, U = int(tl[i]), int(yl[i])
        yy = y[i, :U].cpu()
        lpb = lp[i, :T, :U + 1, blank]
        lpy = torch.full((T, U + 1), float("-inf"), dtype=torch.float64)
        if U:
            lpy[:, :U] = lp[i, :T, torch.arange(U), yy]
        a, be = lattice(lpb, lpy, T, U)
        ll = a[T - 1, U] + lpb[T - 1, U]
        bt = torch.full((T, U + 1), float("-inf"), dtype=torch.float64)
        bt[:-1] = be[1:]
        bt[T - 1, U] = 0.0
        bu = torch.full((T, U + 1), float("-inf"), dtype=torch.float64)
        bu[:, :-1] = be[:, 1:]
        gb = torch.exp(a + lpb + bt - ll)
        gy = (1 + LAM) * torch.exp(a + lpy + bu - ll)
        d = lp[i, :T, :U + 1].exp() * (gb + gy)[..., None]
        d[..., blank] -= gb
        if U:
            d[:, torch.arange(U), yy] -= gy[:, :U]
        dlog[i, :T, :U + 1] = d / B
        lls.append(ll)
    logits.backward(dlog.to(dev))
    return -torch.stack(lls).mean(), f.grad, g.grad, W.grad, b.grad


def nemo_path(f, g, W, b, y, tl, yl, keep, amp):
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    loss_fn = RNNTLoss(num_classes=W.shape[0] - 1, reduction="mean_batch", loss_name="warprnnt_numba",
                       loss_kwargs=dict(fastemit_lambda=LAM, clamp=-1.0))
    dt = torch.float32 if not amp else f.dtype
    f, g = (x.to(dt).detach().requires_grad_() for x in (f, g))
    W, b = (x.detach().clone().requires_grad_() for x in (W, b))
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        x = torch.relu(f[:, :, None] + g[:, None])
        if keep is not None:
            x = x * keep.to(x.dtype)
        logits = Fn.linear(x, W, b)
        loss = loss_fn(log_probs=logits, targets=y, input_lengths=tl, target_lengths=yl)
    loss.backward()
    return loss.detach(), f.grad, g.grad, W.grad, b.grad


def ours(f, g, W, b, y, tl, yl, p, seed, e_budget=rj._E_BUDGET, grad=True):
    f, g = (x.detach().clone().requires_grad_(grad) for x in (f, g))
    W, b = (x.detach().clone().requires_grad_(grad) for x in (W, b))
    with AMP, torch.set_grad_enabled(grad):
        loss, _ = rj.rnnt_joint_loss(f, g, W, b, y, tl, yl, fastemit_lambda=LAM, dropout=p, seed=seed,
                                     e_budget=e_budget)
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


def case(name, B, T, U, H, V, p, wscale=1.0):
    print(f"\n== {name}: B={B} T={T} U={U} H={H} V={V} dropout={p} wscale={wscale}", flush=True)
    f, g, W, b, y, tl, yl = data(B, T, U, H, V, wscale)
    seed = 1234
    keep = keep_mask(f, g, y, tl, yl, p, seed) if p > 0 else None
    gt = ground_truth(f, g, W, b, y, tl, yl, keep)
    try:
        nf32 = nemo_path(f, g, W, b, y, tl, yl, keep, amp=False)
        e0 = [rel(a, r) for a, r in zip(nf32, gt)]
        check("0 GT == NeMo fp32", max(e0) < 1e-5, " ".join(f"{n} {e:.1e}" for n, e in zip(NAMES, e0)))
        nb = nemo_path(f, g, W, b, y, tl, yl, keep, amp=True)
        en = [rel(a, r) for a, r in zip(nb, gt)]
    except ImportError:
        print("  (no NeMo: gate 0 skipped, gate 1 uses rel < 2e-3 only)")
        en = [0.0] * 5
    o = ours(f, g, W, b, y, tl, yl, p, seed)
    eo = [rel(a, r) for a, r in zip(o, gt)]
    for n, a, c in zip(NAMES, eo, en):
        check(f"1 {n}", a <= max(c * 1.05, 2e-3), f"ours {a:.2e}  nemo-bf16 {c:.2e}")
    o2 = ours(f, g, W, b, y, tl, yl, p, seed)
    check("2 repeatable", all(torch.equal(a, c) for a, c in zip(o, o2)), "bitwise")
    o3 = ours(f, g, W, b, y, tl, yl, p, seed, e_budget=0)
    check("3 recompute == stored", all(torch.equal(a, c) for a, c in zip(o, o3)), "bitwise")
    o4 = ours(f, g, W, b, y, tl, yl, 0.0, seed, grad=False)
    if p == 0:
        check("4 no-grad loss", rel(o4[0], o[0]) < 1e-6, f"{o4[0].item():.7f} vs {o[0].item():.7f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true", help="add the real joint size (H 640, V 4097)")
    a = ap.parse_args()
    for p in (0.0, 0.2):
        case("small", 4, 9, 5, 64, 33, p)
        case("odd", 3, 13, 7, 100, 61, p)
    case("out-of-window", 3, 9, 5, 64, 33, 0.0, wscale=200.0)
    if a.big:
        for p in (0.0, 0.2):
            case("real", 4, 40, 16, 640, 4097, p)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
