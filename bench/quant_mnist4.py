"""4-bit on the MNIST residual MLP: quantized TRAINING (W4A8 / W4A4) vs PTQ vs QAT.

    python -m bench.quant_mnist4 [--epochs 4] [--seeds 3]

Same model / data / optimizer as bench.quant_mnist (the MLP-block GEMMs are the quantized ones).
  MXFP8        e4m3 + e8m0 / 32 on every operand (the 8-bit reference)
  W4A8 X       weights X (MXFP4 = e2m1 + e8m0 / 32, NVFP4 = e2m1 + e4m3 / 16 x fp32 tensor scale),
               activations and gradients MXFP8, for the WHOLE run
  W4A4 X [+H]  every operand X, optionally block-Hadamard on each GEMM's reduction dim
  PTQ X        fp32 training, weights rounded to X only at test time
  QAT X        fp32 training, weights rounded to X for the LAST 20% of steps (straight-through:
               the backward uses the rounded weights for dx, dW goes to the fp32 master), test with X
"""
import argparse
import math
import statistics
import time

import torch
import torch.nn.functional as F

import bench.quant_mnist as Q

MXFP8, MXFP4, NVFP4 = ("e4m3", "e8m0", 32), ("e2m1", "e8m0", 32), ("e2m1", "2L-e4m3", 16)
STATE = {"wq_on": True}
_HAD = {}


def had(t, dim, n):
    """Block-Hadamard along dim (orthogonal; applied to both GEMM operands it leaves the product exact)."""
    if not n:
        return t
    H = _HAD.get(n)
    if H is None:
        H = torch.ones(1, 1, device=t.device)
        while H.shape[0] < n:
            H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
        H = _HAD[n] = H / math.sqrt(n)
    x = t.float().movedim(dim, -1)
    return (x.reshape(*x.shape[:-1], -1, n) @ H).reshape(*x.shape).movedim(-1, dim)


class Q4(torch.autograd.Function):
    """cfg = (act/grad recipe or None, weight recipe or None, hadamard n, gated, track)."""

    @staticmethod
    def forward(ctx, x, w, cfg):
        ctx.save_for_backward(x, w)
        ctx.cfg = cfg
        ar, wr, h, gated, track = cfg
        wr = wr if (wr is not None and (not gated or STATE["wq_on"])) else None
        ctx.wr = wr
        xa = had(x, 1, h) if ar is None else Q.fq(had(x, 1, h), 1, *ar, "x" if track else None)
        wa = had(w, 1, h) if wr is None else Q.fq(had(w, 1, h), 1, *wr, "W" if track else None)
        return xa @ wa.t()

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        ar, _, h, _, track = ctx.cfg
        wr = ctx.wr

        def qa(t, d, tag):
            return had(t, d, h) if ar is None else Q.fq(had(t, d, h), d, *ar, tag)
        wq = had(w, 0, h) if wr is None else Q.fq(had(w, 0, h), 0, *wr, None)
        dx = qa(dy, 1, "dy" if track else None) @ wq
        dw = qa(dy, 0, None).t() @ qa(x, 0, None)
        return dx, dw, None


def run(arm, seed, tr, te, epochs, bs=256, lr=2e-3):
    ar, wr, h, qat = arm["a"], arm["w"], arm.get("h", 0), arm.get("qat")
    torch.manual_seed(seed)
    net = Q.Net().to(Q.dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    xs, ys = tr
    per = xs.shape[0] // bs
    steps = epochs * per
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator(device=Q.dev).manual_seed(seed)
    losses = []
    gated = qat is not None
    for s in range(steps):
        if s % per == 0:
            perm = torch.randperm(xs.shape[0], device=Q.dev, generator=g)
        STATE["wq_on"] = (not gated) or s >= qat * steps
        i = perm[(s % per) * bs:][:bs]
        loss = F.cross_entropy(net(xs[i], (ar, wr, h, gated, s % 50 == 0)), ys[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        losses.append(loss.detach())
    STATE["wq_on"] = True                       # test WITH the quantized weights (PTQ / QAT / W4 arms)
    with torch.no_grad():
        lo = net(te[0], (ar, wr, h, gated, False))
        tl_, acc = F.cross_entropy(lo, te[1]).item(), (lo.argmax(-1) == te[1]).float().mean().item()
    return torch.stack(losses[-100:]).mean().item(), tl_, acc, not math.isfinite(tl_)


ARMS = [("fp32", dict(a=None, w=None)),
        ("MXFP8", dict(a=MXFP8, w=MXFP8)),
        ("W4A8 MXFP4", dict(a=MXFP8, w=MXFP4)),
        ("W4A8 NVFP4", dict(a=MXFP8, w=NVFP4)),
        ("W4A4 MXFP4", dict(a=MXFP4, w=MXFP4)),
        ("W4A4 NVFP4", dict(a=NVFP4, w=NVFP4)),
        ("W4A4 MXFP4 +H32", dict(a=MXFP4, w=MXFP4, h=32)),
        ("W4A4 NVFP4 +H16", dict(a=NVFP4, w=NVFP4, h=16)),
        ("PTQ MXFP4 (W only)", dict(a=None, w=MXFP4, qat=1.0)),
        ("PTQ NVFP4 (W only)", dict(a=None, w=NVFP4, qat=1.0)),
        ("QAT MXFP4 last 20%", dict(a=None, w=MXFP4, qat=0.8)),
        ("QAT NVFP4 last 20%", dict(a=None, w=NVFP4, qat=0.8))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    Q.QLinearFn = Q4                            # Net's blocks call quant_mnist.QLinearFn
    tr, te = Q.data()
    print(f"{'arm':22s} {'train loss':>10s} {'test loss':>10s} {'test acc':>9s} {'x zero%':>8s} {'W zero%':>8s} "
          f"{'dy zero%':>9s} {'s/run':>6s}", flush=True)
    base = None
    for name, arm in ARMS:
        Q.STATS.clear()
        t0 = time.time()
        res = [run(arm, 1000 + sd, tr, te, a.epochs) for sd in range(a.seeds)]
        ok = [r for r in res if not r[3]]
        st = lambda k: (100 * Q.STATS[k][0] / Q.STATS[k][2]) if k in Q.STATS and Q.STATS[k][2] else float("nan")
        if not ok:
            print(f"{name:22s} all seeds diverged", flush=True)
            continue
        tel = statistics.mean(r[1] for r in ok)
        base = tel if base is None else base
        print(f"{name:22s} {statistics.mean(r[0] for r in ok):10.4f} {tel:10.4f} {100 * statistics.mean(r[2] for r in ok):8.2f}% "
              f"{st('x'):8.3f} {st('W'):8.3f} {st('dy'):9.3f} {(time.time() - t0) / a.seeds:6.1f}"
              f"   dtest {tel - base:+.4f} (sd {statistics.pstdev(r[1] for r in ok):.4f})"
              + (f"  DIVERGED {len(res) - len(ok)}/{len(res)}" if len(ok) < len(res) else ""), flush=True)
    print("QUANT_MNIST4_DONE")


if __name__ == "__main__":
    main()
