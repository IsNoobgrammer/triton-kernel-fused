"""Which 8-bit recipe trains best? MNIST residual MLP with the MLP blocks quantized in ALL three GEMMs.

    python -m bench.quant_mnist [--epochs 4] [--seeds 3]

Model: 784 -> 512 input layer (full precision), 4 residual blocks x + W2 silu(W1 rmsnorm(x)) with
512 -> 2048 -> 512 (QUANTIZED), 512 -> 10 head (full precision) -- the BiBo split in miniature:
embedding / head high precision, MLP (expert) GEMMs 8-bit. AdamW, bs 256, cosine lr.

Fake quantization (quantize -> dequantize in fp32, matmul in fp32 = an exact fp32 accumulator, which
the sm120 fp8 MMA was measured to have). Every operand of every GEMM is blocked along THAT GEMM's
reduction dim (1D blocks, never across tokens):
  fwd   y  = q(x  | blocks along in)  @ q(W  | along in)^T
  dgrad dx = q(dy | along out)        @ q(W  | along out)
  wgrad dW = q(dy^T | along tokens)   @ q(x^T | along tokens)
Grid: element e4m3 / e5m2 x scale e8m0 (power of two, rounded up) / fp32 / fp16 x block 32 / 64 / 128,
plus fp32 and bf16 baselines and one 2D-weight arm. Reported: final train loss (mean of the last
100 steps), test loss / accuracy (mean over seeds), and per-operand underflow (non-zero value ->
0) and saturation (|x / s| > max) rates averaged over training.
"""
import argparse
import itertools
import math
import statistics
import time

import torch
import torch.nn.functional as F

dev = "cuda"
FMT = {"e4m3": (torch.float8_e4m3fn, 448.0), "e5m2": (torch.float8_e5m2, 57344.0)}
STATS = {}


def fq(t, dim, fmt, scale, blk, tag):
    """Fake-quantize t with 1D blocks of `blk` along `dim`."""
    dt, emax = FMT[fmt]
    x = t.float().movedim(dim, -1)
    shp = x.shape
    K = shp[-1]
    pad = (-K) % blk
    if pad:
        x = F.pad(x, (0, pad))
    xb = x.reshape(*x.shape[:-1], -1, blk)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if scale == "e8m0":
        s = torch.exp2(torch.ceil(torch.log2(amax / emax)).clamp(-127, 127))
    elif scale == "fp32":
        s = amax / emax
    elif scale == "fp16":
        s = (amax / emax * (1 + 2 ** -10)).to(torch.float16).float().clamp_min(2 ** -24)
        bad = (~torch.isfinite(s)) | (s <= 2 ** -24)          # overflowed to inf / pinned at the floor
    elif scale == "bf16":
        s = (amax / emax * (1 + 2 ** -7)).to(torch.bfloat16).float()
    r = xb / s
    y = (r.to(dt).float() * s).reshape(*x.shape)
    if pad:
        y = y[..., :K]
    if tag is not None:
        nz = xb != 0
        st = STATS.setdefault(tag, [0.0, 0.0, 0, 0.0])
        st[0] += ((y.reshape(xb.shape) == 0) & nz).float().sum().item() / max(nz.sum().item(), 1)
        st[1] += (r.abs() > emax * 1.0001).float().mean().item()
        st[2] += 1
        st[3] += bad.float().mean().item() if scale == "fp16" else 0.0
    return y.reshape(shp).movedim(-1, dim)


def fq2d(t, fmt, scale, blk):
    """2D blk x blk blocks (weights): W and W^T quantize to the SAME values."""
    R, C = t.shape
    xb = t.float().reshape(R // blk, blk, C // blk, blk)
    dt, emax = FMT[fmt]
    amax = xb.abs().amax((1, 3), keepdim=True).clamp_min(1e-30)
    s = torch.exp2(torch.ceil(torch.log2(amax / emax)).clamp(-127, 127)) if scale == "e8m0" else amax / emax
    return ((xb / s).to(dt).float() * s).reshape(R, C)


class QLinearFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, cfg):
        ctx.save_for_backward(x, w)
        ctx.cfg = cfg
        fmt, scale, blk, w2d, track = cfg
        if fmt == "fp32":
            return x @ w.t()
        if fmt == "bf16":
            return (x.bfloat16() @ w.bfloat16().t()).float()
        wq = fq2d(w, fmt, scale, blk) if w2d else fq(w, 1, fmt, scale, blk, "W" if track else None)
        return fq(x, 1, fmt, scale, blk, "x" if track else None) @ wq.t()

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        fmt, scale, blk, w2d, track = ctx.cfg
        if fmt == "fp32":
            return dy @ w, dy.t() @ x, None
        if fmt == "bf16":
            b = torch.bfloat16
            return (dy.to(b) @ w.to(b)).float(), (dy.t().to(b) @ x.to(b)).float(), None
        t = "dy" if track else None
        wq = fq2d(w, fmt, scale, blk) if w2d else fq(w, 0, fmt, scale, blk, None)
        dx = fq(dy, 1, fmt, scale, blk, t) @ wq                       # reduce over out
        dw = fq(dy, 0, fmt, scale, blk, None).t() @ fq(x, 0, fmt, scale, blk, None)   # reduce over tokens
        return dx, dw, None


class Block(torch.nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.n = torch.nn.RMSNorm(d)
        self.w1 = torch.nn.Parameter(torch.randn(h, d) / math.sqrt(d))
        self.w2 = torch.nn.Parameter(torch.randn(d, h) / math.sqrt(h) * 0.5)

    def forward(self, x, cfg):
        h = F.silu(QLinearFn.apply(self.n(x), self.w1, cfg))
        return x + QLinearFn.apply(h, self.w2, cfg)


class Net(torch.nn.Module):
    def __init__(self, d=512, h=2048, L=4):
        super().__init__()
        self.inp = torch.nn.Linear(784, d)
        self.blocks = torch.nn.ModuleList(Block(d, h) for _ in range(L))
        self.no = torch.nn.RMSNorm(d)
        self.head = torch.nn.Linear(d, 10)

    def forward(self, x, cfg):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x, cfg)
        return self.head(self.no(x))


def data():
    import torchvision
    out = []
    for train in (True, False):
        ds = torchvision.datasets.MNIST("/home/marimo/work/mnist", train=train, download=True)
        x = ds.data.float().div(255).sub(0.1307).div(0.3081).reshape(-1, 784).to(dev)
        out.append((x, ds.targets.to(dev)))
    return out


def run(cfg, seed, tr, te, epochs, bs=256, lr=2e-3):
    torch.manual_seed(seed)
    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.01)
    xs, ys = tr
    steps = epochs * (xs.shape[0] // bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * s / steps)))
    g = torch.Generator(device=dev).manual_seed(seed)
    losses = []
    for s in range(steps):
        if s % (xs.shape[0] // bs) == 0:
            perm = torch.randperm(xs.shape[0], device=dev, generator=g)
        i = perm[(s % (xs.shape[0] // bs)) * bs:][:bs]
        track = s % 50 == 0
        loss = F.cross_entropy(net(xs[i], (*cfg, track)), ys[i])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step(); sched.step()
        losses.append(loss.detach())
    with torch.no_grad():
        lo = net(te[0], (*cfg, False))
        tl_, acc = F.cross_entropy(lo, te[1]).item(), (lo.argmax(-1) == te[1]).float().mean().item()
    return torch.stack(losses[-100:]).mean().item(), tl_, acc, not math.isfinite(tl_)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    tr, te = data()
    arms = [("fp32", None, 0, False), ("bf16", None, 0, False)]
    arms += [(f, s, b, False) for f, s, b in itertools.product(("e4m3", "e5m2"), ("e8m0", "fp32", "fp16"), (32, 64, 128))]
    arms += [("e4m3", "e8m0", 32, True)]
    print(f"{'arm':30s} {'train loss':>11s} {'test loss':>10s} {'test acc':>9s} "
          f"{'x zero%':>8s} {'W zero%':>8s} {'dy zero%':>9s} {'dy sat%':>8s} {'s/run':>6s}", flush=True)
    base = None
    for fmt, scale, blk, w2d in arms:
        STATS.clear()
        res, t0 = [], time.time()
        for sd in range(a.seeds):
            res.append(run((fmt, scale, blk, w2d), 1000 + sd, tr, te, a.epochs))
        ok = [r for r in res if not r[3]]
        div = f"  DIVERGED {len(res) - len(ok)}/{len(res)} seeds" if len(ok) < len(res) else ""
        if not ok:
            print(f"{fmt} {scale} blk{blk}: all seeds diverged (NaN); fp16-scale bad blocks "
                  f"x {100 * STATS.get('x', [0, 0, 1, 0])[3] / max(STATS.get('x', [0, 0, 1, 0])[2], 1):.3f}% "
                  f"dy {100 * STATS.get('dy', [0, 0, 1, 0])[3] / max(STATS.get('dy', [0, 0, 1, 0])[2], 1):.3f}%", flush=True)
            continue
        trl = statistics.mean(r[0] for r in ok)
        tel = statistics.mean(r[1] for r in ok)
        acc = statistics.mean(r[2] for r in ok)
        sd_tel = statistics.pstdev(r[1] for r in ok)
        if fmt == "fp32":
            base = tel
        st = lambda k, j: (100 * STATS[k][j] / STATS[k][2]) if k in STATS else float("nan")
        name = fmt if scale is None else f"{fmt} {scale} blk{blk}{' W2D' if w2d else ''}"
        print(f"{name:30s} {trl:11.4f} {tel:10.4f} {100 * acc:8.2f}% {st('x', 0):8.3f} {st('W', 0):8.3f} "
              f"{st('dy', 0):9.3f} {st('dy', 1):8.4f} {(time.time() - t0) / a.seeds:6.1f}"
              + (f"   dtest {tel - base:+.4f} (seed sd {sd_tel:.4f})" if base is not None and fmt != "fp32" else "")
              + (f"  fp16-scale bad blocks dy {st('dy', 3):.3f}% x {st('x', 3):.3f}%" if scale == "fp16" else "") + div,
              flush=True)
    print("QUANT_MNIST_DONE")


if __name__ == "__main__":
    main()
