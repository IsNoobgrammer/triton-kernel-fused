"""Fused RNN-T joint + loss vs NeMo's training path, forward and backward timed separately, peak memory.

  nemo fbs=2   what run1 trains with: fuse_loss_wer, 2 utterances per sub-batch (narrow to the pair's max lengths,
               autocast joint, fp32 cast, numba loss), mean over the batch
  nemo fbs=B   the same, whole batch at once (OOM is reported, not fatal)
  ours         kernels/sm120/rnnt_joint.py

Shapes are Lhotse 1200 s buckets of the BiBo ASR mix (8x subsampling = 12.5 frames/s, ~4 tokens/s): lengths are
drawn inside the bucket, H = 640, V = 4097 (4096 BPE + blank), dropout 0.2, FastEmit 0.005.

    python bench/bench_rnnt_joint.py [--only short,mid,long] [--iters 10]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as Fn

import kernels.sm120.rnnt_joint as rj

dev = "cuda"
H, V, P, LAM = 640, 4097, 0.2, 0.005
SHAPES = {"short": (240, 50, 16), "mid": (150, 100, 32), "long": (48, 312, 100)}


def data(B, T, U, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    tl = (T * (0.85 + 0.15 * torch.rand(B, device=dev, generator=gen))).long().clamp(1, T)
    yl = (U * (0.7 + 0.3 * torch.rand(B, device=dev, generator=gen))).long().clamp(0, U)
    tl[0], yl[0] = T, U
    y = torch.randint(0, V - 1, (B, U), device=dev, generator=gen)
    f = torch.randn(B, T, H, device=dev, generator=gen).to(torch.bfloat16).requires_grad_()
    g = torch.randn(B, U + 1, H, device=dev, generator=gen).to(torch.bfloat16).requires_grad_()
    W = (torch.randn(V, H, device=dev, generator=gen) * H ** -0.5).requires_grad_()
    b = torch.zeros(V, device=dev).requires_grad_()
    return f, g, W, b, y, tl, yl


def nemo_fwd(fbs):
    from nemo.collections.asr.losses.rnnt import RNNTLoss
    loss_fn = RNNTLoss(num_classes=V - 1, reduction="mean_batch", loss_name="warprnnt_numba",
                       loss_kwargs=dict(fastemit_lambda=LAM, clamp=-1.0))

    def run(f, g, W, b, y, tl, yl):
        B = f.shape[0]
        n = B if fbs is None else fbs
        losses, lens = [], []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for i in range(0, B, n):                                   # NeMo RNNTJoint.forward, fused branch
                t_, u_ = int(tl[i:i + n].max()), int(yl[i:i + n].max())
                x = torch.relu(f[i:i + n, :t_, None] + g[i:i + n, None, :u_ + 1])
                logits = Fn.linear(Fn.dropout(x, P, True), W, b)
                loss_fn.reduction = None
                losses.append(loss_fn(log_probs=logits, targets=y[i:i + n, :u_], input_lengths=tl[i:i + n],
                                      target_lengths=yl[i:i + n]))
                lens.append(yl[i:i + n])
                loss_fn.reduction = "mean_batch"
            return loss_fn.reduce(losses, lens)
    return run


def ours_fwd(f, g, W, b, y, tl, yl):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return rj.rnnt_joint_loss(f, g, W, b, y, tl, yl, fastemit_lambda=LAM, dropout=P)[0]


def timeit(fn, args, iters):
    ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
    fw, bw = [], []
    for i in range(iters + 3):
        for a in args[:4]:
            a.grad = None
        torch.cuda.synchronize()
        if i == 3:
            torch.cuda.reset_peak_memory_stats()
        ev[0].record()
        loss = fn(*args)
        ev[1].record()
        loss.backward()
        ev[2].record()
        torch.cuda.synchronize()
        if i >= 3:
            fw.append(ev[0].elapsed_time(ev[1]))
            bw.append(ev[1].elapsed_time(ev[2]))
    bad = [n for n, t in zip("f g W b".split(), args[:4]) if not torch.isfinite(t.grad).all()]
    assert not bad, f"non-finite grads: {bad}"                     # a fast NaN is not a result
    med = lambda x: sorted(x)[len(x) // 2]
    return med(fw), med(bw), torch.cuda.max_memory_allocated() / 2 ** 30, loss.item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="short,mid,long")
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    print(f"{torch.cuda.get_device_name()}  torch {torch.__version__}", flush=True)
    for name in a.only.split(","):
        B, T, U = SHAPES[name]
        args = data(B, T, U)
        tl, yl = args[5], args[6]
        pts = int((tl * (yl + 1)).sum())
        audio = float(tl.sum()) * 0.08
        print(f"\n== {name}: B={B} T<={T} U<={U}  {audio:.0f} s audio, {pts / 1e6:.2f}M lattice points", flush=True)
        print(f"  {'path':12s} {'fwd ms':>8s} {'bwd ms':>8s} {'total':>8s} {'peak GB':>8s} {'loss':>9s}", flush=True)
        base = None
        for label, fn in (("nemo fbs=2", nemo_fwd(2)), ("nemo fbs=B", nemo_fwd(None)), ("ours triton", ours_fwd),
                          ("ours", ours_fwd)):
            rj._CUBLAS_LOGITS = label != "ours triton"           # the logits GEMM: Triton kernel vs cuBLAS + rows
            try:
                fw, bw, mem, loss = timeit(fn, args, a.iters)
            except torch.OutOfMemoryError:
                print(f"  {label:12s} OOM", flush=True)
                torch.cuda.empty_cache()
                continue
            base = base or (fw, bw, fw + bw)
            print(f"  {label:12s} {fw:8.1f} {bw:8.1f} {fw + bw:8.1f} {mem:8.1f} {loss:9.4f}   "
                  f"x{base[0] / fw:.2f} fwd  x{base[1] / bw:.2f} bwd  x{base[2] / (fw + bw):.2f} total", flush=True)
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
