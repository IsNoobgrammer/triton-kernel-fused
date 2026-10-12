"""tdt_joint vs rnnt_joint at a real 1200 s train batch (126 clips, ~120 frames, ~25 tokens: ~390k lattice rows,
H 640, V 2048 + 5 durations): forward + backward wall time (median of 10 after warm-up) and the top GPU kernels.

    python bench/bench_tdt_joint.py [--profile]
"""
import argparse
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

import kernels.sm120.rnnt_joint as rj
import kernels.sm120.tdt_joint as tj

dev = "cuda"


def batch(B=126, T=120, U=25, H=640, V=2048, ND=5, seed=0):
    gen = torch.Generator(device=dev).manual_seed(seed)
    tl = torch.randint(T // 2, T + 1, (B,), device=dev, generator=gen)
    yl = torch.randint(U // 2, U + 1, (B,), device=dev, generator=gen)
    y = torch.randint(0, V - 1, (B, U), device=dev, generator=gen)
    f = torch.randn(B, T, H, device=dev, generator=gen).to(torch.bfloat16).requires_grad_()
    g = torch.randn(B, U + 1, H, device=dev, generator=gen).to(torch.bfloat16).requires_grad_()
    W = (torch.randn(V + ND, H, device=dev, generator=gen) * H ** -0.5).requires_grad_()
    b = (torch.randn(V + ND, device=dev, generator=gen) * 0.1).requires_grad_()
    n = int((tl.long() * (yl.long() + 1)).sum())
    return f, g, W, b, y, tl, yl, n, V


def step(kind, f, g, W, b, y, tl, yl, n, V):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        if kind == "rnnt":
            loss, _ = rj.rnnt_joint_loss(f, g, W[:V], b[:V], y, tl, yl, 0.0, 0.2, 1234, n_rows=n)
        else:
            loss, _ = tj.tdt_joint_loss(f, g, W, b, y, tl, yl, durations=(0, 1, 2, 3, 4), sigma=0.05, omega=0.0,
                                        dropout=0.2, seed=1234, n_rows=n)
    loss.backward()
    for x in (f, g, W, b):
        x.grad = None


def timeit(kind, data, reps=10):
    for _ in range(3):
        step(kind, *data)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        step(kind, *data)
        e.record()
        torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    data = batch()
    print(f"lattice rows N = {data[7]}", flush=True)
    tr = timeit("rnnt", data)
    tt = timeit("tdt", data)
    print(f"BENCH rnnt_joint {tr:.2f} ms | tdt_joint {tt:.2f} ms | tdt / rnnt {tt / tr:.3f}", flush=True)
    if a.profile:
        from torch.profiler import ProfilerActivity, profile
        for kind in ("rnnt", "tdt"):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                step(kind, *data)
                torch.cuda.synchronize()
            print(f"\n== {kind}: top GPU kernels")
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=14, max_name_column_width=60))


if __name__ == "__main__":
    main()
