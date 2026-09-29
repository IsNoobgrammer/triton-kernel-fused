"""One MoE layer fwd+bwd, bf16 path (moe_per_expert) vs MXFP8 path (moe_fp8), no harness glue: the
exact step training runs (bf16 input, fp32 master params, grads accumulated in .grad). Per-kernel
profile split into forward / backward.

    python -m bench.bench_moe_fp8_layer [--N 65536]
"""
import argparse
import collections
import importlib
import statistics

import torch

K75 = importlib.import_module("kernels.sm75.moe")
F8 = importlib.import_module("kernels.sm120.moe_fp8")
from bench.bench_moe_layer import make  # noqa: E402


def timed(fn, it=20, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def step(fn, x, idx, w, gu, dn, codes, theta):
    def f():
        x.grad = gu.grad = dn.grad = theta.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fn(x, idx, w, gu, dn, codes, act_params=theta)
        y.backward(torch.ones_like(y))
    return f


def profile(f, fwd_only):
    from torch.profiler import profile as prof, ProfilerActivity
    f(); torch.cuda.synchronize()
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        fwd_only(); torch.cuda.synchronize()
    n_fwd = sum(1 for e in p.events() if e.device_type.name == "CUDA")
    with prof(activities=[ProfilerActivity.CUDA]) as p:
        f(); torch.cuda.synchronize()
    ev = sorted((e for e in p.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start)
    agg = collections.defaultdict(float)
    for i, e in enumerate(ev):
        agg[("fwd " if i < n_fwd else "bwd ") + e.name[:48]] += e.time_range.elapsed_us() / 1e3
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=65536)
    a = ap.parse_args()
    x, idx, w, gu, dn, codes, theta, maxvio = make(N=a.N)
    theta.data.uniform_(-2, 2)
    w = w.clone().requires_grad_(True)
    res = {}
    for name, fn in (("bf16", K75.moe_per_expert), ("fp8", F8.moe_fp8)):
        f = step(fn, x, idx, w, gu, dn, codes, theta)

        def fwd_only(fn=fn):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return fn(x, idx, w, gu, dn, codes, act_params=theta)
        torch.cuda.reset_peak_memory_stats()
        t = timed(f)
        mem = torch.cuda.max_memory_allocated() / 2 ** 30
        agg = profile(f, fwd_only)
        res[name] = (t, mem, agg)
    tb, t8 = res["bf16"][0], res["fp8"][0]
    print(f"N={a.N}, E=64 top-6, MaxVio {maxvio:.2f}: layer fwd+bwd bf16 {tb:.2f} ms | fp8 {t8:.2f} ms | "
          f"{tb / t8:.2f}x (target 1.44x = {tb / 1.44:.2f} ms) | peak mem {res['bf16'][1]:.2f} / {res['fp8'][1]:.2f} GB")
    for name in ("bf16", "fp8"):
        agg = res[name][2]
        fw = sum(v for k, v in agg.items() if k.startswith("fwd"))
        bw = sum(v for k, v in agg.items() if k.startswith("bwd"))
        print(f"\n{name}: kernel sum fwd {fw:.2f} + bwd {bw:.2f} = {fw + bw:.2f} ms")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:16]:
            print(f"   {v:7.3f}  {k}")
    print("BENCH_MOE_FP8_LAYER_DONE")


if __name__ == "__main__":
    main()
