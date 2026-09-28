"""One BiBo MoE layer (the training path: moe_per_expert, radial, gather, deterministic), stage by
stage at the board shapes: N=65536 tokens, H=512, E=64, top-6, I=768, fp32 master weights under
bf16 autocast. Prints per-stage ms and TFLOPS / GB/s, next to rooflines:
  * cuBLAS dense GEMM of the same (M, K, N)    -- what a perfect grouped GEMM could approach
  * plain Triton matmul at 8192^3              -- Triton's ceiling on this card

    python -m bench.bench_moe_layer [--skew 0.35] [--iters 10]
"""
import argparse
import statistics
from collections import defaultdict

import torch
import triton
import triton.language as tl

from kernels.sm120.moe import moe_per_expert

dev = "cuda"


def timed(fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def make(N=65536, H=512, E=64, k=6, I=768, skew=0.35, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = (torch.randn(N, H, device=dev, generator=g)).to(torch.bfloat16).requires_grad_(True)
    logits = torch.randn(N, E, device=dev, generator=g) + skew * torch.randn(E, device=dev, generator=g)
    w, idx = torch.softmax(logits, -1).topk(k, -1)
    w = (w / w.sum(-1, keepdim=True)).float()
    gu = (torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5).requires_grad_(True)
    dn = (torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5).requires_grad_(True)
    codes = torch.full((E,), 8, device=dev, dtype=torch.int32)
    theta = torch.zeros(E, device=dev, requires_grad=True)
    cnt = torch.bincount(idx.reshape(-1), minlength=E).float()
    return x, idx, w, gu, dn, codes, theta, (cnt.max() / cnt.mean() - 1).item()


def layer_step(x, idx, w, gu, dn, codes, theta):
    def f():
        x.grad = gu.grad = dn.grad = theta.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = moe_per_expert(x, idx, w, gu, dn, codes, act_params=theta)
        y.float().sum().backward()
    return f


@triton.jit
def _mm(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, G: tl.constexpr):
    pid = tl.program_id(0)
    nm, nn = tl.cdiv(M, BM), tl.cdiv(N, BN)
    gid = pid // (G * nn)
    fm = gid * G
    gs = tl.minimum(nm - fm, G)
    pm = fm + (pid % (G * nn)) % gs
    pn = (pid % (G * nn)) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    a = A + rm[:, None].to(tl.int64) * K + rk[None, :]
    b = B + rk[:, None] * N + rn[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for _ in range(0, K, BK):
        acc = tl.dot(tl.load(a), tl.load(b), acc)
        a += BK
        b += BK * N
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(tl.bfloat16))


def triton_ceiling():
    n = 8192
    a = torch.randn(n, n, device=dev).bfloat16()
    b = torch.randn(n, n, device=dev).bfloat16()
    c = torch.empty(n, n, device=dev, dtype=torch.bfloat16)
    best = None
    for BM, BN, BK, w, s in [(128, 256, 64, 8, 3), (128, 128, 64, 4, 4), (256, 128, 64, 8, 3),
                              (128, 256, 32, 8, 4), (128, 128, 32, 4, 5), (256, 128, 32, 8, 4)]:
        try:
            ms = timed(lambda: _mm[(triton.cdiv(n, BM) * triton.cdiv(n, BN),)](a, b, c, n, n, n, BM, BN, BK, 8,
                                                                               num_warps=w, num_stages=s), it=5)
            tf = 2 * n ** 3 / ms / 1e9
            best = max(best or (0, None), (tf, (BM, BN, BK, w, s)))
        except Exception:
            pass
    cub = 2 * n ** 3 / timed(lambda: a @ b, it=5) / 1e9
    return best, cub


def stages(x, idx, w, gu, dn, codes, theta):
    """per-kernel GPU time in launch order, from the profiler, mapped to the pipeline stages"""
    f = layer_step(x, idx, w, gu, dn, codes, theta)
    f(); f()
    torch.cuda.synchronize()
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        f()
        torch.cuda.synchronize()
    ev = sorted((e for e in p.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start)
    return [(e.name, e.time_range.elapsed_us() / 1e3) for e in ev]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skew", type=float, default=0.35)
    ap.add_argument("--iters", type=int, default=10)
    a = ap.parse_args()
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M = N * k
    x, idx, w, gu, dn, codes, theta, maxvio = make(skew=a.skew)
    print(f"N={N} H={H} E={E} top{k} I={I} -> M={M} rows, MaxVio {maxvio:.2f}", flush=True)
    tot = timed(layer_step(x, idx, w, gu, dn, codes, theta), it=a.iters)
    print(f"layer fwd+bwd: {tot:.2f} ms", flush=True)
    ev = stages(x, idx, w, gu, dn, codes, theta)
    agg = defaultdict(lambda: [0.0, 0])
    for n_, ms in ev:
        key = n_[:60]
        agg[key][0] += ms
        agg[key][1] += 1
    print("\nkernels in launch order (ms):")
    for n_, ms in ev:
        if ms > 0.02:
            print(f"  {ms:8.3f}  {n_[:90]}")
    print(f"  sum {sum(ms for _, ms in ev):.2f} ms over {len(ev)} kernels")
    # GEMM stages: (label, M, K, N)
    gemms = [("F1 gate_up  x@Wgu^T", M, H, 2 * I), ("F3 down     it@Wdn^T", M, I, H),
             ("B2 dW_down  ge^T@it", M, H, I), ("B3 d_inter  ge@Wdn", M, H, I),
             ("B5 dW_gu    dgu^T@x", M, 2 * I, H), ("B6 d_x      dgu@Wgu", M, 2 * I, H)]
    print("\ncuBLAS dense-GEMM roofline at the same total shapes:")
    for lab, m, kk, n in gemms:
        A = torch.randn(m, kk, device=dev).bfloat16()
        B = torch.randn(kk, n, device=dev).bfloat16()
        ms = timed(lambda: A @ B, it=5)
        print(f"  {lab:24s} {2 * m * kk * n / 1e9:7.0f} GF  {ms:6.3f} ms  {2 * m * kk * n / ms / 1e9:5.0f} TF")
        del A, B
    best, cub = triton_ceiling()
    print(f"\nTriton plain matmul 8192^3: {best[0]:.0f} TF {best[1]} | cuBLAS {cub:.0f} TF")
    print("MOE_BENCH_DONE")


if __name__ == "__main__":
    main()
