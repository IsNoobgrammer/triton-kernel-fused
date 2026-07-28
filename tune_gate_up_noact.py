"""Tile sweep for the gate_up GEMM in act=False mode (the RMS-normed codes 2/6/7).

_BM/_BN/_BK were autotuned with the GLU epilogue attached. Without it the kernel writes 2I instead
of 3I and does no transcendentals, so the balance shifts -- this checks whether the shipped config
is still the right one, against cuBLAS `torch._grouped_mm` as the baseline it has to beat.

Context: fusing the normed activation into the DOWN-projection GEMM's prologue was built and
measured here first, and lost. rms is a per-row scalar so the activation is tile-local once it is
known, but the prologue forces BN=N=512, whose 512xBK bf16 B-tile in shared memory caps BK at 16;
the crippled GEMM gave back everything the saved round-trip won (1.636 ms vs 0.815 + 0.773 = 1.588
for _glu_fwd plus a normal grouped_gemm). The gate_up GEMM itself was the entire win -- 1.903 ms
cuBLAS -> 1.643 ms Triton, 1.16x -- which is what this file now tunes. Run on the box.
"""
import importlib
import time

import torch

importlib.import_module("kernels.sm120.moe")           # see parity_normed_tiles for why sm120 first
M_MOD = importlib.import_module("kernels.sm75.moe")
FG = importlib.import_module("kernels.sm120.moe_fused_glu")

E, H, I, CODE = 64, 512, 768, 2
ROWS = 262144                                          # 32 x 1024 tokens x top-8


def make(dev="cuda"):
    g = torch.Generator(device=dev).manual_seed(0)
    # the real load spread, not a uniform one: imbalance is what the tile map exists for
    share = torch.rand(E, generator=g, device=dev) * 0.6 + 0.7
    counts = (share / share.sum() * ROWS).long()
    counts[-1] += ROWS - int(counts.sum())
    x = torch.randn(int(counts.sum()), H, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    w1 = torch.randn(E, 2 * I, H, generator=g, device=dev, dtype=torch.bfloat16) * (H ** -0.5)
    return x, w1, counts.tolist(), counts.clone()


def timeit(fn, n=20):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def main():
    dev = "cuda"
    x, w1, counts, counts_t = make(dev)
    offs = counts_t.cumsum(0).to(torch.int32)
    w1t = w1.transpose(1, 2)

    base = timeit(lambda: torch._grouped_mm(x, w1t, offs=offs))
    ref = torch._grouped_mm(x, w1t, offs=offs).float()
    print(f"cuBLAS torch._grouped_mm  {base:7.3f} ms\n")
    shipped = (FG._BM, FG._BN, FG._BK, FG._WARPS, FG._STAGES)

    best = None
    print(f"{'BM':>4}{'BN':>5}{'BK':>4}{'w':>3}{'st':>3}{'ms':>9}{'vs cuBLAS':>11}  err")
    for bm in (32, 64, 128):
        for bn in (128, 256):
            if I % bn:
                continue
            for bk in (32, 64):
                for warps in (4, 8):
                    for stages in (2, 3, 4):
                        FG._BM, FG._BN, FG._BK, FG._WARPS, FG._STAGES = bm, bn, bk, warps, stages
                        tm = FG.build_tile_map(counts, counts_t, dev, bm=bm)
                        fn = lambda: FG.fused_gate_up_glu(x, w1, tm, CODE, want_gu=True, act=False)
                        try:
                            got = fn()[0].float()
                        except Exception as exc:
                            continue                    # out of resources / invalid tiling
                        err = ((got - ref).norm() / ref.norm()).item()
                        ms = timeit(fn)
                        mark = " <-- SHIPPED" if (bm, bn, bk, warps, stages) == shipped else ""
                        if best is None or ms < best[0]:
                            best = (ms, (bm, bn, bk, warps, stages))
                            mark += "  <-- best"
                        print(f"{bm:>4}{bn:>5}{bk:>4}{warps:>3}{stages:>3}{ms:>9.3f}"
                              f"{base / ms:>10.2f}x  {err:.1e}{mark}")
    FG._BM, FG._BN, FG._BK, FG._WARPS, FG._STAGES = shipped
    if best:
        print(f"\nBEST {best[1]}  {best[0]:.3f} ms = {base / best[0]:.2f}x cuBLAS"
              f"   (shipped config is {shipped})")


if __name__ == "__main__":
    main()
