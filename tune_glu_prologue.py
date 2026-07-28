"""Tune the normed-GLU prologue GEMM and check it actually beats the two-kernel path.

Path A (shipped): _grouped_mm gate_up -> _glu_fwd -> grouped_gemm(it @ W2)
Path B (new):     fused gate_up + row sumsq -> grouped_gemm_glu (activation in the GEMM prologue)

BN must equal N so the prologue reads gu once per row, and BN=N=512 puts a 512xBK bf16 B-tile in
shared memory -- (BM*BK + BK*BN) * 2 * stages has to stay under ~100 KB, which is what the sweep is
actually exploring. Run on the box:  python tune_glu_prologue.py
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
    counts_t = counts.clone()
    x = torch.randn(int(counts.sum()), H, generator=g, device=dev, dtype=torch.bfloat16) * 0.5
    w1 = (torch.randn(E, 2 * I, H, generator=g, device=dev, dtype=torch.bfloat16) * (H ** -0.5))
    w2 = (torch.randn(E, H, I, generator=g, device=dev, dtype=torch.bfloat16) * (I ** -0.5))
    row_act = torch.repeat_interleave(
        torch.full((E,), CODE, device=dev, dtype=torch.int32), counts_t).to(torch.int32)
    return x, w1, w2, counts.tolist(), counts_t, row_act


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
    x, w1, w2, counts, counts_t, row_act = make(dev)
    offs = counts_t.cumsum(0).to(torch.int32)
    dp_t = w2.transpose(1, 2).contiguous()
    tm_gg = FG.build_tile_map(counts, counts_t, dev, bm=FG._GG[0])
    tm64 = FG.build_tile_map(counts, counts_t, dev, bm=64)

    def path_a():
        gu = torch._grouped_mm(x, w1.transpose(1, 2), offs=offs)
        it = M_MOD._glu_fwd(gu, row_act, code_hint=CODE)
        return FG.grouped_gemm(it, dp_t, tm_gg)

    ref = path_a()
    base = timeit(path_a)
    print(f"path A (shipped)  {base:7.3f} ms\n")

    best = None
    print(f"{'BM':>4}{'BN':>5}{'BK':>4}{'w':>3}{'st':>3}{'smem KB':>9}{'ms':>9}{'vs A':>8}  err")
    for bm in (32, 64):
        for bk in (16, 32, 64):
            for warps in (4, 8):
                for stages in (2, 3, 4):
                    smem = (bm * bk + bk * 512) * 2 * stages / 1024
                    if smem > 99:
                        continue
                    FG._GGA = (bm, 512, bk, warps, stages)
                    tm = FG.build_tile_map(counts, counts_t, dev, bm=bm)

                    def path_b():
                        gu, rms = FG.fused_gate_up_glu(x, w1, tm64, CODE, want_gu=True, normed=True)
                        return FG.grouped_gemm_glu(gu, rms, dp_t, tm, CODE)

                    try:
                        got = path_b()
                    except Exception as exc:                # OOR / illegal config
                        print(f"{bm:>4}{512:>5}{bk:>4}{warps:>3}{stages:>3}{smem:>9.1f}"
                              f"{'-':>9}{'-':>8}  {type(exc).__name__}")
                        continue
                    err = ((got[0].float() - ref.float()).norm() / ref.float().norm()).item()
                    ms = timeit(path_b)
                    flag = "  <-- best" if (best is None or ms < best[0]) else ""
                    if best is None or ms < best[0]:
                        best = (ms, FG._GGA)
                    print(f"{bm:>4}{512:>5}{bk:>4}{warps:>3}{stages:>3}{smem:>9.1f}{ms:>9.3f}"
                          f"{base / ms:>7.2f}x  {err:.1e}{flag}")
    if best:
        print(f"\nBEST {best[1]}  {best[0]:.3f} ms  = {base / best[0]:.2f}x path A")
    else:
        print("\nno config fit in shared memory")


if __name__ == "__main__":
    main()
