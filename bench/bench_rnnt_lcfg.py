"""Tile sweep for rnnt_joint's logits pass (store E) at the real joint shape: K = 672 (640 + bias, padded),
V = 4097 (E rows padded to 4160), N = the long bucket's 1.18M lattice rows.

    python bench/bench_rnnt_lcfg.py [--n 1180000]
"""
import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton

import kernels.sm120.rnnt_joint as rj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_180_000)
    a = ap.parse_args()
    N, K, V = a.n, 672, 4097
    VS = triton.cdiv(V, 64) * 64
    X = (torch.randn(N, K, device="cuda") * 0.5).to(torch.bfloat16)
    Wa = torch.zeros(VS, K, device="cuda", dtype=torch.bfloat16)
    Wa[:V] = (torch.randn(V, K, device="cuda") * K ** -0.5).to(torch.bfloat16)
    lab = torch.randint(0, V - 1, (N,), device="cuda")
    E = torch.empty(N, VS, device="cuda", dtype=torch.bfloat16)
    flop = 2 * N * K * VS
    res = []
    for BM, BN, BK, nw, ns in itertools.product((64, 128), (64, 128, 256), (32, 64), (4, 8), (2, 3, 4)):
        if BM * BN > 128 * 256 or (BN == 256 and nw == 4) or (BM * BK + BN * BK) * 2 * ns > 160 * 1024:
            continue
        rj._LCFG = (BM, BN, BK, 8, nw, ns)
        try:
            ms = triton.testing.do_bench(lambda: rj._stats(X, Wa, lab, V - 1, True, E), warmup=5, rep=40)
        except Exception as e:                                     # out of shared memory / registers
            print(f"  {rj._LCFG} skip: {type(e).__name__}", flush=True)
            continue
        res.append((ms, rj._LCFG))
        print(f"  {rj._LCFG}  {ms:7.2f} ms  {flop / ms / 1e9:6.1f} TFLOPS", flush=True)
    rj._LCFG = None
    print("\nbest 5:")
    for ms, c in sorted(res)[:5]:
        print(f"  {c}  {ms:7.2f} ms  {flop / ms / 1e9:6.1f} TFLOPS")
    base = triton.testing.do_bench(lambda: rj._stats(X, Wa, lab, V - 1, True, E), warmup=5, rep=40)
    print(f"current default {rj._lcfg(K)}: {base:.2f} ms;  cuBLAS mm alone (no exp/stats/E store): "
          f"{triton.testing.do_bench(lambda: torch.mm(X, Wa.t()), warmup=5, rep=40):.2f} ms")


if __name__ == "__main__":
    main()
