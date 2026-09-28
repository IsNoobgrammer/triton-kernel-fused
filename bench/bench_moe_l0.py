"""BiBo layer 0 (8 all-active experts x 576, the _dense_fwd path): stage profile, plus the two weight
gradients through the Triton grouped_wgrad (one 65536-row group) vs cuBLAS addmm into an fp32 grad.

    python -m bench.bench_moe_l0
"""
import torch

import kernels.sm120.moe_fused_glu as FG
from bench.bench_moe_gemm import timed
from bench.bench_moe_layer import layer_step, stages

dev = "cuda"


def main():
    N, H, E, I = 65536, 512, 8, 576
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16).requires_grad_(True)
    logits = torch.randn(N, E, device=dev, generator=g)
    w, idx = torch.softmax(logits, -1).topk(E, -1)
    w = w.float()
    gu = (torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5).requires_grad_(True)
    dn = (torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5).requires_grad_(True)
    codes = torch.full((E,), 8, device=dev, dtype=torch.int32)
    theta = torch.zeros(E, device=dev, requires_grad=True)
    f = layer_step(x, idx, w, gu, dn, codes, theta)
    print(f"L0 all-active {E}x{I}: layer fwd+bwd {timed(f):.3f} ms")
    for n_, ms in stages(x, idx, w, gu, dn, codes, theta):
        if ms > 0.02:
            print(f"  {ms:8.3f}  {n_[:90]}")
    one = torch.full((1,), N, device=dev, dtype=torch.int32)
    for lab, n1, n2 in (("dW_down  its^T@go", E * I, H), ("dW_gu   dgu^T@x", E * 2 * I, H)):
        a = (torch.randn(N, n1, device=dev, generator=g) * 1e-2).to(torch.bfloat16)
        b = (torch.randn(N, n2, device=dev, generator=g) * 1e-2).to(torch.bfloat16)
        acc = torch.zeros(1, n1, n2, device=dev)
        tt = timed(lambda: FG.grouped_wgrad(a, b, one, out=acc, accumulate=True))
        acc2 = torch.zeros(n1, n2, device=dev)
        tc = timed(lambda: torch.addmm(acc2, a.t(), b, out_dtype=torch.float32, out=acc2))
        fl = 2 * N * n1 * n2
        print(f"{lab}: triton grouped_wgrad {tt:.3f} ms ({fl / tt / 1e9:.0f} TF) | cuBLAS addmm {tc:.3f} ms ({fl / tc / 1e9:.0f} TF)")
    print("L0_DONE")


if __name__ == "__main__":
    main()
