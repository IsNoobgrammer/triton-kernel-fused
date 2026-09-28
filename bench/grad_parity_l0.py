"""Gradient parity of the all-active layer-0 path: Triton one-group wgrad (old) vs cuBLAS (new), both
against an fp32 eager reference (moe_eager on fp32 copies of the same bf16 inputs/weights).

    python -m bench.grad_parity_l0 [N]
"""
import importlib
import sys

import torch

K75 = importlib.import_module("kernels.sm75.moe")
dev = "cuda"


def run(mode, x0, idx, w, gu0, dn0, codes, th0):
    K75.DENSE_WGRAD = mode
    x, gu, dn, th = (t.clone().requires_grad_(True) for t in (x0, gu0, dn0, th0))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = K75.moe_per_expert(x, idx, w, gu, dn, codes, act_params=th)
    gy = torch.randn(y.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(1)).to(y.dtype)
    (y.float() * gy.float()).sum().backward()
    return [t.grad.float() for t in (x, gu, dn, th)]


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
    H, E, I = 512, 8, 576
    g = torch.Generator(device=dev).manual_seed(0)
    x0 = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16)
    w, idx = torch.softmax(torch.randn(N, E, device=dev, generator=g), -1).topk(E, -1)
    w = w.float()
    # weights pre-rounded to bf16 so the fp32 reference sees exactly the operands the kernels see
    gu0 = (torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5).bfloat16().float()
    dn0 = (torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5).bfloat16().float()
    codes = torch.full((E,), 8, device=dev, dtype=torch.int32)
    th0 = torch.randn(E, device=dev, generator=g) * 0.3
    old = run("triton", x0, idx, w, gu0, dn0, codes, th0)
    new = run("cublas", x0, idx, w, gu0, dn0, codes, th0)
    new2 = run("cublas", x0, idx, w, gu0, dn0, codes, th0)
    # fp32 reference
    x, gu, dn, th = (t.clone().float().requires_grad_(True) for t in (x0, gu0, dn0, th0))
    y = K75.moe_eager(x, idx, w, gu, dn, codes, act_params=th)
    gy = torch.randn(y.shape, device=dev, generator=torch.Generator(device=dev).manual_seed(1)).to(torch.bfloat16)
    (y * gy.float()).sum().backward()
    ref = [t.grad for t in (x, gu, dn, th)]
    rel = lambda a, r: ((a - r).norm() / r.norm()).item()
    print(f"L0 all-active N={N} E={E} I={I}: gradient error vs fp32 eager (relative Frobenius)")
    for nm, o, n, r in zip(("d_x", "d_gate_up", "d_down", "d_theta"), old, new, ref):
        print(f"  {nm:10s} old triton {rel(o, r):.3e} | new cublas {rel(n, r):.3e} | old vs new {rel(n, o):.2e}")
    print("  new run-to-run bitwise:", all(torch.equal(a, b) for a, b in zip(new, new2)))
    print("GRADPAR_DONE")


if __name__ == "__main__":
    main()
