"""Fused AR kernel == K3's `_apply_attn_res`, and a standalone profile at real shapes.

NO MODEL NEEDED. Dummy tensors at the shapes BiBo actually runs, so the kernel can be iterated on
without a 20-minute training probe in the loop.

Shapes come from the 524M stack: micro-batch 16 x seq 1024 = 16384 tokens, hidden 512, and N =
blocks+1 sweeping 2..11 (block_size=3 gives N up to 5; block_size=1 gives N up to 11).

    python -m parity_check.parity_attn_res
"""
from . import _paths  # noqa: F401

import torch
import triton

from kernels.sm75.attn_res import fused_attn_res, attn_res_reference

DEV = "cuda"
EPS = 1e-6


def _opt_torch(block_residual, prefix_sum, score_weight, eps=EPS):
    """What BiBo runs today: RMS factored out of the contraction, no normalized copy."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = v.float()
    sq = torch.linalg.vector_norm(vf, dim=-1).square()
    inv_rms = torch.rsqrt(sq / vf.shape[-1] + eps)
    scores = torch.matmul(vf, score_weight.float()) * inv_rms
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(v.dtype)


def _mk(T, N, H, dtype):
    torch.manual_seed(0)
    br = torch.randn(T, N - 1, H, device=DEV, dtype=dtype)
    ps = torch.randn(T, H, device=DEV, dtype=dtype)
    w = torch.randn(H, device=DEV, dtype=torch.float32) * 0.05
    return br, ps, w


def parity():
    print("=== parity vs K3 reference ===")
    print(f"{'T':>7}{'N':>4}{'H':>6}{'dtype':>10}{'rel err':>12}{'cached sq':>12}")
    bad = 0
    for dtype in (torch.bfloat16, torch.float32):
        for T, N, H in ((4096, 2, 512), (4096, 5, 512), (16384, 5, 512),
                        (16384, 11, 512), (2048, 4, 1024), (1024, 8, 256)):
            br, ps, w = _mk(T, N, H, dtype)
            ref = attn_res_reference(br, ps, w, EPS).float()
            got = fused_attn_res(br, ps, w, EPS).float()
            rel = (got - ref).abs().max().item() / ref.abs().max().item()

            # cached block squared-norms must give the identical answer
            bsq = torch.zeros(T, N, device=DEV, dtype=torch.float32)
            bsq[:, : N - 1] = br.float().pow(2).sum(-1)
            got2 = fused_attn_res(br, ps, w, EPS, block_sq_sum=bsq).float()
            rel2 = (got2 - ref).abs().max().item() / ref.abs().max().item()

            tol = 3e-2 if dtype is torch.bfloat16 else 1e-5
            ok = rel < tol and rel2 < tol
            bad += not ok
            print(f"{T:>7}{N:>4}{H:>6}{str(dtype).split('.')[-1]:>10}"
                  f"{rel:>12.2e}{rel2:>12.2e}" + ("" if ok else "   <-- FAIL"))
    return bad


def _bench(fn, *a, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    return triton.testing.do_bench(lambda: fn(*a), warmup=20, rep=100)


def profile():
    print()
    print("=== forward profile, bf16, T=16384 H=512 (micro-batch 16 x seq 1024) ===")
    print(f"{'N':>4}{'K3 naive':>12}{'opt torch':>12}{'fused':>10}"
          f"{'vs naive':>10}{'vs opt':>9}{'GB/s':>9}")
    T, H = 16384, 512
    for N in (2, 3, 5, 8, 11):
        br, ps, w = _mk(T, N, H, torch.bfloat16)
        t_ref = _bench(attn_res_reference, br, ps, w)
        t_opt = _bench(_opt_torch, br, ps, w)
        t_fus = _bench(fused_attn_res, br, ps, w)
        # ideal traffic: read V once (bf16) + write out once
        gb = (T * N * H * 2 + T * H * 2) / 1e9
        print(f"{N:>4}{t_ref:>11.3f}m{t_opt:>11.3f}m{t_fus:>9.3f}m"
              f"{t_ref / t_fus:>9.2f}x{t_opt / t_fus:>8.2f}x{gb / (t_fus * 1e-3):>9.0f}")

    print()
    print("=== per-STEP cost, block3 pattern (10 layers, 2 sites/layer + 1 output = 21 mixes) ===")
    # N at layer l is floor(l/3)+1, +1 for the prefix row
    Ns = [min(l // 3 + 1, 4) + 1 for l in range(10) for _ in range(2)] + [5]
    tot = {"naive": 0.0, "opt": 0.0, "fused": 0.0}
    for N in Ns:
        br, ps, w = _mk(T, N, H, torch.bfloat16)
        tot["naive"] += _bench(attn_res_reference, br, ps, w)
        tot["opt"] += _bench(_opt_torch, br, ps, w)
        tot["fused"] += _bench(fused_attn_res, br, ps, w)
    for k, v in tot.items():
        print(f"  {k:<6} {v:7.2f} ms/step-of-AR   ({v / tot['fused']:.2f}x fused)")
    print(f"  a 1527 ms/step baseline step means AR-naive adds ~{tot['naive'] / 1527 * 100:.1f}% "
          f"forward-only, AR-fused ~{tot['fused'] / 1527 * 100:.1f}%")


if __name__ == "__main__":
    bad = parity()
    print(f"\n{'PARITY FAIL' if bad else 'PARITY OK'}")
    if not bad:
        profile()
