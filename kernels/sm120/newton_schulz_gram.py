import math

import torch
import triton
import triton.language as tl

from kernels.sm75.muon import _DSV4_COEFFS, newton_schulz as _newton_schulz_cublas
from kernels.sm120.newton_schulz_symmul import (
    SYMMUL_MIN_DIM, _bmmt_configs, symmul, symmul_axpy, newton_schulz_symmul,
)

GRAM_MIN_RATIO = 1.5

GRAM_RESTART_AT = (4, 6)


@triton.autotune(configs=_bmmt_configs(), key=["M", "K"])
@triton.jit
def _bssm_kernel(
    s1_ptr, s2_ptr, y_ptr,
    M, K,
    stride_1b, stride_1m, stride_1k,
    stride_2b, stride_2m, stride_2k,
    stride_yb, stride_ym, stride_yn,
    BM: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    num_pid_m = tl.cdiv(M, BM)
    num_pid_n = num_pid_m
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m > pid_n:
        return

    s1_ptr += bid * stride_1b
    s2_ptr += bid * stride_2b
    y_ptr += bid * stride_yb

    offs_m = (pid_m * BM + tl.arange(0, BM)) % M
    offs_n = (pid_n * BM + tl.arange(0, BM)) % M
    offs_k = tl.arange(0, BK)
    a_ptrs = s1_ptr + (offs_m[:, None] * stride_1m + offs_k[None, :] * stride_1k)
    b_ptrs = s2_ptr + (offs_n[:, None] * stride_2m + offs_k[None, :] * stride_2k)

    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = offs_k[None, :] < K - k * BK
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=kmask, other=0.0)
        acc = tl.dot(a, tl.permute(b, (1, 0)), acc)
        a_ptrs += BK * stride_1k
        b_ptrs += BK * stride_2k
    c = acc.to(y_ptr.dtype.element_ty)

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BM + tl.arange(0, BM)
    c_ptrs = y_ptr + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)
    if pid_m < pid_n:
        ct_ptrs = y_ptr + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def symmul2(S1, S2, out=None):
    B, M, _ = S1.shape
    if M < SYMMUL_MIN_DIM:
        return torch.bmm(S1, S2) if out is None else torch.bmm(S1, S2, out=out)
    S1 = S1.contiguous()
    S2 = S2.contiguous()
    Y = torch.empty_like(S1) if out is None else out
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(M, meta["BM"]), B)
    _bssm_kernel[grid](
        S1, S2, Y, M, M,
        S1.stride(0), S1.stride(1), S1.stride(2),
        S2.stride(0), S2.stride(1), S2.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
    )
    return Y


def newton_schulz_gram(G, coeffs=_DSV4_COEFFS, ns_dtype=torch.bfloat16, eps=1e-7,
                       gram_dtype=None, restart_at=GRAM_RESTART_AT, force_eager=False):
    n, m = G.shape[-2], G.shape[-1]
    r = max(n, m) / min(n, m)
    if min(n, m) < SYMMUL_MIN_DIM or r < GRAM_MIN_RATIO:
        return newton_schulz_symmul(G, coeffs, ns_dtype, eps, force_eager=force_eager)

    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = (X.to(ns_dtype) / nrm.to(ns_dtype)).contiguous()

    resets = () if not restart_at else (
        (restart_at,) if isinstance(restart_at, int) else tuple(restart_at))
    gdt = gram_dtype or ns_dtype
    R = symmul(X).to(gdt)
    Q = None
    last = len(coeffs) - 1
    for k, (a, b, c) in enumerate(coeffs):
        C = symmul_axpy(R, b, c)
        C.diagonal(dim1=-2, dim2=-1).add_(a)
        Q = C if Q is None else symmul2(C, Q)
        if k != last:
            R = symmul2(symmul(C), R)
        if k + 1 in resets and k != last:
            X = torch.bmm(Q.to(ns_dtype), X)
            R = symmul(X).to(gdt)
            Q = None
    X = torch.bmm(Q.to(ns_dtype), X)

    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


class GramNewtonSchulz:

    def __init__(self, ns_coefficients=_DSV4_COEFFS,
                 gram_newton_schulz_reset_iterations=GRAM_RESTART_AT,
                 ns_dtype=torch.bfloat16, gram_dtype=None):
        self.coeffs = tuple(tuple(float(v) for v in row) for row in ns_coefficients)
        self.resets = tuple(gram_newton_schulz_reset_iterations or ())
        if any(not 1 <= r < len(self.coeffs) for r in self.resets):
            raise ValueError(f"reset iterations must be in [1, {len(self.coeffs) - 1}] "
                             f"(a reset after the last iteration is a no-op): {self.resets}")
        self.ns_dtype = ns_dtype
        self.gram_dtype = gram_dtype

    def __call__(self, X):
        return newton_schulz_gram(X, self.coeffs, self.ns_dtype,
                                  gram_dtype=self.gram_dtype, restart_at=self.resets)


def autotune_restarts(coeffs, num_restarts=1, shape=(2048, 8192), kappas=(1e2, 1e4, 1e6),
                      ns_dtype=torch.bfloat16, seed=0, verbose=True, bench=True):
    from itertools import combinations
    n, m = shape
    if min(n, m) < SYMMUL_MIN_DIM or max(n, m) / min(n, m) < GRAM_MIN_RATIO:
        raise ValueError(f"shape {shape} is below the gram gates "
                         f"(dim >= {SYMMUL_MIN_DIM}, r >= {GRAM_MIN_RATIO}) — it would "
                         "dispatch to symmul and tune nothing")
    coeffs = tuple(tuple(float(v) for v in row) for row in coeffs)
    torch.manual_seed(seed)
    cases = []
    for kappa in kappas:
        g = torch.randn(n, m, device="cuda", dtype=torch.float32)
        U, _, Vh = torch.linalg.svd(g, full_matrices=False)
        s = torch.logspace(0, -math.log10(kappa), n, device="cuda")
        X = ((U * s) @ Vh).to(ns_dtype)
        truth = (U @ Vh).double()
        ref = _newton_schulz_cublas(X, coeffs, ns_dtype)
        e_ref = ((ref.double() - truth).norm() / truth.norm()).item()
        cases.append((kappa, X, truth, e_ref))
    best, best_score = None, float("inf")
    for resets in combinations(range(1, len(coeffs)), num_restarts):
        errs, score = [], 0.0
        for kappa, X, truth, e_ref in cases:
            out = newton_schulz_gram(X, coeffs, ns_dtype, restart_at=resets)
            e = ((out.double() - truth).norm() / truth.norm()).item()
            errs.append((kappa, e))
            ratio = e / max(e_ref, 1e-12)
            score = max(score, ratio if math.isfinite(ratio) else float("inf"))
        if verbose:
            detail = "  ".join(f"kappa=1e{int(math.log10(k))}: {e:.4e}" for k, e in errs)
            print(f"restarts {list(resets)}: worst-ratio-vs-champion {score:.4f}  ({detail})", flush=True)
        if score < best_score:
            best, best_score = list(resets), score
    if verbose:
        print(f"best: {best}  (worst ratio {best_score:.4f}; 1.0 = champion-grade)")
    if bench:
        from triton.testing import do_bench
        Xb = cases[0][1]
        t_gram = do_bench(lambda: newton_schulz_gram(Xb, coeffs, ns_dtype, restart_at=tuple(best)),
                          warmup=10, rep=50)
        t_sym = do_bench(lambda: newton_schulz_symmul(Xb, coeffs, ns_dtype), warmup=10, rep=50)
        t_cub = do_bench(lambda: _newton_schulz_cublas(Xb, coeffs, ns_dtype), warmup=10, rep=50)
        if verbose:
            print(f"speed @ {shape}: gram{best} {t_gram:.3f} ms | "
                  f"symmul {t_sym:.3f} ms ({t_sym / t_gram:.2f}x) | "
                  f"cuBLAS {t_cub:.3f} ms ({t_cub / t_gram:.2f}x)", flush=True)
    return best


def _selfcheck_and_bench():
    from triton.testing import do_bench
    torch.manual_seed(0)
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    variants = {
        "champion (cuBLAS)":  lambda X: _newton_schulz_cublas(X),
        "symmul NS":          lambda X: newton_schulz_symmul(X),
        "gram NS (default)":  lambda X: newton_schulz_gram(X),
        "gram NS no-restart": lambda X: newton_schulz_gram(X, restart_at=()),
        "gram NS fp32-gram":  lambda X: newton_schulz_gram(X, gram_dtype=torch.float32),
    }
    shapes = [(2048, 8192), (2048, 4096), (2048, 2048), (3072, 8192)]
    for n, m in shapes:
        X0 = torch.randn(n, m, device=dev, dtype=torch.float16)
        U, _, Vh = torch.linalg.svd(X0.double(), full_matrices=False)
        truth = (U @ Vh)
        ref = variants["champion (cuBLAS)"](X0)
        print(f"\n({n} x {m})  r={m/n:.1f}")
        for name, fn in variants.items():
            out = fn(X0)
            sv = torch.linalg.svdvals(out.float())
            err_truth = (out.double() - truth).norm() / truth.norm()
            err_champ = (out - ref).float().norm() / ref.float().norm()
            ms = do_bench(lambda f=fn: f(X0), warmup=25, rep=50)
            print(f"  {name:<20} {ms:7.2f} ms  vs-champ {err_champ:.2e}  vs-truth {err_truth:.2e}"
                  f"  sv[min/mean/max] {sv.min():.3f}/{sv.mean():.3f}/{sv.max():.3f}")


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser(description="gram NS self-check/bench, or restart autotune")
    _ap.add_argument("--autotune-restarts", action="store_true",
                     help="grid-search restart placements instead of the self-check")
    _ap.add_argument("--num-restarts", type=int, default=1)
    _ap.add_argument("--coefs", type=str, default=None,
                     help='per-iteration a,b,c rows joined by ";", e.g. "4.08,-6.89,2.93;..."'
                          " (default: the shipped Polar-Express coefficients)")
    _args = _ap.parse_args()
    if _args.autotune_restarts or _args.coefs:
        _coeffs = (tuple(tuple(float(v) for v in row.split(",")) for row in _args.coefs.split(";"))
                   if _args.coefs else _DSV4_COEFFS)
        autotune_restarts(_coeffs, num_restarts=_args.num_restarts)
    else:
        _selfcheck_and_bench()
