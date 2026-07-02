"""Gram-space Newton-Schulz ("gram NS") — iterate on the n x n Gram matrix, not on X.

Math (same identity as Dao-AILab/gram-newton-schulz): one X-space NS step is
X <- C X with C = aI + bA + cA^2, A = X X^T. Every C_k and A_k is a polynomial in
G = X0 X0^T, so they are ALL symmetric and ALL commute. Consequences:

  1. A (here `R`) can be updated in Gram space: R <- C R C^T = C^2 R. The five
     rectangular `B X` GEMMs (cost r*n^3 each, r = m/n) collapse into ONE final
     apply X = Q X0, with Q = prod C_k accumulated as Q <- C Q.
  2. Every product in the loop (C^2, C^2 R, C Q) has a symmetric result, so the
     triangle+mirror symmul kernels halve ALL of them (symmul2 below adds the
     two-input case S1 @ S2).

FLOPs (units of n^3, symmul-halved): gram NS = 1.5r + 8.5 vs symmul NS = 7.5r + 2.5
-> tie at r=1, 1.52x at r=2, 2.24x at r=4. Gate on r: square matrices keep symmul NS.

Numerics: kappa(Gram) = kappa(X)^2, and a pure Gram loop never re-reads X, so the
NS self-correction is lost — on ILL-CONDITIONED inputs (kappa>=1e2; real momentum
matrices) plain gram drifts (vs-truth 0.193 vs champion 0.122 at kappa=1e2) and
fp32 does NOT fix it (0.207 — the Gram's small eigenvalues are sigma^2, already
crushed at fp16 formation; the failure is algorithmic, not roundoff). Dao's
RESTART is the fix: materialize X = Q X mid-run, refresh R, reset Q — re-anchoring
restores self-correction. MEASURED (RTX PRO 6000): restart@3 matches the champion
to the 4th digit at kappa = 1e2/1e4/1e6, costs ~1.5r once. restart_at=3 is the
DEFAULT; gram_dtype=torch.float32 is kept as a flag but measured unnecessary.

Self-check + local bench: python -m kernels.sm120.newton_schulz_gram
"""
import math

import torch
import triton
import triton.language as tl

from kernels.sm75.muon import _PE_COEFFS, newton_schulz as _newton_schulz_cublas
from kernels.sm120.newton_schulz_symmul import (
    SYMMUL_MIN_DIM, _bmmt_configs, symmul, symmul_axpy, newton_schulz_symmul,
)

# The Gram algorithm wins iff r = m/n > 1 (FLOP tie at r=1, and the extra kernel
# launches lose the tie in practice). MEASURED on RTX PRO 6000 (fp16, gram=2048):
# r=1.0 0.93x + parity 2.1e-2 (loss), r=1.25 0.99x, r=1.5 1.20x, r=1.75 1.24x,
# r=2 1.27-1.30x, r=2.7 1.65x, r=4 1.81x (2.24x batched) -> knee at 1.5.
GRAM_MIN_RATIO = 1.5

# Restart after this iteration (1-based; Dao recommends 1 restart for 5 NS steps).
# Ill-conditioned eval: restart@3 == champion parity at kappa 1e2..1e6; without it
# (and with fp32) gram drifts. None disables (only safe for well-conditioned input).
GRAM_RESTART_AT = 3


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
    """Batched y[b] = S1[b] @ S2[b] for SYMMETRIC, COMMUTING S1/S2 (M x M) -> symmetric y.

    Same triangle+mirror walk as _bmmt_kernel; the only change is two input pointers.
    Column n of S2 is row n (symmetry), so the B tile loads rows of S2 and tl.dot
    takes its transpose — no strided column loads.
    """
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
    if pid_m > pid_n:                                   # lower triangle: mirror of upper
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
    """Batched S1 @ S2 for symmetric commuting inputs (result symmetric) — halved FLOPs.

    (B, M, M) x (B, M, M) -> (B, M, M). Below SYMMUL_MIN_DIM cuBLAS bmm wins -> fall back.
    """
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


def newton_schulz_gram(G, coeffs=_PE_COEFFS, ns_dtype=torch.float16, eps=1e-7,
                       gram_dtype=None, restart_at=GRAM_RESTART_AT, force_eager=False):
    """Polar-Express NS via the Gram recurrence: R <- C^2 R, Q <- C Q, X_out = Q X0.

    Same normalization/orientation/coeffs as the champion. Gates: falls back to
    newton_schulz_symmul when r = m/n < GRAM_MIN_RATIO (no FLOP win at r~1), which
    itself falls back to the cuBLAS champion below the gram-dim knee.

    gram_dtype: dtype of the n^3 Gram loop (default ns_dtype; measured NOT a stabilizer).
    restart_at: 1-based iteration(s) after which to refresh X/R/Q — an int, an iterable
        of ints ([2, 4] restarts after iterations 2 AND 4), or None/() for no restarts
        (only safe for well-conditioned input). Default GRAM_RESTART_AT (3 of 5).
    """
    n, m = G.shape[-2], G.shape[-1]
    r = max(n, m) / min(n, m)
    if min(n, m) < SYMMUL_MIN_DIM or r < GRAM_MIN_RATIO:
        return newton_schulz_symmul(G, coeffs, ns_dtype, eps, force_eager=force_eager)

    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)                  # iterate the Gram of the SMALLER side
    if transposed:
        X = X.transpose(1, 2)
    X = (X.to(ns_dtype) / nrm.to(ns_dtype)).contiguous()

    resets = () if not restart_at else (
        (restart_at,) if isinstance(restart_at, int) else tuple(restart_at))
    gdt = gram_dtype or ns_dtype
    R = symmul(X).to(gdt)                               # R0 = X X^T  (0.5 r)
    Q = None
    last = len(coeffs) - 1
    for k, (a, b, c) in enumerate(coeffs):
        C = symmul_axpy(R, b, c)                        # bR + cR^2   (0.5)
        C.diagonal(dim1=-2, dim2=-1).add_(a)            # C = aI + bR + cR^2
        Q = C if Q is None else symmul2(C, Q)           # Q <- C Q    (0.5)
        if k != last:
            R = symmul2(symmul(C), R)                   # R <- C^2 R  (1.0)
        if k + 1 in resets and k != last:
            X = torch.bmm(Q.to(ns_dtype), X)            # materialize, refresh, reset
            R = symmul(X).to(gdt)
            Q = None
    X = torch.bmm(Q.to(ns_dtype), X)                    # the ONE rectangular apply (r)

    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


class GramNewtonSchulz:
    """Dao-style callable API (mirrors Dao-AILab/gram-newton-schulz):

        gram_NS = GramNewtonSchulz(ns_coefficients=_PE_COEFFS,
                                   gram_newton_schulz_reset_iterations=[3])
        Y = gram_NS(X)

    ns_coefficients: list of (a, b, c) per NS iteration. reset_iterations: 1-based
    iterations immediately after which to restart ([2, 4] = after the 2nd AND 4th).
    Find placements for custom coefficients with `autotune_restarts`.
    """

    def __init__(self, ns_coefficients=_PE_COEFFS,
                 gram_newton_schulz_reset_iterations=(GRAM_RESTART_AT,),
                 ns_dtype=torch.float16, gram_dtype=None):
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
                      ns_dtype=torch.float16, seed=0, verbose=True):
    """Grid-search restart placement(s) for a coefficient set (GPU required).

    Scores every combination of `num_restarts` positions in [1, len(coeffs)-1] on
    ill-conditioned inputs (log-spaced singular values at each kappa) by the WORST
    error ratio vs the cuBLAS champion NS running the SAME coefficients — ratio 1.0
    means the restarts fully restore champion-grade stability (all placements cost
    the same ~1.5r, so accuracy is the only criterion). Returns the best placement
    as a list, e.g. [3], ready for GramNewtonSchulz(..., reset_iterations=best).
    """
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
            score = max(score, e / max(e_ref, 1e-12))
        if verbose:
            detail = "  ".join(f"kappa=1e{int(math.log10(k))}: {e:.4e}" for k, e in errs)
            print(f"restarts {list(resets)}: worst-ratio-vs-champion {score:.4f}  ({detail})", flush=True)
        if score < best_score:
            best, best_score = list(resets), score
    if verbose:
        print(f"best: {best}  (worst ratio {best_score:.4f}; 1.0 = champion-grade)")
    return best


def _selfcheck_and_bench():                             # pragma: no cover
    """Parity vs the cuBLAS champion + fp64-SVD ground truth, then do_bench on this GPU."""
    from triton.testing import do_bench
    torch.manual_seed(0)
    dev = "cuda"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    variants = {
        "champion (cuBLAS)": lambda X: _newton_schulz_cublas(X),
        "symmul NS":         lambda X: newton_schulz_symmul(X),
        "gram NS":           lambda X: newton_schulz_gram(X),
        "gram NS restart@3": lambda X: newton_schulz_gram(X, restart_at=3),
        "gram NS fp32-gram": lambda X: newton_schulz_gram(X, gram_dtype=torch.float32),
        "gram NS fp32+r@3":  lambda X: newton_schulz_gram(X, gram_dtype=torch.float32, restart_at=3),
    }
    shapes = [(2048, 8192), (2048, 4096), (2048, 2048), (3072, 8192)]
    for n, m in shapes:
        X0 = torch.randn(n, m, device=dev, dtype=torch.float16)
        # fp64 ground truth: the exact polar factor U V^T
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


if __name__ == "__main__":                              # pragma: no cover
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
                   if _args.coefs else _PE_COEFFS)
        autotune_restarts(_coeffs, num_restarts=_args.num_restarts)
    else:
        _selfcheck_and_bench()
