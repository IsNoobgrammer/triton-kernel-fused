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

Numerics: kappa(Gram) = kappa(X)^2 and Q is a chain of 5 low-precision products.
Two stabilizers, both flagged so the bench picks:
  restart_at=k : Dao's fix — materialize X = Q X, refresh R from it, reset Q (costs
                 ~1.5r extra rectangular work once).
  gram_dtype=torch.float32 : run the n^3 Gram loop in fp32 (tf32 tensor cores on
                 Ampere+); the loop is n^3 not n^2*m, so at r>=2 this still wins
                 while being MORE precise than the fp16 X-space loop.

Self-check + local bench: python -m kernels.sm120.newton_schulz_gram
"""
import torch
import triton
import triton.language as tl

from kernels.sm75.muon import _PE_COEFFS, newton_schulz as _newton_schulz_cublas
from kernels.sm120.newton_schulz_symmul import (
    SYMMUL_MIN_DIM, _bmmt_configs, symmul, symmul_axpy, newton_schulz_symmul,
)

# The Gram algorithm wins iff r = m/n > 1 (FLOP tie at r=1, and the extra kernel
# launches lose the tie in practice). Knee is a ratio, not a dim; measured locally.
GRAM_MIN_RATIO = 2.0


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
                       gram_dtype=None, restart_at=None):
    """Polar-Express NS via the Gram recurrence: R <- C^2 R, Q <- C Q, X_out = Q X0.

    Same normalization/orientation/coeffs as the champion. Gates: falls back to
    newton_schulz_symmul when r = m/n < GRAM_MIN_RATIO (no FLOP win at r~1), which
    itself falls back to the cuBLAS champion below the gram-dim knee.

    gram_dtype: dtype of the n^3 Gram loop (default ns_dtype; float32 = stabilized).
    restart_at: 1-based iteration count after which to refresh X/R/Q (Dao: 3 of 5).
    """
    n, m = G.shape[-2], G.shape[-1]
    r = max(n, m) / min(n, m)
    if min(n, m) < SYMMUL_MIN_DIM or r < GRAM_MIN_RATIO:
        return newton_schulz_symmul(G, coeffs, ns_dtype, eps)

    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)                  # iterate the Gram of the SMALLER side
    if transposed:
        X = X.transpose(1, 2)
    X = (X.to(ns_dtype) / nrm.to(ns_dtype)).contiguous()

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
        if restart_at is not None and k + 1 == restart_at and k != last:
            X = torch.bmm(Q.to(ns_dtype), X)            # materialize, refresh, reset
            R = symmul(X).to(gdt)
            Q = None
    X = torch.bmm(Q.to(ns_dtype), X)                    # the ONE rectangular apply (r)

    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


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
    _selfcheck_and_bench()
