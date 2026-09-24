"""Newton-Schulz with a fused-epilogue batched GEMM: the same math as kernels.sm75.muon.newton_schulz
without torch.baddbmm.

WHY. torch.baddbmm(C, A, B) runs as a device-to-device copy of C into the output followed by a
beta != 0 GEMM that re-reads it. On the BiBo Newton-Schulz shape (576, 512, 1536) on an RTX PRO 6000
that is 1.30 ms copy + 2.17 ms GEMM, against 1.69 ms for a bare cuBLAS bmm. `bgemm_epi` computes
D = alpha * A @ B + beta * C in one Triton kernel (C read once, in the epilogue, C != D): 1.73 ms, and
the same relative error vs an fp32 GEMM as baddbmm (it rounds the same fp32 accumulator once).
It is NOT a faster GEMM than cuBLAS -- a bare cuBLAS bmm is ~2% faster -- the win is the fusion.
So X @ X^T, which has no epilogue, stays on cuBLAS bmm (1.30 vs 1.51 ms).

Per iteration:  A = X X^T (cuBLAS)   B = c A A + b A (epi)   X = B X + a X (epi)
"""
import torch
import triton
import triton.language as tl

from kernels.sm75.muon import _DSV4_COEFFS

_CONFIGS = [
    triton.Config({"BM": bm, "BN": bn, "BK": bk, "GROUP_M": 8}, num_stages=s, num_warps=w)
    for bm, bn, bk, s, w in [
        (128, 128, 64, 3, 4), (128, 128, 64, 3, 8), (128, 128, 32, 4, 4), (128, 256, 64, 2, 8),
        (256, 128, 64, 2, 8), (128, 64, 64, 4, 4), (64, 128, 64, 4, 4), (128, 128, 64, 2, 4),
        (256, 64, 64, 3, 8), (64, 256, 64, 3, 8), (64, 64, 64, 4, 4),
    ]
]
EPI_BK = 64          # every config's BK divides this, so K % EPI_BK == 0 is the only shape rule
# Below this many elements per NS call the epi path loses to cuBLAS: its 16 Triton launches per call
# cost a ~0.64 ms floor (measured: router 9x64x512 0.33 -> 0.64 ms). The two paths are bit-identical,
# so routing small calls to cuBLAS is free.
EPI_MIN_ELEMS = 4 * 1024 * 1024


# ponytail: key is (M, N, K, HAS_C) only, never the batch -- a grid-size key re-autotunes per batch
@triton.autotune(configs=_CONFIGS, key=["M", "N", "K", "HAS_C"])
@triton.jit
def _bgemm_epi(A, B, C, D, M, N, K, alpha, beta,
               sab, sam, sak, sbb, sbk, sbn, scb, scm, scn, sdb, sdm, sdn,
               HAS_C: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
               GROUP_M: tl.constexpr):
    pid = tl.program_id(0)
    bat = tl.program_id(1).to(tl.int64)
    num_m = tl.cdiv(M, BM)
    num_n = tl.cdiv(N, BN)
    group = GROUP_M * num_n
    first_m = (pid // group) * GROUP_M
    gsz = tl.minimum(num_m - first_m, GROUP_M)
    pm = first_m + (pid % group) % gsz
    pn = (pid % group) // gsz
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    mm, mn = rm < M, rn < N
    a_ptr = A + bat * sab + rm[:, None] * sam + rk[None, :] * sak
    b_ptr = B + bat * sbb + rk[:, None] * sbk + rn[None, :] * sbn
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for _ in range(0, K, BK):
        a = tl.load(a_ptr, mask=mm[:, None], other=0.0)
        b = tl.load(b_ptr, mask=mn[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
        a_ptr += BK * sak
        b_ptr += BK * sbk
    acc = acc * alpha
    msk = mm[:, None] & mn[None, :]
    if HAS_C:
        c = tl.load(C + bat * scb + rm[:, None] * scm + rn[None, :] * scn, mask=msk, other=0.0)
        acc += beta * c.to(tl.float32)
    tl.store(D + bat * sdb + rm[:, None] * sdm + rn[None, :] * sdn, acc.to(D.dtype.element_ty), mask=msk)


def bgemm_epi(A, B, C=None, alpha=1.0, beta=0.0, out=None):
    """alpha * A @ B + beta * C for (n, M, K) @ (n, K, N), any strides. `out` must not alias C."""
    n, M, K = A.shape
    N = B.shape[2]
    if K % EPI_BK or B.shape[:2] != (n, K):
        raise ValueError(f"bgemm_epi needs K % {EPI_BK} == 0 and matching batch/K, got {A.shape} @ {B.shape}")
    D = torch.empty((n, M, N), device=A.device, dtype=A.dtype) if out is None else out
    Cc = C if C is not None else D
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(N, meta["BN"]), n)
    _bgemm_epi[grid](A, B, Cc, D, M, N, K, float(alpha), float(beta),
                     *A.stride(), *B.stride(), *Cc.stride(), *D.stride(), HAS_C=C is not None)
    return D


def newton_schulz_epi(G, coeffs=_DSV4_COEFFS, ns_dtype=torch.bfloat16, eps=1e-7):
    """Drop-in for kernels.sm75.muon.newton_schulz (same normalize / orientation / dtype rules)."""
    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype) / nrm.to(ns_dtype)
    n, m, _ = X.shape
    if m % EPI_BK or X.numel() < EPI_MIN_ELEMS:      # the small side is the K of both epi GEMMs
        from kernels.sm75.muon import newton_schulz
        return newton_schulz(G, coeffs, ns_dtype, eps)
    X = X.contiguous()
    A = torch.empty((n, m, m), device=X.device, dtype=ns_dtype)
    Bm = torch.empty_like(A)
    Xb = torch.empty_like(X)
    for a, b, c in coeffs:
        torch.bmm(X, X.transpose(1, 2), out=A)
        bgemm_epi(A, A, A, alpha=c, beta=b, out=Bm)
        bgemm_epi(Bm, X, X, alpha=1.0, beta=a, out=Xb)
        X, Xb = Xb, X
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)
