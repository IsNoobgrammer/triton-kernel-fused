"""Symmetric-matmul ("symmul") Newton-Schulz — the amalgamated lever (sm120 / Blackwell).

NEW, ADDITIVE, FLAGGED. The champion `kernels.sm75.muon.newton_schulz` (cuBLAS bmm + baddbmm
fold) is UNTOUCHED. This adds an ORTHOGONAL optimization dimension on top of it: the two NS
GEMMs `A = X X^T` and `A A` are SYMMETRIC, so we compute only the upper triangle of tiles and
mirror them by a register->global transpose-copy at the epilogue (~half the GEMM FLOPs). The
non-symmetric `B X` stays cuBLAS `baddbmm`.

Thesis (see .autoresearch/scope.md): our FusedMuon wins on dimensions ORTHOGONAL to the GEMM
FLOP count (foreach launch-collapse, batched same-shape state, baddbmm epilogue fold). The
symmetric FLOP cut is a DIFFERENT axis. Stacking them is potentially multiplicative in the
compute-bound / large-matrix regime — exactly where flash-muon measures ~1.5-1.8x on the
symmetric matmul alone (A100/H800/4090 at dim>=2048) and where our fused-vs-compiled gap
shrank to ~1.24x.

The triangle+transpose-copy kernel is adapted from nil0x9/flash-muon's 2D `mmt_kernel`
(itself from the Triton matmul tutorial), with a BATCH dimension added (program_id axis 1 +
batch strides) so it serves Muon's batched same-shape state in one launch. The transpose-copy
is the documented correctness risk (Laker Newhouse's ThunderKittens version had a
transpose-store bug; flash-muon claims fixed) -> the frozen eval gates parity HARD.

Toolchain: Triton-only (no nvcc on the box). fp16/bf16 inputs, fp32 accumulate.
"""
import torch
import triton
import triton.language as tl

from kernels.sm75.muon import _PE_COEFFS, newton_schulz as _newton_schulz_cublas  # noqa: F401


# Shape-dispatch threshold on the GRAM dim (min(rows,cols)): at/above this the symmetric FLOP cut
# beats cuBLAS, below it loses. MEASURED on RTX PRO 6000 (fp16): gram=1024 symmul=0.71x (loss),
# gram=2048 symmul=1.22x (win) -> knee at 2048. Below the knee newton_schulz_symmul returns the
# CHAMPION verbatim (cuBLAS+baddbmm) so the batched-small / small-matrix regime never regresses.
SYMMUL_MIN_DIM = 2048


def _bmmt_configs():
    return [
        triton.Config({"BM": bm, "BK": bk, "GROUP_M": 8}, num_stages=ns, num_warps=nw)
        for bm in (64, 128, 256)
        for bk in (32, 64)
        for ns in (3, 4)
        for nw in (4, 8)
    ]


@triton.autotune(configs=_bmmt_configs(), key=["M", "K"])
@triton.jit
def _bmmt_kernel(
    x_ptr, y_ptr,
    M, K,
    stride_xb, stride_xm, stride_xk,
    stride_yb, stride_ym, stride_yn,
    BM: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    """Batched y[b] = x[b] @ x[b].T, computing only upper-triangle tiles and mirroring them.

    Adapted from flash-muon mmt_kernel: axis-0 walks the M x M output tiles with group swizzle;
    axis-1 is the batch. Lower-triangle tiles early-exit; the upper tile is computed, stored, and
    (off-diagonal) transpose-copied to its mirror. Output is EXACTLY symmetric (diagonal tiles are
    computed in full; off-diagonal mirrored by copy) so a second symmul(A)=A@A^T=A@A is valid.
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
    if pid_m > pid_n:                                   # lower triangle: skip, it's a mirror
        return

    x_ptr += bid * stride_xb
    y_ptr += bid * stride_yb

    offs_xm = (pid_m * BM + tl.arange(0, BM)) % M
    offs_xn = (pid_n * BM + tl.arange(0, BM)) % M
    offs_k = tl.arange(0, BK)
    a_ptrs = x_ptr + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x_ptr + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    acc = tl.zeros((BM, BM), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BK)):
        kmask = offs_k[None, :] < K - k * BK
        a = tl.load(a_ptrs, mask=kmask, other=0.0)
        b = tl.load(b_ptrs, mask=kmask, other=0.0)
        acc = tl.dot(a, tl.permute(b, (1, 0)), acc)
        a_ptrs += BK * stride_xk
        b_ptrs += BK * stride_xk
    c = acc.to(y_ptr.dtype.element_ty)

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BM + tl.arange(0, BM)
    c_ptrs = y_ptr + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    if pid_m < pid_n:                                   # mirror upper tile into the lower triangle
        ct_ptrs = y_ptr + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def symmul(X, out=None):
    """Batched symmetric product X @ X^T via the triangle+mirror kernel.

    X: (B, M, K) or (M, K). Returns (B, M, M) (or (M, M) for 2D input). When M < SYMMUL_MIN_DIM
    the symmetric cut does not pay -> dispatch to cuBLAS bmm. `out` lets the caller reuse a buffer.
    """
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)
    B, M, K = X.shape
    if M < SYMMUL_MIN_DIM:                              # below the knee: cuBLAS wins, don't launch Triton
        Y = torch.bmm(X, X.transpose(1, 2)) if out is None else torch.bmm(X, X.transpose(1, 2), out=out)
        return Y.squeeze(0) if squeeze else Y
    X = X.contiguous()
    Y = torch.empty((B, M, M), device=X.device, dtype=X.dtype) if out is None else out
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(M, meta["BM"]), B)
    _bmmt_kernel[grid](
        X, Y, M, K,
        X.stride(0), X.stride(1), X.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
    )
    return Y.squeeze(0) if squeeze else Y


def newton_schulz_symmul(G, coeffs=_PE_COEFFS, ns_dtype=torch.float16, eps=1e-7):
    """Polar-Express Newton-Schulz with the two SYMMETRIC GEMMs done by the symmul kernel.

    Bit-for-bit the same algorithm as `kernels.sm75.muon.newton_schulz` (same PE coeffs, same
    normalization/orientation, same `B X` via cuBLAS). The ONLY change: `A = X X^T` and `A A`
    use `symmul` instead of `bmm`. The polynomial `b*A + c*AA` is an explicit axpy here (we lose
    the baddbmm fold on those two terms — the tradeoff the loop measures), but the GEMM FLOPs are
    ~halved. `symmul` self-dispatches to cuBLAS below SYMMUL_MIN_DIM so small matrices never regress.
    """
    # Gate on the Gram dim min(rows,cols): below the knee the symmetric cut loses, so return the
    # champion verbatim (cuBLAS + baddbmm fold) -> the batched-small / small-matrix regime never
    # regresses, by construction (identical op to the champion, not a re-implementation).
    gram = min(G.shape[-2], G.shape[-1])
    if gram < SYMMUL_MIN_DIM:
        return _newton_schulz_cublas(G, coeffs, ns_dtype, eps)

    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)                  # iterate on the smaller Gram
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype) / nrm.to(ns_dtype)
    X = X.contiguous()
    Bsz, M, _ = X.shape
    # Preallocate every buffer ONCE and reuse across all iterations (zero per-iter allocation), to
    # out-plan inductor's buffer reuse and hold peak at/under `compiled`. A/AA are the two Gram
    # buffers; the polynomial b*A + c*AA folds IN-PLACE into AA (B aliases it). The B@X result
    # PING-PONGS between X and Xb (baddbmm out=) so the iteration never allocates a fresh X. Live
    # set = {X, Xb, A, AA} for the whole loop, fixed.
    A = torch.empty((Bsz, M, M), device=X.device, dtype=ns_dtype)
    AA = torch.empty_like(A)
    Xb = torch.empty_like(X)
    for a, b, c in coeffs:
        symmul(X, out=A)                                # A = X X^T   (symmetric -> half FLOPs)
        symmul(A, out=AA)                               # AA = A A^T = A A  (A is exactly symmetric)
        AA.mul_(c).add_(A, alpha=b)                     # B = b*A + c*AA, in-place into AA (no new alloc)
        torch.baddbmm(X, AA, X, beta=a, alpha=1.0, out=Xb)  # Xb = a*X + B X  (out= -> reuse, no alloc)
        X, Xb = Xb, X                                   # ping-pong: next iter overwrites the old X
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)
