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


@triton.autotune(configs=_bmmt_configs(), key=["M", "K"])
@triton.jit
def _bmmt_axpy_kernel(
    x_ptr, y_ptr,
    M, K, SA, SAA,
    stride_xb, stride_xm, stride_xk,
    stride_yb, stride_ym, stride_yn,
    BM: tl.constexpr, BK: tl.constexpr, GROUP_M: tl.constexpr,
):
    """Fused: batched y[b] = SAA*(A[b] @ A[b]^T) + SA*A[b], for SYMMETRIC square A (M==K).

    Same triangle+mirror as _bmmt_kernel, but the epilogue also loads the (m,n) block of A itself
    and folds the polynomial b*A + c*(A A) in-register before the store -> no separate AA buffer and
    no elementwise mul/add passes. Output B is symmetric (A symmetric), so the mirror tile is the
    transpose, exactly as for the plain symmul.
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
    if pid_m > pid_n:
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

    offs_cm = pid_m * BM + tl.arange(0, BM)
    offs_cn = pid_n * BM + tl.arange(0, BM)
    # load the (m,n) block of A itself for the SA*A term (A is square M x M here)
    ablk_ptrs = x_ptr + stride_xm * offs_cm[:, None] + stride_xk * offs_cn[None, :]
    ablk_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    ablk = tl.load(ablk_ptrs, mask=ablk_mask, other=0.0).to(tl.float32)
    c = (SAA * acc + SA * ablk).to(y_ptr.dtype.element_ty)

    c_ptrs = y_ptr + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)
    if pid_m < pid_n:                                   # B is symmetric -> mirror tile is the transpose
        ct_ptrs = y_ptr + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def symmul_axpy(A, sa, saa, out=None):
    """B = saa*(A A^T) + sa*A for a symmetric square A (B,M,M)/(M,M). Fuses the NS polynomial."""
    squeeze = A.ndim == 2
    if squeeze:
        A = A.unsqueeze(0)
    B, M, K = A.shape
    if M < SYMMUL_MIN_DIM:
        AA = torch.baddbmm(A, A, A, beta=sa, alpha=saa)   # cuBLAS fold below the knee
        return AA.squeeze(0) if squeeze else AA
    A = A.contiguous()
    Y = torch.empty((B, M, M), device=A.device, dtype=A.dtype) if out is None else out
    grid = lambda meta: (triton.cdiv(M, meta["BM"]) * triton.cdiv(M, meta["BM"]), B)
    _bmmt_axpy_kernel[grid](
        A, Y, M, K, float(sa), float(saa),
        A.stride(0), A.stride(1), A.stride(2),
        Y.stride(0), Y.stride(1), Y.stride(2),
    )
    return Y.squeeze(0) if squeeze else Y


# ── custom ops so torch.compile can plan buffers AROUND the Triton kernels (inductor owns X reuse) ──
@torch.library.custom_op("symmul_muon::mmt", mutates_args=())
def _mmt_op(X: torch.Tensor) -> torch.Tensor:
    return symmul(X)


@_mmt_op.register_fake
def _(X):
    return X.new_empty((X.shape[0], X.shape[1], X.shape[1]))


@torch.library.custom_op("symmul_muon::mmt_axpy", mutates_args=())
def _mmt_axpy_op(A: torch.Tensor, sa: float, saa: float) -> torch.Tensor:
    return symmul_axpy(A, sa, saa)


@_mmt_axpy_op.register_fake
def _(A, sa, saa):
    return torch.empty_like(A)


def _amalg_core(X, coeffs):
    """Functional NS core: symmul + fused symmul-axpy + cuBLAS B@X. Compiled so inductor plans the
    X-reuse in-place (matching `compiled`'s peak) while the two symmetric GEMMs stay halved."""
    for a, b, c in coeffs:
        A = torch.ops.symmul_muon.mmt(X)                # A = X X^T
        B = torch.ops.symmul_muon.mmt_axpy(A, b, c)     # B = b*A + c*(A A), one fused kernel
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)   # a*X + B X
    return X


_amalg_compiled = torch.compile(_amalg_core)


def _amalg_eager(X, coeffs):
    """Eager path (fallback): preallocate + reuse A/B/Xb, fused symmul-axpy kills the AA buffer."""
    Bsz, M, _ = X.shape
    A = torch.empty((Bsz, M, M), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    Xb = torch.empty_like(X)
    for a, b, c in coeffs:
        symmul(X, out=A)
        symmul_axpy(A, b, c, out=B)
        torch.baddbmm(X, B, X, beta=a, alpha=1.0, out=Xb)
        X, Xb = Xb, X
    return X


# Use the torch.compile path by default (inductor buffer planning); fall back to eager on any error.
AMALG_COMPILE = True


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
    X = (X.to(ns_dtype) / nrm.to(ns_dtype)).contiguous()
    if AMALG_COMPILE:
        try:
            X = _amalg_compiled(X, coeffs)
        except Exception:                               # graph-unfriendly env -> eager (keeps the speed win)
            X = _amalg_eager(X, coeffs)
    else:
        X = _amalg_eager(X, coeffs)
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)
