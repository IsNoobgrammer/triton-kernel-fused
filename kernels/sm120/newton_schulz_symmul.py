import torch
import triton
import triton.language as tl

from kernels.sm75.muon import _DSV4_COEFFS, newton_schulz as _newton_schulz_cublas


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


def symmul(X, out=None):
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)
    B, M, K = X.shape
    if M < SYMMUL_MIN_DIM:
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
    ablk_ptrs = x_ptr + stride_xm * offs_cm[:, None] + stride_xk * offs_cn[None, :]
    ablk_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    ablk = tl.load(ablk_ptrs, mask=ablk_mask, other=0.0).to(tl.float32)
    c = (SAA * acc + SA * ablk).to(y_ptr.dtype.element_ty)

    c_ptrs = y_ptr + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)
    if pid_m < pid_n:
        ct_ptrs = y_ptr + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def symmul_axpy(A, sa, saa, out=None):
    squeeze = A.ndim == 2
    if squeeze:
        A = A.unsqueeze(0)
    B, M, K = A.shape
    if M < SYMMUL_MIN_DIM:
        AA = torch.baddbmm(A, A, A, beta=sa, alpha=saa)
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
    for a, b, c in coeffs:
        A = torch.ops.symmul_muon.mmt(X)
        B = torch.ops.symmul_muon.mmt_axpy(A, b, c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    return X


try:
    _amalg_compiled = torch.compile(_amalg_core)
except Exception:
    _amalg_compiled = None


def _amalg_eager(X, coeffs):
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


AMALG_COMPILE = _amalg_compiled is not None


def newton_schulz_symmul(G, coeffs=_DSV4_COEFFS, ns_dtype=torch.bfloat16, eps=1e-7, force_eager=False):
    gram = min(G.shape[-2], G.shape[-1])
    if gram < SYMMUL_MIN_DIM:
        return _newton_schulz_cublas(G, coeffs, ns_dtype, eps)

    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = (X.to(ns_dtype) / nrm.to(ns_dtype)).contiguous()
    if AMALG_COMPILE and not force_eager:
        try:
            X = _amalg_compiled(X, coeffs)
        except Exception:
            X = _amalg_eager(X, coeffs)
    else:
        X = _amalg_eager(X, coeffs)
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)
