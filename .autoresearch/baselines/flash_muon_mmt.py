"""BASELINE (do not optimize) — flash-muon's 2D single-matrix symmetric matmul, verbatim.

Source: nil0x9/flash-muon `matmul_transpose_triton.py` (Triton default impl; Tianyang Lin, 2025).
Vendored UNCHANGED as the "triu" reference baseline in the frozen eval. This is the symmetric
lever WITHOUT batching — it serves one (M,K) matrix per launch. Our amalgamated candidate adds a
batch dimension (kernels/sm120/newton_schulz_symmul.py); comparing the two isolates the value of
batching the symmetric kernel for Muon's same-shape state.

Idea originally proposed by Laker Newhouse et al. (faster symmul with ThunderKittens).
"""
import torch
import triton
import triton.language as tl


def get_autotune_config():
    return [
        triton.Config({"BLOCK_SIZE_M": blk_m, "BLOCK_SIZE_K": blk_k, "GROUP_SIZE_M": grp_sz},
                      num_stages=n_stages, num_warps=n_warps)
        for blk_m in [32, 64, 128]
        for blk_k in [32, 64]
        for grp_sz in [8]
        for n_stages in [3, 4, 5]
        for n_warps in [4, 8]
    ]


@triton.autotune(configs=get_autotune_config(), key=["M", "K"])
@triton.jit
def mmt_kernel(
    x, y,
    M, K,
    stride_xm, stride_xk,
    stride_ym, stride_yn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    if pid_m > pid_n:
        return

    offs_xm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_xn = (pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = x + (offs_xm[:, None] * stride_xm + offs_k[None, :] * stride_xk)
    b_ptrs = x + (offs_xn[:, None] * stride_xm + offs_k[None, :] * stride_xk)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_M), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, tl.permute(b, (1, 0)), accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_xk
        b_ptrs += BLOCK_SIZE_K * stride_xk
    c = accumulator.to(x.dtype.element_ty)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    c_ptrs = y + stride_ym * offs_cm[:, None] + stride_yn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < M)
    tl.store(c_ptrs, c, mask=c_mask)

    if pid_m < pid_n:
        ct_ptrs = y + stride_ym * offs_cn[:, None] + stride_yn * offs_cm[None, :]
        ct_mask = (offs_cn[:, None] < M) & (offs_cm[None, :] < M)
        tl.store(ct_ptrs, tl.permute(c, (1, 0)), mask=ct_mask)


def matmul_transpose_assign(d_in, d_out):
    assert d_in.is_cuda and d_out.is_cuda and d_in.ndim == 2 and d_out.ndim == 2
    d_in = d_in.contiguous()
    M, K = d_in.shape
    grid = lambda META: (triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(M, META["BLOCK_SIZE_M"]),)
    with torch.cuda.device(d_in.device.index):
        mmt_kernel[grid](d_in, d_out, M, K, d_in.stride(0), d_in.stride(1), d_out.stride(0), d_out.stride(1))


def matmul_transpose(d_in):
    M, _ = d_in.shape
    d_out = torch.empty((M, M), device=d_in.device, dtype=d_in.dtype)
    matmul_transpose_assign(d_in, d_out)
    return d_out


# ── flash-muon's EXACT Newton-Schulz + optimizer (nil0x9/flash-muon muon.py, verbatim) ──
import torch.optim as _optim


def fast_newtonschulz(G, steps=5):
    """nil0x9/flash-muon muon.py fast_newtonschulz, VERBATIM (Jordan quintic coeffs, bf16,
    buf1/buf2 reuse, matmul_transpose_assign for the two symmetric GEMMs). 2D only."""
    assert G.ndim >= 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    if G.size(-2) > G.size(-1):
        X = X.mT
    buf1 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    buf2 = torch.empty(X.size(0), X.size(0), dtype=X.dtype, device=X.device)
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    for _ in range(steps):
        matmul_transpose_assign(X, buf1)
        matmul_transpose_assign(buf1, buf2)
        B = b * buf1 + c * buf2
        X = a * X + B @ X
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X


class FlashMuon(_optim.Optimizer):
    """flash-muon's optimizer NS used single-GPU: per-param momentum + their exact fast_newtonschulz
    (the distributed all_gather is stripped; for world_size=1 it changes nothing). 3D expert stacks
    are looped per slice since their NS is 2D-only. This is the apples-to-apples 'their triu' baseline
    at the optimizer level (no cross-param batching, their Jordan/bf16 NS)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      weight_decay=weight_decay, ns_steps=ns_steps))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, wd, nesterov, steps = (group["lr"], group["momentum"], group["weight_decay"],
                                                 group["nesterov"], group["ns_steps"])
            for p in group["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                g = p.grad
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.lerp_(g, 1 - momentum)
                u = g.lerp_(buf, momentum) if nesterov else buf
                if u.ndim == 3:
                    out = torch.stack([fast_newtonschulz(u[i], steps) for i in range(u.shape[0])])
                else:
                    out = fast_newtonschulz(u, steps)
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)
                p.add_(out.to(p.dtype).reshape(p.shape), alpha=-lr * scale)
