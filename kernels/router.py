"""Whole conv MoE router — the T4 'cudnn' winner (1.11-1.17x fwd+bwd vs torch.compile, exact grads).

MiMo-V2.5 / DeepSeek-V3 auxiliary-loss-free sigmoid gate, conv variant, folded behind one node:

    logits  = causal_conv(x)        # cuDNN conv (padding=K-1, no F.pad copy)
    scores  = sigmoid(logits)       # } fused in ONE Triton epilogue kernel, in-register —
    sel     = scores + bias         # } bias = SELECTION ONLY (DeepSeek-V3), no grad
    idx     = top-k argmax(sel)     # } replaces native torch.topk (the ~295us seam compile can't fuse)
    weights = scores[idx]           # } UNBIASED gathered weights (differentiable)
    counts  = bincount(idx)         # in-kernel atomic-add side-output (non-autograd)

backward: cuDNN convolution_backward called DIRECTLY on a saved-once contiguous (B,H,S) input
(autograd otherwise copies x->contiguous twice + casts/transposes) + a fused epilogue-bwd kernel.
`norm_topk_prob` (÷Σ) and `routed_scaling_factor` (×c) stay in EAGER (a tiny k-wide reduction).

WHY cuDNN and not a Triton conv: the conv is tiny/bandwidth-bound but on T4 (sm_75) cuDNN owns it —
inductor itself falls back to extern cuDNN, and every hand-rolled Triton conv we tried lost. See
`.autoresearch/reflections.md` (Round 4-5) for the full ledger of refuted approaches (tl.dot conv,
cuBLAS K-GEMM, channels-last, merged-contraction — all T4-dead or SRAM-bound) and what to revisit on
Ampere/Hopper, where the bandwidth-optimal transpose-free Triton conv has real headroom.

Scope of the fused path: gate_type='sigmoid', router_activation='none' (the conv-router default).
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_router", "router_bias_update", "FusedConvRouterCuDNN"]


@triton.jit
def _router_epilogue_fwd_kernel(Logit_ptr, Bias_ptr, Idx_ptr, W_ptr, N, sln, sle,
                                HAS_BIAS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    """sigmoid + (selection)bias + top-k argmax + unbiased gather in ONE pass. sel=scores+bias picks;
    weights=scores (UNBIASED) at the picks. BLOCK_N rows/program (vectorized argmax over E)."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    mask_n = offs_n < N
    mask_e = offs_e < E
    logit = tl.load(Logit_ptr + offs_n[:, None] * sln + offs_e[None, :] * sle,
                    mask=mask_n[:, None] & mask_e[None, :], other=0.0).to(tl.float32)
    scores = 1.0 / (1.0 + tl.exp(-logit))
    sel = scores
    if HAS_BIAS:
        b = tl.load(Bias_ptr + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        sel = sel + b[None, :]
    sel = tl.where(mask_e[None, :], sel, -1e30)
    for k in tl.static_range(TOPK):
        am = tl.argmax(sel, axis=1)
        onehot = offs_e[None, :] == am[:, None]
        w_k = tl.sum(tl.where(onehot, scores, 0.0), axis=1)
        tl.store(Idx_ptr + offs_n * TOPK + k, am.to(tl.int64), mask=mask_n)
        tl.store(W_ptr + offs_n * TOPK + k, w_k, mask=mask_n)
        sel = tl.where(onehot, -1e30, sel)


@triton.jit
def _router_epilogue_bwd_kernel(Logit_ptr, Idx_ptr, Gw_ptr, Gout_ptr, N,
                                sln, sle, sin, sik, sgn, sgk, son, soe,
                                E: tl.constexpr, TOPK: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    """grad_logits = scatter(grad_w -> picked idx slots) * sigmoid'(logit). One kernel."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    mask_n = offs_n < N
    mask_e = offs_e < E
    logit = tl.load(Logit_ptr + offs_n[:, None] * sln + offs_e[None, :] * sle,
                    mask=mask_n[:, None] & mask_e[None, :], other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-logit))
    sp = s * (1.0 - s)
    gscore = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        ik = tl.load(Idx_ptr + offs_n * sin + k * sik, mask=mask_n, other=0).to(tl.int32)
        gk = tl.load(Gw_ptr + offs_n * sgn + k * sgk, mask=mask_n, other=0.0)
        gscore += tl.where(offs_e[None, :] == ik[:, None], gk[:, None], 0.0)
    gout = gscore * sp
    tl.store(Gout_ptr + offs_n[:, None] * son + offs_e[None, :] * soe,
             gout.to(Gout_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_e[None, :])


def _epilogue_fwd(logits, bias, top_k):
    N, E = logits.shape
    idx = torch.empty(N, top_k, device=logits.device, dtype=torch.long)
    w = torch.empty(N, top_k, device=logits.device, dtype=torch.float32)
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _router_epilogue_fwd_kernel[grid](
        logits, bias if bias is not None else logits, idx, w, N,
        logits.stride(0), logits.stride(1),
        HAS_BIAS=bias is not None, E=E, TOPK=top_k,
        BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
    return idx, w


@triton.jit
def _count_experts_kernel(Idx_ptr, Count_ptr, NK, E: tl.constexpr, BLOCK: tl.constexpr):
    """counts[e] += #{selected slots == e}. idx flattened (B*S*k,). atomic_add -> non-autograd."""
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NK
    e = tl.load(Idx_ptr + offs, mask=mask, other=-1)
    for ei in tl.static_range(E):
        tl.atomic_add(Count_ptr + ei, tl.sum(tl.where(e == ei, 1, 0).to(tl.int32)))


def _count_experts(idx, num_experts):
    """idx (.., k) long -> counts (E,) int32 on-device. Per-rank local count (pre-all_reduce)."""
    flat = idx.reshape(-1).to(torch.int32)
    NK = flat.numel()
    counts = torch.zeros(num_experts, device=idx.device, dtype=torch.int32)
    BLOCK = 1024
    _count_experts_kernel[(triton.cdiv(NK, BLOCK),)](flat, counts, NK, E=num_experts, BLOCK=BLOCK)
    return counts


class FusedConvRouterCuDNN(torch.autograd.Function):
    """Whole conv router (conv + sigmoid + bias-select + top-k + gather + in-kernel count) as one node.
    Returns (idx (B*S,k) long, weights (B*S,k) fp32 UNBIASED, counts (E,) int32 per-rank). Only
    `weights` is differentiable. cuDNN conv fwd + fused epilogue; cuDNN convolution_backward + fused
    epilogue-bwd. grad_x exact; grad_w = cuDNN's (matches compiled)."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        import torch.nn.functional as F
        B, S, H = x.shape
        E, _, K = weight.shape
        xc = x.transpose(1, 2).contiguous()                     # (B,H,S) once, reused in bwd
        conv = F.conv1d(xc, weight, padding=K - 1)[..., :S]     # (B,E,S) causal, no F.pad copy
        logits = conv.transpose(1, 2).reshape(B * S, E)         # (B*S,E)
        idx, weights = _epilogue_fwd(logits, bias, top_k)
        counts = _count_experts(idx, num_experts)
        ctx.save_for_backward(xc, weight, logits, idx)
        ctx.K, ctx.S, ctx.E = K, S, E
        ctx.mark_non_differentiable(idx, counts)
        return idx, weights, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        import torch.nn.functional as F
        xc, weight, logits, idx = ctx.saved_tensors
        K, S, E = ctx.K, ctx.S, ctx.E
        N, top_k = idx.shape
        grad_logits = torch.empty(N, E, device=xc.device, dtype=xc.dtype)
        gw = grad_weights.contiguous()
        BLOCK_N = 128
        _router_epilogue_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            logits, idx, gw, grad_logits, N,
            logits.stride(0), logits.stride(1), idx.stride(0), idx.stride(1),
            gw.stride(0), gw.stride(1), grad_logits.stride(0), grad_logits.stride(1),
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
        B = xc.shape[0]
        grad_full = F.pad(grad_logits.view(B, S, E).transpose(1, 2), (0, K - 1))   # (B,E,S+K-1)
        grad_xc, grad_w = torch.ops.aten.convolution_backward(
            grad_full, xc, weight, [0], [1], [K - 1], [1], False, [0], 1, [True, True, False])[:2]
        return grad_xc.transpose(1, 2), grad_w, None, None, None   # (B,H,S)->(B,S,H)


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    """Whole conv router. x (B,S,H), conv_weight (E,H,K) from nn.Conv1d(H,E,K), bias (E,) fp32 or None.
    Returns (idx (B,S,k) long, norm_weights (B,S,k) fp32) — or (..., counts (E,) int32) if
    return_counts. norm_topk_prob / routed_scaling applied in eager (autograd carries the Jacobian)."""
    B, S, _ = x.shape
    idx, w, counts = FusedConvRouterCuDNN.apply(x, conv_weight, bias, top_k, num_experts)
    if top_k > 1 and norm_topk_prob:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)            # MiMo/DeepSeek-V3 top-k sum-to-1
    w = w * routed_scaling_factor                            # 1.0 = no-op (MiMo-V2.5)
    idx = idx.view(B, S, top_k)
    w = w.view(B, S, top_k)
    return (idx, w, counts) if return_counts else (idx, w)


@torch.no_grad()
def router_bias_update(bias, counts, u):
    """Heuristic DeepSeek-V3 bias update: b += u·sign(mean_load − load). NON-autograd, in-place.
    `counts` must already be the GLOBAL load — caller does `dist.all_reduce(counts, SUM)` first
    (the one collective; it cannot run inside a Triton kernel). sign() is scale-invariant so SUM
    across ranks is fine and the update is identical on every rank."""
    if u <= 0:
        return
    tpe = counts.detach().float()
    bias.add_(u * (tpe.mean() - tpe).sign())
