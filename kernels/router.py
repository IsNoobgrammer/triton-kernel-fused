"""Whole-router fusion — MiMo-V2.5 / DeepSeek-V3 sigmoid gate, conv variant fused in Triton.

Folds the entire eager router pipeline behind ONE autograd node:

    logits  = causal_conv(x)                 # Triton conv (transpose-free) ...
    scores  = sigmoid(logits)                # ... + sigmoid fused into the store epilogue (fp32)
    sel     = scores + bias                  # bias = SELECTION ONLY (DeepSeek-V3), no grad
    idx     = topk(sel, k)                   # torch.topk — robust tie-break, grad-free
    weights = scores.gather(idx)             # UNBIASED gathered weights (differentiable)
    counts  = bincount(idx)                  # in-kernel atomic-add side-output (non-autograd)

`norm_topk_prob` (÷Σ) and `routed_scaling_factor` (×c) stay in EAGER in `fused_router` so autograd
carries their Jacobian — they are a tiny reduction over k (k=2), nothing to fuse.

MLP router: NOT fused here — it's a small cuBLAS GEMM + sigmoid/bias/topk that torch.compile already
fuses for free. Use the eager module + torch.compile for `router_type="mlp"`. The conv win is real
because the transpose-free Triton conv removes cuDNN's NCHW transpose + left-pad HBM copies, which
torch.compile cannot touch (cuDNN owns the layout).

Multi-node bias update (heuristic, NON-autograd):
    counts is per-rank local. Caller does the ONE cross-rank collective — `dist.all_reduce(counts,
    SUM)` — then `router_bias_update(bias, counts, u)` applies `b += u·sign(mean−load)` identically
    on every rank. The collective cannot live inside a Triton kernel (it's NCCL), so it stays the
    thin Python step; everything else (the count, the sign update) is on-device and off autograd.

Scope of the fused path: gate_type='sigmoid', router_activation='none' (the conv-router default).
Backward: grad_x exact; grad_w correct up to fp32 long-reduction order (documented dw-kernel caveat).
"""
import torch
import triton
import triton.language as tl

# Reuse the transpose-free conv backward kernels (identical math) — don't duplicate.
from .causal_conv1d_router import _conv_router_dx_kernel, _conv_router_dw_kernel

__all__ = ["fused_router", "router_bias_update", "FusedConvRouter"]


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64, "BLOCK_H": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64, "BLOCK_H": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 32, "BLOCK_H": 64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 128}, num_warps=8, num_stages=2),
    ],
    key=["H"],
)
@triton.jit
def _conv_router_fwd_sigmoid_kernel(
    X_ptr, W_ptr, Out_ptr, B, S, H,
    sxb, sxs, sxh, swe, swh, swk, som, soe,
    K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr, APPLY_SIGMOID: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, BLOCK_E)
    mask_s = offs_s < S
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_S, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(K):
        src = offs_s - (K - 1) + k
        mask_src = (src >= 0) & mask_s
        for h0 in range(0, H, BLOCK_H):
            offs_h = h0 + tl.arange(0, BLOCK_H)
            mask_h = offs_h < H
            xv = tl.load(X_ptr + pid_b * sxb + src[:, None] * sxs + offs_h[None, :] * sxh,
                         mask=mask_src[:, None] & mask_h[None, :], other=0.0)
            wv = tl.load(W_ptr + offs_e[:, None] * swe + offs_h[None, :] * swh + k * swk,
                         mask=mask_e[:, None] & mask_h[None, :], other=0.0)
            acc += tl.dot(xv, tl.trans(wv))
    # Fused sigmoid epilogue: never round-trip raw logits through HBM. acc is fp32 (matches eager's
    # logits.float() before sigmoid) — actually MORE precise than eager's fp16-rounded logits.
    if APPLY_SIGMOID:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    out_row = pid_b * S + offs_s
    tl.store(Out_ptr + out_row[:, None] * som + offs_e[None, :] * soe,
             acc.to(Out_ptr.dtype.element_ty), mask=mask_s[:, None] & mask_e[None, :])


def _conv_router_scores(x, weight, apply_sigmoid=True):
    """x (B,S,H), weight (E,H,K) -> scores (B*S, E) fp32 (sigmoid(causal_conv) when apply_sigmoid)."""
    B, S, Hd = x.shape
    E, _, K = weight.shape
    out = torch.empty(B * S, E, device=x.device, dtype=torch.float32)
    grid = lambda meta: (B, triton.cdiv(S, meta["BLOCK_S"]))
    _conv_router_fwd_sigmoid_kernel[grid](
        x, weight, out, B, S, Hd,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        out.stride(0), out.stride(1),
        K=K, E=E, BLOCK_E=max(16, triton.next_power_of_2(E)), APPLY_SIGMOID=apply_sigmoid)
    return out


@triton.jit
def _count_experts_kernel(Idx_ptr, Count_ptr, NK, E: tl.constexpr, BLOCK: tl.constexpr):
    """counts[e] += #{selected slots == e}. idx is (B*S*k,) flattened. atomic_add -> non-autograd."""
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


class FusedConvRouter(torch.autograd.Function):
    """Whole conv router (conv+sigmoid+bias-select+topk+gather + in-kernel count) as one node.
    Returns (idx (B*S,k) long, weights (B*S,k) fp32 UNBIASED, counts (E,) int32 per-rank).
    Only `weights` is differentiable (idx/counts discrete)."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        scores = _conv_router_scores(x, weight, apply_sigmoid=True)        # (B*S, E) fp32
        sel = scores + bias if bias is not None else scores               # bias = selection only
        _, idx = torch.topk(sel, top_k, dim=-1)                           # robust tie-break, grad-free
        weights = scores.gather(-1, idx)                                  # UNBIASED, differentiable
        counts = _count_experts(idx, num_experts)                         # in-kernel atomic-add count
        ctx.save_for_backward(x, weight, scores, idx)
        ctx.mark_non_differentiable(idx, counts)
        return idx, weights, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        x, weight, scores, idx = ctx.saved_tensors
        B, S, Hd = x.shape
        E, _, K = weight.shape
        # gather^T: scatter grad into selected score slots. Within a row the top-k idx are distinct
        # -> scatter_add == set (no contention). sigmoid': scores*(1-scores) (scores ARE the sigmoid).
        grad_scores = torch.zeros_like(scores)
        grad_scores.scatter_add_(-1, idx, grad_weights.float())
        grad_logits = grad_scores * scores * (1.0 - scores)               # fp32
        go = grad_logits.to(x.dtype).contiguous()
        BE = max(16, triton.next_power_of_2(E))
        gx = torch.empty(B, S, Hd, device=x.device, dtype=x.dtype)
        gw = torch.empty(E, Hd, K, device=x.device, dtype=x.dtype)
        gridx = lambda m: (B, triton.cdiv(S, m["BLOCK_S"]), triton.cdiv(Hd, m["BLOCK_H"]))
        _conv_router_dx_kernel[gridx](go, weight, gx, B, S, Hd,
            go.stride(0), go.stride(1), weight.stride(0), weight.stride(1), weight.stride(2),
            gx.stride(0), gx.stride(1), gx.stride(2), K=K, E=E, BLOCK_E=BE)
        gridw = lambda m: (K, triton.cdiv(Hd, m["BLOCK_H"]))
        _conv_router_dw_kernel[gridw](go, x, gw, B, S, Hd,
            go.stride(0), go.stride(1), x.stride(0), x.stride(1), x.stride(2),
            gw.stride(0), gw.stride(1), gw.stride(2), K=K, E=E, BLOCK_E=BE)
        return gx, gw, None, None, None   # x, weight, bias, top_k, num_experts


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    """Whole conv router. x (B,S,H), conv_weight (E,H,K) from nn.Conv1d(H,E,K), bias (E,) fp32 or None.

    Returns (idx (B,S,k) long, norm_weights (B,S,k) fp32) — or (..., counts (E,) int32) if
    return_counts. norm_topk_prob/routed_scaling applied in eager (autograd carries the Jacobian).
    """
    B, S, _ = x.shape
    idx, w, counts = FusedConvRouter.apply(x, conv_weight, bias, top_k, num_experts)
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
