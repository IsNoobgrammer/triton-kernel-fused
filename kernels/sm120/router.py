"""sm120 (Blackwell) conv MoE router — transpose-free, cuDNN-free K-GEMM conv.

CANDIDATE (autoresearch router_sm120, candidate #1). The sm75 `cudnn` champion REGRESSED on Blackwell
(0.82x uncompiled / 0.91-0.96x compiled): the profile showed `aten::copy_` = 48% of CUDA + ~870us of
cuDNN nchwToNhwc/nhwcToNchw layout transposes that are UNAVOIDABLE while cuDNN owns the conv. So this
backend removes cuDNN entirely.

A causal conv1d with K taps is K matmuls:  logit[b,s,e] = Σ_k Σ_h x[b, s-(K-1)+k, h] · W[e,h,k].
For tap k (shift = K-1-k):  logits[:, shift:, :] += x[:, :S-shift, :] @ W[:,:,k].T   — x stays (B,S,H),
the matmul contracts H, NO transpose to (B,H,S), NO cuDNN, NO layout copy. Backward is the same shape:
  grad_x[:, :S-shift, :] += grad_logit[:, shift:, :] @ W[:,:,k]              (K matmuls)
  grad_W[:,:,k]          = grad_logit[:, shift:, :]ᵀ @ x[:, :S-shift, :]      (K matmuls, fat contraction)
All bf16 tensor-core GEMMs (fp32 accumulate); the proven fused epilogue (sigmoid+selection-bias+top-k+
unbiased gather, and its scatter+sigmoid' backward) is REUSED verbatim from sm75. ~15 launches vs cuDNN's
~75, zero layout transposes. Correct-by-construction (standard conv-as-GEMM).

T4 NOTE: the K-GEMM conv was 0.35x on T4 (fp32 RMW + launch-bound on Turing). This is a Blackwell-only
backend — sm75 keeps the cudnn champion untouched. A/B on Blackwell decides whether it ships.
"""
import torch
import triton

# Reuse the proven sm75 epilogue + side-kernels verbatim (single source of truth).
from kernels.sm75.router import (  # noqa: F401
    _epilogue_fwd, _router_epilogue_bwd_kernel, _count_experts, router_bias_update,
    FusedConvRouterCuDNN,  # kept available as a fallback backend
)

__all__ = ["fused_router", "router_bias_update", "FusedConvRouterCuDNN",
           "FusedConvRouterGEMM", "_count_experts"]


class FusedConvRouterGEMM(torch.autograd.Function):
    """Conv router with the conv expressed as K shifted matmuls (no cuDNN, no layout transpose).
    Returns (idx (B*S,k) long, weights (B*S,k) fp32 UNBIASED, counts (E,) int32). Only `weights`
    is differentiable. grad_x / grad_W are the exact conv-transpose / correlation, also as K matmuls."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        B, S, H = x.shape
        E, _, K = weight.shape
        xd = x.contiguous()                                      # (B,S,H) — its NATIVE layout; no transpose
        # Causal conv as K matmuls, accumulated in fp32. No (B,H,S) copy, no cuDNN, no nchwToNhwc.
        logits = torch.zeros(B, S, E, device=x.device, dtype=torch.float32)
        for k in range(K):
            shift = (K - 1) - k                                  # output s uses input s-shift
            if shift >= S:
                continue
            Wk = weight[:, :, k].to(x.dtype)                     # (E,H)
            # (B, S-shift, H) @ (H, E) -> (B, S-shift, E); bf16 tensor-core matmul, fp32 accumulate
            contrib = torch.matmul(xd[:, : S - shift, :], Wk.t())
            logits[:, shift:, :] += contrib.float()
        logits2d = logits.reshape(B * S, E)
        idx, weights = _epilogue_fwd(logits2d, bias, top_k)
        counts = _count_experts(idx, num_experts)
        ctx.save_for_backward(xd, weight, logits2d, idx)
        ctx.dims = (B, S, H, E, K)
        ctx.mark_non_differentiable(idx, counts)
        return idx, weights, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        xd, weight, logits2d, idx = ctx.saved_tensors
        B, S, H, E, K = ctx.dims
        N, top_k = idx.shape
        # grad_logits via the SAME fused epilogue-bwd kernel as sm75 (scatter grad_w to picks * sigmoid').
        grad_logits = torch.empty(N, E, device=xd.device, dtype=torch.float32)
        gw = grad_weights.contiguous()
        BLOCK_N = 128
        _router_epilogue_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            logits2d, idx, gw, grad_logits, N,
            logits2d.stride(0), logits2d.stride(1), idx.stride(0), idx.stride(1),
            gw.stride(0), gw.stride(1), grad_logits.stride(0), grad_logits.stride(1),
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))

        gl = grad_logits.reshape(B, S, E).to(xd.dtype)           # bf16 for the tensor-core matmuls
        grad_x = torch.zeros(B, S, H, device=xd.device, dtype=xd.dtype)
        grad_w = torch.zeros(E, H, K, device=xd.device, dtype=weight.dtype)
        for k in range(K):
            shift = (K - 1) - k
            if shift >= S:
                continue
            Wk = weight[:, :, k].to(xd.dtype)                    # (E,H)
            # grad_x[:, :S-shift, :] += grad_logit[:, shift:, :] @ W[:,:,k]   (B,S-shift,E)@(E,H)
            grad_x[:, : S - shift, :] += torch.matmul(gl[:, shift:, :], Wk)
            # grad_W[:,:,k] = grad_logit[:, shift:, :]ᵀ @ x[:, :S-shift, :]    fat contraction B*(S-shift)
            a = gl[:, shift:, :].reshape(-1, E)                  # (M,E)
            b = xd[:, : S - shift, :].reshape(-1, H)             # (M,H)
            grad_w[:, :, k] = torch.matmul(a.t(), b).to(weight.dtype)
        return grad_x, grad_w, None, None, None


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    """sm120 conv router (transpose-free K-GEMM). Same API/semantics as sm75.fused_router; only the
    conv backend differs. x (B,S,H), conv_weight (E,H,K), bias (E,) fp32 or None."""
    B, S, _ = x.shape
    idx, w, counts = FusedConvRouterGEMM.apply(x, conv_weight, bias, top_k, num_experts)
    if top_k > 1 and norm_topk_prob:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * routed_scaling_factor
    idx = idx.view(B, S, top_k)
    w = w.view(B, S, top_k)
    return (idx, w, counts) if return_counts else (idx, w)
