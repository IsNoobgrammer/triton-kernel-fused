"""sm120 (Blackwell) conv MoE router — candidate #2: single-launch FUSED Triton conv (no cuDNN).

Candidate #1 (K torch matmuls) was correct but 0.54x — a launch/copy explosion (161 launches/iter, 240
aten::mm, 760 aten::copy_). Lesson: any multi-torch-op conv loses on this tiny op. So this backend does
the whole conv in FUSED Triton kernels — one launch each, x read once, no layout transpose, no cuDNN, no
intermediate copies:

  forward  (`_conv_router_fwd_kernel`): per (b, s-tile) accumulate the K-tap H-contraction via tl.dot
      (x[b, s-(K-1)+k, :] @ W[:,:,k]ᵀ over k), then sigmoid + selection-bias + top-k argmax + unbiased
      gather IN-REGISTER -> idx, weights, and saved logits. Counts via the reused atomic kernel.
  backward grad_x (`_conv_router_dx_kernel`): grad_x[:,s',:] = Σ_k grad_logit[:,s'+shift_k,:] @ W[:,:,k]
      (contraction over E), one fused kernel.
  backward grad_W (`_conv_router_dw_kernel`): grad_W[:,:,k] = Σ_n grad_logit[n,:]ᵀ · x[n-shift_k,:],
      a fused reduction over n=B*S (no sliced-matmul copies).

All tl.dot in bf16/fp16 with fp32 accumulate. ~4 launches vs cuDNN's ~75, zero layout transposes. The
fused epilogue (top-k) is the same edge as sm75; only the conv is now Triton instead of cuDNN. T4-dead
(SRAM); Blackwell (~228KB SRAM + bf16 TC) is the revisit case. A/B on Blackwell decides if it ships.
"""
import torch
import triton
import triton.language as tl

from kernels.sm75.router import _count_experts, router_bias_update, FusedConvRouterCuDNN  # noqa: F401

__all__ = ["fused_router", "router_bias_update", "FusedConvRouterCuDNN",
           "FusedConvRouterFused", "_count_experts"]


@triton.jit
def _conv_router_fwd_kernel(X, W, Bias, Idx, Wt, Logit, S, sxb, sxs, sxh, swe, swh, swk,
                            sln, sle, HAS_BIAS: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
                            TOPK: tl.constexpr, H: tl.constexpr, BLOCK_S: tl.constexpr,
                            BLOCK_E: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    st = tl.program_id(1)
    offs_s = st * BLOCK_S + tl.arange(0, BLOCK_S)          # output positions
    offs_e = tl.arange(0, BLOCK_E)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < S
    mask_e = offs_e < E
    mask_d = offs_d < H
    acc = tl.zeros((BLOCK_S, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(K):
        in_s = offs_s - (K - 1) + k                        # causal input position for this tap
        m_in = (in_s >= 0) & (in_s < S)
        xk = tl.load(X + b * sxb + in_s[:, None] * sxs + offs_d[None, :] * sxh,
                     mask=m_in[:, None] & mask_d[None, :], other=0.0)            # (BLOCK_S, BLOCK_D)
        wk = tl.load(W + offs_d[:, None] * swh + offs_e[None, :] * swe + k * swk,
                     mask=mask_d[:, None] & mask_e[None, :], other=0.0)          # (BLOCK_D, BLOCK_E)
        acc = tl.dot(xk, wk, acc)                          # (BLOCK_S,BLOCK_D)@(BLOCK_D,BLOCK_E)
    # save logits (for backward)
    n = b * S + offs_s
    tl.store(Logit + n[:, None] * sln + offs_e[None, :] * sle,
             acc.to(Logit.dtype.element_ty), mask=mask_s[:, None] & mask_e[None, :])
    # fused epilogue: sigmoid + selection-bias + top-k argmax + unbiased gather
    scores = 1.0 / (1.0 + tl.exp(-acc))
    sel = scores
    if HAS_BIAS:
        bb = tl.load(Bias + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        sel = sel + bb[None, :]
    sel = tl.where(mask_e[None, :], sel, -1e30)
    for kk in tl.static_range(TOPK):
        am = tl.argmax(sel, axis=1)
        onehot = offs_e[None, :] == am[:, None]
        w_kk = tl.sum(tl.where(onehot, scores, 0.0), axis=1)
        tl.store(Idx + n * TOPK + kk, am.to(tl.int64), mask=mask_s)
        tl.store(Wt + n * TOPK + kk, w_kk, mask=mask_s)
        sel = tl.where(onehot, -1e30, sel)


@triton.jit
def _conv_router_dx_kernel(GL, W, GX, S, sgn, sge, swe, swh, swk, sxb, sxs, sxh,
                           E: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                           BLOCK_S: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    st = tl.program_id(1)
    hb = tl.program_id(2)                                  # H tiled to bound the accumulator size
    offs_s = st * BLOCK_S + tl.arange(0, BLOCK_S)          # output positions s'
    offs_e = tl.arange(0, BLOCK_E)
    offs_d = hb * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_s = offs_s < S
    mask_e = offs_e < E
    mask_d = offs_d < H
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for k in tl.static_range(K):
        shift = (K - 1) - k
        src = offs_s + shift                               # grad_logit position
        m_src = src < S
        n = b * S + src
        gl = tl.load(GL + n[:, None] * sgn + offs_e[None, :] * sge,
                     mask=m_src[:, None] & mask_e[None, :], other=0.0)           # (BLOCK_S, BLOCK_E) fp32
        wk = tl.load(W + offs_e[:, None] * swe + offs_d[None, :] * swh + k * swk,
                     mask=mask_e[:, None] & mask_d[None, :], other=0.0)          # (BLOCK_E, BLOCK_D)
        acc = tl.dot(gl.to(wk.dtype), wk, acc)             # (BLOCK_S,BLOCK_E)@(BLOCK_E,BLOCK_D)
    tl.store(GX + b * sxb + offs_s[:, None] * sxs + offs_d[None, :] * sxh,
             acc.to(GX.dtype.element_ty), mask=mask_s[:, None] & mask_d[None, :])


@triton.jit
def _conv_router_dw_kernel(GL, X, GW, N, S, sgn, sge, sxn, sxh, gwe, gwh, gwk,
                           E: tl.constexpr, H: tl.constexpr, BLOCK_E: tl.constexpr,
                           BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, K: tl.constexpr):
    k = tl.program_id(0)
    hb = tl.program_id(1)
    shift = (K - 1) - k
    offs_e = tl.arange(0, BLOCK_E)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_e = offs_e < E
    mask_h = offs_h < H
    acc = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
    for n0 in range(0, N, BLOCK_N):
        nn = n0 + tl.arange(0, BLOCK_N)                    # flat (b*S+s) for grad_logit
        s_in = nn % S
        m = (nn < N) & (s_in >= shift)                     # causal: x[n-shift] valid only when s>=shift
        gl_t = tl.load(GL + offs_e[:, None] * sge + nn[None, :] * sgn,
                       mask=mask_e[:, None] & m[None, :], other=0.0)             # (BLOCK_E, BLOCK_N)
        x_t = tl.load(X + (nn - shift)[:, None] * sxn + offs_h[None, :] * sxh,
                      mask=m[:, None] & mask_h[None, :], other=0.0)              # (BLOCK_N, BLOCK_H)
        acc = tl.dot(gl_t.to(x_t.dtype), x_t, acc)         # (BLOCK_E,BLOCK_N)@(BLOCK_N,BLOCK_H)
    tl.store(GW + offs_e[:, None] * gwe + offs_h[None, :] * gwh + k * gwk,
             acc.to(GW.dtype.element_ty), mask=mask_e[:, None] & mask_h[None, :])


class FusedConvRouterFused(torch.autograd.Function):
    """Whole conv router as fused Triton kernels (no cuDNN, no layout transpose). Returns
    (idx (B*S,k) long, weights (B*S,k) fp32 UNBIASED, counts (E,) int32); only `weights` differentiable."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        B, S, H = x.shape
        E, _, K = weight.shape
        x = x.contiguous()
        weight = weight.contiguous()
        N = B * S
        idx = torch.empty(N, top_k, device=x.device, dtype=torch.long)
        wt = torch.empty(N, top_k, device=x.device, dtype=torch.float32)
        logits = torch.empty(N, E, device=x.device, dtype=torch.float32)
        BLOCK_S = 64
        BLOCK_E = max(16, triton.next_power_of_2(E))
        BLOCK_D = triton.next_power_of_2(H)
        grid = (B, triton.cdiv(S, BLOCK_S))
        _conv_router_fwd_kernel[grid](
            x, weight, bias if bias is not None else x, idx, wt, logits, S,
            x.stride(0), x.stride(1), x.stride(2), weight.stride(0), weight.stride(1), weight.stride(2),
            logits.stride(0), logits.stride(1), HAS_BIAS=bias is not None,
            E=E, K=K, TOPK=top_k, H=H, BLOCK_S=BLOCK_S, BLOCK_E=BLOCK_E, BLOCK_D=BLOCK_D)
        counts = _count_experts(idx, num_experts)
        ctx.save_for_backward(x, weight, logits, idx)
        ctx.dims = (B, S, H, E, K, top_k)
        ctx.mark_non_differentiable(idx, counts)
        return idx, wt, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        x, weight, logits, idx = ctx.saved_tensors
        B, S, H, E, K, top_k = ctx.dims
        N = B * S
        from kernels.sm75.router import _router_epilogue_bwd_kernel
        grad_logits = torch.empty(N, E, device=x.device, dtype=torch.float32)
        gw_in = grad_weights.contiguous()
        BLOCK_N_EP = 128
        BLOCK_E = max(16, triton.next_power_of_2(E))
        _router_epilogue_bwd_kernel[(triton.cdiv(N, BLOCK_N_EP),)](
            logits, idx, gw_in, grad_logits, N,
            logits.stride(0), logits.stride(1), idx.stride(0), idx.stride(1),
            gw_in.stride(0), gw_in.stride(1), grad_logits.stride(0), grad_logits.stride(1),
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N_EP, BLOCK_E=BLOCK_E)

        grad_x = torch.empty(B, S, H, device=x.device, dtype=x.dtype)
        BLOCK_S = 64
        BLOCK_D = 128                                      # tile H so the (BLOCK_S, BLOCK_D) acc stays small
        _conv_router_dx_kernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_D))](
            grad_logits, weight, grad_x, S,
            grad_logits.stride(0), grad_logits.stride(1),
            weight.stride(0), weight.stride(1), weight.stride(2),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
            E=E, K=K, H=H, BLOCK_S=BLOCK_S, BLOCK_E=BLOCK_E, BLOCK_D=BLOCK_D)

        grad_w = torch.empty(E, H, K, device=x.device, dtype=weight.dtype)
        xflat = x.view(N, H)
        BLOCK_H = 128
        _conv_router_dw_kernel[(K, triton.cdiv(H, BLOCK_H))](
            grad_logits, xflat, grad_w, N, S,
            grad_logits.stride(0), grad_logits.stride(1), xflat.stride(0), xflat.stride(1),
            grad_w.stride(0), grad_w.stride(1), grad_w.stride(2),
            E=E, H=H, BLOCK_E=BLOCK_E, BLOCK_H=BLOCK_H, BLOCK_N=128, K=K)
        return grad_x, grad_w, None, None, None


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    """sm120 conv router (fused Triton conv). Same API/semantics as sm75.fused_router; conv is Triton."""
    B, S, _ = x.shape
    idx, w, counts = FusedConvRouterFused.apply(x, conv_weight, bias, top_k, num_experts)
    if top_k > 1 and norm_topk_prob:
        w = w / (w.sum(-1, keepdim=True) + 1e-20)
    w = w * routed_scaling_factor
    idx = idx.view(B, S, top_k)
    w = w.view(B, S, top_k)
    return (idx, w, counts) if return_counts else (idx, w)
