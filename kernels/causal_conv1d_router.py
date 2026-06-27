"""Fused causal-conv1d router projection (Triton fwd + transpose-free Triton bwd).

A causal depthwise-over-channels conv1d used as a token router / projection:
reads x in native (B, S, H) layout and writes logits (B*S, E) directly, fusing away the
(B,S,H)->(B,H,S) transpose + left-pad copies that the cuDNN path pays.

    out[b,s,e] = sum_k sum_h x[b, s-(K-1)+k, h] * W[e, h, k]      (causal: src row <0 -> 0)

`W` is an `nn.Conv1d(H, E, K)` weight of shape (E, H, K). Backward is transpose-free
(custom dx/dw Triton kernels), grad_x exact; grad_w correct up to fp32 long-reduction order.

Drop-in (replaces permute -> pad -> F.conv1d -> reshape):
    from kernels.causal_conv1d_router import causal_conv1d_router
    logits = causal_conv1d_router(x, conv.weight)     # x (B,S,H), weight (E,H,K) -> (B*S, E)

NOTE: this is the projection only (no sigmoid/top-k) — those stay in your router in eager
so autograd handles them. Useful for conv-augmented routers / short-kernel token mixing.
"""
import torch
import triton
import triton.language as tl

__all__ = ["causal_conv1d_router", "CausalConv1dRouter"]


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
def _conv_router_fwd_kernel(X_ptr, W_ptr, Out_ptr, B, S, H,
                            sxb, sxs, sxh, swe, swh, swk, som, soe,
                            K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
                            BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr):
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
    out_row = pid_b * S + offs_s
    tl.store(Out_ptr + out_row[:, None] * som + offs_e[None, :] * soe,
             acc.to(Out_ptr.dtype.element_ty), mask=mask_s[:, None] & mask_e[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_S": 64, "BLOCK_H": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64, "BLOCK_H": 128}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_S": 128, "BLOCK_H": 128}, num_warps=8, num_stages=2),
    ], key=["H"])
@triton.jit
def _conv_router_dx_kernel(GO_ptr, W_ptr, GX_ptr, B, S, H,
                           sgm, sge, swe, swh, swk, sxb, sxs, sxh,
                           K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
                           BLOCK_S: tl.constexpr, BLOCK_H: tl.constexpr):
    # grad_x[b,j,h] = sum_m sum_e grad_out[b, j+m, e] * W[e, h, K-1-m]  (j+m<S)
    pid_b = tl.program_id(0); pid_s = tl.program_id(1); pid_h = tl.program_id(2)
    offs_j = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_e = tl.arange(0, BLOCK_E)
    mask_j = offs_j < S; mask_h = offs_h < H; mask_e = offs_e < E
    acc = tl.zeros((BLOCK_S, BLOCK_H), dtype=tl.float32)
    for m in tl.static_range(K):
        s = offs_j + m
        mask_s = (s < S) & mask_j
        kw = K - 1 - m
        go = tl.load(GO_ptr + (pid_b * S + s)[:, None] * sgm + offs_e[None, :] * sge,
                     mask=mask_s[:, None] & mask_e[None, :], other=0.0)
        wv = tl.load(W_ptr + offs_e[:, None] * swe + offs_h[None, :] * swh + kw * swk,
                     mask=mask_e[:, None] & mask_h[None, :], other=0.0)
        acc += tl.dot(go, wv)
    tl.store(GX_ptr + pid_b * sxb + offs_j[:, None] * sxs + offs_h[None, :] * sxh,
             acc.to(GX_ptr.dtype.element_ty), mask=mask_j[:, None] & mask_h[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_H": 64, "BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 128, "BLOCK_S": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_H": 64, "BLOCK_S": 128}, num_warps=4, num_stages=3),
    ], key=["H"])
@triton.jit
def _conv_router_dw_kernel(GO_ptr, X_ptr, GW_ptr, B, S, H,
                           sgm, sge, sxb, sxs, sxh, gwe, gwh, gwk,
                           K: tl.constexpr, E: tl.constexpr, BLOCK_E: tl.constexpr,
                           BLOCK_H: tl.constexpr, BLOCK_S: tl.constexpr):
    # grad_w[e,h,k] = sum_{b,s} grad_out[b,s,e] * x[b, s-(K-1)+k, h]  (src>=0)
    pid_k = tl.program_id(0); pid_h = tl.program_id(1)
    offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    offs_e = tl.arange(0, BLOCK_E)
    mask_h = offs_h < H; mask_e = offs_e < E
    acc = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
    for b in range(B):
        for s0 in range(0, S, BLOCK_S):
            offs_s = s0 + tl.arange(0, BLOCK_S)
            src = offs_s - (K - 1) + pid_k
            mask_s = offs_s < S
            mask_src = (src >= 0) & mask_s
            go = tl.load(GO_ptr + (b * S + offs_s)[:, None] * sgm + offs_e[None, :] * sge,
                         mask=mask_s[:, None] & mask_e[None, :], other=0.0)
            xv = tl.load(X_ptr + b * sxb + src[:, None] * sxs + offs_h[None, :] * sxh,
                         mask=mask_src[:, None] & mask_h[None, :], other=0.0)
            acc += tl.dot(tl.trans(go), xv)
    tl.store(GW_ptr + offs_e[:, None] * gwe + offs_h[None, :] * gwh + pid_k * gwk,
             acc.to(GW_ptr.dtype.element_ty), mask=mask_e[:, None] & mask_h[None, :])


def _fwd(x, weight):
    B, S, Hd = x.shape
    E, _, K = weight.shape
    out = torch.empty(B * S, E, device=x.device, dtype=x.dtype)
    grid = lambda meta: (B, triton.cdiv(S, meta["BLOCK_S"]))
    _conv_router_fwd_kernel[grid](
        x, weight, out, B, S, Hd,
        x.stride(0), x.stride(1), x.stride(2),
        weight.stride(0), weight.stride(1), weight.stride(2),
        out.stride(0), out.stride(1),
        K=K, E=E, BLOCK_E=max(16, triton.next_power_of_2(E)))
    return out


class CausalConv1dRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x, weight)
        return _fwd(x, weight)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight = ctx.saved_tensors
        B, S, Hd = x.shape
        E, _, K = weight.shape
        go = grad_out.contiguous().to(x.dtype)
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
        return gx, gw


def causal_conv1d_router(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x (B,S,H), weight (E,H,K) from nn.Conv1d -> causal conv logits (B*S, E)."""
    return CausalConv1dRouter.apply(x, weight)
