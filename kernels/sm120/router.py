import torch
import triton
import triton.language as tl

from kernels.sm75.router import _count_experts, router_bias_update, FusedConvRouterCuDNN

__all__ = ["fused_router", "router_bias_update", "FusedConvRouterCuDNN",
           "FusedConvRouterFused", "_count_experts"]


@triton.jit
def _conv_router_fwd_kernel(X, W, Bias, Idx, Wt, Logit, S, sxb, sxs, sxh, swe, swh, swk,
                            sln, sle, scale, HAS_BIAS: tl.constexpr, NORM: tl.constexpr,
                            E: tl.constexpr, K: tl.constexpr,
                            TOPK: tl.constexpr, TOPK_P2: tl.constexpr, H: tl.constexpr,
                            BLOCK_S: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    st = tl.program_id(1)
    offs_s = st * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, BLOCK_E)
    offs_d = tl.arange(0, BLOCK_D)
    mask_s = offs_s < S
    mask_e = offs_e < E
    mask_d = offs_d < H
    acc = tl.zeros((BLOCK_S, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(K):
        in_s = offs_s - (K - 1) + k
        m_in = (in_s >= 0) & (in_s < S)
        xk = tl.load(X + b * sxb + in_s[:, None] * sxs + offs_d[None, :] * sxh,
                     mask=m_in[:, None] & mask_d[None, :], other=0.0)
        wk = tl.load(W + offs_d[:, None] * swh + offs_e[None, :] * swe + k * swk,
                     mask=mask_d[:, None] & mask_e[None, :], other=0.0)
        acc = tl.dot(xk, wk, acc, input_precision="ieee")
    n = b * S + offs_s
    tl.store(Logit + n[:, None] * sln + offs_e[None, :] * sle,
             acc.to(Logit.dtype.element_ty), mask=mask_s[:, None] & mask_e[None, :])
    scores = 1.0 / (1.0 + tl.exp(-acc))
    sel = scores
    if HAS_BIAS:
        bb = tl.load(Bias + offs_e, mask=mask_e, other=0.0).to(tl.float32)
        sel = sel + bb[None, :]
    sel = tl.where(mask_e[None, :], sel, -1e30)
    offs_k = tl.arange(0, TOPK_P2)
    wmat = tl.zeros((BLOCK_S, TOPK_P2), dtype=tl.float32)
    for kk in tl.static_range(TOPK):
        am = tl.argmax(sel, axis=1)
        onehot = offs_e[None, :] == am[:, None]
        w_kk = tl.sum(tl.where(onehot, scores, 0.0), axis=1)
        tl.store(Idx + n * TOPK + kk, am.to(tl.int64), mask=mask_s)
        wmat = tl.where(offs_k[None, :] == kk, w_kk[:, None], wmat)
        sel = tl.where(onehot, -1e30, sel)
    if NORM:
        t = tl.sum(wmat, axis=1) + 1e-20
        wmat = wmat / t[:, None]
    wmat = wmat * scale
    tl.store(Wt + n[:, None] * TOPK + offs_k[None, :], wmat,
             mask=mask_s[:, None] & (offs_k < TOPK)[None, :])


@triton.jit
def _conv_router_dx_kernel(GL, W, GX, S, sgn, sge, swe, swh, swk, sxb, sxs, sxh,
                           E: tl.constexpr, K: tl.constexpr, H: tl.constexpr,
                           BLOCK_S: tl.constexpr, BLOCK_E: tl.constexpr, BLOCK_D: tl.constexpr):
    b = tl.program_id(0)
    st = tl.program_id(1)
    hb = tl.program_id(2)
    offs_s = st * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, BLOCK_E)
    offs_d = hb * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_s = offs_s < S
    mask_e = offs_e < E
    mask_d = offs_d < H
    acc = tl.zeros((BLOCK_S, BLOCK_D), dtype=tl.float32)
    for k in tl.static_range(K):
        shift = (K - 1) - k
        src = offs_s + shift
        m_src = src < S
        n = b * S + src
        gl = tl.load(GL + n[:, None] * sgn + offs_e[None, :] * sge,
                     mask=m_src[:, None] & mask_e[None, :], other=0.0)
        wk = tl.load(W + offs_e[:, None] * swe + offs_d[None, :] * swh + k * swk,
                     mask=mask_e[:, None] & mask_d[None, :], other=0.0)
        acc = tl.dot(gl.to(wk.dtype), wk, acc, input_precision="ieee")
    tl.store(GX + b * sxb + offs_s[:, None] * sxs + offs_d[None, :] * sxh,
             acc.to(GX.dtype.element_ty), mask=mask_s[:, None] & mask_d[None, :])


@triton.jit
def _conv_router_dw_kernel(GL, X, GW, N, S, sgn, sge, sxn, sxh, gwe, gwh, gwk,
                           N_PER, E: tl.constexpr, H: tl.constexpr, BLOCK_E: tl.constexpr,
                           BLOCK_H: tl.constexpr, BLOCK_N: tl.constexpr, K: tl.constexpr):
    k = tl.program_id(0)
    hb = tl.program_id(1)
    sn = tl.program_id(2)
    shift = (K - 1) - k
    offs_e = tl.arange(0, BLOCK_E)
    offs_h = hb * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_e = offs_e < E
    mask_h = offs_h < H
    acc = tl.zeros((BLOCK_E, BLOCK_H), dtype=tl.float32)
    n_start = sn * N_PER
    for n0 in range(n_start, n_start + N_PER, BLOCK_N):
        nn = n0 + tl.arange(0, BLOCK_N)
        s_in = nn % S
        m = (nn < N) & (s_in >= shift)
        gl_t = tl.load(GL + offs_e[:, None] * sge + nn[None, :] * sgn,
                       mask=mask_e[:, None] & m[None, :], other=0.0)
        x_t = tl.load(X + (nn - shift)[:, None] * sxn + offs_h[None, :] * sxh,
                      mask=m[:, None] & mask_h[None, :], other=0.0)
        acc = tl.dot(gl_t.to(x_t.dtype), x_t, acc, input_precision="ieee")
    tl.atomic_add(GW + offs_e[:, None] * gwe + offs_h[None, :] * gwh + k * gwk,
                  acc, mask=mask_e[:, None] & mask_h[None, :])


class FusedConvRouterFused(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts, norm_topk=True, scale=1.0):
        B, S, H = x.shape
        E, _, K = weight.shape
        x = x.contiguous()
        weight = weight.contiguous()
        N = B * S
        idx = torch.empty(N, top_k, device=x.device, dtype=torch.long)
        wt = torch.empty(N, top_k, device=x.device, dtype=torch.float32)
        logits = torch.empty(N, E, device=x.device, dtype=torch.float32)
        BLOCK_S = 32
        BLOCK_E = max(16, triton.next_power_of_2(E))
        BLOCK_D = triton.next_power_of_2(H)
        grid = (B, triton.cdiv(S, BLOCK_S))
        _conv_router_fwd_kernel[grid](
            x, weight, bias if bias is not None else x, idx, wt, logits, S,
            x.stride(0), x.stride(1), x.stride(2), weight.stride(0), weight.stride(1), weight.stride(2),
            logits.stride(0), logits.stride(1), float(scale), HAS_BIAS=bias is not None,
            NORM=bool(norm_topk and top_k > 1),
            E=E, K=K, TOPK=top_k, TOPK_P2=max(1, triton.next_power_of_2(top_k)), H=H,
            BLOCK_S=BLOCK_S, BLOCK_E=BLOCK_E, BLOCK_D=BLOCK_D,
            num_stages=1, num_warps=4)
        counts = _count_experts(idx, num_experts)
        ctx.save_for_backward(x, weight, logits, idx)
        ctx.dims = (B, S, H, E, K, top_k)
        ctx.norm, ctx.scale = bool(norm_topk and top_k > 1), float(scale)
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
            ctx.scale, NORM=ctx.norm,
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N_EP, BLOCK_E=BLOCK_E)

        grad_x = torch.empty(B, S, H, device=x.device, dtype=x.dtype)
        BLOCK_S = 32
        BLOCK_D = 128
        _conv_router_dx_kernel[(B, triton.cdiv(S, BLOCK_S), triton.cdiv(H, BLOCK_D))](
            grad_logits, weight, grad_x, S,
            grad_logits.stride(0), grad_logits.stride(1),
            weight.stride(0), weight.stride(1), weight.stride(2),
            grad_x.stride(0), grad_x.stride(1), grad_x.stride(2),
            E=E, K=K, H=H, BLOCK_S=BLOCK_S, BLOCK_E=BLOCK_E, BLOCK_D=BLOCK_D,
            num_stages=2, num_warps=4)

        grad_w_acc = torch.zeros(E, H, K, device=x.device, dtype=torch.float32)
        xflat = x.view(N, H)
        BLOCK_H = 64
        BLOCK_N = 128
        SPLIT_N = 16
        N_PER = triton.cdiv(triton.cdiv(N, SPLIT_N), BLOCK_N) * BLOCK_N
        _conv_router_dw_kernel[(K, triton.cdiv(H, BLOCK_H), SPLIT_N)](
            grad_logits, xflat, grad_w_acc, N, S,
            grad_logits.stride(0), grad_logits.stride(1), xflat.stride(0), xflat.stride(1),
            grad_w_acc.stride(0), grad_w_acc.stride(1), grad_w_acc.stride(2), N_PER,
            E=E, H=H, BLOCK_E=BLOCK_E, BLOCK_H=BLOCK_H, BLOCK_N=BLOCK_N, K=K,
            num_stages=2, num_warps=4)
        grad_w = grad_w_acc.to(weight.dtype)
        return grad_x, grad_w, None, None, None, None, None


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    B, S, _ = x.shape
    idx, w, counts = FusedConvRouterFused.apply(x, conv_weight, bias, top_k, num_experts,
                                                norm_topk_prob, routed_scaling_factor)
    idx = idx.view(B, S, top_k)
    w = w.view(B, S, top_k)
    return (idx, w, counts) if return_counts else (idx, w)


@triton.jit
def _topk_norm_fwd_kernel(W, O, N, sn, sk, SCALE, EPS, K: tl.constexpr,
                          BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m = (offs_n < N)[:, None] & (offs_k < K)[None, :]
    w = tl.load(W + offs_n[:, None] * sn + offs_k[None, :] * sk, mask=m, other=0.0).to(tl.float32)
    s = tl.sum(w, axis=1) + EPS
    o = w * (SCALE / s)[:, None]
    tl.store(O + offs_n[:, None] * sn + offs_k[None, :] * sk, o.to(O.dtype.element_ty), mask=m)


@triton.jit
def _topk_norm_bwd_kernel(W, GO, GW, N, sn, sk, SCALE, EPS, K: tl.constexpr,
                          BLOCK_K: tl.constexpr, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    m = (offs_n < N)[:, None] & (offs_k < K)[None, :]
    w = tl.load(W + offs_n[:, None] * sn + offs_k[None, :] * sk, mask=m, other=0.0).to(tl.float32)
    go = tl.load(GO + offs_n[:, None] * sn + offs_k[None, :] * sk, mask=m, other=0.0).to(tl.float32)
    s = tl.sum(w, axis=1) + EPS
    p = w / s[:, None]
    gdot = tl.sum(go * p, axis=1)
    gw = (SCALE / s)[:, None] * (go - gdot[:, None])
    tl.store(GW + offs_n[:, None] * sn + offs_k[None, :] * sk, gw.to(GW.dtype.element_ty), mask=m)


class _TopkNormalize(torch.autograd.Function):

    @staticmethod
    def forward(ctx, w, scale, eps):
        w = w.contiguous()
        N, K = w.shape
        o = torch.empty_like(w)
        BLOCK_N, BLOCK_K = 128, max(1, triton.next_power_of_2(K))
        _topk_norm_fwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            w, o, N, w.stride(0), w.stride(1), float(scale), float(eps),
            K=K, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N)
        ctx.save_for_backward(w)
        ctx.scale, ctx.eps = float(scale), float(eps)
        return o

    @staticmethod
    def backward(ctx, go):
        (w,) = ctx.saved_tensors
        N, K = w.shape
        go = go.contiguous()
        gw = torch.empty_like(w)
        BLOCK_N, BLOCK_K = 128, max(1, triton.next_power_of_2(K))
        _topk_norm_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            w, go, gw, N, w.stride(0), w.stride(1), ctx.scale, ctx.eps,
            K=K, BLOCK_K=BLOCK_K, BLOCK_N=BLOCK_N)
        return gw, None, None


def _topk_normalize(w, scale, eps=1e-20):
    return _TopkNormalize.apply(w, scale, eps)
