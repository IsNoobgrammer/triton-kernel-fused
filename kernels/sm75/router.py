import torch
import triton
import triton.language as tl

__all__ = ["fused_router", "fused_mlp_router", "router_bias_update",
           "FusedConvRouterCuDNN", "FusedMLPRouter"]


@triton.jit
def _router_epilogue_fwd_kernel(Logit_ptr, Bias_ptr, Idx_ptr, W_ptr, Count_ptr, N, sln, sle, scale,
                                HAS_BIAS: tl.constexpr, COUNT: tl.constexpr, NORM: tl.constexpr,
                                E: tl.constexpr, TOPK: tl.constexpr, TOPK_P2: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    offs_k = tl.arange(0, TOPK_P2)
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
    cnt = tl.zeros((BLOCK_E,), dtype=tl.int32)
    wmat = tl.zeros((BLOCK_N, TOPK_P2), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        am = tl.argmax(sel, axis=1)
        onehot = offs_e[None, :] == am[:, None]
        w_k = tl.sum(tl.where(onehot, scores, 0.0), axis=1)
        tl.store(Idx_ptr + offs_n * TOPK + k, am.to(tl.int64), mask=mask_n)
        wmat = tl.where(offs_k[None, :] == k, w_k[:, None], wmat)
        if COUNT:
            cnt += tl.sum(tl.where(mask_n[:, None] & onehot, 1, 0).to(tl.int32), axis=0)
        sel = tl.where(onehot, -1e30, sel)
    if NORM:
        t = tl.sum(wmat, axis=1) + 1e-20
        wmat = wmat / t[:, None]
    wmat = wmat * scale
    tl.store(W_ptr + offs_n[:, None] * TOPK + offs_k[None, :], wmat,
             mask=mask_n[:, None] & (offs_k < TOPK)[None, :])
    if COUNT:
        tl.atomic_add(Count_ptr + offs_e, cnt, mask=mask_e)


@triton.jit
def _router_epilogue_bwd_kernel(Logit_ptr, Idx_ptr, Gw_ptr, Gout_ptr, N,
                                sln, sle, sin, sik, sgn, sgk, son, soe, scale,
                                NORM: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    mask_n = offs_n < N
    mask_e = offs_e < E
    logit = tl.load(Logit_ptr + offs_n[:, None] * sln + offs_e[None, :] * sle,
                    mask=mask_n[:, None] & mask_e[None, :], other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-logit))
    sp = s * (1.0 - s)
    if NORM:
        t = tl.zeros((BLOCK_N,), dtype=tl.float32)
        dot = tl.zeros((BLOCK_N,), dtype=tl.float32)
        for k in tl.static_range(TOPK):
            ik = tl.load(Idx_ptr + offs_n * sin + k * sik, mask=mask_n, other=0).to(tl.int32)
            gk = tl.load(Gw_ptr + offs_n * sgn + k * sgk, mask=mask_n, other=0.0).to(tl.float32)
            wk = tl.sum(tl.where(offs_e[None, :] == ik[:, None], s, 0.0), axis=1)
            t += wk
            dot += gk * wk
        t += 1e-20
    gscore = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        ik = tl.load(Idx_ptr + offs_n * sin + k * sik, mask=mask_n, other=0).to(tl.int32)
        gk = tl.load(Gw_ptr + offs_n * sgn + k * sgk, mask=mask_n, other=0.0).to(tl.float32)
        if NORM:
            graw = (scale / t) * (gk - dot / t)
        else:
            graw = gk * scale
        gscore += tl.where(offs_e[None, :] == ik[:, None], graw[:, None], 0.0)
    gout = gscore * sp
    tl.store(Gout_ptr + offs_n[:, None] * son + offs_e[None, :] * soe,
             gout.to(Gout_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_e[None, :])


def _epilogue_fwd(logits, bias, top_k, num_experts=None, norm_topk=True, scale=1.0):
    N, E = logits.shape
    idx = torch.empty(N, top_k, device=logits.device, dtype=torch.long)
    w = torch.empty(N, top_k, device=logits.device, dtype=torch.float32)
    counts = torch.zeros(num_experts, device=logits.device, dtype=torch.int32) if num_experts else None
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _router_epilogue_fwd_kernel[grid](
        logits, bias if bias is not None else logits, idx, w,
        counts if counts is not None else logits, N,
        logits.stride(0), logits.stride(1), float(scale),
        HAS_BIAS=bias is not None, COUNT=counts is not None,
        NORM=bool(norm_topk and top_k > 1), E=E, TOPK=top_k,
        TOPK_P2=max(1, triton.next_power_of_2(top_k)),
        BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
    return idx, w, counts


@triton.jit
def _count_experts_kernel(Idx_ptr, Count_ptr, NK, E: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < NK
    e = tl.load(Idx_ptr + offs, mask=mask, other=-1)
    for ei in tl.static_range(E):
        tl.atomic_add(Count_ptr + ei, tl.sum(tl.where(e == ei, 1, 0).to(tl.int32)))


def _count_experts(idx, num_experts):
    flat = idx.reshape(-1).to(torch.int32)
    NK = flat.numel()
    counts = torch.zeros(num_experts, device=idx.device, dtype=torch.int32)
    BLOCK = 1024
    _count_experts_kernel[(triton.cdiv(NK, BLOCK),)](flat, counts, NK, E=num_experts, BLOCK=BLOCK)
    return counts


class FusedConvRouterCuDNN(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts, norm_topk=True, scale=1.0):
        import torch.nn.functional as F
        if torch.is_autocast_enabled("cuda"):
            _dt = torch.get_autocast_dtype("cuda")
            x, weight = x.to(_dt), weight.to(_dt)
        B, S, H = x.shape
        E, _, K = weight.shape
        xc = x.transpose(1, 2).contiguous()
        conv = F.conv1d(xc, weight, padding=K - 1)[..., :S]
        logits = conv.transpose(1, 2).reshape(B * S, E)
        idx, weights, counts = _epilogue_fwd(logits, bias, top_k, num_experts,
                                             norm_topk=norm_topk, scale=scale)
        ctx.save_for_backward(xc, weight, logits, idx)
        ctx.K, ctx.S, ctx.E = K, S, E
        ctx.norm, ctx.scale = bool(norm_topk and top_k > 1), float(scale)
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
            ctx.scale, NORM=ctx.norm,
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
        B = xc.shape[0]
        grad_full = F.pad(grad_logits.view(B, S, E).transpose(1, 2), (0, K - 1))
        grad_xc, grad_w = torch.ops.aten.convolution_backward(
            grad_full, xc, weight, [0], [1], [K - 1], [1], False, [0], 1, [True, True, False])[:2]
        return grad_xc.transpose(1, 2), grad_w, None, None, None, None, None


class FusedMLPRouter(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts, norm_topk=True, scale=1.0):
        if torch.is_autocast_enabled("cuda"):
            _dt = torch.get_autocast_dtype("cuda")
            x, weight = x.to(_dt), weight.to(_dt)
        xf = x.reshape(-1, x.shape[-1]).contiguous()
        logits = xf @ weight.t()
        idx, weights, counts = _epilogue_fwd(logits, bias, top_k, num_experts,
                                             norm_topk=norm_topk, scale=scale)
        ctx.xshape = x.shape
        ctx.save_for_backward(xf, weight, logits, idx)
        ctx.norm, ctx.scale = bool(norm_topk and top_k > 1), float(scale)
        ctx.mark_non_differentiable(idx, counts)
        return idx, weights, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        xf, weight, logits, idx = ctx.saved_tensors
        N, top_k = idx.shape
        E = weight.shape[0]
        grad_logits = torch.empty(N, E, device=xf.device, dtype=xf.dtype)
        gw = grad_weights.contiguous()
        BLOCK_N = 128
        _router_epilogue_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            logits, idx, gw, grad_logits, N,
            logits.stride(0), logits.stride(1), idx.stride(0), idx.stride(1),
            gw.stride(0), gw.stride(1), grad_logits.stride(0), grad_logits.stride(1),
            ctx.scale, NORM=ctx.norm,
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
        grad_x = grad_logits @ weight
        grad_w = grad_logits.t() @ xf
        return grad_x.view(ctx.xshape), grad_w, None, None, None, None, None


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    B, S, _ = x.shape
    idx, w, counts = FusedConvRouterCuDNN.apply(x, conv_weight, bias, top_k, num_experts,
                                                norm_topk_prob, routed_scaling_factor)
    idx = idx.view(B, S, top_k)
    w = w.view(B, S, top_k)
    return (idx, w, counts) if return_counts else (idx, w)


def fused_mlp_router(x, gate_weight, bias, top_k, num_experts,
                     norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False):
    idx, w, counts = FusedMLPRouter.apply(x, gate_weight, bias, top_k, num_experts,
                                          norm_topk_prob, routed_scaling_factor)
    if x.ndim == 3:
        B, S, _ = x.shape
        idx = idx.view(B, S, top_k)
        w = w.view(B, S, top_k)
    return (idx, w, counts) if return_counts else (idx, w)


@torch.no_grad()
def router_bias_update(bias, counts, u, mode="prop"):
    # "prop": share = counts/sum(counts); bias += u*(mean(share) - share). This is BiBo's balancer
    # as of Aug 1 2026 -- proportional control on the NORMALIZED share, which has a fixed point and
    # cannot drift common-mode. "sign" is the older DeepSeek bang-bang rule on RAW counts, kept
    # because every run before that date used it and u is NOT comparable between the two.
    if u <= 0:
        return
    tpe = counts.detach().float()
    if mode == "sign":
        bias.add_(u * (tpe.mean() - tpe).sign())
        return
    share = tpe / tpe.sum().clamp_min(1.0)
    bias.add_(u * (share.mean() - share))
