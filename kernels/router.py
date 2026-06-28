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

__all__ = ["fused_router", "router_bias_update", "FusedConvRouter", "FusedConvRouterCuDNN"]


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


@triton.autotune(
    # SRAM-safe for T4 (sm_75, 64KB): operand tiles ~ (BLOCK_S+BLOCK_E)*BLOCK_C*2*num_stages bytes.
    # BLOCK_C capped at 128 (a 128x512 fp16 tile = 128KB > 64KB crashes). This also means the merged
    # contraction can't go "fat" on Turing -> expect ~tldot perf (SRAM-bound). All configs < 40KB.
    configs=[
        triton.Config({"BLOCK_S": 64, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_C": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 64, "BLOCK_C": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_S": 128, "BLOCK_C": 128}, num_warps=8, num_stages=1),
        triton.Config({"BLOCK_S": 256, "BLOCK_C": 64}, num_warps=8, num_stages=1),
    ],
    key=["H"],
)
@triton.jit
def _conv_router_fwd_merged_kernel(
    X_ptr, W2_ptr, Out_ptr, B, S, H,
    sxb, sxs, sxh, sw2e, sw2c, som, soe,
    K: tl.constexpr, E: tl.constexpr, KH: tl.constexpr, BLOCK_E: tl.constexpr,
    BLOCK_S: tl.constexpr, BLOCK_C: tl.constexpr, APPLY_SIGMOID: tl.constexpr,
):
    # Transpose-free causal conv with the K taps FOLDED into one (k,h)=K*H contraction:
    #   out[s,e] = sum_{c=0..KH-1} xim[s,c] * W2[e,c],  c=k*H+h,  xim[s,c]=x[s-(K-1)+k, h]
    # vs the K separate skinny dots in _conv_router_fwd_sigmoid_kernel — fatter/fewer tl.dots over a
    # contiguous 2048 K-dim (better MMA util on T4). x reread via the shifted gather (L2-served).
    pid_b = tl.program_id(0)
    pid_s = tl.program_id(1)
    offs_s = pid_s * BLOCK_S + tl.arange(0, BLOCK_S)
    offs_e = tl.arange(0, BLOCK_E)
    mask_s = offs_s < S
    mask_e = offs_e < E
    acc = tl.zeros((BLOCK_S, BLOCK_E), dtype=tl.float32)
    for c0 in range(0, KH, BLOCK_C):
        offs_c = c0 + tl.arange(0, BLOCK_C)
        kk = offs_c // H
        hh = offs_c % H
        src = offs_s[:, None] - (K - 1) + kk[None, :]
        mc = offs_c < KH
        xmask = (src >= 0) & mask_s[:, None] & mc[None, :]
        xim = tl.load(X_ptr + pid_b * sxb + src * sxs + hh[None, :] * sxh, mask=xmask, other=0.0)
        w2 = tl.load(W2_ptr + offs_e[:, None] * sw2e + offs_c[None, :] * sw2c,
                     mask=mask_e[:, None] & mc[None, :], other=0.0)
        acc += tl.dot(xim, tl.trans(w2))
    if APPLY_SIGMOID:
        acc = 1.0 / (1.0 + tl.exp(-acc))
    out_row = pid_b * S + offs_s
    tl.store(Out_ptr + out_row[:, None] * som + offs_e[None, :] * soe,
             acc.to(Out_ptr.dtype.element_ty), mask=mask_s[:, None] & mask_e[None, :])


def _conv_router_scores_merged(x, weight, apply_sigmoid=True):
    """Transpose-free scores via the merged (k,h) contraction kernel. weight (E,H,K) -> W2 (E,K*H)
    with c=k*H+h (one tiny permute+reshape). Same output as _conv_router_scores."""
    B, S, Hd = x.shape
    E, _, K = weight.shape
    W2 = weight.permute(0, 2, 1).reshape(E, K * Hd).contiguous()   # (E,K,H)->(E,K*H), c=k*H+h
    out = torch.empty(B * S, E, device=x.device, dtype=torch.float32)
    grid = lambda meta: (B, triton.cdiv(S, meta["BLOCK_S"]))
    _conv_router_fwd_merged_kernel[grid](
        x, W2, out, B, S, Hd,
        x.stride(0), x.stride(1), x.stride(2), W2.stride(0), W2.stride(1),
        out.stride(0), out.stride(1),
        K=K, E=E, KH=K * Hd, BLOCK_E=max(16, triton.next_power_of_2(E)), APPLY_SIGMOID=apply_sigmoid)
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


def _conv_router_grads(x, weight, scores, idx, grad_weights):
    """Shared transpose-free backward for the Triton conv routers (tldot + readonce share it).
    gather^T scatter -> sigmoid' -> dx/dw tl.dot kernels. grad_x exact; grad_w up to fp32 order."""
    B, S, Hd = x.shape
    E, _, K = weight.shape
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
    return gx, gw


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
        gx, gw = _conv_router_grads(x, weight, scores, idx, grad_weights)
        return gx, gw, None, None, None   # x, weight, bias, top_k, num_experts


class FusedConvRouterMerged(torch.autograd.Function):
    """Round-5 iter1: same whole-router fusion as FusedConvRouter (transpose-free), but the forward
    conv folds the K taps into ONE K*H contraction (_conv_router_fwd_merged_kernel). Backward shared
    (transpose-free dx/dw). Candidate to beat cudnn's forward — A/B on T4, cudnn champion untouched."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        scores = _conv_router_scores_merged(x, weight, apply_sigmoid=True)
        sel = scores + bias if bias is not None else scores
        _, idx = torch.topk(sel, top_k, dim=-1)
        weights = scores.gather(-1, idx)
        counts = _count_experts(idx, num_experts)
        ctx.save_for_backward(x, weight, scores, idx)
        ctx.mark_non_differentiable(idx, counts)
        return idx, weights, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        x, weight, scores, idx = ctx.saved_tensors
        gx, gw = _conv_router_grads(x, weight, scores, idx, grad_weights)
        return gx, gw, None, None, None


class FusedConvRouterCuBLAS(torch.autograd.Function):
    """Same whole-router fusion, but the conv is K cuBLAS GEMMs on the NATIVE (B,S,H) layout —
    NO transpose, NO pad materialization (the two HBM round-trips F.conv1d/cuDNN can't avoid).

        logits[:, K-1-k:, :] += x[:, :S-(K-1-k), :] @ W[:,:,k].T     for k in 0..K-1   (causal)

    GEMMs run in x.dtype (fp16 tensor cores on T4), accumulated in an fp32 buffer. This is the
    CE/MoE lever (cuBLAS > tl.dot on Turing) applied to the router. Epilogue (sigmoid/scatter/
    sigmoid') is the fused glue; topk/gather/count identical to the tl.dot path."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        B, S, H = x.shape
        E, _, K = weight.shape
        logits = torch.zeros(B, S, E, device=x.device, dtype=torch.float32)
        for k in range(K):
            d = K - 1 - k                                            # causal shift for tap k
            logits[:, d:, :] += (x[:, :S - d, :] @ weight[:, :, k].t()).float()
        scores = torch.sigmoid(logits)
        sel = scores + bias if bias is not None else scores
        _, idx = torch.topk(sel, top_k, dim=-1)
        weights = scores.gather(-1, idx)
        counts = _count_experts(idx, num_experts)
        ctx.save_for_backward(x, weight, scores, idx)
        ctx.mark_non_differentiable(idx, counts)
        return idx, weights, counts

    @staticmethod
    def backward(ctx, grad_idx, grad_weights, grad_counts):
        x, weight, scores, idx = ctx.saved_tensors
        B, S, H = x.shape
        E, _, K = weight.shape
        grad_scores = torch.zeros_like(scores)
        grad_scores.scatter_add_(-1, idx, grad_weights.float())
        grad_logits = grad_scores * scores * (1.0 - scores)          # (B,S,E) fp32
        gl = grad_logits.to(x.dtype)                                 # cast once for fp16 tensor cores
        gx = torch.zeros(B, S, H, device=x.device, dtype=torch.float32)
        gw = torch.empty(E, H, K, device=x.device, dtype=weight.dtype)
        for k in range(K):
            d = K - 1 - k
            # grad_x[:, :S-d] += grad_logits[:, d:] @ W[:,:,k]   (E,H), fp16 GEMM, fp32 accumulate
            gx[:, :S - d, :] += (gl[:, d:, :] @ weight[:, :, k]).float()
            # grad_w[:,:,k] = (Σ_bs gl·x) = gl[:,d:]ᵀ @ x[:,:S-d] as ONE (E,N)x(N,H) cuBLAS GEMM
            # over N=B·(S-d) (fp16 in, fp32 internal accumulate). reshape copies the slice contiguous.
            gw[:, :, k] = gl[:, d:, :].reshape(-1, E).t() @ x[:, :S - d, :].reshape(-1, H)
        return gx.to(x.dtype), gw, None, None, None


@triton.jit
def _router_epilogue_kernel(Logit_ptr, Bias_ptr, Idx_ptr, W_ptr, N, sln, sle,
                            HAS_BIAS: tl.constexpr, E: tl.constexpr, TOPK: tl.constexpr,
                            BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    """Fused router epilogue: sigmoid + (selection)bias + top-k argmax + unbiased gather, in ONE pass.
    Replaces [sigmoid, add, torch.topk, gather] — the native topk is a ~295us general bitonic kernel
    for top-2-of-11; a bespoke argmax over E=11 is ~free, and compile CANNOT fuse a native topk.
    BLOCK_N rows/program (vectorized argmax over E) for occupancy. sel = scores+bias picks; weights
    = scores (UNBIASED) gathered at the picks. logits read fp16, computed fp32 (matches eager)."""
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
    sel = tl.where(mask_e[None, :], sel, -1e30)            # mask padded experts out of the argmax
    for k in tl.static_range(TOPK):
        am = tl.argmax(sel, axis=1)                        # (BLOCK_N,) — top expert this round
        onehot = offs_e[None, :] == am[:, None]
        w_k = tl.sum(tl.where(onehot, scores, 0.0), axis=1)   # unbiased score at the pick
        tl.store(Idx_ptr + offs_n * TOPK + k, am.to(tl.int64), mask=mask_n)
        tl.store(W_ptr + offs_n * TOPK + k, w_k, mask=mask_n)
        sel = tl.where(onehot, -1e30, sel)                 # remove the pick, next round finds the next
    return


@triton.jit
def _router_epilogue_bwd_kernel(Logit_ptr, Idx_ptr, Gw_ptr, Gout_ptr, N,
                                sln, sle, sin, sik, sgn, sgk, son, soe,
                                E: tl.constexpr, TOPK: tl.constexpr,
                                BLOCK_N: tl.constexpr, BLOCK_E: tl.constexpr):
    """grad_logits (N,E) = scatter(grad_w -> picked idx slots) * sigmoid(logit)*(1-sigmoid(logit)).
    Fuses the epilogue backward (sigmoid' + gather^T scatter) into ONE kernel — replaces sigmoid +
    zeros + scatter_add + 2 muls + cast. BLOCK_N rows/program."""
    pid = tl.program_id(0)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_e = tl.arange(0, BLOCK_E)
    mask_n = offs_n < N
    mask_e = offs_e < E
    logit = tl.load(Logit_ptr + offs_n[:, None] * sln + offs_e[None, :] * sle,
                    mask=mask_n[:, None] & mask_e[None, :], other=0.0).to(tl.float32)
    s = 1.0 / (1.0 + tl.exp(-logit))
    sp = s * (1.0 - s)                                     # sigmoid'
    gscore = tl.zeros((BLOCK_N, BLOCK_E), dtype=tl.float32)
    for k in tl.static_range(TOPK):
        ik = tl.load(Idx_ptr + offs_n * sin + k * sik, mask=mask_n, other=0).to(tl.int32)
        gk = tl.load(Gw_ptr + offs_n * sgn + k * sgk, mask=mask_n, other=0.0)
        gscore += tl.where(offs_e[None, :] == ik[:, None], gk[:, None], 0.0)   # distinct picks -> set
    gout = gscore * sp
    tl.store(Gout_ptr + offs_n[:, None] * son + offs_e[None, :] * soe,
             gout.to(Gout_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_e[None, :])


def _epilogue_fwd(logits, bias, top_k):
    """logits (N,E) -> idx (N,k) long, weights (N,k) fp32 unbiased — fused sigmoid+bias+top-k+gather."""
    N, E = logits.shape
    idx = torch.empty(N, top_k, device=logits.device, dtype=torch.long)
    w = torch.empty(N, top_k, device=logits.device, dtype=torch.float32)
    BLOCK_N = 128
    grid = (triton.cdiv(N, BLOCK_N),)
    _router_epilogue_kernel[grid](
        logits, bias if bias is not None else logits, idx, w, N,
        logits.stride(0), logits.stride(1),
        HAS_BIAS=bias is not None, E=E, TOPK=top_k,
        BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
    return idx, w


class FusedConvRouterCuDNN(torch.autograd.Function):
    """WIN backend (Round 4): cuDNN conv + fused top-k epilogue + MERGED manual backward.
    Forward: padding=K-1 cuDNN conv (no F.pad copy -> fwd 1.15x on T4) + fused epilogue (kills native
    topk). Backward: cuDNN convolution_backward called DIRECTLY on the saved-once contiguous (B,H,S)
    input (autograd would copy x->contiguous twice + cast/transpose around it = the 0.83x glue) + the
    fused epilogue-bwd kernel. grad_x exact; grad_w = cuDNN's (bit-matches compiled's bwd)."""

    @staticmethod
    def forward(ctx, x, weight, bias, top_k, num_experts):
        import torch.nn.functional as F
        B, S, H = x.shape
        E, _, K = weight.shape
        # NCHW-contiguous (B,H,S) for cuDNN. (Tried channels-last to skip cuDNN's ~482us nchwToNhwc
        # transposes — T4-refuted: cuDNN copies to its layout anyway + the strided input ADDED copies
        # and slowed convolution_backward 613->948us. The transpose tax isn't removable via cuDNN.)
        xc = x.transpose(1, 2).contiguous()                     # (B,H,S) — ONE copy, reused in bwd
        conv = F.conv1d(xc, weight, padding=K - 1)[..., :S]      # (B,E,S) causal, no F.pad copy
        logits = conv.transpose(1, 2).reshape(B * S, E)          # (B*S,E)
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
        B, H, _ = xc.shape
        N, top_k = idx.shape
        grad_logits = torch.empty(N, E, device=xc.device, dtype=xc.dtype)
        gw = grad_weights.contiguous()
        BLOCK_N = 128
        _router_epilogue_bwd_kernel[(triton.cdiv(N, BLOCK_N),)](
            logits, idx, gw, grad_logits, N,
            logits.stride(0), logits.stride(1), idx.stride(0), idx.stride(1),
            gw.stride(0), gw.stride(1), grad_logits.stride(0), grad_logits.stride(1),
            E=E, TOPK=top_k, BLOCK_N=BLOCK_N, BLOCK_E=max(16, triton.next_power_of_2(E)))
        # to conv grad layout (B,E,S), pad right K-1 (the sliced-off outputs get zero grad)
        grad_full = F.pad(grad_logits.view(B, S, E).transpose(1, 2), (0, K - 1))   # (B,E,S+K-1)
        grad_xc, grad_w = torch.ops.aten.convolution_backward(
            grad_full, xc, weight, [0], [1], [K - 1], [1], False, [0], 1, [True, True, False])[:2]
        return grad_xc.transpose(1, 2), grad_w, None, None, None   # (B,H,S)->(B,S,H)


def _cudnn_router(x, weight, bias, top_k, num_experts):
    return FusedConvRouterCuDNN.apply(x, weight, bias, top_k, num_experts)


def _ref_router(x, weight, bias, top_k, num_experts):
    """REFERENCE backend = the torch.compile dump's recipe, hand-assembled and UNCOMPILED: cuDNN
    conv (the extern op inductor itself falls back to) + plain torch glue. Pure-autograd (no custom
    Function) so cuDNN convolution_backward runs the backward. Measures the bar a hand kernel must
    beat, and how much of compiled's win is just inductor fusing the glue vs the conv itself."""
    import torch.nn.functional as F
    B, S, H = x.shape
    E, _, K = weight.shape
    xp = F.pad(x.transpose(1, 2), (K - 1, 0))                 # (B,H,S+K-1)
    logits = F.conv1d(xp, weight).transpose(1, 2).reshape(B * S, E).float()   # cuDNN -> (B*S,E)
    scores = torch.sigmoid(logits)
    sel = scores + bias if bias is not None else scores
    _, idx = torch.topk(sel, top_k, dim=-1)
    weights = scores.gather(-1, idx)
    counts = _count_experts(idx, num_experts)
    return idx, weights, counts


_BACKENDS = {"tldot": FusedConvRouter, "tlconv": FusedConvRouterMerged,
             "cublas": FusedConvRouterCuBLAS}
_FN_BACKENDS = {"ref": _ref_router, "cudnn": _cudnn_router}   # plain-autograd backends (cuDNN conv)


def fused_router(x, conv_weight, bias, top_k, num_experts,
                 norm_topk_prob=True, routed_scaling_factor=1.0, return_counts=False,
                 backend="cudnn"):
    """Whole conv router. x (B,S,H), conv_weight (E,H,K) from nn.Conv1d(H,E,K), bias (E,) fp32 or None.

    backend (T4 vs torch.compile, Round 4):
      'cudnn'  — DEFAULT/best: cuDNN conv (autograd convolution_backward) + FusedTop2Epilogue (one
                 Triton kernel killing the ~295us native topk+gather). Ties compiled on speed (~1.0x
                 fwd+bwd), exact grads, mem parity. The conv itself can't beat cuDNN (inductor punts
                 to cuDNN too), so the win is the fused topk seam — tie+better-grads is the ceiling.
      'tldot'  — one transpose-free Triton conv; 0.73x speed but 1.12x LESS mem (pick when mem-bound).
      'cublas' — K cuBLAS GEMMs, native layout; dominated (0.35x), kept for reference.
    Returns (idx (B,S,k) long, norm_weights (B,S,k) fp32) — or (..., counts (E,) int32). norm_topk_prob
    / routed_scaling applied in eager.
    """
    B, S, _ = x.shape
    if backend in _FN_BACKENDS:
        idx, w, counts = _FN_BACKENDS[backend](x, conv_weight, bias, top_k, num_experts)
    else:
        idx, w, counts = _BACKENDS[backend].apply(x, conv_weight, bias, top_k, num_experts)
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
