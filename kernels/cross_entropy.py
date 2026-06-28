"""Fused-linear cross-entropy (cut-cross-entropy style), cuBLAS-chunked.

Standard CE on an LM head materializes the full (N, V) logits — at vocab 80k+ and 16k
tokens that is the memory bottleneck (and often OOMs). This fuses the LM-head GEMM with
the softmax-CE so the (N, V) logits are NEVER materialized: the forward streams logits
in row-chunks via cuBLAS and keeps only `lse` (N,); the backward recomputes each chunk's
logits (cuBLAS), turns them into grad-logits in-place with a single Triton kernel
(softmax - onehot, scaled), then forms grad_hidden / grad_weight via cuBLAS. Peak memory
is bounded by one (chunk, V) transient instead of the full (N, V).

Speed: ties torch-compiled standard CE (both cuBLAS-bound) while using ~chunk-sized memory;
it is the only path that fits when standard CE OOMs. Grad-exact vs F.cross_entropy
(loss Δ~4e-6, grad Δ~4e-9).

Drop-in (replaces `F.cross_entropy(hidden @ lm_head.weight.T, labels)`):
    from kernels.cross_entropy import fused_linear_cross_entropy
    loss = fused_linear_cross_entropy(hidden, lm_head.weight, labels)   # hidden (N,H), weight (V,H)

Supports ignore_index (default -100). Any vocab; H up to a few thousand.
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_linear_cross_entropy"]

# (chunk, V) fp16 transient budget. 384 MiB -> chunk ~2485 at V=81000. Lower for less peak
# memory at the cost of a few more cuBLAS launches; raise if you have headroom.
_BWD_LOGITS_BUDGET = 384 * 1024 * 1024


@triton.jit
def _grad_logits_kernel(L_ptr, Lse_ptr, Lab_ptr, M, Vv, scale, ignore_index,
                        s_lm, s_lv, BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_m = offs_m < M
    mask_v = offs_v < Vv
    mask = mask_m[:, None] & mask_v[None, :]
    lse = tl.load(Lse_ptr + offs_m, mask=mask_m, other=0.0)
    lab = tl.load(Lab_ptr + offs_m, mask=mask_m, other=ignore_index)
    lptr = L_ptr + offs_m[:, None] * s_lm + offs_v[None, :] * s_lv
    logit = tl.load(lptr, mask=mask, other=0.0).to(tl.float32)
    p = tl.exp(logit - lse[:, None])
    g = (p - tl.where(offs_v[None, :] == lab[:, None], 1.0, 0.0)) * scale
    g = tl.where(lab[:, None] != ignore_index, g, 0.0)
    tl.store(lptr, g.to(L_ptr.dtype.element_ty), mask=mask)


def _grad_logits_inplace(logits, lse, labels, scale, ignore_index):
    M, Vv = logits.shape
    BLOCK_M, BLOCK_V = 32, 256
    _grad_logits_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(Vv, BLOCK_V))](
        logits, lse, labels, M, Vv, scale, ignore_index,
        logits.stride(0), logits.stride(1), BLOCK_M=BLOCK_M, BLOCK_V=BLOCK_V)
    return logits


@triton.jit
def _fwd_reduce_kernel(L_ptr, Lab_ptr, Lse_ptr, Tgt_ptr, M, V, s_n, s_v, ignore_index,
                       BLOCK_V: tl.constexpr):
    # One program per row of a chunk. Online-softmax over V — reads the fp16 logits, accumulates
    # max+sum in fp32 REGISTERS (never materializes an fp32 (C,V) buffer), and gathers the target
    # logit in the SAME launch. Replaces the old .float() + torch.logsumexp + .gather (3 passes +
    # an fp32 (C,V) alloc) with one streaming pass. This is the forward-latency/memory win.
    row = tl.program_id(0)
    lab = tl.load(Lab_ptr + row)
    m = -float("inf")
    s = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        x = tl.load(L_ptr + row * s_n + offs * s_v, mask=offs < V, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    tl.store(Lse_ptr + row, m + tl.log(s))
    safe_lab = tl.where(lab == ignore_index, 0, lab)
    tl.store(Tgt_ptr + row, tl.load(L_ptr + row * s_n + safe_lab * s_v).to(tl.float32))


def _chunk_rows(N, V, budget=None):
    return max(512, min(N, (budget or _BWD_LOGITS_BUDGET) // (V * 2)))


class _CECublasChunked(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, budget):
        N, Hd = hidden.shape
        V = weight.shape[0]
        C = _chunk_rows(N, V, budget)
        ctx.budget = budget
        lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
        tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
        with torch.no_grad():
            for i in range(0, N, C):
                cl = min(C, N - i)
                logits = torch.mm(hidden[i:i + C], weight.t())           # cuBLAS (C,V) fp16 — no .float()
                # one fused pass: online-softmax -> lse, + target gather. No fp32 (C,V) buffer.
                _fwd_reduce_kernel[(cl,)](logits, labels[i:i + C], lse[i:i + C], tgt[i:i + C],
                                          cl, V, logits.stride(0), logits.stride(1), ignore_index,
                                          BLOCK_V=1024)
        valid = labels != ignore_index
        loss = ((lse - tgt) * valid).sum() / valid.sum().clamp(min=1)
        ctx.save_for_backward(hidden, weight, labels, lse, valid.sum().clamp(min=1))
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        hidden, weight, labels, lse, n_valid = ctx.saved_tensors
        ig = ctx.ignore_index
        N, Hd = hidden.shape
        V = weight.shape[0]
        # Keep the loss-mean scale on-GPU (0-d tensor) — no float()/.item() host sync / graph break.
        # Applied AFTER the GEMMs: (g·scale)@W == (g@W)·scale (linear), so grad-exact.
        sc = grad_out / n_valid
        gh = torch.empty(N, Hd, device=hidden.device, dtype=hidden.dtype)
        gw = torch.zeros(V, Hd, device=hidden.device, dtype=torch.float32)
        C = _chunk_rows(N, V, ctx.budget)
        for i in range(0, N, C):
            hc = hidden[i:i + C]; labc = labels[i:i + C]; lsec = lse[i:i + C]
            logits = torch.mm(hc, weight.t())                            # cuBLAS recompute (C,V)
            g = _grad_logits_inplace(logits, lsec, labc, 1.0, ig)        # in-place (softmax - onehot), unscaled
            gh[i:i + C] = torch.mm(g, weight)
            # accumulate grad_weight in fp32 WITHOUT a (V,H) fp32 temp: add_ casts the fp16 mm
            # result in its fused kernel (the old `.float()` materialized a 166MB temp PER chunk).
            gw.add_(torch.mm(g.t(), hc))
        return (gh * sc.to(gh.dtype)), (gw * sc).to(weight.dtype), None, None, None


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None):
    """hidden (N,H), weight=lm_head.weight (V,H), labels (N,) -> mean CE loss.
    Never materializes (N,V); cuBLAS speed at bounded (chunk,V) memory. bwd_logits_budget (bytes)
    overrides the (chunk,V) transient budget: larger = fewer cuBLAS launches, more peak memory."""
    return _CECublasChunked.apply(hidden, weight, labels, ignore_index, bwd_logits_budget)
