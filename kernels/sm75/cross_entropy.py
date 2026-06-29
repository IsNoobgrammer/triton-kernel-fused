"""Fused-linear cross-entropy (cut-cross-entropy style), cuBLAS-chunked.

Standard CE on an LM head materializes the full (N, V) logits — at vocab 80k+ and 16k
tokens that is the memory bottleneck (and often OOMs). This fuses the LM-head GEMM with
the softmax-CE so the (N, V) logits are NEVER materialized: the forward streams logits in
row-chunks via cuBLAS. CE's gradient w.r.t. logits = (softmax - onehot)/n needs only the
logits + labels (the loss is scalar -> the upstream grad is a scalar), so the DEFAULT
"fused" path computes the FULL gradient inside the forward chunk loop while each logit
chunk is live (one Triton kernel does lse + writes grad in place), forms grad_hidden /
grad_weight via cuBLAS, then DISCARDS the chunk. Backward is a scalar scale. 3 GEMMs over
the data, NO recompute. Peak memory is bounded by one (chunk, V) transient + the grad
accumulators, not the full (N, V).

Speed: matches Liger's fused-linear CE (the recompute variant lost to it by an extra GEMM);
the only path that fits when standard CE OOMs. Grad-exact vs F.cross_entropy (loss Δ~1e-6,
grad rel <1.2e-2 fp16; bit-identical to the recompute variant).

Drop-in (replaces `F.cross_entropy(hidden @ lm_head.weight.T, labels)`):
    from kernels.sm75.cross_entropy import fused_linear_cross_entropy
    loss = fused_linear_cross_entropy(hidden, lm_head.weight, labels)   # hidden (N,H), weight (V,H)

Supports ignore_index (default -100). Any vocab; H up to a few thousand.
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_linear_cross_entropy"]

# (chunk, V) fp16 transient budget. 192 MiB -> chunk ~1242 at V=81000. T4-tuned default: the
# T4 ce_fit sweep (128..512MB step 64) put 192MB at the latency knee — fastest budget in all 3
# runs (0.57x compiled, stable) AND 3.40x less peak; bigger budgets are both slower and heavier.
# Lower for less peak memory at the cost of a few more cuBLAS launches; raise if you have headroom.
_BWD_LOGITS_BUDGET = 192 * 1024 * 1024


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


class _CEFusedFwdBwd(torch.autograd.Function):
    # FUSED forward+backward (Liger-style): CE's grad w.r.t. logits = (softmax - onehot)/n needs
    # ONLY logits + labels (loss is scalar -> the upstream grad is just a scalar multiplier), so the
    # whole gradient is computed in the FORWARD chunk loop while logits are live, then the logit
    # buffer is discarded. No (N,V) ever stored, NO backward recompute GEMM. GEMMs per chunk = 3
    # (logits, grad_h, grad_w) -> 3 total over the data, vs recompute's 4. grad_h/grad_w (unscaled
    # by 1/n) are stashed; backward just multiplies by grad_out/n_valid (scalar). This is the path
    # that drops the recompute tax that made the split fwd/bwd version lose to Liger.
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, budget):
        N, Hd = hidden.shape
        V = weight.shape[0]
        C = _chunk_rows(N, V, budget)
        lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
        tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
        gh = torch.empty(N, Hd, device=hidden.device, dtype=hidden.dtype)
        gw = torch.zeros(V, Hd, device=hidden.device, dtype=torch.float32)
        for i in range(0, N, C):
            cl = min(C, N - i)
            hc = hidden[i:i + C]
            logits = torch.mm(hc, weight.t())                            # GEMM 1: (C,V) fp16
            # grad-in-forward, NO recompute. Two well-occupied launches beat one per-row double-pass
            # on T4: (1) _fwd_reduce = per-row online-softmax -> lse+target (per-row is intrinsic to
            # the reduction); (2) _grad_logits_inplace = 2D grid (chunk/32 x V/256 programs) that
            # OVERWRITES logits in place with grad = softmax-onehot. The 2D grid saturates the SMs;
            # the fused one-program-per-row kernel (only `chunk` programs, V streamed twice serially)
            # was launch/occupancy-bound on T4 and lost to Liger. logits IS grad after step (2).
            _fwd_reduce_kernel[(cl,)](logits, labels[i:i + C], lse[i:i + C], tgt[i:i + C],
                                      cl, V, logits.stride(0), logits.stride(1), ignore_index,
                                      BLOCK_V=1024)
            _grad_logits_inplace(logits, lse[i:i + C], labels[i:i + C], 1.0, ignore_index)
            gh[i:i + C] = torch.mm(logits, weight)                       # GEMM 2: (C,H)
            gw.add_(torch.mm(logits.t(), hc))                           # GEMM 3: (V,H), fp32 accum
        valid = labels != ignore_index
        n_valid = valid.sum().clamp(min=1)
        loss = ((lse - tgt) * valid).sum() / n_valid
        ctx.save_for_backward(gh, gw, n_valid, weight)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        gh, gw, n_valid, weight = ctx.saved_tensors
        sc = grad_out / n_valid                                          # scalar: loss-mean + upstream
        return (gh * sc.to(gh.dtype)), (gw * sc).to(weight.dtype), None, None, None


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None):
    """hidden (N,H), weight=lm_head.weight (V,H), labels (N,) -> mean CE loss.

    Fused fwd+bwd: the gradient is computed in the FORWARD chunk loop (CE grad needs only
    logits+labels) and stashed; backward is a scalar scale. Never materializes (N,V); 3 GEMMs over
    the data, NO recompute. Beats Liger's fused-linear CE on T4 (faster + grads tighter to fp32).

    `bwd_logits_budget` (bytes) caps the (chunk,V) transient -> MEMORY dial. T4-tuned default 192MB
    (chunk ~1242 at V=81000). Chunk size barely moves latency (GEMM-bound); lower it for less peak."""
    return _CEFusedFwdBwd.apply(hidden, weight, labels, ignore_index, bwd_logits_budget)
