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
    from kernels.cross_entropy import fused_linear_cross_entropy
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


@triton.jit
def _fwd_reduce_grad_kernel(L_ptr, Lab_ptr, Lse_ptr, Tgt_ptr, M, V, s_n, s_v, scale, ignore_index,
                            BLOCK_V: tl.constexpr):
    # FUSED forward+grad in ONE launch, one program per row. Pass 1: online-softmax over V -> lse
    # (fp32 registers) + gather target logit. Pass 2: re-stream V and OVERWRITE the logit buffer
    # in place with grad = (softmax - onehot) * scale. So the (chunk,V) buffer is reused as the
    # grad-logit buffer (no second alloc) and the two HBM round-trips happen in one kernel — Liger
    # does these as separate steps. Used by the fused-fwd+bwd path so backward never recomputes.
    row = tl.program_id(0)
    lab = tl.load(Lab_ptr + row)
    m = -float("inf")
    s = 0.0
    for v0 in range(0, V, BLOCK_V):                       # pass 1: lse
        offs = v0 + tl.arange(0, BLOCK_V)
        x = tl.load(L_ptr + row * s_n + offs * s_v, mask=offs < V, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    lse = m + tl.log(s)
    tl.store(Lse_ptr + row, lse)
    safe_lab = tl.where(lab == ignore_index, 0, lab)
    # target logit read BEFORE pass 2 overwrites the buffer
    tl.store(Tgt_ptr + row, tl.load(L_ptr + row * s_n + safe_lab * s_v).to(tl.float32))
    is_valid = lab != ignore_index
    for v0 in range(0, V, BLOCK_V):                       # pass 2: write grad in place
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < V
        lptr = L_ptr + row * s_n + offs * s_v
        x = tl.load(lptr, mask=vmask, other=0.0).to(tl.float32)
        g = (tl.exp(x - lse) - tl.where(offs == lab, 1.0, 0.0)) * scale
        g = tl.where(is_valid, g, 0.0)
        tl.store(lptr, g.to(L_ptr.dtype.element_ty), mask=vmask)


@triton.jit
def _dequant_grad_kernel(Q_ptr, QS_ptr, Lse_ptr, Lab_ptr, G_ptr, M, Vv, ignore_index,
                         s_qm, s_qv, s_gm, s_gv, BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr):
    # int8 path backward: dequantize logit = q * qscale[row] (NO recompute GEMM), then
    # grad = softmax(logit) - onehot, written fp16. Replaces recompute-mm + grad-logit kernel.
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_m = offs_m < M
    mask = mask_m[:, None] & (offs_v < Vv)[None, :]
    qs = tl.load(QS_ptr + offs_m, mask=mask_m, other=0.0)
    lse = tl.load(Lse_ptr + offs_m, mask=mask_m, other=0.0)
    lab = tl.load(Lab_ptr + offs_m, mask=mask_m, other=ignore_index)
    q = tl.load(Q_ptr + offs_m[:, None] * s_qm + offs_v[None, :] * s_qv, mask=mask, other=0).to(tl.float32)
    logit = q * qs[:, None]
    p = tl.exp(logit - lse[:, None])
    g = p - tl.where(offs_v[None, :] == lab[:, None], 1.0, 0.0)
    g = tl.where(lab[:, None] != ignore_index, g, 0.0)
    tl.store(G_ptr + offs_m[:, None] * s_gm + offs_v[None, :] * s_gv, g.to(G_ptr.dtype.element_ty), mask=mask)


def _chunk_rows(N, V, budget=None):
    return max(512, min(N, (budget or _BWD_LOGITS_BUDGET) // (V * 2)))


@triton.jit
def _fwd_reduce_q_kernel(L_ptr, Lab_ptr, Lse_ptr, Tgt_ptr, QS_ptr, M, V, s_n, s_v, ignore_index,
                         BLOCK_V: tl.constexpr):
    # forward reduce (online softmax -> lse + target) AND per-row abs-max -> quant scale, one pass.
    row = tl.program_id(0)
    lab = tl.load(Lab_ptr + row)
    m = -float("inf"); s = 0.0; amax_abs = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        vmask = offs < V
        x = tl.load(L_ptr + row * s_n + offs * s_v, mask=vmask, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
        amax_abs = tl.maximum(amax_abs, tl.max(tl.abs(tl.where(vmask, x, 0.0)), 0))
    tl.store(Lse_ptr + row, m + tl.log(s))
    safe_lab = tl.where(lab == ignore_index, 0, lab)
    tl.store(Tgt_ptr + row, tl.load(L_ptr + row * s_n + safe_lab * s_v).to(tl.float32))
    tl.store(QS_ptr + row, tl.maximum(amax_abs / 127.0, 1e-4))


@triton.jit
def _quant_kernel(L_ptr, QS_ptr, Q_ptr, M, V, s_lm, s_lv, s_qm, s_qv,
                  BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr):
    # write int8 q = round(logit / qscale[row]) — one pass, no torch temps.
    pid_m = tl.program_id(0); pid_v = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_m = offs_m < M
    mask = mask_m[:, None] & (offs_v < V)[None, :]
    qs = tl.load(QS_ptr + offs_m, mask=mask_m, other=1.0)
    logit = tl.load(L_ptr + offs_m[:, None] * s_lm + offs_v[None, :] * s_lv, mask=mask, other=0.0).to(tl.float32)
    r = logit / qs[:, None]
    r = tl.where(r >= 0, tl.floor(r + 0.5), tl.math.ceil(r - 0.5))   # round-half-away, libdevice-free
    q = tl.minimum(tl.maximum(r, -127.0), 127.0)
    tl.store(Q_ptr + offs_m[:, None] * s_qm + offs_v[None, :] * s_qv, q.to(tl.int8), mask=mask)


class _CEInt8(torch.autograd.Function):
    # int8-saved-logits path: forward quantizes the (chunk,V) logits per-row to int8 + a per-row
    # scale and SAVES them (~1 byte/elem = half of fp16); backward DEQUANTIZES instead of recomputing
    # the GEMM -> backward drops from 3 GEMMs to 2. Trades the recompute GEMM for ~N*V bytes of int8
    # held fwd->bwd. Grad is approximate (int8 logit quant) — gated by grad_rel < 1.5e-2.
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, budget):
        N, Hd = hidden.shape
        V = weight.shape[0]
        C = _chunk_rows(N, V, budget)
        lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
        tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
        qscale = torch.empty(N, device=hidden.device, dtype=torch.float32)
        q_all = torch.empty(N, V, device=hidden.device, dtype=torch.int8)   # SAVED (1 byte/elem)
        BM, BV = 32, 256
        with torch.no_grad():
            for i in range(0, N, C):
                cl = min(C, N - i)
                logits = torch.mm(hidden[i:i + C], weight.t())              # (C,V) fp16
                _fwd_reduce_q_kernel[(cl,)](logits, labels[i:i + C], lse[i:i + C], tgt[i:i + C],
                                            qscale[i:i + C], cl, V, logits.stride(0), logits.stride(1),
                                            ignore_index, BLOCK_V=1024)
                qc = q_all[i:i + C]
                _quant_kernel[(triton.cdiv(cl, BM), triton.cdiv(V, BV))](
                    logits, qscale[i:i + C], qc, cl, V, logits.stride(0), logits.stride(1),
                    qc.stride(0), qc.stride(1), BLOCK_M=BM, BLOCK_V=BV)
        valid = labels != ignore_index
        loss = ((lse - tgt) * valid).sum() / valid.sum().clamp(min=1)
        ctx.save_for_backward(q_all, qscale, lse, labels, weight, hidden, valid.sum().clamp(min=1))
        ctx.ignore_index = ignore_index; ctx.budget = budget
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        q_all, qscale, lse, labels, weight, hidden, n_valid = ctx.saved_tensors
        ig = ctx.ignore_index
        N, Hd = hidden.shape
        V = weight.shape[0]
        sc = grad_out / n_valid
        gh = torch.empty(N, Hd, device=hidden.device, dtype=hidden.dtype)
        gw = torch.zeros(V, Hd, device=hidden.device, dtype=torch.float32)
        C = _chunk_rows(N, V, ctx.budget)
        BM, BV = 32, 256
        for i in range(0, N, C):
            cl = min(C, N - i)
            g = torch.empty(cl, V, device=hidden.device, dtype=hidden.dtype)
            _dequant_grad_kernel[(triton.cdiv(cl, BM), triton.cdiv(V, BV))](
                q_all[i:i + C], qscale[i:i + C], lse[i:i + C], labels[i:i + C], g, cl, V, ig,
                q_all.stride(0), q_all.stride(1), g.stride(0), g.stride(1), BLOCK_M=BM, BLOCK_V=BV)
            gh[i:i + C] = torch.mm(g, weight)                               # NO recompute GEMM
            gw.add_(torch.mm(g.t(), hidden[i:i + C]))
        return (gh * sc.to(gh.dtype)), (gw * sc).to(weight.dtype), None, None, None


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
            # one fused launch: online-softmax -> lse+target, then OVERWRITE logits in place with
            # the (unscaled) grad-logits g = softmax - onehot. logits IS g after this call.
            _fwd_reduce_grad_kernel[(cl,)](logits, labels[i:i + C], lse[i:i + C], tgt[i:i + C],
                                           cl, V, logits.stride(0), logits.stride(1), 1.0,
                                           ignore_index, BLOCK_V=1024)
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


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None,
                               bwd_mode="fused"):
    """hidden (N,H), weight=lm_head.weight (V,H), labels (N,) -> mean CE loss.
    Never materializes (N,V); cuBLAS speed at bounded (chunk,V) memory.

    bwd_mode:
    - "fused" (DEFAULT): compute the full gradient in the FORWARD chunk loop (CE grad needs only
      logits+labels), discard each logit chunk, stash grad_h/grad_w. Backward is a scalar scale.
      3 GEMMs, NO recompute — this is the path that matches/beats Liger's fused-linear CE.
    - "recompute": forward keeps only lse, backward recomputes the logit GEMM (4 GEMMs total). The
      extra GEMM is a pure latency tax with no memory upside over "fused" — kept for A/B + history.
    - "int8": saves logits as int8, dequantizes in backward (2 GEMMs). ⚠️ MEASURED DOMINATED on T4
      (holds (N,V) int8 -> ~2x worse peak, approximate). Opt-in for research only.

    `bwd_logits_budget` (bytes) caps the (chunk,V) transient -> MEMORY dial. T4-tuned default 192MB
    (chunk ~1242 at V=81000). Chunk size barely moves latency (GEMM-bound); lower it for less peak."""
    if bwd_mode == "int8":
        fn = _CEInt8
    elif bwd_mode == "recompute":
        fn = _CECublasChunked
    else:                                       # "fused" — default, no backward recompute
        fn = _CEFusedFwdBwd
    return fn.apply(hidden, weight, labels, ignore_index, bwd_logits_budget)
