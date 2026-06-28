"""Fused PolyGLU MoE experts — the hard one. Per-expert + grouped paths + naive-eager reference.

PolyGLU: experts are GLU MLPs with *heterogeneous* activations — each expert carries an
activation code (0=SiLU, 1=ReLU², 2=Tanh), e.g. groups of three [SiLU, ReLU², Tanh]. Pass an
`act_codes` (E,) int32 tensor alongside the expert weights.

Why MoE can't be ONE fused kernel (unlike SwiGLU / XSA / CE)
------------------------------------------------------------
SwiGLU/XSA/CE are *dense*: every row does identical work, so one kernel with a fixed grid
covers them. An MoE is *data-dependent* — the router sends each token to a runtime-chosen
subset of experts, so the work is a ragged collection of per-expert GEMMs whose sizes aren't
known until the router fires. That breaks single-kernel fusion at three points, "from weights,
to dispatch, to the final summed tensor":

  1. DISPATCH (gather): tokens for expert e are scattered across the batch; you must gather them
     into a contiguous block before a GEMM can touch them.
  2. RAGGED GEMM: expert e gets `count[e]` tokens — runtime-shaped, different every step. A plain
     batched GEMM needs equal sizes; here each "batch" is a different M. You loop (one GEMM per
     expert) or block-schedule a grouped GEMM over sorted tokens.
  3. COMBINE (scatter): each token went to top-k experts, so the output is a weighted sum of k
     expert outputs scattered back to its row — an index-add reduction, not a plain write.

So a real MoE is a *pipeline* of fused stages wired by a sort, not one kernel. The router itself
stays in your model; pass its top-k indices/weights in. Two expert-pipeline drop-ins:

  moe_per_expert(...) — sort by expert, then per expert: cuBLAS gate_up GEMM -> fused PolyGLU
        activation (Triton) -> cuBLAS down GEMM -> weighted scatter. Pure composition,
        autograd-correct by construction. Best at LOW token counts (loop overhead small).
  moe_grouped(...)    — ONE block-scheduled grouped-GEMM over all sorted tokens (Triton tl.dot)
        + matched grouped-GEMM backward. Best at HIGH token counts. ⚠ tl.dot: re-bench per arch.
  moe(...)            — auto: grouped at >= GROUPED_MIN_TOKENS rows, else per-expert.

Why naive eager is so slow
---------------------------
`moe_eager` is the hand-written version: loop experts, boolean-mask each (`idx == e`), gather,
two `F.linear`s, activation, scatter. Slow because (a) the per-expert boolean-mask/index forces
a GPU→CPU **sync every iteration** (the launch queue drains E times per layer), (b) the GLU
activation is unfused elementwise kernels + an intermediate write, (c) zero GEMM batching — E
tiny GEMMs each under-utilizing the device. The fused paths kill all three: one sort instead of
E masks, a fused-activation Triton kernel, and (grouped) a single batched GEMM.

Weights: gate_up_proj (E, 2*I, H), down_proj (E, H, I), act_codes (E,) int32.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["moe", "moe_per_expert", "moe_grouped", "moe_grouped_cublas", "moe_eager",
           "BatchedGLU", "GROUPED_MIN_TOKENS"]

GROUPED_MIN_TOKENS = 4096
SCHED_BLOCK_M = 64


# ───────────────────────── PolyGLU activation (per-row act code) ─────────────────────────
@triton.jit
def _glu_fwd_kernel(GateUp_ptr, Act_ptr, Out_ptr, M, I, s_gu_m, s_gu_i, s_o_m, s_o_i,
                    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0); pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M; mask = mask_m[:, None] & (offs_i < I)[None, :]
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + offs_m, mask=mask_m, other=0)[:, None]
    silu = gate * (1.0 / (1.0 + tl.exp(-gate)))
    relu = tl.maximum(gate, 0.0)
    tnh = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * gate))) - 1.0
    act = tl.where(at == 0, silu, tl.where(at == 1, relu * relu, tnh))
    tl.store(Out_ptr + offs_m[:, None] * s_o_m + offs_i[None, :] * s_o_i,
             (act * up).to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _glu_bwd_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, GradGateUp_ptr, M, I,
                    s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i,
                    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0); pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M; mask = mask_m[:, None] & (offs_i < I)[None, :]
    go = tl.load(GradOut_ptr + offs_m[:, None] * s_go_m + offs_i[None, :] * s_go_i, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + offs_m, mask=mask_m, other=0)[:, None]
    sig = 1.0 / (1.0 + tl.exp(-gate)); silu = gate * sig; dsilu = sig * (1.0 + gate * (1.0 - sig))
    relu = tl.maximum(gate, 0.0); relu2 = relu * relu; drelu2 = 2.0 * relu
    tnh = 2.0 * (1.0 / (1.0 + tl.exp(-2.0 * gate))) - 1.0; dtanh = 1.0 - tnh * tnh
    act = tl.where(at == 0, silu, tl.where(at == 1, relu2, tnh))
    dact = tl.where(at == 0, dsilu, tl.where(at == 1, drelu2, dtanh))
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + offs_i[None, :] * s_ggu_i, go * up * dact, mask=mask)
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + (I + offs_i)[None, :] * s_ggu_i, go * act, mask=mask)


def _glu_fwd(gate_up, row_act):
    M, twoI = gate_up.shape; I = twoI // 2
    out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_fwd_kernel[grid](gate_up, row_act, out, M, I, gate_up.stride(0), gate_up.stride(1),
                          out.stride(0), out.stride(1), BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    return out


def _glu_bwd(grad_out, gate_up, row_act):
    M, twoI = gate_up.shape; I = twoI // 2
    ggu = torch.empty_like(gate_up)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_bwd_kernel[grid](grad_out, gate_up, row_act, ggu, M, I, grad_out.stride(0), grad_out.stride(1),
                          gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    return ggu


class BatchedGLU(torch.autograd.Function):
    """PolyGLU activation: out = act_{row}(gate) * up, with a per-row activation code."""
    @staticmethod
    def forward(ctx, gate_up, row_act):
        ctx.save_for_backward(gate_up, row_act)
        return _glu_fwd(gate_up, row_act)

    @staticmethod
    def backward(ctx, grad_out):
        gate_up, row_act = ctx.saved_tensors
        return _glu_bwd(grad_out.contiguous(), gate_up, row_act), None


# ───────────────────────── grouped GEMM kernels ─────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ], key=["K", "N"])
@triton.jit
def _grouped_mm_kernel(X_ptr, W_ptr, Out_ptr, TileExpert_ptr, TileStart_ptr, ExpertEnd_ptr,
                       K, N, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                       stride_om, stride_on,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0); pid_n = tl.program_id(1)
    e = tl.load(TileExpert_ptr + pid_t); m0 = tl.load(TileStart_ptr + pid_t); m_end = tl.load(ExpertEnd_ptr + e)
    offs_m = m0 + tl.arange(0, BLOCK_M); offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m_end; mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    W_e = W_ptr + e * stride_we
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K); mask_k = offs_k < K
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W_e + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": 64}, num_warps=4, num_stages=2),
    ], key=["N", "K"])
@triton.jit
def _grouped_wgrad_kernel(A_ptr, B_ptr, GW_ptr, ExpertStart_ptr, ExpertEnd_ptr, N, K,
                          stride_am, stride_an, stride_bm, stride_bk, stride_ge, stride_gn, stride_gk,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_e = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    m_start = tl.load(ExpertStart_ptr + pid_e); m_end = tl.load(ExpertEnd_ptr + pid_e)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N; mask_k = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    m = m_start
    while m < m_end:
        offs_m = m + tl.arange(0, BLOCK_M); mask_m = offs_m < m_end
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(tl.trans(a), b)
        m += BLOCK_M
    tl.store(GW_ptr + pid_e * stride_ge + offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk,
             acc.to(GW_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_k[None, :])


def _build_schedule(counts, bounds, E, device, block_m=SCHED_BLOCK_M):
    tile_expert, tile_start = [], []
    for e in range(E):
        for ti in range((counts[e] + block_m - 1) // block_m):
            tile_expert.append(e); tile_start.append(bounds[e] + ti * block_m)
    if not tile_expert:
        return None, None
    return (torch.tensor(tile_expert, dtype=torch.int32, device=device),
            torch.tensor(tile_start, dtype=torch.int32, device=device))


def _grouped_mm(x_sorted, W, te, ts, e_end, N, trans_w=False):
    M, K = x_sorted.shape
    out = torch.empty(M, N, device=x_sorted.device, dtype=x_sorted.dtype)
    s_we, s_wn, s_wk = (W.stride(0), W.stride(2), W.stride(1)) if trans_w else (W.stride(0), W.stride(1), W.stride(2))
    grid = lambda meta: (te.numel(), triton.cdiv(N, meta["BLOCK_N"]))
    _grouped_mm_kernel[grid](x_sorted, W, out, te, ts, e_end, K, N, x_sorted.stride(0), x_sorted.stride(1),
                             s_we, s_wn, s_wk, out.stride(0), out.stride(1))
    return out


def _grouped_wgrad(A, B, e_start, e_end, E, N, K):
    gW = torch.zeros(E, N, K, device=A.device, dtype=A.dtype)
    grid = lambda meta: (E, triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))
    _grouped_wgrad_kernel[grid](A, B, gW, e_start, e_end, N, K, A.stride(0), A.stride(1),
                                B.stride(0), B.stride(1), gW.stride(0), gW.stride(1), gW.stride(2))
    return gW


def _sort_by_expert(idx, wt, E):
    ntok, top_k = idx.shape
    flat_t = torch.arange(ntok, device=idx.device).unsqueeze(1).expand_as(idx).flatten()
    sorted_e, order = idx.flatten().sort()
    counts = torch.bincount(sorted_e, minlength=E).tolist()
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    return flat_t[order], wt.flatten()[order], order, counts, bounds


# ───────────────────────── grouped path ─────────────────────────
class _GroupedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, wt, gate_up_proj, down_proj, act_codes):
        ntok, H = x.shape
        top_k = idx.shape[1]; E = gate_up_proj.shape[0]; I = gate_up_proj.shape[1] // 2
        dev = x.device
        st, sw, order, counts, bounds = _sort_by_expert(idx, wt, E)
        e_start = torch.tensor(bounds[:E], dtype=torch.int32, device=dev)
        e_end = torch.tensor(bounds[1:], dtype=torch.int32, device=dev)
        te, ts = _build_schedule(counts, bounds, E, dev)
        counts_t = torch.tensor(counts, device=dev)
        row_act = torch.repeat_interleave(act_codes, counts_t).to(torch.int32)
        x_s = x[st].contiguous()
        gate_up = _grouped_mm(x_s, gate_up_proj, te, ts, e_end, 2 * I)
        inter = _glu_fwd(gate_up, row_act)
        eo = _grouped_mm(inter, down_proj, te, ts, e_end, H)
        out = torch.zeros(ntok, H, device=dev, dtype=torch.float32)   # fp32 accumulate (MiMo)
        out.index_add_(0, st, (eo * sw.unsqueeze(-1)).float())
        out = out.to(x.dtype)
        ctx.save_for_backward(x_s, gate_up, inter, eo, st, sw, order, te, ts, e_start, e_end,
                              row_act, gate_up_proj, down_proj)
        ctx.shapes = (ntok, H, I, top_k, E)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x_s, gate_up, inter, eo, st, sw, order, te, ts, e_start, e_end, row_act,
         gate_up_proj, down_proj) = ctx.saved_tensors
        ntok, H, I, top_k, E = ctx.shapes
        go_s = grad_out[st].contiguous()
        grad_w_s = (go_s.float() * eo.float()).sum(-1).to(grad_out.dtype)
        grad_eo = go_s * sw.unsqueeze(-1)
        grad_inter = _grouped_mm(grad_eo, down_proj, te, ts, e_end, I, trans_w=True)
        grad_down_proj = _grouped_wgrad(grad_eo, inter, e_start, e_end, E, H, I)
        grad_gate_up = _glu_bwd(grad_inter, gate_up, row_act)
        grad_x_s = _grouped_mm(grad_gate_up, gate_up_proj, te, ts, e_end, H, trans_w=True)
        grad_gate_up_proj = _grouped_wgrad(grad_gate_up, x_s, e_start, e_end, E, 2 * I, H)
        grad_x = torch.zeros(ntok, H, device=grad_out.device, dtype=grad_out.dtype)
        grad_x.index_add_(0, st, grad_x_s)
        grad_wt = torch.zeros(ntok * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = grad_w_s
        return grad_x, None, grad_wt.view(ntok, top_k), grad_gate_up_proj, grad_down_proj, None


def moe_grouped(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Block-scheduled grouped-GEMM PolyGLU MoE. hidden (N,H), indices/weights (N,k),
    gate_up_proj (E,2I,H), down_proj (E,H,I), act_codes (E,) int32 -> (N,H)."""
    return _GroupedMoE.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)


# ── candidate: grouped GEMM via torch._grouped_mm (cuBLAS, GPU-resident, no host sync) ──
# This is the Turing candidate: a cuBLAS grouped GEMM instead of the slow tl.dot one, with the
# dispatch built entirely on-GPU (cumsum offsets, no .tolist()/Python schedule loop). Composition
# of autograd-native ops, so no custom backward — IF torch._grouped_mm is differentiable. Requires
# torch with _grouped_mm (>= ~2.5/2.8); raises otherwise (the bench catches it and reports FAILED).
def moe_grouped_cublas(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    # NOTE (measured T4, round 1): torch._grouped_mm is **bf16/fp8-only** and needs bf16 tensor
    # cores → **sm_80+ (Ampere/Hopper) only**. On Turing (T4, sm_75) it cannot run; we hard-skip.
    # Where it DOES run, the grouped GEMM is cuBLAS + the whole dispatch is GPU-resident (cumsum
    # offsets, no .tolist()/Python loop). GEMMs go through bf16 (cast in/out); grad flows through
    # _grouped_mm (autograd-native). UNTESTED end-to-end — no sm_80+ box in the loop yet.
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm unavailable in this torch build")
    if torch.cuda.get_device_capability(hidden.device)[0] < 8:
        raise RuntimeError("torch._grouped_mm needs bf16 tensor cores (sm_80+); skipped on this GPU")
    N, H = hidden.shape
    E = gate_up_proj.shape[0]
    flat_t = torch.arange(N, device=hidden.device).unsqueeze(1).expand_as(top_k_indices).flatten()
    sorted_e, order = top_k_indices.flatten().sort()
    st = flat_t[order]
    sw = top_k_weights.flatten()[order]
    counts = torch.bincount(sorted_e, minlength=E)                        # GPU
    offs = counts.cumsum(0).to(torch.int32)                               # GPU end-exclusive offsets
    row_act = torch.repeat_interleave(act_codes, counts).to(torch.int32)  # GPU
    x_s = hidden[st].contiguous()                                         # (M,H)
    bf = torch.bfloat16
    gate_up = torch._grouped_mm(x_s.to(bf), gate_up_proj.transpose(-2, -1).to(bf), offs=offs).to(hidden.dtype)
    inter = BatchedGLU.apply(gate_up, row_act)                            # (M,I) in model dtype
    eo = torch._grouped_mm(inter.to(bf), down_proj.transpose(-2, -1).to(bf), offs=offs).to(hidden.dtype)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)   # fp32 accumulate (MiMo)
    out.index_add_(0, st, (eo * sw.unsqueeze(-1)).float())
    return out.to(hidden.dtype)


# ───────────────────────── per-expert path (composition, autograd-native) ─────────────────────────
def moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Sorted dispatch + cuBLAS GEMMs + fused PolyGLU activation + weighted scatter.
    No custom backward — composition of autograd-correct ops. Wins at low token counts."""
    N, H = hidden.shape
    E = gate_up_proj.shape[0]
    st, sw, _order, _counts, bounds = _sort_by_expert(top_k_indices, top_k_weights, E)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)   # fp32 accumulate (MiMo)
    for e in range(E):
        s, en = bounds[e], bounds[e + 1]
        if en == s:
            continue
        tok = st[s:en]; w = sw[s:en]
        gate_up = hidden[tok] @ gate_up_proj[e].t()
        row_act = act_codes[e:e + 1].expand(en - s).contiguous()   # contiguous: kernel indexes Act_ptr+offs_m (stride 1)
        inter = BatchedGLU.apply(gate_up, row_act)
        eo = inter @ down_proj[e].t()
        out = out.index_add_(0, tok, (eo * w.unsqueeze(-1)).float())  # in-place: no full-accumulator clone per expert
    return out.to(hidden.dtype)


def moe(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Auto: grouped at >= GROUPED_MIN_TOKENS rows (N*top_k) on Ampere+ (sm_80+), else per-expert.
    The grouped path's tl.dot GEMMs are catastrophic on Turing (T4, sm_75) — measured ~0.1x vs
    compiled eager — so it is NEVER chosen on sm_<80. per-expert (cuBLAS) wins there."""
    cap_major = torch.cuda.get_device_capability(hidden.device)[0]
    if top_k_indices.numel() >= GROUPED_MIN_TOKENS and cap_major >= 8:
        return moe_grouped(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)
    return moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)


# ───────────────────────── naive eager reference (the slow baseline) ─────────────────────────
def _act_eager(gate, code):
    if code == 0:
        return F.silu(gate)
    if code == 1:
        return F.relu(gate) ** 2
    return torch.tanh(gate)


def moe_eager(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Hand-written MoE: per-expert boolean mask (GPU sync each iter), unfused activation,
    E tiny GEMMs. Correct, and deliberately the slow baseline the fused paths beat."""
    N, H = hidden.shape
    E, twoI, _ = gate_up_proj.shape
    I = twoI // 2
    codes = act_codes.tolist()
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)   # fp32 accumulate (MiMo)
    for e in range(E):
        rows = (top_k_indices == e).any(-1)
        if not bool(rows.any()):
            continue
        w = (top_k_weights * (top_k_indices == e)).sum(-1)[rows]
        gate_up = hidden[rows] @ gate_up_proj[e].t()
        inter = _act_eager(gate_up[:, :I], codes[e]) * gate_up[:, I:]
        out[rows] += ((inter @ down_proj[e].t()) * w.unsqueeze(-1)).float()
    return out.to(hidden.dtype)
