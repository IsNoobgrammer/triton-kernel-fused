"""Fused RMSNorm + router: phase 1 of the MoE megakernel.

One kernel does `h = rmsnorm(x) * w`, `scores = sigmoid(h @ Wr)`, top-k over `scores + bias`,
gather, and sum-normalise. Unfused this is two passes over a [T, H] tensor -- RMSNorm reads 67 MB
and writes 67 MB, then the router reads that 67 MB straight back. Measured at T=65536, H=512:
rmsnorm 0.796 ms + router 0.550 ms = 23.8% of the whole MoE block.

EXACT semantics of src/modeling/ffn/router.py, which are load-bearing:

    scores = sigmoid(gate_proj(h).float())     # logits AND sigmoid in fp32, not bf16
    sel    = scores + bias                     # bias steers SELECTION ONLY
    idx    = topk(sel, K).indices              # eager passes sorted=False -> order unspecified
    w      = scores.gather(-1, idx)            # gathered from the UNBIASED scores
    w      = w / (w.sum(-1) + 1e-20)

`bias` is the load-balancing term: requires_grad=False, mutated by `.add_()` outside the optimizer.
It is an INPUT here and is never updated by this kernel -- the update is a side effect that must
fire exactly once per step, so the backward reuses the saved indices rather than re-running the
router. Recomputing it would apply the balancing twice and silently double the balancing rate.

Unlike eager we emit indices in ASCENDING EXPERT ORDER. eager's `sorted=False` leaves the order
undefined, so there is nothing to be bit-identical to; a defined order is what makes the downstream
dispatch and combine deterministic. Parity is therefore checked as an expert->weight MAPPING, not
as positional arrays.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _norm_router_fwd(X, NW, RW, B, HN, IDX, WGT, RSTD,
                     T, H: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
                     EPS: tl.constexpr, BLOCK_T: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    m_r = rows < T
    h_off = tl.arange(0, H)
    e_off = tl.arange(0, E)

    # ---- RMSNorm. fp32 accumulate: the sum of squares over H=512 in bf16 loses ~3 bits and the
    # reciprocal sqrt amplifies it straight into every downstream logit.
    x = tl.load(X + rows[:, None] * H + h_off[None, :], mask=m_r[:, None], other=0.0).to(tl.float32)
    rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=1) / H + EPS)
    nw = tl.load(NW + h_off).to(tl.float32)
    hn = x * rstd[:, None] * nw[None, :]
    tl.store(HN + rows[:, None] * H + h_off[None, :], hn.to(HN.dtype.element_ty), mask=m_r[:, None])
    tl.store(RSTD + rows, rstd, mask=m_r)

    # ---- router logits. eager casts the projection output to fp32 BEFORE the sigmoid; matching
    # that matters because sigmoid saturates and a bf16 logit quantises the score it produces.
    rw = tl.load(RW + h_off[:, None] * E + e_off[None, :]).to(tl.float32)
    logits = tl.dot(hn.to(RW.dtype.element_ty), rw.to(RW.dtype.element_ty), out_dtype=tl.float32)
    scores = tl.sigmoid(logits)
    bias = tl.load(B + e_off).to(tl.float32)
    sel = scores + bias[None, :]

    # ---- top-k by repeated arg-max over E. E is small (64) so K linear passes beat a sort.
    # Ties break toward the LOWER expert index, deterministically -- eager's sorted=False makes no
    # promise here, so we pick a rule and keep it.
    cur = sel
    NEG = float("-inf")
    acc_sum = tl.zeros([BLOCK_T], dtype=tl.float32)
    chosen = tl.zeros([BLOCK_T, E], dtype=tl.int1)
    for _ in tl.static_range(K):
        best = tl.max(cur, axis=1)
        is_best = (cur == best[:, None]) & (~chosen)
        first = tl.argmax(is_best.to(tl.int32), axis=1)          # lowest index among ties
        hit = e_off[None, :] == first[:, None]
        chosen = chosen | hit
        cur = tl.where(hit, NEG, cur)
        acc_sum += tl.sum(tl.where(hit, scores, 0.0), axis=1)

    # ---- emit in ascending expert order, and sum-normalise with eager's exact epsilon.
    inv = 1.0 / (acc_sum + 1e-20)
    rank = tl.cumsum(chosen.to(tl.int32), axis=1) - 1              # 0..K-1 among selected
    for k in tl.static_range(K):
        slot = chosen & (rank == k)
        idx_k = tl.sum(tl.where(slot, e_off[None, :], 0), axis=1)
        w_k = tl.sum(tl.where(slot, scores, 0.0), axis=1) * inv
        tl.store(IDX + rows * K + k, idx_k.to(IDX.dtype.element_ty), mask=m_r)
        tl.store(WGT + rows * K + k, w_k.to(WGT.dtype.element_ty), mask=m_r)


def norm_router_forward(x, norm_weight, router_weight, bias, top_k, eps=1e-6,
                        out_dtype=torch.bfloat16, block_t=32):
    """x [T,H] -> (h_norm [T,H], idx [T,K] int32, weights [T,K] fp32, rstd [T] fp32)."""
    assert x.ndim == 2, f"expected [T,H], got {tuple(x.shape)}"
    T, H = x.shape
    E = router_weight.shape[1]
    assert router_weight.shape[0] == H, f"router weight {tuple(router_weight.shape)} vs H={H}"
    assert bias.numel() == E, f"bias {bias.numel()} vs E={E}"
    assert top_k <= E, f"top_k={top_k} > E={E}"
    assert H & (H - 1) == 0, f"H={H} must be a power of two for the block load"
    assert E & (E - 1) == 0, f"E={E} must be a power of two for the block load"

    hn = torch.empty((T, H), device=x.device, dtype=out_dtype)
    idx = torch.empty((T, top_k), device=x.device, dtype=torch.int32)
    wgt = torch.empty((T, top_k), device=x.device, dtype=torch.float32)
    rstd = torch.empty((T,), device=x.device, dtype=torch.float32)
    # grid keyed on T only; T is a grid dim so it must never be an autotune key (a stale cache
    # keyed on a grid dim cost a 4.1x eval stall once already)
    grid = (triton.cdiv(T, block_t),)
    _norm_router_fwd[grid](x, norm_weight, router_weight, bias, hn, idx, wgt, rstd,
                           T, H=H, E=E, K=top_k, EPS=eps, BLOCK_T=block_t)
    return hn, idx, wgt, rstd


def norm_router_reference(x, norm_weight, router_weight, bias, top_k, eps=1e-6, dtype=None):
    """Eager reference. Pass dtype=torch.float64 for the ground truth the kernel is graded against."""
    dt = dtype or x.dtype
    xf = x.to(dt)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    hn = xf * rstd * norm_weight.to(dt)
    scores = torch.sigmoid(hn @ router_weight.to(dt))
    sel = scores + bias.to(dt)
    _, idx = torch.topk(sel, top_k, dim=-1, sorted=False)
    idx, _ = torch.sort(idx, dim=-1)                    # ascending, to match the kernel's order
    w = scores.gather(-1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return hn, idx.to(torch.int32), w, rstd.squeeze(-1)
