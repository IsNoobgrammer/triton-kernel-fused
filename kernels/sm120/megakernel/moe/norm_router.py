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
def _norm_router_fwd(X, NW, RW, B, HN, IDX, WGT, RSTD, COUNTS,
                     T, H: tl.constexpr, E: tl.constexpr, K: tl.constexpr,
                     EPS: tl.constexpr, BLOCK_T: tl.constexpr, BLOCK_H: tl.constexpr,
                     WRITE_HN: tl.constexpr, ROUTER_FP32: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_T + tl.arange(0, BLOCK_T)
    m_r = rows < T
    e_off = tl.arange(0, E)

    # H is TILED, not loaded whole. In fp32 the router weight alone is H*E*4 = 131 KB at H=512,
    # E=64, over the 101 KB shared-memory limit, so holding [H,E] resident is impossible once the
    # projection runs in fp32. Two passes over x instead: one to accumulate the sum of squares,
    # one to build hn and the logits chunk by chunk. x re-reads hit L2, which is far cheaper than
    # dropping the projection back to bf16.
    ss = tl.zeros([BLOCK_T], dtype=tl.float32)
    for h0 in tl.range(0, H, BLOCK_H):
        hc = h0 + tl.arange(0, BLOCK_H)
        xc = tl.load(X + rows[:, None] * H + hc[None, :],
                     mask=m_r[:, None] & (hc[None, :] < H), other=0.0).to(tl.float32)
        ss += tl.sum(xc * xc, axis=1)
    # fp32 accumulate: the sum of squares over H in bf16 loses ~3 bits and the reciprocal sqrt
    # amplifies it straight into every downstream logit.
    rstd = 1.0 / tl.sqrt(ss / H + EPS)
    tl.store(RSTD + rows, rstd, mask=m_r)

    # ---- router logits, accumulated over H tiles. eager casts the projection output to fp32
    # BEFORE the sigmoid; matching that matters because sigmoid saturates and a bf16 logit
    # quantises the score it produces.
    #
    # ROUTER_FP32 does the projection in true IEEE fp32 rather than on bf16 tensor cores. This is
    # THE lever on selection flips: top-k over `scores + bias` is decided by near-ties, and a bf16
    # mantissa on the logits is what makes two nearly-equal experts swap order versus exact
    # arithmetic. It is nearly free -- the projection is [T,512]x[512,64], 4.3 GFLOP at T=65536,
    # against the ~2 TFLOP of the expert GEMMs it feeds.
    logits = tl.zeros([BLOCK_T, E], dtype=tl.float32)
    for h0 in tl.range(0, H, BLOCK_H):
        hc = h0 + tl.arange(0, BLOCK_H)
        m_h = hc < H
        xc = tl.load(X + rows[:, None] * H + hc[None, :],
                     mask=m_r[:, None] & m_h[None, :], other=0.0).to(tl.float32)
        nwc = tl.load(NW + hc, mask=m_h, other=0.0).to(tl.float32)
        hnc = xc * rstd[:, None] * nwc[None, :]
        # h_norm is NOT written in the megakernel path: the expert phase gathers raw x rows
        # permuted by expert and re-applies rstd*nw inline. Writing it would cost a 67 MB store
        # here and a 67 MB load there -- the exact round-trip this fusion removes. WRITE_HN is
        # for grading against eager only.
        if WRITE_HN:
            tl.store(HN + rows[:, None] * H + hc[None, :], hnc.to(HN.dtype.element_ty),
                     mask=m_r[:, None] & m_h[None, :])
        rwc = tl.load(RW + hc[:, None] * E + e_off[None, :], mask=m_h[:, None], other=0.0)
        if ROUTER_FP32:
            logits += tl.dot(hnc, rwc.to(tl.float32), out_dtype=tl.float32,
                             input_precision="ieee")
        else:
            logits += tl.dot(hnc.to(RW.dtype.element_ty), rwc.to(RW.dtype.element_ty),
                             out_dtype=tl.float32)
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

    # ---- per-expert counts for the tile map. Accumulate a BLOCK-LOCAL histogram first, then do
    # E atomics per block instead of K per token: at T=65536, K=6 that is 131k atomics on 64
    # counters rather than 393k, on the same 64 contended addresses.
    hist = tl.sum(tl.where(m_r[:, None], chosen.to(tl.int32), 0), axis=0)
    tl.atomic_add(COUNTS + e_off, hist)


def norm_router_forward(x, norm_weight, router_weight, bias, top_k, eps=1e-6,
                        out_dtype=torch.bfloat16, block_t=32, block_h=128, write_hn=False,
                        router_fp32=True):
    """x [T,H] -> (h_norm|None, idx [T,K] int32, weights [T,K] fp32, rstd [T] fp32, counts [E] int32).

    `write_hn=False` by default: the expert phase recomputes the norm from x while gathering, so
    materializing h_norm is pure wasted bandwidth. Set True only to grade against eager.
    """
    assert x.ndim == 2, f"expected [T,H], got {tuple(x.shape)}"
    T, H = x.shape
    E = router_weight.shape[1]
    assert router_weight.shape[0] == H, f"router weight {tuple(router_weight.shape)} vs H={H}"
    assert bias.numel() == E, f"bias {bias.numel()} vs E={E}"
    assert top_k <= E, f"top_k={top_k} > E={E}"
    assert H & (H - 1) == 0, f"H={H} must be a power of two for the block load"
    assert E & (E - 1) == 0, f"E={E} must be a power of two for the block load"

    hn = torch.empty((T, H), device=x.device, dtype=out_dtype) if write_hn else x  # dummy ptr
    idx = torch.empty((T, top_k), device=x.device, dtype=torch.int32)
    wgt = torch.empty((T, top_k), device=x.device, dtype=torch.float32)
    rstd = torch.empty((T,), device=x.device, dtype=torch.float32)
    # zeroed every call: atomic_add accumulates, so a reused buffer would keep counting. Output
    # tensors that are atomically accumulated into MUST be reset -- that has bitten this repo.
    counts = torch.zeros((E,), device=x.device, dtype=torch.int32)
    # grid keyed on T only; T is a grid dim so it must never be an autotune key (a stale cache
    # keyed on a grid dim cost a 4.1x eval stall once already)
    grid = (triton.cdiv(T, block_t),)
    _norm_router_fwd[grid](x, norm_weight, router_weight, bias, hn, idx, wgt, rstd, counts,
                           T, H=H, E=E, K=top_k, EPS=eps, BLOCK_T=block_t, BLOCK_H=block_h,
                           WRITE_HN=write_hn, ROUTER_FP32=router_fp32)
    return (hn if write_hn else None), idx, wgt, rstd, counts


def norm_router_reference(x, norm_weight, router_weight, bias, top_k, eps=1e-6, dtype=None):
    """Eager reference, matching src/modeling/norm.py::BiBoRMSNorm and ffn/router.py EXACTLY.

    The ordering is load-bearing and easy to get wrong: BiBoRMSNorm upcasts to fp32, takes the
    variance and rsqrt in fp32, then casts back to the INPUT dtype and only then multiplies by the
    weight. Writing it as one fp32 expression makes the reference more accurate than the model and
    turns this grader into a comparison against a strawman.

    Pass dtype=torch.float64 for the ground truth.
    """
    dt = dtype or x.dtype
    xf = x.to(torch.float32) if dt != torch.float64 else x.to(torch.float64)
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    hn = (xf * rstd).to(dt) * norm_weight.to(dt)      # cast BEFORE the weight multiply
    scores = torch.sigmoid((hn @ router_weight.to(dt)).float() if dt != torch.float64
                           else hn @ router_weight.to(dt))
    sel = scores + bias.to(dt)
    _, idx = torch.topk(sel, top_k, dim=-1, sorted=False)
    idx, _ = torch.sort(idx, dim=-1)                    # ascending, to match the kernel's order
    w = scores.gather(-1, idx)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return hn, idx.to(torch.int32), w, rstd.squeeze(-1)
