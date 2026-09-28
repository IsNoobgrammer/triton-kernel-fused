"""Fused linear cross-entropy, FACTORED softmax gradient (sm120). Forward and both gradients in one pass.

The chunked kernel (kernels/sm75/cross_entropy.py) spends 24% of its time on a pure-memory grad pass:
it rewrites the (C, V) logits into dL = softmax - onehot (2 GB r+w per 1 GB chunk) before the two
gradient GEMMs. Here the logits GEMM epilogue stores E = exp(L - c) instead of L (bf16 has fp32's
exponent range), with c a per-row constant (0 unless the row is out of window). Then, per row r:

    dL    = s_r * E - w_r * onehot,          s_r = exp(c_r - lse_r) * w_r
    gh    = s * (E @ W) - w * W[label]                     cuBLAS mm + a (C, H) row epilogue
    gw   += E^T @ (s * h) - scatter(label, w * h)          cuBLAS addmm on a pre-scaled (C, H) h

w_r is the per-row loss weight (valid / n_valid for plain CE; head_weight_k / n_valid_k for MTP heads
concatenated into one call). No grad pass, all three big GEMMs stay cuBLAS, no atomics anywhere.

Window: c = 0 is safe while the row max logit m is in [_LO, _HI]. Rows outside it get a second pass
of the same logits kernel over just those rows with c = m exactly; they are found on the device
(stable argsort of the flag), so there is no host sync, and blocks with no such row exit at once. Without
grad (val) the stats kernel keeps the classic per-tile (max, sum) and never stores E.

    loss = fused_linear_cross_entropy(hidden, weight, labels)                       # drop-in
    total, per_head = fused_linear_cross_entropy_heads([h1, h2], weight, [y1, y2], [1.0, 0.3])
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_linear_cross_entropy", "fused_linear_cross_entropy_heads"]

_BUDGET = 1 << 30                       # bytes of E per chunk
# c = 0 window on the row max logit m. Upper: V * e^m and E @ W stay < fp32 max up to V = 256k
# (m + ln V < 77). Lower: E's top entries stay normal, and s = e^-lse * w stays representable after _HS.
_LO, _HI = -30.0, 64.0
_HS = 2.0 ** 32                         # exact rescale of s*h into bf16's normal range


def _lcfg(K):
    """(BM, BN, BK, GROUP, warps, stages), swept per hidden size on the RTX PRO 6000 at V=81920
    (bench_ce_lcfg.py). Beats cuBLAS + a separate exp/stats pass at every K up to 16384."""
    return (128, 128, 32, 8, 4, 4) if K < 2048 else (128, 256, 64, 8, 8, 3)


@triton.jit
def _logits_kernel(X, W, E, PA, PB, TGT, LAB, ORDER, CROW, M, V, NT, K,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                   STORE_E: tl.constexpr, FIX: tl.constexpr, EVEN_K: tl.constexpr, EVEN_V: tl.constexpr):
    # FIX: second pass over the out-of-window rows only (sorted first by ORDER), E = exp(L - c_row).
    # Blocks with no such row exit at once, so the pass is ~free when every row is in window.
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    width = GROUP * NT
    g = pid // width
    first = g * GROUP
    gs = tl.minimum(nm - first, GROUP)
    pm = first + (pid % width) % gs
    pn = (pid % width) // gs
    j = pm * BM + tl.arange(0, BM)
    if FIX:
        rm = tl.load(ORDER + j, mask=j < M, other=0)
        c = tl.load(CROW + rm, mask=j < M, other=0.0)
        mm = (j < M) & (c != 0.0)
        if tl.max(mm.to(tl.int32), 0) == 0:
            return
    else:
        rm = j
        mm = j < M
    rn = pn * BN + tl.arange(0, BN)
    mn = rn < V
    rk = tl.arange(0, BK)
    xp = X + rm[:, None].to(tl.int64) * K + rk[None, :]
    wp = W + rn[:, None].to(tl.int64) * K + rk[None, :]
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        if EVEN_K:
            x = tl.load(xp, mask=mm[:, None], other=0.0)
            if EVEN_V:
                w = tl.load(wp)
            else:
                w = tl.load(wp, mask=mn[:, None], other=0.0)
        else:
            mk = (k0 + rk) < K
            x = tl.load(xp, mask=mm[:, None] & mk[None, :], other=0.0)
            w = tl.load(wp, mask=mn[:, None] & mk[None, :], other=0.0)
        acc = tl.dot(x, tl.trans(w), acc)
        xp += BK
        wp += BK
    xf = acc.to(tl.bfloat16).to(tl.float32)        # the bf16 logits every other path sees
    if not EVEN_V:
        xf = tl.where(mn[None, :], xf, -float("inf"))
    mx = tl.max(xf, axis=1)
    if STORE_E:
        if FIX:
            e = tl.exp(xf - c[:, None])
        else:
            e = tl.exp(xf)                           # c = 0
        tl.store(E + rm[:, None].to(tl.int64) * V + rn[None, :], e.to(tl.bfloat16),
                 mask=mm[:, None] & mn[None, :])
        tl.store(PA + rm * NT + pn, tl.sum(e, axis=1), mask=mm)
    else:
        tl.store(PA + rm * NT + pn, tl.sum(tl.exp(xf - mx[:, None]), axis=1), mask=mm)
    if not FIX:
        tl.store(PB + rm * NT + pn, mx, mask=mm)
        lab = tl.load(LAB + rm, mask=mm, other=-1)
        hit = rn[None, :] == lab[:, None]
        tl.store(TGT + rm, tl.sum(tl.where(hit, xf, 0.0), axis=1),
                 mask=mm & (lab >= pn * BN) & (lab < pn * BN + BN))


@triton.jit
def _combine_kernel(PA, PB, LSE, MX, CROW, NT, BT: tl.constexpr, STORE_E: tl.constexpr, FIX: tl.constexpr):
    r = tl.program_id(0)
    t = tl.arange(0, BT)
    mk = t < NT
    a = tl.load(PA + r * NT + t, mask=mk, other=0.0)
    if STORE_E:
        if FIX:
            tl.store(LSE + r, tl.load(CROW + r) + tl.log(tl.sum(a, 0)))
        else:
            tl.store(LSE + r, tl.log(tl.sum(a, 0)))
    else:
        b = tl.load(PB + r * NT + t, mask=mk, other=-float("inf"))
        mx = tl.max(b, 0)
        tl.store(LSE + r, mx + tl.log(tl.sum(a * tl.exp(b - mx), 0)))
    if not FIX:
        tl.store(MX + r, tl.max(tl.load(PB + r * NT + t, mask=mk, other=-float("inf")), 0))


@triton.jit
def _scale_rows_kernel(X, S, OUT, H, HS, BH: tl.constexpr):
    # OUT = bf16(X * (S[r] * HS)), one read + one write of the (C, H) chunk
    r = tl.program_id(0)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    mc = cols < H
    sv = tl.load(S + r) * HS
    x = tl.load(X + r.to(tl.int64) * H + cols, mask=mc, other=0.0).to(tl.float32)
    tl.store(OUT + r.to(tl.int64) * H + cols, (x * sv).to(OUT.dtype.element_ty), mask=mc)


@triton.jit
def _gh_epi_kernel(G, S, W, SAFE, WR, OUT, H, BH: tl.constexpr):
    # gh = bf16(G * s - w * W[label]); G = E @ W in fp32 from cuBLAS
    r = tl.program_id(0)
    cols = tl.program_id(1) * BH + tl.arange(0, BH)
    mc = cols < H
    g = tl.load(G + r.to(tl.int64) * H + cols, mask=mc, other=0.0) * tl.load(S + r)
    wrow = tl.load(W + tl.load(SAFE + r).to(tl.int64) * H + cols, mask=mc, other=0.0).to(tl.float32)
    tl.store(OUT + r.to(tl.int64) * H + cols, (g - wrow * tl.load(WR + r)).to(OUT.dtype.element_ty), mask=mc)


@triton.jit
def _scatter_kernel(GW, LABS, ROWS, X, WR, n, V, H, BH: tl.constexpr):
    # gw[label] -= sum of its rows' w*h (h read in bf16, w per row). Labels sorted (stable); the
    # first row of each run sums the run in order. Deterministic, no atomics. Ignored rows carry the sentinel label V.
    i = tl.program_id(0)
    hb = tl.program_id(1)
    lab = tl.load(LABS + i)
    prev = tl.load(LABS + i - 1, mask=i > 0, other=-1)
    if (lab < V) & (lab != prev):
        cols = hb * BH + tl.arange(0, BH)
        mc = cols < H
        acc = tl.zeros((BH,), tl.float32)
        j = i
        cur = lab
        while cur == lab:
            row = tl.load(ROWS + j)
            acc += tl.load(X + row.to(tl.int64) * H + cols, mask=mc, other=0.0).to(tl.float32) * tl.load(WR + row)
            j += 1
            cur = tl.load(LABS + j, mask=j < n, other=-1)
        ptr = GW + lab.to(tl.int64) * H + cols
        tl.store(ptr, tl.load(ptr, mask=mc) - acc, mask=mc)


def _chunk(N, V, budget):
    rows = max(512, min(N, (budget or _BUDGET) // (V * 2), (2 ** 31 - 1) // V))
    n = -(-N // rows)
    return -(-N // n)                                   # balanced: no tiny tail chunk


def _stats(hc, W, lab, store_e, E=None):
    """logits pass for one chunk -> (lse, tgt, crow). With store_e, E = exp(L - crow) is filled and
    out-of-window rows got a second, exact pass (crow = their max logit; 0 elsewhere)."""
    M, K = hc.shape
    V = W.shape[0]
    BM, BN, BK, G, nw, ns = _lcfg(K)
    NT = triton.cdiv(V, BN)
    dev = hc.device
    PA = torch.empty(M, NT, device=dev, dtype=torch.float32)
    PB = torch.empty(M, NT, device=dev, dtype=torch.float32)
    tgt = torch.zeros(M, device=dev, dtype=torch.float32)
    lse = torch.empty(M, device=dev, dtype=torch.float32)
    mx = torch.empty(M, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * NT,)
    ev = dict(EVEN_K=K % BK == 0, EVEN_V=V % BN == 0, num_warps=nw, num_stages=ns)
    _logits_kernel[grid](hc, W, E if store_e else PA, PA, PB, tgt, lab, lab, PA, M, V, NT, K,
                         BM, BN, BK, G, store_e, False, **ev)
    _combine_kernel[(M,)](PA, PB, lse, mx, PA, NT, triton.next_power_of_2(NT), store_e, False, num_warps=4)
    if not store_e:
        return lse, tgt, None
    crow = torch.where((mx < _LO) | (mx > _HI), mx, torch.zeros_like(mx))
    order = torch.argsort((crow != 0).to(torch.int8), descending=True, stable=True)
    _logits_kernel[grid](hc, W, E, PA, PB, tgt, lab, order, crow, M, V, NT, K, BM, BN, BK, G, True, True, **ev)
    _combine_kernel[(M,)](PA, PB, lse, mx, crow, NT, triton.next_power_of_2(NT), True, True, num_warps=4)
    return lse, tgt, crow


def _nll_nograd(hidden, weight, labels, budget):
    N = hidden.shape[0]
    V = weight.shape[0]
    NT = triton.cdiv(V, _lcfg(hidden.shape[1])[1])
    rows = max(512, min(N, (budget or _BUDGET) // (NT * 8)))       # only (rows, NT) partials live
    lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
    tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
    for i in range(0, N, rows):
        lse[i:i + rows], tgt[i:i + rows], _ = _stats(hidden[i:i + rows], weight, labels[i:i + rows], False)
    return lse - tgt


class _FactoredCE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, rw, ignore_index, budget):
        wdt = weight.dtype
        if torch.is_autocast_enabled("cuda"):
            dt = torch.get_autocast_dtype("cuda")
            hidden, weight = hidden.to(dt), weight.to(dt)
        hidden, weight = hidden.contiguous(), weight.contiguous()
        N, H = hidden.shape
        V = weight.shape[0]
        C = _chunk(N, V, budget)
        dev = hidden.device
        nll = torch.empty(N, device=dev, dtype=torch.float32)
        gh = torch.empty(N, H, device=dev, dtype=hidden.dtype)
        gw = torch.zeros(V, H, device=dev, dtype=torch.float32)
        E = torch.empty(C, V, device=dev, dtype=torch.bfloat16)
        for i in range(0, N, C):
            hc, lab, w = hidden[i:i + C], labels[i:i + C], rw[i:i + C]
            M = hc.shape[0]
            Ec = E[:M]
            lse, tgt, crow = _stats(hc, weight, lab, True, Ec)
            nll[i:i + M] = lse - tgt
            s = torch.exp(crow - lse) * w
            valid = lab != ignore_index
            safe = torch.where(valid, lab, 0)
            BH = min(1024, triton.next_power_of_2(H))
            rg = (M, triton.cdiv(H, BH))
            g = torch.mm(Ec, weight, out_dtype=torch.float32)
            _gh_epi_kernel[rg](g, s, weight, safe, w, gh[i:i + M], H, BH, num_warps=4)
            # s*h can sit far below bf16's normal range (s ~ e^-lse / n_valid): lift it by an exact
            # power of two and take it back through cuBLAS alpha
            hs = torch.empty_like(hc)
            _scale_rows_kernel[rg](hc, s, hs, H, _HS, BH, num_warps=4)
            torch.addmm(gw, Ec.t(), hs, alpha=1.0 / _HS, out_dtype=torch.float32, out=gw)
            labs, rows = torch.sort(torch.where(valid, lab, V), stable=True)
            _scatter_kernel[rg](gw, labs, rows, hc, w, M, V, H, BH, num_warps=4)
        ctx.save_for_backward(gh, gw)
        ctx.wdt = wdt
        ctx.mark_non_differentiable(nll)
        return (nll * rw).sum(), nll

    @staticmethod
    def backward(ctx, g, _g_nll):
        gh, gw = ctx.saved_tensors
        return gh * g.to(gh.dtype), (gw * g).to(ctx.wdt), None, None, None, None


def _row_weights(labels_list, head_weights, ignore_index):
    out = []
    for y, hw in zip(labels_list, head_weights):
        v = (y != ignore_index).float()
        out.append(v * (hw / v.sum().clamp(min=1)))
    return torch.cat(out) if len(out) > 1 else out[0]


def fused_linear_cross_entropy_heads(hiddens, weight, labels, head_weights, ignore_index=-100,
                                     bwd_logits_budget=None):
    """sum_k head_weights[k] * CE_k in ONE pass over the vocab (MTP). Returns (total, [CE_k] detached)."""
    y = torch.cat(labels) if len(labels) > 1 else labels[0]
    rw = _row_weights(labels, head_weights, ignore_index)
    if torch.is_grad_enabled() and (weight.requires_grad or any(h.requires_grad for h in hiddens)):
        h = torch.cat(hiddens) if len(hiddens) > 1 else hiddens[0]
        total, nll = _FactoredCE.apply(h, weight, y, rw, ignore_index, bwd_logits_budget)
    else:
        with torch.no_grad():
            nll = torch.cat([_nll_nograd(*_cast(hh, weight), yy, bwd_logits_budget) for hh, yy in zip(hiddens, labels)])
        total = (nll * rw).sum()
    nll = nll.detach()
    per, o = [], 0
    for yy in labels:
        n = yy.shape[0]
        v = (yy != ignore_index).float()
        per.append((nll[o:o + n] * v).sum() / v.sum().clamp(min=1))
        o += n
    return total, per


def _cast(h, w):
    if torch.is_autocast_enabled("cuda"):
        dt = torch.get_autocast_dtype("cuda")
        h, w = h.to(dt), w.to(dt)
    return h.contiguous(), w.contiguous()


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None):
    """Drop-in for kernels.sm120.cross_entropy.fused_linear_cross_entropy (mean over valid rows)."""
    return fused_linear_cross_entropy_heads([hidden], weight, [labels], [1.0], ignore_index, bwd_logits_budget)[0]
