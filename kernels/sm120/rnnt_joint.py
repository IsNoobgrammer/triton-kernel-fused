"""Fused RNN-T joint + loss (sm120): NeMo RNNTJoint.joint_after_projection + RNNTLoss(warprnnt_numba), one op.

NeMo (fuse_loss_wer, fused_batch_size=2) loops over the batch two utterances at a time: per pair it builds the
(2, T, U+1, V) bf16 logits, casts them to fp32, runs the numba loss (which writes a same-size fp32 grad tensor) and
syncs the host for the max lengths. Here the logits never exist:

  pack     only the real (b, t, u) lattice points (t < T_b, u <= U_b), N rows, no padding:
           X[r] = [dropout(relu(bf16(f[b,t] + g[b,u]))), 1, 0...] in bf16, K = H + 1 rounded up to 32
           (the 1 column carries the bias: W_aug = [W | bias | 0], so the GEMM adds it in fp32 like cuBLAS does)
  stats    one GEMM pass over V (kernels/sm120/ce_factored.py's logits kernel + a blank column): per row
           lse, logit[label], logit[blank]; with grad, E = exp(L - c) in bf16 (c = 0 unless out of window)
  lattice  alpha (and beta, with grad) per utterance: a T-step loop, each step one log-space associative scan
           over u (x_u = logaddexp(c_u, x_{u-1} + d_u) composes as (d1, c1)(d2, c2) = (d1+d2, lae(c1+d2, c2)))
  grads    NeMo's numba gradient, FastEmit included, factored per row:
              dL/dlogit = p * (gb + gy) - gb * [v = blank] - gy * [v = label]
              gb = w e^(alpha + lp_blank + beta(t+1,u) - ll),   gy = w (1 + lambda) e^(alpha + lp_label + beta(t,u+1) - ll)
           Both one-hot terms are folded into E itself (E'[y] = e^(L_y - c) - gy/s, E'[blank] likewise, fp32 then one
           bf16 rounding), s = (gb + gy) e^(c - lse), so the backward is two cuBLAS GEMMs and nothing else:
           gX = s (E' @ W), gW = E'^T (s X). Then dropout/relu backward and two segment sums give df (over u), dg (over t).

The forward stores E for the whole batch when it fits `e_budget` (the user's "keep the logits, online softmax"
plan: 3 GEMMs per step); above that the backward recomputes E per chunk (4 GEMMs). Deterministic: no atomics.

    loss, nll = rnnt_joint_loss(f, g, weight, bias, targets, f_len, y_len, fastemit_lambda=0.005, dropout=0.2)
f: (B, T, H) projected encoder, g: (B, U+1, H) projected prednet, weight (V, H), bias (V,), blank = V - 1.
loss = mean over the batch of (1 + lambda) * -log P(y|x), exactly NeMo's reported value (its FastEmit scales the
cost, the gradient is the formula above); nll = per-utterance, detached.
"""
import torch
import triton
import triton.language as tl

from kernels.sm120.ce_factored import _HI, _HS, _LO, _lcfg

__all__ = ["rnnt_joint_loss"]

_E_BUDGET = 24 << 30                    # bytes of E kept from forward to backward (N * V * 2)
_NEG = float("-inf")
_LCFG = None                            # (BM, BN, BK, GROUP, warps, stages) override for bench/bench_rnnt_lcfg.py


@triton.jit
def _hidden_kernel(F, G, Y, OFF, TLEN, YLEN, X, LAB, seed, p, scale, T, U1, YS, Hd, K,
                   BU: tl.constexpr, BK: tl.constexpr, DROP: tl.constexpr):
    # one program per (b, t): rows off[b] + t (U_b + 1) + u, u = 0..U_b
    b = tl.program_id(0)
    t = tl.program_id(1)
    if t >= tl.load(TLEN + b):
        return
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    row0 = tl.load(OFF + b) + t * ub1
    fptr = F + (b * T + t).to(tl.int64) * Hd
    for u0 in range(0, ub1, BU):
        u = u0 + tl.arange(0, BU)
        mu = u < ub1
        rows = row0 + u
        lab = tl.load(Y + b.to(tl.int64) * YS + u, mask=mu & (u < ub1 - 1), other=-1)
        tl.store(LAB + rows, lab, mask=mu)
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            mk = k < Hd
            fv = tl.load(fptr + k, mask=mk, other=0.0).to(tl.float32)
            gv = tl.load(G + (b * U1 + u[:, None]).to(tl.int64) * Hd + k[None, :],
                         mask=mu[:, None] & mk[None, :], other=0.0).to(tl.float32)
            x = tl.maximum((fv[None, :] + gv).to(tl.bfloat16).to(tl.float32), 0.0)   # bf16 add, like eager
            if DROP:
                r = tl.rand(seed, (rows[:, None] * Hd + k[None, :]).to(tl.int32))
                x = tl.where(r >= p, x * scale, 0.0)
            x = tl.where(k[None, :] == Hd, 1.0, x)                                      # bias column
            tl.store(X + rows[:, None].to(tl.int64) * K + k[None, :], x.to(tl.bfloat16),
                     mask=mu[:, None] & (k[None, :] < K))


@triton.jit
def _logits_kernel(X, W, E, PA, PB, TGT, BLK, LAB, ORDER, CROW, M, V, VS, NT, K, BLANK,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr,
                   STORE_E: tl.constexpr, FIX: tl.constexpr, EVEN_K: tl.constexpr, EVEN_V: tl.constexpr):
    # ce_factored._logits_kernel plus the blank logit, E rows padded to VS (zeros past V: 16-byte aligned rows keep
    # cuBLAS on its sm120 kernels -- V = 4097 dropped it to cutlass_75 align1). FIX: out-of-window rows only.
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
    xf = acc.to(tl.bfloat16).to(tl.float32)        # the bf16 logits NeMo's autocast joint produces
    if not EVEN_V:
        xf = tl.where(mn[None, :], xf, -float("inf"))
    mx = tl.max(xf, axis=1)
    if STORE_E:
        if FIX:
            e = tl.exp(xf - c[:, None])
        else:
            e = tl.exp(xf)
        tl.store(E + rm[:, None].to(tl.int64) * VS + rn[None, :], e.to(tl.bfloat16),
                 mask=mm[:, None] & (rn[None, :] < VS))
        tl.store(PA + rm * NT + pn, tl.sum(e, axis=1), mask=mm)
    else:
        tl.store(PA + rm * NT + pn, tl.sum(tl.exp(xf - mx[:, None]), axis=1), mask=mm)
    if not FIX:
        tl.store(PB + rm * NT + pn, mx, mask=mm)
        lab = tl.load(LAB + rm, mask=mm, other=-1)
        hit = rn[None, :] == lab[:, None]
        tl.store(TGT + rm, tl.sum(tl.where(hit, xf, 0.0), axis=1),
                 mask=mm & (lab >= pn * BN) & (lab < pn * BN + BN))
        if (BLANK >= pn * BN) & (BLANK < pn * BN + BN):
            tl.store(BLK + rm, tl.sum(tl.where(rn[None, :] == BLANK, xf, 0.0), axis=1), mask=mm)


@triton.jit
def _lae(a, b):
    m = tl.maximum(a, b)
    m = tl.where(m == float("-inf"), 0.0, m)
    return m + tl.log(tl.exp(a - m) + tl.exp(b - m))


@triton.jit
def _comb(d1, c1, d2, c2):
    # x -> lae(x + d, c), composed left then right
    return d1 + d2, _lae(c1 + d2, c2)


@triton.jit
def _lattice_kernel(LPB, LPY, OFF, TLEN, YLEN, ALPHA, BETA, LL, BU: tl.constexpr, BETA_ON: tl.constexpr):
    # one program per utterance; alpha[t, u] = lae(alpha[t-1, u] + lpb[t-1, u], alpha[t, u-1] + lpy[t, u-1])
    b = tl.program_id(0)
    tb = tl.load(TLEN + b).to(tl.int32)
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    row0 = tl.load(OFF + b)
    j = tl.arange(0, BU)
    mj = j < ub1
    prev = tl.where(j == 0, 0.0, float("-inf"))
    lpb_prev = tl.zeros([BU], tl.float32)
    for t in range(0, tb):
        rows = row0 + t * ub1 + j
        c = tl.where(mj, prev + lpb_prev, float("-inf"))
        d = tl.load(LPY + rows - 1, mask=mj & (j > 0), other=0.0)
        _, a = tl.associative_scan((d, c), 0, _comb)
        tl.store(ALPHA + rows, a, mask=mj)
        lpb_prev = tl.load(LPB + rows, mask=mj, other=float("-inf"))
        prev = a
    tl.store(LL + b, tl.sum(tl.where(j == ub1 - 1, prev + lpb_prev, 0.0), 0))
    if BETA_ON:
        # beta[t, u] = lae(beta[t+1, u] + lpb[t, u], beta[t, u+1] + lpy[t, u]); lanes run u = U..0
        u = ub1 - 1 - j
        nxt = tl.where(j == 0, 0.0, float("-inf"))
        for tt in range(0, tb):
            rows = row0 + (tb - 1 - tt) * ub1 + u
            c = tl.where(mj, nxt + tl.load(LPB + rows, mask=mj, other=float("-inf")), float("-inf"))
            d = tl.load(LPY + rows, mask=mj & (j > 0), other=0.0)
            _, x = tl.associative_scan((d, c), 0, _comb)
            tl.store(BETA + rows, x, mask=mj)
            nxt = x


@triton.jit
def _rowgrad_kernel(ALPHA, BETA, LPB, LPY, LL, WB, LSE, CROW, OFF, TLEN, YLEN, GB, GY, S, X, XS, lam1, K, HS,
                    BU: tl.constexpr, BK: tl.constexpr):
    # per (b, t): the two lattice weights of every row, s = (gb + gy) e^(c - lse), and XS = bf16(X s HS) for dW
    b = tl.program_id(0)
    t = tl.program_id(1)
    tb = tl.load(TLEN + b).to(tl.int32)
    if t >= tb:
        return
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    row0 = tl.load(OFF + b) + t * ub1
    ll = tl.load(LL + b)
    w = tl.load(WB)
    for u0 in range(0, ub1, BU):
        u = u0 + tl.arange(0, BU)
        mu = u < ub1
        rows = row0 + u
        a = tl.load(ALPHA + rows, mask=mu, other=0.0)
        bt = tl.load(BETA + rows + ub1, mask=mu & (t < tb - 1), other=float("-inf"))
        bt = tl.where((t == tb - 1) & (u == ub1 - 1), 0.0, bt)
        bu = tl.load(BETA + rows + 1, mask=mu & (u < ub1 - 1), other=float("-inf"))
        gb = w * tl.exp(a + tl.load(LPB + rows, mask=mu, other=0.0) + bt - ll)
        gy = w * lam1 * tl.exp(a + tl.load(LPY + rows, mask=mu, other=0.0) + bu - ll)
        s = (gb + gy) * tl.exp(tl.load(CROW + rows, mask=mu, other=0.0) - tl.load(LSE + rows, mask=mu, other=0.0))
        tl.store(GB + rows, gb, mask=mu)
        tl.store(GY + rows, gy, mask=mu)
        tl.store(S + rows, s, mask=mu)
        sh = s * HS
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            m2 = mu[:, None] & (k[None, :] < K)
            off = rows[:, None].to(tl.int64) * K + k[None, :]
            x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
            tl.store(XS + off, (x * sh[:, None]).to(tl.bfloat16), mask=m2)


@triton.jit
def _fix_e_kernel(E, LAB, TGT, BLK, CROW, LSE, GB, GY, r0, M, VS, BLANK, BR: tl.constexpr):
    # fold the one-hot terms into E so the two GEMMs give the whole gradient (no scatter, no GEMV):
    #   s E'[r, v] = s E[r, v] - gy [v = label] - gb [v = blank],  E'[y] = e^(L_y - c) - (gy / (gb + gy)) e^(lse - c)
    # computed in fp32 from the bf16 logit and rounded ONCE (NeMo rounds its fp32 dlogits to bf16 the same way)
    i = tl.program_id(0) * BR + tl.arange(0, BR)
    mi = i < M
    r = r0 + i
    gb = tl.load(GB + r, mask=mi, other=0.0)
    gy = tl.load(GY + r, mask=mi, other=0.0)
    tot = gb + gy
    # lattice points with occupancy < 1e-30 exist (7.6e-43 seen at V=4097, T=40): 1/tot of a denormal is inf -> NaN.
    # Such a row's whole gradient is < 1e-30, far under any bf16 gradient it sums with: leave its E unfolded.
    live = mi & (tot > 1e-30)
    c = tl.load(CROW + r, mask=mi, other=0.0)
    big = tl.exp(tl.load(LSE + r, mask=mi, other=0.0) - c)
    inv = tl.where(live, 1.0 / tot, 0.0)
    erow = E + i.to(tl.int64) * VS
    lab = tl.load(LAB + r, mask=mi, other=-1)
    ey = tl.exp(tl.load(TGT + r, mask=mi, other=0.0) - c) - gy * inv * big
    eb = tl.exp(tl.load(BLK + r, mask=mi, other=0.0) - c) - gb * inv * big
    tl.store(erow + tl.maximum(lab, 0), ey.to(tl.bfloat16), mask=live & (lab >= 0))
    tl.store(erow + BLANK, eb.to(tl.bfloat16), mask=live)


@triton.jit
def _dfg_kernel(GM, S, F, G, OFF, TLEN, YLEN, DF, DGP, seed, p, scale, T, U1, NTB, Hd, K,
                TB: tl.constexpr, BU: tl.constexpr, BH: tl.constexpr, DROP: tl.constexpr):
    # per (b, block of TB frames, h-block), every u at once: dX = s (E' @ W) read ONCE from the GEMM output,
    # relu + dropout backward, df[b, t] = sum over u (complete per t), dg partial over this block's t (summed after)
    b = tl.program_id(0)
    tbi = tl.program_id(1)
    tb = tl.load(TLEN + b).to(tl.int32)
    t0 = tbi * TB
    if t0 >= tb:
        return
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    base = tl.load(OFF + b)
    cols = tl.program_id(2) * BH + tl.arange(0, BH)
    mc = cols < Hd
    u = tl.arange(0, BU)
    mu = u < ub1
    m2 = mu[:, None] & mc[None, :]
    gv = tl.load(G + (b * U1 + u[:, None]).to(tl.int64) * Hd + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    accg = tl.zeros([BU, BH], tl.float32)
    for t in range(t0, tl.minimum(t0 + TB, tb)):
        rows = base + t * ub1 + u
        fv = tl.load(F + (b * T + t).to(tl.int64) * Hd + cols, mask=mc, other=0.0).to(tl.float32)
        x = (fv[None, :] + gv).to(tl.bfloat16).to(tl.float32)
        sr = tl.load(S + rows, mask=mu, other=0.0)
        d = tl.load(GM + rows[:, None].to(tl.int64) * K + cols[None, :], mask=m2, other=0.0) * sr[:, None]
        d = tl.where(m2 & (x > 0.0), d, 0.0)
        if DROP:
            r = tl.rand(seed, (rows[:, None] * Hd + cols[None, :]).to(tl.int32))
            d = tl.where(r >= p, d * scale, 0.0)
        tl.store(DF + (b * T + t).to(tl.int64) * Hd + cols, tl.sum(d, axis=0), mask=mc)
        accg += d
    tl.store(DGP + ((b * NTB + tbi) * U1 + u[:, None]).to(tl.int64) * Hd + cols[None, :], accg, mask=m2)


@triton.jit
def _combine_rows_kernel(PA, PB, LSE, MX, CROW, M, NT, BR: tl.constexpr, BT: tl.constexpr,
                         STORE_E: tl.constexpr, FIX: tl.constexpr):
    # ce_factored._combine_kernel, BR rows per program (one row per program was 1.18M programs: 1.9 ms -> ~0.3)
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = r < M
    t = tl.arange(0, BT)
    m2 = mr[:, None] & (t[None, :] < NT)
    off = r[:, None].to(tl.int64) * NT + t[None, :]
    a = tl.load(PA + off, mask=m2, other=0.0)
    if STORE_E:
        if FIX:
            lse = tl.load(CROW + r, mask=mr, other=0.0) + tl.log(tl.sum(a, 1))
        else:
            lse = tl.log(tl.sum(a, 1))
    else:
        b = tl.load(PB + off, mask=m2, other=-float("inf"))
        mx = tl.max(b, 1)
        lse = mx + tl.log(tl.sum(a * tl.exp(b - mx[:, None]), 1))
    tl.store(LSE + r, lse, mask=mr)
    if not FIX:
        tl.store(MX + r, tl.max(tl.load(PB + off, mask=m2, other=-float("inf")), 1), mask=mr)


def _stats(X, Wa, lab, blank, store_e, E=None):
    """logits pass -> (lse, logit[label], logit[blank], crow). With store_e, E = exp(L - crow) is filled and the
    out-of-window rows got a second, exact pass; without it crow is what that pass WOULD use (for a recompute)."""
    M, K = X.shape
    VS = Wa.shape[0]
    V = blank + 1
    BM, BN, BK, G, nw, ns = _LCFG or _lcfg(K)
    NT = triton.cdiv(VS, BN)
    dev = X.device
    PA = torch.empty(M, NT, device=dev, dtype=torch.float32)
    PB = torch.empty(M, NT, device=dev, dtype=torch.float32)
    tgt = torch.zeros(M, device=dev, dtype=torch.float32)
    blk = torch.empty(M, device=dev, dtype=torch.float32)
    lse = torch.empty(M, device=dev, dtype=torch.float32)
    mx = torch.empty(M, device=dev, dtype=torch.float32)
    grid = (triton.cdiv(M, BM) * NT,)
    ev = dict(EVEN_K=K % BK == 0, EVEN_V=V % BN == 0, num_warps=nw, num_stages=ns)
    _logits_kernel[grid](X, Wa, E if store_e else PA, PA, PB, tgt, blk, lab, lab, PA, M, V, VS, NT, K, blank,
                         BM, BN, BK, G, store_e, False, **ev)
    cgrid = (triton.cdiv(M, 128),)
    _combine_rows_kernel[cgrid](PA, PB, lse, mx, PA, M, NT, 128, triton.next_power_of_2(NT), store_e, False,
                                num_warps=4)
    crow = torch.where((mx < _LO) | (mx > _HI), mx, torch.zeros_like(mx))
    if store_e:
        order = torch.argsort((crow != 0).to(torch.int8), descending=True, stable=True)
        _logits_kernel[grid](X, Wa, E, PA, PB, tgt, blk, lab, order, crow, M, V, VS, NT, K, blank,
                             BM, BN, BK, G, True, True, **ev)
        _combine_rows_kernel[cgrid](PA, PB, lse, mx, crow, M, NT, 128, triton.next_power_of_2(NT), True, True,
                                    num_warps=4)
    return lse, tgt, blk, crow


def _pack(f, g, y, tlen, ylen, p, seed):
    """-> X (N, K) bf16 packed hidden rows with the bias column, LAB (N,) int64 (-1 = no label), off (B,), N."""
    B, T, Hd = f.shape
    U1 = g.shape[1]
    K = triton.cdiv(Hd + 1, 32) * 32
    n = tlen * (ylen + 1)
    off = torch.cumsum(n, 0) - n
    N = int(n.sum())                                            # the op's one host sync
    assert p == 0 or N * Hd < 2 ** 31, "dropout offsets are int32"   # eval (p=0): no rand, no limit
    X = torch.empty(N, K, device=f.device, dtype=torch.bfloat16)
    LAB = torch.empty(N, device=f.device, dtype=torch.int64)
    _hidden_kernel[(B, T)](f, g, y, off, tlen, ylen, X, LAB, seed, p, 1.0 / (1.0 - p), T, U1, y.stride(0), Hd, K,
                           BU=min(32, triton.next_power_of_2(U1)), BK=64, DROP=p > 0, num_warps=4)
    return X, LAB, off, N


class _RNNTJoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, g, weight, bias, y, tlen, ylen, lam, p, seed, e_budget):
        fdt, gdt = f.dtype, g.dtype
        f = f.to(torch.bfloat16).contiguous()
        g = g.to(torch.bfloat16).contiguous()
        B, T, Hd = f.shape
        V = weight.shape[0]
        tlen, ylen = tlen.to(torch.int64), ylen.to(torch.int64)
        X, LAB, off, N = _pack(f, g, y.to(torch.int64), tlen, ylen, p, seed)
        K = X.shape[1]
        Wa = torch.zeros(triton.cdiv(V, 64) * 64, K, device=f.device, dtype=torch.bfloat16)
        Wa[:V, :Hd] = weight
        Wa[:V, Hd] = bias
        need = any(ctx.needs_input_grad[:4])
        store_e = need and N * V * 2 <= e_budget
        E = torch.empty(N, Wa.shape[0], device=f.device, dtype=torch.bfloat16) if store_e else None
        lse, tgt, blk, crow = _stats(X, Wa, LAB, V - 1, store_e, E)
        lpb = blk - lse
        lpy = torch.where(LAB >= 0, tgt - lse, _NEG)
        alpha = torch.empty(N, device=f.device, dtype=torch.float32)
        beta = torch.empty(N, device=f.device, dtype=torch.float32) if need else alpha
        ll = torch.empty(B, device=f.device, dtype=torch.float32)
        _lattice_kernel[(B,)](lpb, lpy, off, tlen, ylen, alpha, beta, ll,
                              BU=triton.next_power_of_2(g.shape[1]), BETA_ON=need, num_warps=4)
        nll = -(1.0 + lam) * ll                                 # NeMo's numba reports cost (1 + lambda) * -ll
        if need:
            ctx.save_for_backward(f, g, X, LAB, off, tlen, ylen, Wa, E, lse, crow, tgt, blk, lpb, lpy, alpha, beta, ll)
            ctx.cfg = (fdt, gdt, weight.dtype, bias.dtype, lam, p, seed, V)
        ctx.mark_non_differentiable(nll)
        return nll.mean(), nll

    @staticmethod
    def backward(ctx, gout, _g_nll):
        f, g, X, LAB, off, tlen, ylen, Wa, E, lse, crow, tgt, blk, lpb, lpy, alpha, beta, ll = ctx.saved_tensors
        fdt, gdt, wdt, bdt, lam, p, seed, V = ctx.cfg
        B, T, Hd = f.shape
        U1 = g.shape[1]
        N, K = X.shape
        VS = Wa.shape[0]
        dev = f.device
        w = (gout.float() / B).reshape(1)                       # mean_batch, no host sync
        GB, GY, S = (torch.empty(N, device=dev, dtype=torch.float32) for _ in range(3))
        XS = torch.empty_like(X)
        BU = min(32, triton.next_power_of_2(U1))
        _rowgrad_kernel[(B, T)](alpha, beta, lpb, lpy, ll, w, lse, crow, off, tlen, ylen, GB, GY, S, X, XS, 1.0 + lam,
                                K, _HS, BU=BU, BK=64, num_warps=4)
        GM = torch.empty(N, K, device=dev, dtype=torch.float32)       # E' @ W_aug, then dX in place
        gw = torch.zeros(VS, K, device=dev, dtype=torch.float32)
        BR = 256

        def grads(Ec, i, M):
            _fix_e_kernel[(triton.cdiv(M, BR),)](Ec, LAB, tgt, blk, crow, lse, GB, GY, i, M, VS, V - 1, BR,
                                                 num_warps=4)
            torch.mm(Ec, Wa, out_dtype=torch.float32, out=GM[i:i + M])
            # s X can sit far below bf16's normal range: XS carries it lifted by an exact power of two
            torch.addmm(gw, Ec.t(), XS[i:i + M], alpha=1.0 / _HS, out_dtype=torch.float32, out=gw)

        if E is not None:
            grads(E, 0, N)                                           # E is consumed (one backward per forward)
        else:
            C = max(1024, min(N, _E_BUDGET // (VS * 2)))
            Ebuf = torch.empty(min(C, N), VS, device=dev, dtype=torch.bfloat16)
            for i in range(0, N, C):
                M = min(C, N - i)
                _stats(X[i:i + M], Wa, LAB[i:i + M], V - 1, True, Ebuf[:M])
                grads(Ebuf[:M], i, M)
        df = torch.zeros(B, T, Hd, device=dev, dtype=torch.float32)
        BUg = triton.next_power_of_2(U1)
        BH = max(16, min(128, 8192 // BUg))                     # the (u, h) dg accumulator stays in registers
        TB = 16
        NTB = triton.cdiv(T, TB)
        dgp = torch.zeros(B, NTB, U1, Hd, device=dev, dtype=torch.float32)
        _dfg_kernel[(B, NTB, triton.cdiv(Hd, BH))](GM, S, f, g, off, tlen, ylen, df, dgp, seed, p, 1.0 / (1.0 - p),
                                                  T, U1, NTB, Hd, K, TB=TB, BU=BUg, BH=BH, DROP=p > 0, num_warps=8)
        dg = dgp.sum(1)
        return (df.to(fdt), dg.to(gdt), gw[:V, :Hd].to(wdt), gw[:V, Hd].to(bdt),
                None, None, None, None, None, None, None)


def rnnt_joint_loss(f, g, weight, bias, targets, f_len, y_len, fastemit_lambda=0.0, dropout=0.0, seed=None,
                    e_budget=_E_BUDGET):
    """NeMo joint (relu -> dropout -> linear) + warprnnt_numba loss, mean over the batch. Returns (loss, nll).
    dropout is the caller's (pass 0 in eval, like nn.Dropout outside training)."""
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))           # CPU generator: no device sync
    return _RNNTJoint.apply(f, g, weight, bias, targets, f_len, y_len, float(fastemit_lambda), float(dropout),
                            seed, e_budget)
