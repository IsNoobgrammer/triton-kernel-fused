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
                r = tl.rand(seed, (rows[:, None] * Hd + k[None, :]).to(tl.int64))
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
def _frame_targets_kernel(ALPHA, BETA, LPY, LL, LAB, OFF, TLEN, YLEN, IDX, LOGP, T, BLANK,
                          K: tl.constexpr, BU: tl.constexpr):
    # per (b, t): the transducer's own per-frame emission distribution from the lattice posteriors -- no decoding:
    #   py(u) = P(emit y_(u+1) at frame t | x, y) = exp(alpha(t,u) + lpy(t,u) + beta(t,u+1) - ll)
    # as a CTC-frame target: q(token) = sum of py over every u with that token / max(1, sum py), q(blank) =
    # max(0, 1 - sum py). Written as blank + the top K-1 tokens (log-probs; -inf = empty slot); the tokens not kept
    # are kl_div_topk's rest bucket. Duplicate tokens across u are merged before the top-k.
    b = tl.program_id(0)
    t = tl.program_id(1)
    tb = tl.load(TLEN + b).to(tl.int32)
    if t >= tb:
        return
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    row0 = tl.load(OFF + b) + t * ub1
    ll = tl.load(LL + b)
    u = tl.arange(0, BU)
    mu = u < ub1 - 1                                            # a label transition leaves every u < U
    rows = row0 + u
    a = tl.load(ALPHA + rows, mask=mu, other=float("-inf"))
    bu = tl.load(BETA + rows + 1, mask=mu, other=float("-inf"))
    lp = tl.load(LPY + rows, mask=mu, other=float("-inf"))
    py = tl.where(mu, tl.exp(a + lp + bu - ll), 0.0)
    lab = tl.load(LAB + rows, mask=mu, other=-1)
    tot = tl.sum(py, 0)
    den = tl.maximum(tot, 1.0)
    base = (b * T + t).to(tl.int64) * K
    pb = tl.maximum(1.0 - tot, 0.0)
    tl.store(IDX + base, BLANK)
    tl.store(LOGP + base, tl.where(pb > 0, tl.log(tl.maximum(pb, 1e-30)), float("-inf")))
    rem = py
    for k in tl.static_range(1, K):
        mx = tl.max(rem, 0)
        j = tl.argmax(rem, 0)
        lj = tl.sum(tl.where(u == j, lab, 0), 0)
        same = (lab == lj) & mu
        v = tl.sum(tl.where(same, rem, 0.0), 0)
        rem = tl.where(same, 0.0, rem)
        ok = mx > 0
        tl.store(IDX + base + k, tl.where(ok, lj, BLANK).to(tl.int32))
        tl.store(LOGP + base + k, tl.where(ok, tl.log(tl.maximum(v / den, 1e-30)), float("-inf")))


@triton.jit
def _rowgrad_kernel(ALPHA, BETA, LPB, LPY, LL, WB, LSE, CROW, OFF, TLEN, YLEN, GB, GY, S, X, XS, lam1, K, HS,
                    TPB, TPY, TOT, KW, BU: tl.constexpr, BK: tl.constexpr, KD: tl.constexpr):
    # per (b, t): the two lattice weights of every row, s = (gb + gy) e^(c - lse), and XS = bf16(X s HS) for dW.
    # KD (collapsed 3-way distillation, every node): dKL/dlogit = p * a - gbk [blank] - gyk [label] with
    # r = P_rest / q_rest, a = 1 - r, gbk = P_b - p_b r, gyk = P_y - p_y r -- the same factored form, so the row's
    # scale becomes tot = gb + gy + kw a and the one-hot weights gb + kw gbk, gy + kw gyk (TOT carries tot to fix_e)
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
        lpb = tl.load(LPB + rows, mask=mu, other=0.0)
        lpy = tl.load(LPY + rows, mask=mu, other=0.0)
        gb = w * tl.exp(a + lpb + bt - ll)
        gy = w * lam1 * tl.exp(a + lpy + bu - ll)
        tot = gb + gy
        if KD:
            kw = tl.load(KW)
            pb = tl.exp(lpb)
            py = tl.exp(lpy)                                     # 0 on u = U (lpy = -inf there)
            tpb = tl.exp(tl.load(TPB + rows, mask=mu, other=0.0))
            tpy = tl.exp(tl.load(TPY + rows, mask=mu, other=float("-inf")))
            qr = tl.maximum(1.0 - pb - py, 1e-6)
            r = tl.maximum(1.0 - tpb - tpy, 0.0) / qr
            gb = gb + kw * (tpb - pb * r)
            gy = gy + kw * (tpy - py * r)
            tot = tot + kw * (1.0 - r)
            tot = tl.where(tl.abs(tot) < 1e-6, tl.where(tot < 0, -1e-6, 1e-6), tot)   # factorable: tot != 0
        s = tot * tl.exp(tl.load(CROW + rows, mask=mu, other=0.0) - tl.load(LSE + rows, mask=mu, other=0.0))
        tl.store(GB + rows, gb, mask=mu)
        tl.store(GY + rows, gy, mask=mu)
        tl.store(TOT + rows, tot, mask=mu)
        tl.store(S + rows, s, mask=mu)
        sh = s * HS
        for k0 in range(0, K, BK):
            k = k0 + tl.arange(0, BK)
            m2 = mu[:, None] & (k[None, :] < K)
            off = rows[:, None].to(tl.int64) * K + k[None, :]
            x = tl.load(X + off, mask=m2, other=0.0).to(tl.float32)
            tl.store(XS + off, (x * sh[:, None]).to(tl.bfloat16), mask=m2)


@triton.jit
def _fix_e_kernel(E, LAB, TGT, BLK, CROW, LSE, GB, GY, TOT, r0, M, VS, BLANK, BR: tl.constexpr):
    # fold the one-hot terms into E so the two GEMMs give the whole gradient (no scatter, no GEMV):
    #   s E'[r, v] = s E[r, v] - gy [v = label] - gb [v = blank],  E'[y] = e^(L_y - c) - (gy / (gb + gy)) e^(lse - c)
    # computed in fp32 from the bf16 logit and rounded ONCE (NeMo rounds its fp32 dlogits to bf16 the same way)
    i = tl.program_id(0) * BR + tl.arange(0, BR)
    mi = i < M
    r = r0 + i
    gb = tl.load(GB + r, mask=mi, other=0.0)
    gy = tl.load(GY + r, mask=mi, other=0.0)
    tot = tl.load(TOT + r, mask=mi, other=0.0)                  # gb + gy (plus the KD scale, which can be < 0)
    # lattice points with occupancy < 1e-30 exist (7.6e-43 seen at V=4097, T=40): 1/tot of a denormal is inf -> NaN.
    # Such a row's whole gradient is < 1e-30, far under any bf16 gradient it sums with: leave its E unfolded.
    live = mi & (tl.abs(tot) > 1e-30)
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
            r = tl.rand(seed, (rows[:, None] * Hd + cols[None, :]).to(tl.int64))
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


def _pack(f, g, y, tlen, ylen, p, seed, n_rows=None):
    """-> X (N, K) bf16 packed hidden rows with the bias column, LAB (N,) int64 (-1 = no label), off (B,), N.
    n_rows: the caller's N = sum(tlen * (ylen + 1)) computed on the HOST from lengths it already has there -- skips the
    op's only host sync (int(n.sum())), so the CPU can keep queueing the step."""
    B, T, Hd = f.shape
    U1 = g.shape[1]
    K = triton.cdiv(Hd + 1, 32) * 32
    n = tlen * (ylen + 1)
    off = torch.cumsum(n, 0) - n
    N = int(n.sum()) if n_rows is None else int(n_rows)        # n.sum() = the op's one host sync
    # dropout offsets are int64: Philox counter (lo, hi) = (offset, offset >> 32), so below 2**32 the numbers are
    # bit-identical to the old int32 offsets and batches past 2**31 hidden elements no longer wrap
    X = torch.empty(N, K, device=f.device, dtype=torch.bfloat16)
    LAB = torch.empty(N, device=f.device, dtype=torch.int64)
    _hidden_kernel[(B, T)](f, g, y, off, tlen, ylen, X, LAB, seed, p, 1.0 / (1.0 - p), T, U1, y.stride(0), Hd, K,
                           BU=min(32, triton.next_power_of_2(U1)), BK=64, DROP=p > 0, num_warps=4)
    return X, LAB, off, N


class _RNNTJoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, g, weight, bias, y, tlen, ylen, lam, p, seed, e_budget, n_rows, frame_k, tpb=None, tpy=None,
                hard_w=1.0, kd_w=0.0):
        fdt, gdt = f.dtype, g.dtype
        f = f.to(torch.bfloat16).contiguous()
        g = g.to(torch.bfloat16).contiguous()
        B, T, Hd = f.shape
        V = weight.shape[0]
        tlen, ylen = tlen.to(torch.int64), ylen.to(torch.int64)
        X, LAB, off, N = _pack(f, g, y.to(torch.int64), tlen, ylen, p, seed, n_rows)
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
        beta = torch.empty(N, device=f.device, dtype=torch.float32) if need or frame_k else alpha
        ll = torch.empty(B, device=f.device, dtype=torch.float32)
        _lattice_kernel[(B,)](lpb, lpy, off, tlen, ylen, alpha, beta, ll,
                              BU=triton.next_power_of_2(g.shape[1]), BETA_ON=need or frame_k > 0, num_warps=4)
        nll = -(1.0 + lam) * ll                                 # NeMo's numba reports cost (1 + lambda) * -ll
        idx = torch.full((B, T, max(frame_k, 1)), V - 1, device=f.device, dtype=torch.int32)
        flp = torch.full((B, T, max(frame_k, 1)), _NEG, device=f.device, dtype=torch.float32)
        if frame_k:                                             # per-frame teacher targets for a CTC head
            _frame_targets_kernel[(B, T)](alpha, beta, lpy, ll, LAB, off, tlen, ylen, idx, flp, T, V - 1,
                                          K=frame_k, BU=triton.next_power_of_2(g.shape[1]), num_warps=4)
        kd = tpb is not None
        if kd:                                                  # collapsed KL(teacher || student) per node, summed
            assert tpb.shape == (N,) and tpy.shape == (N,), "teacher node log-probs must use the student's packing"
            pb, py = lpb.exp(), lpy.exp()
            Pb, Py = tpb.float().exp(), tpy.float().exp()
            qr = (1.0 - pb - py).clamp_min(1e-6)
            Pr = (1.0 - Pb - Py).clamp_min(0.0)
            z = torch.zeros_like(pb)
            kl = (torch.where(Pb > 0, Pb * (tpb.float() - lpb), z) + torch.where(Py > 0, Py * (tpy.float() - lpy), z)
                  + torch.where(Pr > 0, Pr * (Pr.clamp_min(1e-30).log() - qr.log()), z))
            cs = torch.cumsum(kl.double(), 0)                   # per-utterance segment sums (deterministic)
            end = off + tlen * (ylen + 1) - 1
            kdu = (cs[end] - torch.where(off > 0, cs[(off - 1).clamp_min(0)], torch.zeros_like(cs[end]))).float()
        else:
            kdu = torch.zeros(B, device=f.device, dtype=torch.float32)
        if need:
            ctx.save_for_backward(f, g, X, LAB, off, tlen, ylen, Wa, E, lse, crow, tgt, blk, lpb, lpy, alpha, beta, ll,
                                  tpb if kd else lpb, tpy if kd else lpy)
            ctx.cfg = (fdt, gdt, weight.dtype, bias.dtype, lam, p, seed, V, kd, float(hard_w), float(kd_w))
        ctx.mark_non_differentiable(nll, idx, flp, kdu)
        loss = nll.mean() if not kd and hard_w == 1.0 else hard_w * nll.mean() + kd_w * kdu.mean()
        return loss, nll, idx, flp, kdu

    @staticmethod
    def backward(ctx, gout, _g_nll, _g_idx, _g_flp, _g_kd):
        f, g, X, LAB, off, tlen, ylen, Wa, E, lse, crow, tgt, blk, lpb, lpy, alpha, beta, ll, tpb, tpy = ctx.saved_tensors
        fdt, gdt, wdt, bdt, lam, p, seed, V, kd, hard_w, kd_w = ctx.cfg
        B, T, Hd = f.shape
        U1 = g.shape[1]
        N, K = X.shape
        VS = Wa.shape[0]
        dev = f.device
        w = (gout.float() / B).reshape(1)                       # mean_batch, no host sync
        kw = w * kd_w
        if hard_w != 1.0:
            w = w * hard_w
        GB, GY, S, TOT = (torch.empty(N, device=dev, dtype=torch.float32) for _ in range(4))
        XS = torch.empty_like(X)
        BU = min(32, triton.next_power_of_2(U1))
        _rowgrad_kernel[(B, T)](alpha, beta, lpb, lpy, ll, w, lse, crow, off, tlen, ylen, GB, GY, S, X, XS, 1.0 + lam,
                                K, _HS, tpb, tpy, TOT, kw, BU=BU, BK=64, KD=kd, num_warps=4)
        GM = torch.empty(N, K, device=dev, dtype=torch.float32)       # E' @ W_aug, then dX in place
        gw = torch.zeros(VS, K, device=dev, dtype=torch.float32)
        BR = 256

        def grads(Ec, i, M):
            _fix_e_kernel[(triton.cdiv(M, BR),)](Ec, LAB, tgt, blk, crow, lse, GB, GY, TOT, i, M, VS, V - 1, BR,
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
                None, None, None, None, None, None, None, None, None, None, None, None, None)


def rnnt_joint_loss(f, g, weight, bias, targets, f_len, y_len, fastemit_lambda=0.0, dropout=0.0, seed=None,
                    e_budget=_E_BUDGET, n_rows=None, frame_targets=0, kd=None, kd_weight=0.0, hard_weight=1.0):
    """NeMo joint (relu -> dropout -> linear) + warprnnt_numba loss, mean over the batch. Returns (loss, nll).
    dropout is the caller's (pass 0 in eval, like nn.Dropout outside training). n_rows: sum(f_len * (y_len + 1)) if the
    caller knows it on the host (no sync); it MUST be exact (the parity test checks the result is bitwise the same).
    frame_targets=K > 0: also return (idx, logp), each (B, T, K): the transducer's per-frame emission distribution from
    the lattice posteriors (blank + the top K-1 tokens, log-probs, -inf = empty), a CTC distillation target for
    kl_div_topk -- no decoding, computed from alpha / beta the loss already has.
    kd=(tpb, tpy): a teacher transducer's per-node log P(blank), log P(next label), packed like this lattice (from
    rnnt_joint_node_logp on the same lengths / targets): loss = hard_weight * mean(nll) + kd_weight * mean(kd), kd per
    utterance = sum over every lattice node of KL over {label, blank, rest} (Panchapagesan et al. 2021, "Efficient
    knowledge distillation for RNN-transducer models"); the student's rest mass is clamped at 1e-6. Returns
    (loss, nll, (idx, logp) or None, kd)."""
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))           # CPU generator: no device sync
    tpb, tpy = kd if kd is not None else (None, None)
    loss, nll, idx, flp, kdu = _RNNTJoint.apply(f, g, weight, bias, targets, f_len, y_len, float(fastemit_lambda),
                                                float(dropout), seed, e_budget, n_rows, int(frame_targets),
                                                tpb, tpy, float(hard_weight), float(kd_weight))
    if kd is not None:
        return loss, nll, ((idx, flp) if frame_targets else None), kdu
    return (loss, nll, (idx, flp)) if frame_targets else (loss, nll)


@torch.no_grad()
def rnnt_joint_node_logp(f, g, weight, bias, targets, f_len, y_len, n_rows=None, frame_targets=0):
    """Teacher side (no grad, no dropout): per lattice node log P(blank), log P(next label) (-inf on u = U), packed in
    rnnt_joint_loss's row order, plus the teacher's per-utterance nll and, with frame_targets=K, its per-frame (idx,
    logp) for a CTC student. -> (lpb, lpy, nll, (idx, logp) or None)"""
    f = f.to(torch.bfloat16).contiguous()
    g = g.to(torch.bfloat16).contiguous()
    B, T, Hd = f.shape
    V = weight.shape[0]
    tlen, ylen = f_len.to(torch.int64), y_len.to(torch.int64)
    X, LAB, off, N = _pack(f, g, targets.to(torch.int64), tlen, ylen, 0.0, 0, n_rows)
    K = X.shape[1]
    Wa = torch.zeros(triton.cdiv(V, 64) * 64, K, device=f.device, dtype=torch.bfloat16)
    Wa[:V, :Hd] = weight
    Wa[:V, Hd] = bias
    lse, tgt, blk, _ = _stats(X, Wa, LAB, V - 1, False)
    lpb = blk - lse
    lpy = torch.where(LAB >= 0, tgt - lse, _NEG)
    alpha = torch.empty(N, device=f.device, dtype=torch.float32)
    beta = torch.empty(N, device=f.device, dtype=torch.float32) if frame_targets else alpha
    ll = torch.empty(B, device=f.device, dtype=torch.float32)
    _lattice_kernel[(B,)](lpb, lpy, off, tlen, ylen, alpha, beta, ll,
                          BU=triton.next_power_of_2(g.shape[1]), BETA_ON=frame_targets > 0, num_warps=4)
    ft = None
    if frame_targets:
        idx = torch.full((B, T, frame_targets), V - 1, device=f.device, dtype=torch.int32)
        flp = torch.full((B, T, frame_targets), _NEG, device=f.device, dtype=torch.float32)
        _frame_targets_kernel[(B, T)](alpha, beta, lpy, ll, LAB, off, tlen, ylen, idx, flp, T, V - 1,
                                      K=frame_targets, BU=triton.next_power_of_2(g.shape[1]), num_warps=4)
        ft = (idx, flp)
    return lpb, lpy, -ll, ft
