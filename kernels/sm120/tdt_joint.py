"""Fused TDT joint + loss (sm120): NeMo RNNTJoint (num_extra_outputs = len(durations)) + RNNTLoss(loss_name="tdt"),
one op -- kernels/sm120/rnnt_joint.py extended to the Token-and-Duration Transducer (arXiv 2304.06795).

The joint's output layer has V token classes (blank last) followed by D duration classes. NeMo's TDT loss (numba,
gpu_rnnt.py / gpu_rnnt_kernel.py) is, per lattice point (t, u):
    token log-probs    lp_v = log_softmax(token logits)_v - sigma        (logit under-normalisation)
    duration log-probs lp_d = log_softmax(duration logits)_d
    blank, duration d > 0      (t, u) -> (t + d, u)       [the last arc: t + d == T at u == U]
    label y_u, duration d >= 0 (t, u) -> (t + d, u + 1)   (t + d < T)
    loss = -log P = -lae over the final blank arcs; with probability omega (one host draw per call, NeMo's) the
    step trains the plain RNN-T loss on the token logits instead (duration gradients zero).

Everything token-side is rnnt_joint's machinery unchanged (packed rows, one logits GEMM pass, E = exp(L - c), the
one-hot terms folded into E, two backward GEMMs), because the token gradient has the RNN-T form:
    dL/dlogit_v = occ * p_v - gb [v = blank] - gy [v = label],   occ = gb + gy
with gb / gy the posterior mass of the node's blank / label arcs summed over their durations. New here:
    lattice   alpha / beta per utterance over t; each frame's row is one associative scan over u (the duration-0
              label arcs stay inside the frame), the arcs from / to frames t -+ d (d > 0) read back from memory
    rowgrad   gb, gy and the D per-duration masses gd (= dL/d log p_d, negated) per row
    durations logits Z = X W_d^T (N x D, the GEMM's bias column included), dL/dZ = occ softmax(Z) - gd, folded into dX
              (dX = s (E' W) + dZ W_d) and dW_d = dZ^T X

    loss, nll = tdt_joint_loss(f, g, weight, bias, targets, f_len, y_len, durations=(0, 1, 2, 3, 4), sigma=0.05,
                               omega=0.1, dropout=0.2)
weight (V + D, H) / bias (V + D,): the joint's output layer (token rows first, blank = V - 1, then durations).
FastEmit is not supported (NeMo's TDT recipe trains without it; ASR recipe: FastEmit 0 since Oct 12 2026).
"""
import random

import torch
import triton
import triton.language as tl

from kernels.sm120.ce_factored import _HS
from kernels.sm120.rnnt_joint import (_E_BUDGET, _NEG, _comb, _fix_e_kernel, _lae, _pack, _stats,
                                      rnnt_joint_loss)

__all__ = ["tdt_joint_loss"]


@triton.jit
def _tdt_alpha_kernel(LPB, LPY, LPD, DUR, OFF, TLEN, YLEN, ALPHA, LL,
                      ND: tl.constexpr, HAS0: tl.constexpr, BU: tl.constexpr):
    # one program per utterance, lanes = u; rows of frame t: row0 + t * ub1 + u
    b = tl.program_id(0)
    tb = tl.load(TLEN + b).to(tl.int32)
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    row0 = tl.load(OFF + b)
    j = tl.arange(0, BU)
    mj = j < ub1
    for t in range(0, tb):
        rows = row0 + t * ub1 + j
        c = tl.where(mj & (j == 0) & (t == 0), 0.0, float("-inf"))
        for i in tl.static_range(ND):
            d = tl.load(DUR + i)
            if d > 0:
                if t >= d:
                    rp = rows - d * ub1                                  # (t - d, u)
                    ab = tl.load(ALPHA + rp, mask=mj, other=float("-inf"))
                    c = _lae(c, ab + tl.load(LPB + rp, mask=mj, other=float("-inf"))
                             + tl.load(LPD + rp * ND + i, mask=mj, other=float("-inf")))
                    ml = mj & (j > 0)                                    # label from (t - d, u - 1)
                    al = tl.load(ALPHA + rp - 1, mask=ml, other=float("-inf"))
                    c = _lae(c, al + tl.load(LPY + rp - 1, mask=ml, other=float("-inf"))
                             + tl.load(LPD + (rp - 1) * ND + i, mask=ml, other=float("-inf")))
        c = tl.where(mj, c, float("-inf"))
        if HAS0:                                                         # duration-0 label: x_u <- x_{u-1} in-frame
            ml = mj & (j > 0)
            dd = tl.load(LPY + rows - 1, mask=ml, other=float("-inf")) + tl.load(LPD + (rows - 1) * ND, mask=ml,
                                                                                 other=float("-inf"))
            dd = tl.where(j > 0, dd, 0.0)
        else:
            dd = tl.where(j > 0, float("-inf"), 0.0)
        _, a = tl.associative_scan((dd, c), 0, _comb)
        tl.store(ALPHA + rows, a, mask=mj)
        tl.debug_barrier()                                               # frame t visible to the lanes of t + d
    # log-likelihood: the final blank arcs (t, U) -> T
    ll = tl.full((), float("-inf"), tl.float32)
    for i in tl.static_range(ND):
        d = tl.load(DUR + i)
        if d > 0:
            if tb >= d:
                r = row0 + (tb - d) * ub1 + ub1 - 1
                ll = _lae(ll, tl.load(ALPHA + r) + tl.load(LPB + r) + tl.load(LPD + r * ND + i))
    tl.store(LL + b, ll)


@triton.jit
def _tdt_beta_kernel(LPB, LPY, LPD, DUR, OFF, TLEN, YLEN, BETA,
                     ND: tl.constexpr, HAS0: tl.constexpr, BU: tl.constexpr):
    # lanes run u = U..0 (u = ub1 - 1 - j) so the in-frame duration-0 label arcs (u -> u + 1) are one forward scan
    b = tl.program_id(0)
    tb = tl.load(TLEN + b).to(tl.int32)
    ub1 = tl.load(YLEN + b).to(tl.int32) + 1
    row0 = tl.load(OFF + b)
    j = tl.arange(0, BU)
    mj = j < ub1
    u = ub1 - 1 - j
    last = u == ub1 - 1
    for tt in range(0, tb):
        t = tb - 1 - tt
        rows = row0 + t * ub1 + u
        lpb = tl.load(LPB + rows, mask=mj, other=float("-inf"))
        ml = mj & (u < ub1 - 1)
        lpy = tl.load(LPY + rows, mask=ml, other=float("-inf"))
        c = tl.full([BU], float("-inf"), tl.float32)
        for i in tl.static_range(ND):
            d = tl.load(DUR + i)
            lpd = tl.load(LPD + rows * ND + i, mask=mj, other=float("-inf"))
            if d > 0:
                if t + d < tb:
                    c = _lae(c, tl.load(BETA + rows + d * ub1, mask=mj, other=float("-inf")) + lpb + lpd)
                    c = _lae(c, tl.load(BETA + rows + d * ub1 + 1, mask=ml, other=float("-inf")) + lpy + lpd)
                if t + d == tb:                                          # the final blank arc
                    c = _lae(c, tl.where(last, lpb + lpd, float("-inf")))
        c = tl.where(mj, c, float("-inf"))
        if HAS0:
            dd = tl.where(j > 0, lpy + tl.load(LPD + rows * ND, mask=ml, other=float("-inf")), 0.0)
        else:
            dd = tl.where(j > 0, float("-inf"), 0.0)
        _, x = tl.associative_scan((dd, c), 0, _comb)
        tl.store(BETA + rows, x, mask=mj)
        tl.debug_barrier()


@triton.jit
def _tdt_rowgrad_kernel(ALPHA, BETA, LPB, LPY, LPD, DUR, LL, WB, LSE, CROW, OFF, TLEN, YLEN, GB, GY, GD, S, X, XS,
                        K, HS, ND: tl.constexpr, BU: tl.constexpr, BK: tl.constexpr):
    # per (b, t): gb / gy (arc masses summed over durations), gd (per duration), s = (gb + gy) e^(c - lse), XS
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
        a = tl.load(ALPHA + rows, mask=mu, other=float("-inf"))
        lpb = tl.load(LPB + rows, mask=mu, other=float("-inf"))
        ml = mu & (u < ub1 - 1)
        lpy = tl.load(LPY + rows, mask=ml, other=float("-inf"))
        gb = tl.zeros([BU], tl.float32)
        gy = tl.zeros([BU], tl.float32)
        for i in tl.static_range(ND):
            d = tl.load(DUR + i)
            lpd = tl.load(LPD + rows * ND + i, mask=mu, other=float("-inf"))
            pb = tl.zeros([BU], tl.float32)
            py = tl.zeros([BU], tl.float32)
            if t + d < tb:
                bu = tl.load(BETA + rows + d * ub1 + 1, mask=ml, other=float("-inf"))
                py = tl.where(ml, w * tl.exp(a + lpy + lpd + bu - ll), 0.0)
                if d > 0:
                    bb = tl.load(BETA + rows + d * ub1, mask=mu, other=float("-inf"))
                    pb = tl.where(mu, w * tl.exp(a + lpb + lpd + bb - ll), 0.0)
            if d > 0:
                if t + d == tb:
                    pb = tl.where(mu & (u == ub1 - 1), w * tl.exp(a + lpb + lpd - ll), 0.0)
            gb += pb
            gy += py
            tl.store(GD + rows * ND + i, pb + py, mask=mu)
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
def _tdt_dfg_kernel(GM, S, DZ, WD, F, G, OFF, TLEN, YLEN, DF, DGP, seed, p, scale, T, U1, NTB, Hd, K,
                    ND: tl.constexpr, TB: tl.constexpr, BU: tl.constexpr, BH: tl.constexpr, DROP: tl.constexpr):
    # rnnt_joint._dfg_kernel + the duration head in-register: dX = s (E' W) + sum_d dZ[r, d] W_d (D = 5 FMAs per
    # element) -- one pass over GM, as for RNN-T (the first version scaled GM and added dZ W_d in two extra passes)
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
        for i in tl.static_range(ND):
            dz = tl.load(DZ + rows * ND + i, mask=mu, other=0.0)
            wd = tl.load(WD + i * K + cols, mask=mc, other=0.0)
            d += dz[:, None] * wd[None, :]
        d = tl.where(m2 & (x > 0.0), d, 0.0)
        if DROP:
            r = tl.rand(seed, (rows[:, None] * Hd + cols[None, :]).to(tl.int64))
            d = tl.where(r >= p, d * scale, 0.0)
        tl.store(DF + (b * T + t).to(tl.int64) * Hd + cols, tl.sum(d, axis=0), mask=mc)
        accg += d
    tl.store(DGP + ((b * NTB + tbi) * U1 + u[:, None]).to(tl.int64) * Hd + cols[None, :], accg, mask=m2)


_DUR = {}


def _dur_tensor(durations, dev):
    key = (durations, str(dev))
    if key not in _DUR:
        _DUR[key] = torch.tensor(durations, device=dev, dtype=torch.int32)
    return _DUR[key]


class _TDTJoint(torch.autograd.Function):
    @staticmethod
    def forward(ctx, f, g, weight, bias, y, tlen, ylen, durations, sigma, p, seed, e_budget, n_rows):
        fdt, gdt = f.dtype, g.dtype
        f = f.to(torch.bfloat16).contiguous()
        g = g.to(torch.bfloat16).contiguous()
        B, T, Hd = f.shape
        ND = len(durations)
        V = weight.shape[0] - ND                                     # token classes, blank = V - 1
        dev = f.device
        tlen, ylen = tlen.to(torch.int64), ylen.to(torch.int64)
        X, LAB, off, N = _pack(f, g, y.to(torch.int64), tlen, ylen, p, seed, n_rows)
        K = X.shape[1]
        Wa = torch.zeros(triton.cdiv(V, 64) * 64, K, device=dev, dtype=torch.bfloat16)
        Wa[:V, :Hd] = weight[:V]
        Wa[:V, Hd] = bias[:V]
        Wd = torch.zeros(ND, K, device=dev, dtype=torch.bfloat16)   # duration rows, bias in the 1 column
        Wd[:, :Hd] = weight[V:]
        Wd[:, Hd] = bias[V:]
        need = any(ctx.needs_input_grad[:4])
        store_e = need and N * Wa.shape[0] * 2 <= e_budget
        E = torch.empty(N, Wa.shape[0], device=dev, dtype=torch.bfloat16) if store_e else None
        lse, tgt, blk, crow = _stats(X, Wa, LAB, V - 1, store_e, E)
        Z = torch.mm(X, Wd.t(), out_dtype=torch.float32)            # (N, D) duration logits, fp32 accumulate
        lpd = torch.log_softmax(Z, -1).contiguous()
        lpb = blk - lse - sigma
        lpy = torch.where(LAB >= 0, tgt - lse - sigma, _NEG)
        dur = _dur_tensor(tuple(durations), dev)                     # cached: a per-call H2D copy is a host sync
        alpha = torch.empty(N, device=dev, dtype=torch.float32)
        ll = torch.empty(B, device=dev, dtype=torch.float32)
        BU = triton.next_power_of_2(g.shape[1])
        has0 = durations[0] == 0
        _tdt_alpha_kernel[(B,)](lpb, lpy, lpd, dur, off, tlen, ylen, alpha, ll, ND=ND, HAS0=has0, BU=BU, num_warps=4)
        nll = -ll
        if need:
            beta = torch.empty(N, device=dev, dtype=torch.float32)
            _tdt_beta_kernel[(B,)](lpb, lpy, lpd, dur, off, tlen, ylen, beta, ND=ND, HAS0=has0, BU=BU, num_warps=4)
            ctx.save_for_backward(f, g, X, LAB, off, tlen, ylen, Wa, Wd, E, lse, crow, tgt, blk, lpb, lpy, lpd, Z,
                                  dur, alpha, beta, ll)
            ctx.cfg = (fdt, gdt, weight.dtype, bias.dtype, p, seed, V, ND)
        ctx.mark_non_differentiable(nll)
        return nll.mean(), nll

    @staticmethod
    def backward(ctx, gout, _g_nll):
        (f, g, X, LAB, off, tlen, ylen, Wa, Wd, E, lse, crow, tgt, blk, lpb, lpy, lpd, Z, dur, alpha, beta,
         ll) = ctx.saved_tensors
        fdt, gdt, wdt, bdt, p, seed, V, ND = ctx.cfg
        B, T, Hd = f.shape
        U1 = g.shape[1]
        N, K = X.shape
        VS = Wa.shape[0]
        dev = f.device
        w = (gout.float() / B).reshape(1)                           # mean_batch, no host sync
        GB, GY, S = (torch.empty(N, device=dev, dtype=torch.float32) for _ in range(3))
        GD = torch.empty(N, ND, device=dev, dtype=torch.float32)
        XS = torch.empty_like(X)
        _tdt_rowgrad_kernel[(B, T)](alpha, beta, lpb, lpy, lpd, dur, ll, w, lse, crow, off, tlen, ylen, GB, GY, GD, S,
                                    X, XS, K, _HS, ND=ND, BU=min(32, triton.next_power_of_2(U1)), BK=64, num_warps=4)
        GM = torch.empty(N, K, device=dev, dtype=torch.float32)      # E' @ W_aug, then the whole dX
        gw = torch.zeros(VS, K, device=dev, dtype=torch.float32)
        BR = 256

        TOT = GB + GY                                           # _fix_e_kernel's row scale (as it computed before)
        def grads(Ec, i, M):
            _fix_e_kernel[(triton.cdiv(M, BR),)](Ec, LAB, tgt, blk, crow, lse, GB, GY, TOT, i, M, VS, V - 1, BR,
                                                 num_warps=4)
            torch.mm(Ec, Wa, out_dtype=torch.float32, out=GM[i:i + M])
            torch.addmm(gw, Ec.t(), XS[i:i + M], alpha=1.0 / _HS, out_dtype=torch.float32, out=gw)

        if E is not None:
            grads(E, 0, N)
        else:
            C = max(1024, min(N, _E_BUDGET // (VS * 2)))
            Ebuf = torch.empty(min(C, N), VS, device=dev, dtype=torch.bfloat16)
            for i in range(0, N, C):
                M = min(C, N - i)
                _stats(X[i:i + M], Wa, LAB[i:i + M], V - 1, True, Ebuf[:M])
                grads(Ebuf[:M], i, M)
        # durations: dL/dZ = occ softmax(Z) - gd (occ = gb + gy = sum_d gd); dX = s (E' W) + dZ W_d, dW_d = dZ^T X
        dZ = ((GB + GY)[:, None] * torch.softmax(Z, -1) - GD).contiguous()
        # dW_d = dZ^T X: one GEMM on bf16 operands, fp32 accumulate (the token rows' dW uses bf16 E' the same way)
        gwd = torch.mm(dZ.t().to(torch.bfloat16), X, out_dtype=torch.float32)
        df = torch.zeros(B, T, Hd, device=dev, dtype=torch.float32)
        BUg = triton.next_power_of_2(U1)
        BH = max(16, min(128, 8192 // BUg))
        TB = 16
        NTB = triton.cdiv(T, TB)
        dgp = torch.zeros(B, NTB, U1, Hd, device=dev, dtype=torch.float32)
        _tdt_dfg_kernel[(B, NTB, triton.cdiv(Hd, BH))](GM, S, dZ, Wd.float().contiguous(), f, g, off, tlen, ylen, df,
                                                      dgp, seed, p, 1.0 / (1.0 - p), T, U1, NTB, Hd, K, ND=ND, TB=TB,
                                                      BU=BUg, BH=BH, DROP=p > 0, num_warps=8)
        dg = dgp.sum(1)
        dW = torch.cat([gw[:V, :Hd], gwd[:, :Hd]], 0)
        db = torch.cat([gw[:V, Hd], gwd[:, Hd]], 0)
        return (df.to(fdt), dg.to(gdt), dW.to(wdt), db.to(bdt),
                None, None, None, None, None, None, None, None, None)


def tdt_joint_loss(f, g, weight, bias, targets, f_len, y_len, durations=(0, 1, 2, 3, 4), sigma=0.0, omega=0.0,
                   dropout=0.0, seed=None, e_budget=_E_BUDGET, n_rows=None, rng=random):
    """NeMo joint (relu -> dropout -> linear, V token + D duration outputs) + TDT loss, mean over the batch.
    Returns (loss, nll). omega: NeMo's sampled RNN-T loss -- with probability omega (one draw from `rng`, Python's
    random by default as NeMo's numba loss) this call is rnnt_joint_loss on the token rows, the durations get no
    gradient. durations ascending, durations[0] in (0, 1) (NeMo's kernels assume the same)."""
    durations = [int(d) for d in durations]
    assert durations == sorted(durations) and durations[0] in (0, 1), durations
    ND = len(durations)
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))
    if omega > 0 and rng.uniform(0, 1) < omega:
        V = weight.shape[0] - ND
        loss, nll = rnnt_joint_loss(f, g, weight[:V], bias[:V], targets, f_len, y_len, 0.0, dropout, seed,
                                    e_budget, n_rows)
        return loss + 0.0 * (weight[V:].sum() + bias[V:].sum()), nll  # durations in the graph, zero gradient
    return _TDTJoint.apply(f, g, weight, bias, targets, f_len, y_len, durations, float(sigma), float(dropout), seed,
                           e_budget, n_rows)
