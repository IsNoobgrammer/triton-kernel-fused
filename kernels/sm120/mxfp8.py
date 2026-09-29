"""MXFP8 building blocks for the expert GEMMs (sm120): e4m3 values, e8m0 (power-of-two) scale per 32
along each GEMM's reduction dim, applied INSIDE the MMA (tl.dot_scaled), fp32 accumulation
(verified exact on sm120: bench/bench_quant_study.py accum).

    quant_rows(x)            (R, K) float -> (q e4m3 (R, K), s e8m0 uint8 (R, K/32)); blocks along K
    quant_weight(w)          (E, R, C) fp32 master -> cached {(E, R, C) and (E, C, R)} e4m3 + scales,
                             from ONE set of 2D 32x32 blocks, so the forward (W) and the input-gradient
                             (W^T) GEMMs see the SAME quantized weight
    grouped_gemm(aq, as_, bq, bs, tile_map, rows=None)
                             C[r] = A[rows[r] or r] @ B[e(r)]^T; B stored (E, N, K) K-contiguous
    qstats(x, q, s)          (flushed-to-0 %, saturated %) of one quantization, for the drift logs

Scale rule: e = ceil(log2(amax / 448)) -- rounded UP, so no value ever exceeds e4m3's 448 (the
round-down rule diverged in NVIDIA's MXFP8 pretraining study). Design record and measurements:
bench/bench_quant_study.py, bench/bench_quant_layer.py, BiBo ablate/tools/quant_layer_study.py.
"""
import weakref

import torch
import triton
import triton.language as tl

F8 = torch.float8_e4m3fn
F8MAX = 448.0


# ------------------------------------------------------------------ activations
@triton.jit
def _quant_rows_kernel(X, Q, S, M, K: tl.constexpr, BM: tl.constexpr, NB: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_k * NB * 32 + tl.arange(0, NB * 32)
    mask = rows[:, None] < M
    x = tl.load(X + rows[:, None].to(tl.int64) * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    xb = tl.reshape(x, (BM, NB, 32))
    amax = tl.maximum(tl.max(tl.abs(xb), axis=2), 1e-30)
    e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
    q = tl.reshape(xb * tl.exp2(-e)[:, :, None], (BM, NB * 32))
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], q.to(tl.float8e4nv), mask=mask)
    sc = pid_k * NB + tl.arange(0, NB)
    tl.store(S + rows[:, None].to(tl.int64) * (K // 32) + sc[None, :], (e + 127.0).to(tl.uint8), mask=mask)


def quant_rows(x, BM=64):
    R, K = x.shape
    assert K % 32 == 0 and x.stride(1) == 1, (x.shape, x.stride())
    kb = K // 32
    nb = next(n for n in (4, 3, 2, 1) if kb % n == 0)
    q = torch.empty(R, K, device=x.device, dtype=F8)
    s = torch.empty(R, kb, device=x.device, dtype=torch.uint8)
    _quant_rows_kernel[(triton.cdiv(R, BM), kb // nb)](x, q, s, R, K, BM, nb, num_warps=4)
    return q, s


def dequant(q, s):
    return q.float() * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)


def qstats(x, q, s):
    """(% of non-zero values flushed to 0, % that would exceed 448 before saturation)."""
    xf = x.float()
    sc = torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)
    nz = xf != 0
    flushed = ((q.float() == 0) & nz).sum().item() / max(nz.sum().item(), 1)
    sat = ((xf.abs() / sc) > F8MAX * 1.0001).float().mean().item()
    return 100 * flushed, 100 * sat


# ------------------------------------------------------------------ weights (2D blocks, cached)
_WCACHE = {}


def _quant_weight_2d(w):
    E, R, C = w.shape
    assert R % 32 == 0 and C % 32 == 0, w.shape
    wb = w.detach().float().reshape(E, R // 32, 32, C // 32, 32)
    amax = wb.abs().amax(dim=(2, 4)).clamp_min(1e-30)                       # (E, R/32, C/32)
    e = torch.ceil(torch.log2(amax / F8MAX)).clamp(-127, 127)
    q = (wb * torch.exp2(-e)[:, :, None, :, None]).reshape(E, R, C).to(F8)
    s = (e + 127).to(torch.uint8)
    s_rc = s.repeat_interleave(32, dim=1).contiguous()                       # (E, R, C/32): blocks along C
    s_cr = s.transpose(1, 2).repeat_interleave(32, dim=1).contiguous()       # (E, C, R/32): blocks along R
    return {"rc": (q, s_rc), "cr": (q.transpose(1, 2).contiguous(), s_cr)}


def quant_weight(w):
    """Per-step cache keyed on the master tensor's identity and _version (Triton optimizer writes
    bump _version explicitly -- the stale-cast lesson, tkf 69157b5)."""
    key = w.untyped_storage().data_ptr()
    hit = _WCACHE.get(key)
    if hit is not None and hit[0]() is w and hit[1] == w._version and hit[2] == tuple(w.shape):
        return hit[3]
    out = _quant_weight_2d(w)
    _WCACHE[key] = (weakref.ref(w), w._version, tuple(w.shape), out)
    if len(_WCACHE) > 64:
        for k in list(_WCACHE)[:32]:
            _WCACHE.pop(k, None)
    return out


# ------------------------------------------------------------------ grouped GEMM
@triton.jit
def _mx_gg_kernel(A, AS, B, BS, C, TE, TS, TM, ROWS, X1, X2, X3, X4, I2, K: tl.constexpr, N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GATHER: tl.constexpr,
                  EPI: tl.constexpr, NP: tl.constexpr):
    """EPI 0: plain.  EPI 1 (F1): per-row partial sum of squares of the GATE columns (as stored, bf16)
    -> X1 (M, NP): radial r without its own pass.  EPI 2 (B3): with X2 = GU (M, 2*I2) bf16 and X3 = r
    (M,), per-row partials of S = sum go*u*df*gn and T = sum go*u*f -> X1 (M, 2*NP): the radial
    backward then needs a single pass. Partials are summed in fixed order: deterministic."""
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    if GATHER:
        ra = tl.load(ROWS + rm, mask=mask_m, other=0)
    else:
        ra = rm
    KS: tl.constexpr = K // 32
    Bb = B + e.to(tl.int64) * (N * K)
    BSb = BS + e.to(tl.int64) * (N * KS)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        rs = k0 // 32 + tl.arange(0, BK // 32)
        a = tl.load(A + ra[:, None].to(tl.int64) * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        a_s = tl.load(AS + ra[:, None].to(tl.int64) * KS + rs[None, :], mask=mask_m[:, None], other=127)
        b = tl.load(Bb + rn[None, :] * K + rk[:, None])
        b_s = tl.load(BSb + rn[:, None] * KS + rs[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
    if EPI == 3:
        # output itself stored as MXFP8 (blocks of 32 along N, scales -> X1 (M, N/32)): the F1 output
        # GU cached in fp8, as DeepSeek-V3 caches the SwiGLU input
        vb = tl.reshape(acc, (BM, BN // 32, 32))
        ex = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(vb), axis=2), 1e-30) / 448.0)), -127.0), 127.0)
        q = tl.reshape(vb * tl.exp2(-ex)[:, :, None], (BM, BN))
        tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], q.to(tl.float8e4nv), mask=mask_m[:, None])
        sc = pid_n * (BN // 32) + tl.arange(0, BN // 32)
        tl.store(X1 + rm[:, None].to(tl.int64) * (N // 32) + sc[None, :], (ex + 127.0).to(tl.uint8), mask=mask_m[:, None])
        if NP > 0:                    # F1: sum of squares of the dequantized gate columns (radial r)
            if pid_n < NP:
                dq = tl.reshape(tl.reshape(q.to(tl.float8e4nv).to(tl.float32), (BM, BN // 32, 32))
                                * tl.exp2(ex)[:, :, None], (BM, BN))
                tl.store(X2 + rm.to(tl.int64) * NP + pid_n, tl.sum(dq * dq, axis=1), mask=mask_m)
        return
    out = acc.to(C.dtype.element_ty)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], out, mask=mask_m[:, None])
    if EPI == 1:
        if pid_n < NP:                                   # a gate tile (columns < I)
            v = out.to(tl.float32)
            tl.store(X1 + rm.to(tl.int64) * NP + pid_n, tl.sum(v * v, axis=1), mask=mask_m)
    if EPI == 2:
        # radial backward S partials: go = this output tile, g / u = GU (fp8 + scales X4) of the same
        # rows and columns, r = X3. T needs no pass (T = w * gw / r^p, see moe_fp8).
        go = out.to(tl.float32)
        gub = X2 + rm[:, None].to(tl.int64) * (2 * I2)
        g = tl.load(gub + rn[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        u = tl.load(gub + I2 + rn[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        sb = X4 + rm[:, None].to(tl.int64) * (2 * I2 // 32)
        cs = pid_n * (BN // 32) + tl.arange(0, BN // 32)
        sg = tl.exp2(tl.load(sb + cs[None, :], mask=mask_m[:, None], other=127).to(tl.float32) - 127.0)
        su = tl.exp2(tl.load(sb + I2 // 32 + cs[None, :], mask=mask_m[:, None], other=127).to(tl.float32) - 127.0)
        g = tl.reshape(tl.reshape(g, (BM, BN // 32, 32)) * sg[:, :, None], (BM, BN))
        u = tl.reshape(tl.reshape(u, (BM, BN // 32, 32)) * su[:, :, None], (BM, BN))
        r = tl.load(X3 + rm, mask=mask_m, other=1.0)
        gn = g / r[:, None]
        sig = 1.0 / (1.0 + tl.exp(-gn))
        tl.store(X1 + rm.to(tl.int64) * NP + pid_n,
                 tl.sum(go * u * sig * (1.0 + gn * (1.0 - sig)) * gn, axis=1), mask=mask_m)


# swept at the board shapes (bench_quant_study / bench_quant_gemm --sweep), keyed by (K, N)
def _cfg(K, N):
    if K >= 1024:
        return (256, 128, 64, 8, 3)
    if K % 128 == 0:
        return (128, 128, 128, 4, 2)
    return (128, 128, 64, 4, 4) if K % 64 == 0 else (128, 128, 32, 4, 4)


_TM_CACHE = {}


def tile_map(counts_t, m_rows, bm):
    """build_tile_map, memoized on the counts tensor object (the same counts serve every GEMM of the
    layer, forward and backward: one build per BM instead of one per call)."""
    from kernels.sm120.moe_fused_glu import build_tile_map
    key = (id(counts_t), counts_t._version, m_rows, bm)
    hit = _TM_CACHE.get(key)
    if hit is not None and hit[0] is counts_t:
        return hit[1]
    tm = build_tile_map(None, counts_t, counts_t.device, bm=bm, m_rows=m_rows)
    _TM_CACHE[key] = (counts_t, tm)
    if len(_TM_CACHE) > 64:
        for k in list(_TM_CACHE)[:32]:
            _TM_CACHE.pop(k, None)
    return tm


def grouped_gemm(aq, as_, bq, bs, counts_t, m_rows, rows=None, out_dtype=torch.bfloat16, cfg=None,
                 epi=0, x1=None, x2=None, x3=None, x4=None, i2=0, np_=1):
    """C (m_rows, N): row r = A[rows[r] if rows is given else r] @ B[expert(r)]^T. A is MXFP8 along
    K; B (E, N, K) e4m3 with scales (E, N, K/32). epi / x1..x3: see _mx_gg_kernel."""
    K = aq.shape[1]
    E, N, _ = bq.shape
    BM, BN, BK, w, st = cfg or _cfg(K, N)
    while N % BN:
        BN //= 2
    while K % BK:
        BK //= 2
    TE, TS, TM = tile_map(counts_t, m_rows, BM)
    c = torch.empty(m_rows, N, device=aq.device, dtype=F8 if epi == 3 else out_dtype)
    grid = (TE.numel() * (N // BN),)
    d = x1 if x1 is not None else c
    _mx_gg_kernel[grid](aq, as_, bq, bs, c, TE, TS, TM, rows if rows is not None else TE,
                        d, x2 if x2 is not None else d, x3 if x3 is not None else d,
                        x4 if x4 is not None else d, i2, K, N,
                        BM, BN, BK, rows is not None, epi, np_, num_warps=w, num_stages=st)
    return c


def gemm_bn(K, N):
    BN = (_cfg(K, N))[1]
    while N % BN:
        BN //= 2
    return BN


# ------------------------------------------------------------------ weight gradients (K = tokens)
@triton.jit
def _mx_wgrad_kernel(A, B, C, ROWS, START, END, N1: tl.constexpr, N2: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     GATHER: tl.constexpr, ACC: tl.constexpr):
    """C[e] (N1, N2) fp32 (+)= A[rows_e]^T @ B[rows_e], reduction over the expert's TOKENS.
    A (M, N1) / B (M or N_tok, N2) are bf16; each tile is quantized to MXFP8 IN REGISTERS along the
    token axis (32-token blocks counted from the expert's first row; rows past its end load as 0,
    which cannot raise a block max), then fed to the block-scaled MMA. No quant pass, no padding."""
    pid = tl.program_id(0)
    TN: tl.constexpr = N2 // BN
    TM: tl.constexpr = N1 // BM
    e = pid // (TM * TN)
    r = pid % (TM * TN)
    pm = r // TN
    pn = r % TN
    s0 = tl.load(START + e)
    s1 = tl.load(END + e)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    NB: tl.constexpr = BK // 32
    for k0 in range(s0, s1, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < s1
        a = tl.load(A + rk[None, :].to(tl.int64) * N1 + rm[:, None], mask=mk[None, :], other=0.0).to(tl.float32)
        if GATHER:
            rb = tl.load(ROWS + rk, mask=mk, other=0)
        else:
            rb = rk
        b = tl.load(B + rb[:, None].to(tl.int64) * N2 + rn[None, :], mask=mk[:, None], other=0.0).to(tl.float32)
        ab = tl.reshape(a, (BM, NB, 32))
        ea = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(ab), axis=2), 1e-30) / 448.0)), -127.0), 127.0)
        aq = tl.reshape(ab * tl.exp2(-ea)[:, :, None], (BM, BK)).to(tl.float8e4nv)
        bb = tl.reshape(b, (NB, 32, BN))
        eb = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(bb), axis=1), 1e-30) / 448.0)), -127.0), 127.0)
        bq = tl.reshape(bb * tl.exp2(-eb)[:, None, :], (BK, BN)).to(tl.float8e4nv)
        acc = tl.dot_scaled(aq, (ea + 127.0).to(tl.uint8), "e4m3", bq, tl.trans((eb + 127.0).to(tl.uint8)), "e4m3", acc)
    cp = C + e.to(tl.int64) * (N1 * N2) + rm[:, None] * N2 + rn[None, :]
    if ACC:
        acc += tl.load(cp)
    tl.store(cp, acc)


_WG8 = {"narrow": (128, 128, 64, 4, 3), "wide": (128, 128, 64, 8, 3)}


def grouped_wgrad(a, b, counts_t, out=None, accumulate=False, b_rows=None, cfg=None):
    """out (E, N1, N2) fp32 = per expert a[rows_e]^T @ b[rows_e] in MXFP8 (token-axis blocks)."""
    M, N1 = a.shape
    N2 = b.shape[1]
    E = counts_t.numel()
    BM, BN, BK, w, st = cfg or _WG8["wide" if N1 >= 1024 else "narrow"]
    assert N1 % BM == 0 and N2 % BN == 0 and a.stride(1) == 1 and b.stride(1) == 1
    end = torch.cumsum(counts_t, 0).to(torch.int32)
    start = (end - counts_t.to(torch.int32)).to(torch.int32)
    if out is None:
        out = torch.empty(E, N1, N2, device=a.device, dtype=torch.float32)
        accumulate = False
    _mx_wgrad_kernel[(E * (N1 // BM) * (N2 // BN),)](
        a, b, out, b_rows if b_rows is not None else start, start, end, N1, N2, BM, BN, BK,
        b_rows is not None, accumulate, num_warps=w, num_stages=st)
    return out


# ------------------------------------------------------------------ weight gradients on K-MAJOR fp8 copies
@triton.jit
def _mx_wgrad_km_kernel(AT, ATS, BT, BTS, C, PST, PCNT, Mp: tl.constexpr, N1: tl.constexpr, N2: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, ACC: tl.constexpr,
                        EVEN: tl.constexpr):
    """C[e] (N1, N2) fp32 (+)= A_e^T @ B_e with the TOKEN axis contiguous in memory: AT (N1, Mp) and
    BT (N2, Mp) are e4m3, scales (Mp/32, N1 | N2) per 32 tokens (token-block-major: contiguous per K step). Expert e owns token columns
    [PST[e], PST[e] + PCNT[e]) (32-aligned, zero-padded by the producer)."""
    pid = tl.program_id(0)
    TN: tl.constexpr = N2 // BN
    TM: tl.constexpr = N1 // BM
    e = pid // (TM * TN)
    r = pid % (TM * TN)
    rm = (r // TN) * BM + tl.arange(0, BM)
    rn = (r % TN) * BN + tl.arange(0, BN)
    p0 = tl.load(PST + e)
    p1 = p0 + tl.load(PCNT + e)
    KS: tl.constexpr = Mp // 32
    p0 = tl.multiple_of(p0, 32)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(p0, p1, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < p1
        rs = k0 // 32 + tl.arange(0, BK // 32)
        ms = rs < p1 // 32
        if EVEN:            # every expert range padded to a multiple of BK: no masks on the K axis
            a = tl.load(AT + rm[:, None].to(tl.int64) * Mp + rk[None, :])
            a_s = tl.load(ATS + rs[None, :].to(tl.int64) * N1 + rm[:, None])
            b = tl.load(BT + rn[None, :].to(tl.int64) * Mp + rk[:, None])
            b_s = tl.load(BTS + rs[None, :].to(tl.int64) * N2 + rn[:, None])
        else:
            a = tl.load(AT + rm[:, None].to(tl.int64) * Mp + rk[None, :], mask=mk[None, :], other=0.0)
            a_s = tl.load(ATS + rs[None, :].to(tl.int64) * N1 + rm[:, None], mask=ms[None, :], other=127)
            b = tl.load(BT + rn[None, :].to(tl.int64) * Mp + rk[:, None], mask=mk[:, None], other=0.0)
            b_s = tl.load(BTS + rs[None, :].to(tl.int64) * N2 + rn[:, None], mask=ms[None, :], other=127)
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
    cp = C + e.to(tl.int64) * (N1 * N2) + rm[:, None] * N2 + rn[None, :]
    if ACC:
        acc += tl.load(cp)
    tl.store(cp, acc)


def wgrad_kmajor(at, ats, bt, bts, pst, pcnt, E, out=None, accumulate=False, cfg=None, even=False):
    N1, Mp = at.shape
    N2 = bt.shape[0]
    BM, BN, BK, w, st = cfg or (128, 128, 128, 4, 3)     # swept: 449 / 516 TF on B2 / B5 (unmasked)
    if out is None:
        out = torch.empty(E, N1, N2, device=at.device, dtype=torch.float32)
        accumulate = False
    _mx_wgrad_km_kernel[(E * (N1 // BM) * (N2 // BN),)](at, ats, bt, bts, out, pst, pcnt, Mp, N1, N2,
                                                        BM, BN, BK, accumulate, even, num_warps=w, num_stages=st)
    return out
