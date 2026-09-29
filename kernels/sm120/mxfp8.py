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
def _mx_gg_kernel(A, AS, B, BS, C, TE, TS, TM, ROWS, K: tl.constexpr, N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GATHER: tl.constexpr):
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
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


# swept at the board shapes (bench_quant_study / bench_quant_gemm --sweep), keyed by (K, N)
def _cfg(K, N):
    if K >= 1024:
        return (256, 128, 64, 8, 3)
    if K % 128 == 0:
        return (128, 128, 128, 4, 2)
    return (128, 128, 64, 4, 4) if K % 64 == 0 else (128, 128, 32, 4, 4)


def grouped_gemm(aq, as_, bq, bs, counts_t, m_rows, rows=None, out_dtype=torch.bfloat16, cfg=None):
    """C (m_rows, N): row r = A[rows[r] if rows is given else r] @ B[expert(r)]^T. A is MXFP8 along
    K; B (E, N, K) e4m3 with scales (E, N, K/32)."""
    from kernels.sm120.moe_fused_glu import build_tile_map
    K = aq.shape[1]
    E, N, _ = bq.shape
    BM, BN, BK, w, st = cfg or _cfg(K, N)
    while N % BN:
        BN //= 2
    while K % BK:
        BK //= 2
    TE, TS, TM = build_tile_map(None, counts_t, aq.device, bm=BM, m_rows=m_rows)
    c = torch.empty(m_rows, N, device=aq.device, dtype=out_dtype)
    grid = (TE.numel() * (N // BN),)
    _mx_gg_kernel[grid](aq, as_, bq, bs, c, TE, TS, TM, rows if rows is not None else TE, K, N,
                        BM, BN, BK, rows is not None, num_warps=w, num_stages=st)
    return c
