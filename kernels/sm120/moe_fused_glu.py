import torch
import triton
import triton.language as tl

__all__ = ["fused_gate_up_glu", "fused_supported", "gemm_supported", "tiles_supported",
           "build_tile_map", "fused_gate_up_radial", "radial_supported"]

_BM, _BN, _BK, _WARPS, _STAGES = 64, 256, 32, 8, 3


@triton.jit
def _gate_up_glu_kernel(X, W, GU, IT, TE, TS, TM,
                        H: tl.constexpr, I: tl.constexpr, CODE: tl.constexpr,
                        WRITE_GU: tl.constexpr, ACT: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    Wb = W + e.to(tl.int64) * (2 * I * H)
    ag = tl.zeros((BM, BN), tl.float32)
    au = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, H, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
        wg = tl.load(Wb + rn[:, None] * H + rk[None, :])
        wu = tl.load(Wb + (I + rn[:, None]) * H + rk[None, :])
        ag = tl.dot(x, tl.trans(wg), ag)
        au = tl.dot(x, tl.trans(wu), au)
    if ACT:
        if CODE == 1:
            act = tl.maximum(ag, 0.0) * tl.maximum(ag, 0.0)
        else:
            act = ag * tl.sigmoid(ag)
        tl.store(IT + rm[:, None] * I + rn[None, :], (act * au).to(tl.bfloat16), mask=mask_m[:, None])
    if WRITE_GU:
        tl.store(GU + rm[:, None] * (2 * I) + rn[None, :], ag.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), au.to(tl.bfloat16),
                 mask=mask_m[:, None])


def tiles_supported(hidden):
    return (hidden.dtype in (torch.bfloat16, torch.float16)
            and hidden.device.type == "cuda"
            and torch.cuda.get_device_capability(hidden.device)[0] >= 8
            and hidden.is_contiguous())


def gemm_supported(hidden, gate_up_proj, codes):
    I = gate_up_proj.shape[1] // 2
    return (tiles_supported(hidden) and hidden.dtype is torch.bfloat16
            and gate_up_proj.is_contiguous()
            and I % _BN == 0 and gate_up_proj.shape[2] % _BK == 0
            and len(set(codes)) == 1 and codes[0] in (0, 2, 8))


def fused_supported(hidden, gate_up_proj, codes):
    return gemm_supported(hidden, gate_up_proj, codes) and codes[0] == 0


def build_tile_map(counts, counts_t, device, bm=None):
    bm = _BM if bm is None else bm
    ntile = [(c + bm - 1) // bm for c in counts]
    total = sum(ntile)
    nt = ((counts_t + (bm - 1)) // bm).to(torch.int32)
    te = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), nt)
    start = torch.cumsum(nt, 0) - nt
    within = torch.arange(total, device=device, dtype=torch.int32) - start[te]
    bnd = torch.cumsum(counts_t, 0) - counts_t
    ts = (bnd[te] + within * bm).to(torch.int32)
    tm = torch.clamp(counts_t[te] - within * bm, max=bm).to(torch.int32)
    return te, ts, tm


def fused_gate_up_glu(x_s, gate_up_proj, tile_map, code, want_gu=True, act=True):
    TE, TS, TM = tile_map
    M, H = x_s.shape
    I = gate_up_proj.shape[1] // 2
    it = torch.empty(M, I, device=x_s.device, dtype=x_s.dtype) if act else None
    gu = torch.empty(M, 2 * I, device=x_s.device, dtype=x_s.dtype) if want_gu else it
    _gate_up_glu_kernel[(TE.numel(), I // _BN)](
        x_s, gate_up_proj, gu, it, TE, TS, TM, H, I, code, want_gu, act,
        _BM, _BN, _BK, num_warps=_WARPS, num_stages=_STAGES)
    return gu, it


_BBM, _BBN, _BBK, _BWARPS, _BSTAGES = 32, 256, 64, 8, 3


@triton.jit
def _dinter_glu_bwd_kernel(GE, W2, GU, DGU, TE, TS, TM,
                           H: tl.constexpr, I: tl.constexpr, CODE: tl.constexpr,
                           BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    Wb = W2 + e.to(tl.int64) * (H * I)
    gi = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, H, BK):
        rk = k0 + tl.arange(0, BK)
        ge = tl.load(GE + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(Wb + rk[:, None] * I + rn[None, :])
        gi = tl.dot(ge, w, gi)
    ag = tl.load(GU + rm[:, None] * (2 * I) + rn[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    au = tl.load(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), mask=mask_m[:, None], other=0.0).to(tl.float32)
    if CODE == 1:
        r = tl.maximum(ag, 0.0)
        d_ag = gi * (2.0 * r) * au
        d_au = gi * (r * r)
    else:
        sg = tl.sigmoid(ag)
        d_ag = gi * (sg * (1.0 + ag * (1.0 - sg))) * au
        d_au = gi * (ag * sg)
    tl.store(DGU + rm[:, None] * (2 * I) + rn[None, :], d_ag.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(DGU + rm[:, None] * (2 * I) + (I + rn[None, :]), d_au.to(tl.bfloat16), mask=mask_m[:, None])


BWD_BM = _BBM


def fused_dinter_glu_bwd(ge, down_proj, gu, tile_map, code):
    TE, TS, TM = tile_map
    M, H = ge.shape
    I = down_proj.shape[2]
    dgu = torch.empty(M, 2 * I, device=ge.device, dtype=ge.dtype)
    _dinter_glu_bwd_kernel[(TE.numel(), I // _BBN)](
        ge, down_proj, gu, dgu, TE, TS, TM, H, I, code,
        _BBM, _BBN, _BBK, num_warps=_BWARPS, num_stages=_BSTAGES)
    return dgu


@triton.jit
def _grouped_gemm_kernel(A, B, C, TE, TS, TM, K: tl.constexpr, N: tl.constexpr,
                         BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    Bb = B + e.to(tl.int64) * (K * N)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        b = tl.load(Bb + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(tl.bfloat16), mask=mask_m[:, None])


_GG = (128, 256, 64, 8, 3)


def grouped_gemm(a, b_enk, tile_map, out=None):
    TE, TS, TM = tile_map
    M, K = a.shape
    N = b_enk.shape[2]
    if N % _GG[1] or K % _GG[2] or TE is None:
        return None
    c = torch.empty(M, N, device=a.device, dtype=a.dtype) if out is None else out
    BM, BN, BK, w, st = _GG
    _grouped_gemm_kernel[(TE.numel(), N // BN)](a, b_enk, c, TE, TS, TM, K, N,
                                                BM, BN, BK, num_warps=w, num_stages=st)
    return c


_DW = (64, 64, 64, 4, 3)


@triton.jit
def _dw_kernel(A, B, C, ROW0, ROWN, N1: tl.constexpr, N2: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    e = tl.program_id(0)
    p1 = tl.program_id(1)
    p2 = tl.program_id(2)
    r0 = tl.load(ROW0 + e)
    nr = tl.load(ROWN + e)
    r1 = p1 * BM + tl.arange(0, BM)
    r2 = p2 * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, nr, BK):
        rk = k0 + tl.arange(0, BK)
        mk = rk < nr
        a = tl.load(A + (r0 + rk[:, None]) * N1 + r1[None, :], mask=mk[:, None], other=0.0)
        b = tl.load(B + (r0 + rk[:, None]) * N2 + r2[None, :], mask=mk[:, None], other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
    tl.store(C + e.to(tl.int64) * (N1 * N2) + r1[:, None] * N2 + r2[None, :], acc.to(tl.bfloat16))


def grouped_dw(a, b, row0, rown, E):
    N1 = a.shape[1]; N2 = b.shape[1]
    BM, BN, BK, w, st = _DW
    if N1 % BM or N2 % BN:
        return None
    c = torch.empty(E, N1, N2, device=a.device, dtype=a.dtype)
    _dw_kernel[(E, N1 // BM, N2 // BN)](a, b, c, row0, rown, N1, N2,
                                        BM, BN, BK, num_warps=w, num_stages=st)
    return c


@triton.jit
def _grouped_gemm_scatter_kernel(A, B, OUT, TOK, TE, TS, TM, K: tl.constexpr, N: tl.constexpr,
                                 BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    t = tl.program_id(0)
    pid_n = tl.program_id(1)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    Bb = B + e.to(tl.int64) * (K * N)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        b = tl.load(Bb + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tok = tl.load(TOK + rm, mask=mask_m, other=0)
    tl.atomic_add(OUT + tok[:, None] * N + rn[None, :], acc, mask=mask_m[:, None])


def grouped_gemm_scatter(a, b_enk, tok, tile_map, n_rows_out):
    TE, TS, TM = tile_map
    M, K = a.shape
    N = b_enk.shape[2]
    BM, BN, BK, w, st = _GG
    if N % BN or K % BK:
        return None
    out = torch.zeros(n_rows_out, N, device=a.device, dtype=torch.float32)
    _grouped_gemm_scatter_kernel[(TE.numel(), N // BN)](a, b_enk, out, tok, TE, TS, TM, K, N,
                                                        BM, BN, BK, num_warps=w, num_stages=st)
    return out


# ───────────────────── radial NormSiLU, fused into the GEMM epilogue ─────────────────────
# r^p * SiLU(g/r) * up, computed in ONE pass over the GEMM output.
#
# The plain epilogue above tiles N at BN=256 and cannot do this: r is an RMS over ALL I gate
# columns, so no single N-tile has the data. Radial therefore fell back to a separate _glu_fwd
# pass, measured at 1.222 ms per call at N=65536/I=768 -- almost exactly the DRAM round trip
# (1.2 GB read of gate_up + 0.6 GB write of inter at ~1.5 TB/s). Pure bandwidth, so only fusing
# removes it.
#
# This kernel gives ONE program the whole gate row by looping the N-tiles internally:
#   phase 1  per n-tile: gate GEMM -> store to GU, accumulate sum(g^2)      (r needs all of I)
#   phase 2  per n-tile: up GEMM, reload gate from GU (L2-hot), apply, store
# Same GEMM FLOPs as before; the 1.8 GB DRAM round trip becomes an L2 re-read of the gate.
_RBM, _RBN, _RBK, _RWARPS, _RSTAGES = 32, 256, 64, 8, 3


@triton.jit
def _gate_up_radial_kernel(X, W, GU, IT, TE, TS, TM, ALPHA,
                           H: tl.constexpr, I: tl.constexpr, EPS: tl.constexpr,
                           WRITE_GU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr):
    t = tl.program_id(0)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    mask_m = tl.arange(0, BM) < mm
    Wb = W + e.to(tl.int64) * (2 * I * H)

    # ---- phase 1: gate GEMM over every N tile, keeping only the running sum of squares.
    ss = tl.zeros((BM,), tl.float32)
    for n0 in range(0, I, BN):
        rn = n0 + tl.arange(0, BN)
        ag = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, H, BK):
            rk = k0 + tl.arange(0, BK)
            x = tl.load(X + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
            wg = tl.load(Wb + rn[:, None] * H + rk[None, :])
            ag = tl.dot(x, tl.trans(wg), ag)
        ss += tl.sum(ag * ag, axis=1)
        # the gate half of GU is written here regardless: the backward needs it, and phase 2
        # reads it back from L2 instead of recomputing the GEMM
        tl.store(GU + rm[:, None] * (2 * I) + rn[None, :], ag.to(GU.dtype.element_ty),
                 mask=mask_m[:, None])

    r = tl.sqrt(ss / I + EPS)                       # RMS over the FULL gate row
    aa = tl.load(ALPHA + rm, mask=mask_m, other=0.0).to(tl.float32)
    p = 1.0 / (1.0 + tl.exp(-aa))                   # code 8: p = sigmoid(theta)
    rp = tl.exp(p * tl.log(r))                      # gain r^p, bounded by [1, r]

    # ---- phase 2: up GEMM, then apply r^p * SiLU(g/r) * up
    for n0 in range(0, I, BN):
        rn = n0 + tl.arange(0, BN)
        au = tl.zeros((BM, BN), tl.float32)
        for k0 in range(0, H, BK):
            rk = k0 + tl.arange(0, BK)
            x = tl.load(X + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
            wu = tl.load(Wb + (I + rn[:, None]) * H + rk[None, :])
            au = tl.dot(x, tl.trans(wu), au)
        ag = tl.load(GU + rm[:, None] * (2 * I) + rn[None, :], mask=mask_m[:, None],
                     other=0.0).to(tl.float32)
        gn = ag / r[:, None]
        f = gn * tl.sigmoid(gn)
        act = rp[:, None] * f
        tl.store(IT + rm[:, None] * I + rn[None, :], (act * au).to(IT.dtype.element_ty),
                 mask=mask_m[:, None])
        if WRITE_GU:
            tl.store(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), au.to(GU.dtype.element_ty),
                     mask=mask_m[:, None])


def radial_supported(hidden, gate_up_proj, codes):
    """Radial (code 8) with the fused epilogue. Same shape constraints as gemm_supported, plus a
    tile map built at _RBM -- the caller must pass the matching map."""
    I = gate_up_proj.shape[1] // 2
    return (tiles_supported(hidden) and hidden.dtype is torch.bfloat16
            and gate_up_proj.is_contiguous()
            and I % _RBN == 0 and gate_up_proj.shape[2] % _RBK == 0
            and len(set(codes)) == 1 and codes[0] == 8)


def fused_gate_up_radial(x_s, gate_up_proj, tile_map, row_alpha, eps=1e-6, want_gu=True):
    """(gu, inter) with radial NormSiLU applied. `row_alpha` is theta PER ROW, not per expert."""
    TE, TS, TM = tile_map
    M, H = x_s.shape
    I = gate_up_proj.shape[1] // 2
    it = torch.empty(M, I, device=x_s.device, dtype=x_s.dtype)
    gu = torch.empty(M, 2 * I, device=x_s.device, dtype=x_s.dtype)
    _gate_up_radial_kernel[(TE.numel(),)](
        x_s, gate_up_proj, gu, it, TE, TS, TM, row_alpha.contiguous(), H, I, eps, want_gu,
        _RBM, _RBN, _RBK, num_warps=_RWARPS, num_stages=_RSTAGES)
    return gu, it
