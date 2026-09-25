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


def build_tile_map(counts, counts_t, device, bm=None, m_rows=None):
    """(TE, TS, TM): expert, first row and row count of every BM-row tile of the expert-sorted rows.

    counts=None builds it entirely on the device, with NO host sync: the grid is sized by the upper
    bound ceil(M/bm) + E (each expert pads at most one partial tile), and the tiles past the real
    count get TM = 0 with TE clamped to a valid expert. Every kernel masks rows by TM, so those
    tiles load and store nothing; the first `total` entries equal the host-built map exactly.
    """
    bm = _BM if bm is None else bm
    if counts is None:
        E = counts_t.numel()
        nt = (counts_t + (bm - 1)) // bm
        t_end = torch.cumsum(nt, 0)
        t = torch.arange((m_rows + bm - 1) // bm + E, device=device)
        te = torch.searchsorted(t_end, t, right=True)
        valid = te < E
        te = te.clamp_max(E - 1)
        within = t - (t_end - nt)[te]
        ts = (torch.cumsum(counts_t, 0) - counts_t)[te] + within * bm
        tm = torch.where(valid, torch.clamp(counts_t[te] - within * bm, max=bm), 0)
        return te.to(torch.int32), torch.where(valid, ts, 0).to(torch.int32), tm.to(torch.int32)
    ntile = [(c + bm - 1) // bm for c in counts]
    total = sum(ntile)
    nt = ((counts_t + (bm - 1)) // bm).to(torch.int32)
    # output_size: without it repeat_interleave reads nt.sum() back to the host (a GPU drain)
    te = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), nt,
                                 output_size=total)
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
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


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


# Grouped weight gradient  out[e] = a[rows_e]^T @ b[rows_e]  (the MoE dW GEMMs), one launch, no host
# sync. Replaces torch._grouped_mm for the 2D x 2D (K-grouped) case, which on sm120 is a HOST LOOP
# of E cuBLAS GEMMs: it reads the offsets back (one sync per call, 80 per board step) and each
# per-expert GEMM is ~24 output tiles, so the GPU runs a quarter full (163-210 TFLOPS).
#
# Work = (expert chunk of <= CH rows) x (BM x BN output tile), all in one grid, heaviest chunks
# first. An expert with more than CH rows is SPLIT: each chunk writes an fp32 partial and
# _wg_reduce sums them in chunk order, so the result is deterministic (no atomics). Experts with
# no rows get one empty chunk, which stores zeros -- the same as torch._grouped_mm.
# swept on the RTX PRO 6000 at the board shapes (bench_grouped_wgrad.py --sweep): grad_down 1.16 ms vs
# torch 1.89 (1.63x), grad_gate_up 2.17 vs 2.95 (1.36x), L0 gate_up 2.30 vs 2.37. Fixed, not autotuned.
_WG = dict(CH=16384, BM=128, BN=128, BK=32, num_warps=4, num_stages=4)


@triton.jit
def _wg_kernel(A, B, C, P, IT_E, IT_S, IT_N, IT_SLOT, ORDER, N1, N2, sa, sb,
               NT2: tl.constexpr, NTILE: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    item = tl.load(ORDER + pid // NTILE)
    e = tl.load(IT_E + item)
    if e < 0:
        return
    tile = pid % NTILE
    r1 = (tile // NT2) * BM + tl.arange(0, BM)
    r2 = (tile % NT2) * BN + tl.arange(0, BN)
    s0 = tl.load(IT_S + item).to(tl.int64)
    n = tl.load(IT_N + item)
    acc = tl.zeros((BM, BN), tl.float32)
    rk = tl.arange(0, BK)
    for k0 in tl.range(0, n, BK):
        mk = (k0 + rk) < n
        rows = s0 + k0 + rk
        a = tl.load(A + rows[:, None] * sa + r1[None, :], mask=mk[:, None], other=0.0)
        b = tl.load(B + rows[:, None] * sb + r2[None, :], mask=mk[:, None], other=0.0)
        acc = tl.dot(tl.trans(a), b, acc)
    slot = tl.load(IT_SLOT + item)
    off = r1[:, None] * N2 + r2[None, :]
    if slot < 0:
        tl.store(C + e.to(tl.int64) * N1 * N2 + off, acc.to(C.dtype.element_ty))
    else:
        tl.store(P + slot.to(tl.int64) * N1 * N2 + off, acc)


@triton.jit
def _wg_reduce(P, C, FIRST, NCH, NN, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    nch = tl.load(NCH + e)
    if nch <= 1:
        return
    first = tl.load(FIRST + e).to(tl.int64)
    o = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = o < NN
    acc = tl.zeros((BLOCK,), tl.float32)
    for j in range(0, nch):                      # chunk order: deterministic
        acc += tl.load(P + (first + j) * NN + o, mask=m, other=0.0)
    tl.store(C + e.to(tl.int64) * NN + o, acc.to(C.dtype.element_ty), mask=m)


def grouped_wgrad(a, b, offs, cfg=None):
    """out (E, N1, N2) = per expert a[s:t]^T @ b[s:t], with offs the int32 END row offsets exactly as
    torch._grouped_mm(a.t(), b, offs=offs) takes them. a (M, N1), b (M, N2), same dtype. Returns
    None when the shape is not tileable (caller falls back)."""
    c = dict(_WG, **(cfg or {}))
    CH, BM, BN, BK = c["CH"], c["BM"], c["BN"], c["BK"]
    M, N1 = a.shape
    N2 = b.shape[1]
    E = offs.numel()
    if N1 % BM or N2 % BN or a.stride(1) != 1 or b.stride(1) != 1:
        return None
    dev = a.device
    end = offs.to(torch.int64)
    cnt = end - torch.cat((end.new_zeros(1), end[:-1]))
    nch = ((cnt + CH - 1) // CH).clamp_min(1)             # an empty expert still stores its zeros
    cend = torch.cumsum(nch, 0)
    NI = (M + CH - 1) // CH + E                            # static upper bound on chunk count
    i = torch.arange(NI, device=dev)
    ie = torch.searchsorted(cend, i, right=True)
    valid = ie < E
    ie = ie.clamp_max(E - 1)
    j = i - (cend - nch)[ie]
    it_s = (end - cnt)[ie] + j * CH
    it_n = torch.where(valid, (cnt[ie] - j * CH).clamp(0, CH), 0)
    it_slot = torch.where(valid & (nch[ie] > 1), i, -1)
    it_e = torch.where(valid, ie, -1)
    order = torch.argsort(it_n, descending=True, stable=True)   # heaviest chunks launch first
    out = torch.empty(E, N1, N2, device=dev, dtype=a.dtype)
    part = torch.empty(NI, N1, N2, device=dev, dtype=torch.float32)
    nt2 = N2 // BN
    ntile = (N1 // BM) * nt2
    _wg_kernel[(NI * ntile,)](a, b, out, part, it_e.to(torch.int32), it_s, it_n.to(torch.int32),
                              it_slot.to(torch.int32), order, N1, N2, a.stride(0), b.stride(0),
                              NT2=nt2, NTILE=ntile, BM=BM, BN=BN, BK=BK,
                              num_warps=c["num_warps"], num_stages=c["num_stages"])
    RB = 1024
    _wg_reduce[(E, triton.cdiv(N1 * N2, RB))](part, out, (cend - nch), nch, N1 * N2, BLOCK=RB,
                                              num_warps=4)
    return out


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
_RBM, _RBN, _RBK, _RWARPS, _RSTAGES = 16, 1024, 32, 8, 1


@triton.jit
def _gate_up_radial_kernel(X, W, GU, IT, TE, TS, TM, ALPHA,
                           H: tl.constexpr, I: tl.constexpr, EPS: tl.constexpr,
                           WRITE_GU: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                           BK: tl.constexpr):
    """One program owns a row block at FULL gate width, so r is available without any reload.

    The first version of this kernel tiled N and looped twice (gate pass, then up pass reloading
    the gate from GU). That was measured SLOWER than not fusing at all -- it traded a 1.8 GB DRAM
    round trip for a 0.6 GB reload plus a second pass of X loads plus a 3x smaller grid. Keeping
    the gate resident is the only arrangement that removes traffic instead of moving it.

    BN spans all of I (padded to a power of two and masked), so BM must stay small: the two fp32
    accumulators are 2 * BM * BN * 4 bytes of registers.
    """
    t = tl.program_id(0)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    mask_n = rn < I
    mask = mask_m[:, None] & mask_n[None, :]
    Wb = W + e.to(tl.int64) * (2 * I * H)

    # TWO k-loops, not one. A single loop needs both weight tiles resident, and at BN=1024 that
    # is 2 x BK x BN x 2 bytes of shared memory -- 264 KB against a 101 KB limit, which is exactly
    # how the one-loop version failed. Split, only one weight tile is live at a time, and `ag`
    # stays in REGISTERS across both loops, so r still needs no reload. X is re-read, but it is
    # BM x H and tiny next to the weights.
    ag = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, H, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
        wg = tl.load(Wb + rn[:, None] * H + rk[None, :], mask=mask_n[:, None], other=0.0)
        ag = tl.dot(x, tl.trans(wg), ag)
    au = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, H, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
        wu = tl.load(Wb + (I + rn[:, None]) * H + rk[None, :], mask=mask_n[:, None], other=0.0)
        au = tl.dot(x, tl.trans(wu), au)

    # r over the FULL gate row -- masked lanes contributed 0 and must not count toward the mean
    ss = tl.sum(tl.where(mask, ag * ag, 0.0), axis=1)
    r = tl.sqrt(ss / I + EPS)
    aa = tl.load(ALPHA + rm, mask=mask_m, other=0.0).to(tl.float32)
    p = 1.0 / (1.0 + tl.exp(-aa))                   # code 8: p = sigmoid(theta)
    rp = tl.exp(p * tl.log(r))
    gn = ag / r[:, None]
    act = rp[:, None] * (gn * tl.sigmoid(gn))
    tl.store(IT + rm[:, None] * I + rn[None, :], (act * au).to(IT.dtype.element_ty), mask=mask)
    if WRITE_GU:
        tl.store(GU + rm[:, None] * (2 * I) + rn[None, :], ag.to(GU.dtype.element_ty), mask=mask)
        tl.store(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), au.to(GU.dtype.element_ty),
                 mask=mask)


def radial_supported(hidden, gate_up_proj, codes):
    """Radial (code 8) with the fused epilogue. OFF unless BIBO_RADIAL_FUSED=1.

    MEASURED LOSS at BiBo's shapes (H=512, I=768), all three designs, against 195.4k tok/s for
    the unfused path:
        v1  tile N, reload gate from GU        192.5k   traded a DRAM round trip for a reload
        v2  full width, one k-loop             would not launch: 264 KB smem vs a 101 KB limit
        v3  full width, two k-loops            177.0k

    The blocker is that I=768 is NOT a power of two, so tl.arange forces BN=1024 and 25% of every
    tensor-core lane is masked -- roughly 33% extra GEMM work, against the ~0.8 ms of activation
    traffic the fusion removes. Fusing radial only pays when I is a power of two (512 or 1024);
    at 768 it cannot. Kept behind a flag so the finding stays reproducible.
    """
    import os
    if os.environ.get("BIBO_RADIAL_FUSED") != "1":
        return False
    I = gate_up_proj.shape[1] // 2
    return (tiles_supported(hidden) and hidden.dtype is torch.bfloat16
            and gate_up_proj.is_contiguous()
            and I <= _RBN and gate_up_proj.shape[2] % _RBK == 0
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


# ───────────────────── deterministic combine: gather instead of atomic scatter ─────────────────────
# Every token owns exactly K expert-sorted rows. Scattering them with atomic_add sums the K
# contributions in whatever order the SMs finish -- run-to-run different fp32 rounding. Here each
# token GATHERS its K rows by the inverse sort permutation and sums them in slot order j = 0..K-1,
# so the result is bitwise reproducible. Same bytes read, and plain stores instead of atomics.

@triton.jit
def _combine_gather_kernel(ROWS, W, INV, OUT, NT, H, s_r, s_o, K: tl.constexpr, HAS_W: tl.constexpr,
                           BT: tl.constexpr, BH: tl.constexpr):
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    mt = t < NT
    m = mt[:, None] & (h < H)[None, :]
    acc = tl.zeros((BT, BH), tl.float32)
    for j in tl.static_range(K):
        r = tl.load(INV + t.to(tl.int64) * K + j, mask=mt, other=0)
        x = tl.load(ROWS + r[:, None] * s_r + h[None, :], mask=m, other=0.0).to(tl.float32)
        if HAS_W:
            x = x * tl.load(W + r, mask=mt, other=0.0).to(tl.float32)[:, None]
        acc += x
    tl.store(OUT + t.to(tl.int64)[:, None] * s_o + h[None, :], acc.to(OUT.dtype.element_ty), mask=m)


def inverse_order(order):
    """order[r] = flat slot (token*K + j) of expert-sorted row r  ->  inv[slot] = r."""
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), device=order.device, dtype=order.dtype)
    return inv


def combine_gather(rows, inv, n_tok, k, w=None, out=None, out_dtype=torch.float32):
    """out[t] = sum_j rows[inv[t*k + j]] * (w[inv[t*k + j]] if w is not None else 1), j in order."""
    H = rows.shape[1]
    out = torch.empty(n_tok, H, device=rows.device, dtype=out_dtype) if out is None else out
    BT, BH = 32, 128
    _combine_gather_kernel[(triton.cdiv(n_tok, BT), triton.cdiv(H, BH))](
        rows, rows if w is None else w, inv, out, n_tok, H, rows.stride(0), out.stride(0),
        K=k, HAS_W=w is not None, BT=BT, BH=BH, num_warps=4)
    return out


def grouped_gemm_gather(a, b_enk, inv, tile_map, n_tok, k):
    """Deterministic grouped_gemm_scatter: the per-row GEMM result goes to an fp32 row buffer, then
    combine_gather sums each token's k rows in slot order. Costs one extra fp32 (M, N) write+read."""
    rows = torch.empty(a.shape[0], b_enk.shape[2], device=a.device, dtype=torch.float32)
    if grouped_gemm(a, b_enk, tile_map, out=rows) is None:
        return None
    return combine_gather(rows, inv, n_tok, k)
