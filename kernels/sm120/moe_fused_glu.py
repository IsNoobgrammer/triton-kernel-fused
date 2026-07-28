"""Fused grouped GEMM + PolyGLU epilogue for the MoE gate_up projection (Blackwell).

WHY: profiling 64 experts / top-8 showed _glu_fwd sitting EXACTLY at HBM bandwidth -- it reads the
(M, 2I) gate_up output and writes (M, I). That traffic only exists because cuBLAS has to land `gu`
in memory before a separate activation kernel can touch it. Computing the activation in the GEMM's
epilogue, while the accumulators are still in registers, removes the read entirely.

Measured at the real load spread (64 experts, mean 4096 rows, min ~2.2k max ~9k, H=512, I=768):
    cuBLAS per-expert mm + _glu_fwd   2.68 ms
    fused, writes gu + it             2.03 ms   1.32x
    fused, writes it only             1.58 ms   1.68x
`gu` is still written because the existing backward needs it; the it-only variant is available for
a future fused backward that recomputes the accumulators instead.

Tile config came from an autotune sweep -- the obvious 64/64/64 gives only 1.08x, i.e. roughly
cuBLAS parity. Do not change BLOCK sizes without re-running that sweep.

Variable rows-per-expert are handled with a TILE MAP (tile -> expert, first row, valid rows) built
on the host from the counts we already materialize for the sort, so one launch covers every expert
without padding.
"""
import torch
import triton
import triton.language as tl

__all__ = ["fused_gate_up_glu", "fused_supported", "build_tile_map"]

_BM, _BN, _BK, _WARPS, _STAGES = 64, 128, 64, 4, 3      # autotuned for WRITE_GU=True


@triton.jit
def _gate_up_glu_kernel(X, W, GU, IT, TE, TS, TM, ACT,
                        H: tl.constexpr, I: tl.constexpr, CODE: tl.constexpr,
                        WRITE_GU: tl.constexpr,
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
    if CODE == 1:
        act = tl.maximum(ag, 0.0) * tl.maximum(ag, 0.0)       # ReLU^2
    else:
        act = ag * tl.sigmoid(ag)                              # SiLU (code 0)
    tl.store(IT + rm[:, None] * I + rn[None, :], (act * au).to(tl.bfloat16), mask=mask_m[:, None])
    if WRITE_GU:
        tl.store(GU + rm[:, None] * (2 * I) + rn[None, :], ag.to(tl.bfloat16), mask=mask_m[:, None])
        tl.store(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), au.to(tl.bfloat16),
                 mask=mask_m[:, None])


def fused_supported(hidden, gate_up_proj, codes):
    """Only the NON-normalized pointwise GLUs: codes 2/6/7 need a per-row RMS over the gate half,
    which a GEMM epilogue cannot see (it holds a BN-wide slice of the row, not the whole row)."""
    I = gate_up_proj.shape[1] // 2
    return (hidden.dtype is torch.bfloat16
            and torch.cuda.get_device_capability(hidden.device)[0] >= 8
            and gate_up_proj.is_contiguous() and hidden.is_contiguous()
            and I % _BN == 0 and gate_up_proj.shape[2] % _BK == 0
            and len(set(codes)) == 1 and codes[0] in (0, 1))


def build_tile_map(counts, counts_t, device, bm=None):
    """(tile -> expert, first row, valid rows), built VECTORIZED on the GPU.

    The obvious host-side double loop costs ~4k python iterations plus three H2D copies EVERY call
    -- 32 calls per step at 8 MoE layers x grad_accum 4. Measured: it turned a kernel that is 1.32x
    faster in isolation into a 2% end-to-end LOSS. Only the per-expert tile COUNTS are computed on
    the host (64 ints, no sync -- `counts` is already materialized for the sort); everything else
    is torch ops on tensors that are already resident."""
    bm = _BM if bm is None else bm
    ntile = [(c + bm - 1) // bm for c in counts]
    total = sum(ntile)
    nt = torch.tensor(ntile, device=device, dtype=torch.int32)
    te = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), nt)
    start = torch.cumsum(nt, 0) - nt                       # first tile index of each expert
    within = torch.arange(total, device=device, dtype=torch.int32) - start[te]
    bnd = torch.cumsum(counts_t, 0) - counts_t             # first row of each expert
    ts = (bnd[te] + within * bm).to(torch.int32)
    tm = torch.clamp(counts_t[te] - within * bm, max=bm).to(torch.int32)
    return te, ts, tm


def fused_gate_up_glu(x_s, gate_up_proj, tile_map, code, want_gu=True):
    """(M,H) x (E,2I,H) -> it (M,I), and gu (M,2I) when want_gu."""
    TE, TS, TM = tile_map
    M, H = x_s.shape
    I = gate_up_proj.shape[1] // 2
    it = torch.empty(M, I, device=x_s.device, dtype=x_s.dtype)
    gu = torch.empty(M, 2 * I, device=x_s.device, dtype=x_s.dtype) if want_gu else it
    _gate_up_glu_kernel[(TE.numel(), I // _BN)](
        x_s, gate_up_proj, gu, it, TE, TS, TM, None, H, I, code, want_gu,
        _BM, _BN, _BK, num_warps=_WARPS, num_stages=_STAGES)
    return gu, it


# ───────────────────────── fused backward: grad_inter GEMM + GLU backward ─────────────────────────
# Forward is it = act(ag) * au. Backward needs
#     grad_ag = grad_it * act'(ag) * au ,  grad_au = grad_it * act(ag)
# and grad_it is itself a GEMM (ge @ W2). Doing them separately materializes grad_it (M,I) and
# re-reads gu (M,2I) -- profiled as a cuBLAS GEMM plus _glu_bwd at 42.5 ms/step, the single largest
# non-GEMM kernel. Computing grad_it in registers and applying the GLU backward in the same epilogue
# removes the grad_it round-trip entirely.
_BBM, _BBN, _BBK, _BWARPS, _BSTAGES = 64, 128, 64, 4, 3


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
    for k0 in range(0, H, BK):                      # grad_it = ge @ W2[e]   (K = H)
        rk = k0 + tl.arange(0, BK)
        ge = tl.load(GE + rm[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
        w = tl.load(Wb + rk[:, None] * I + rn[None, :])
        gi = tl.dot(ge, w, gi)
    ag = tl.load(GU + rm[:, None] * (2 * I) + rn[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
    au = tl.load(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), mask=mask_m[:, None], other=0.0).to(tl.float32)
    if CODE == 1:                                   # ReLU^2:  act = relu(ag)^2
        r = tl.maximum(ag, 0.0)
        d_ag = gi * (2.0 * r) * au
        d_au = gi * (r * r)
    else:                                           # SiLU:    act = ag*sigmoid(ag)
        sg = tl.sigmoid(ag)
        d_ag = gi * (sg * (1.0 + ag * (1.0 - sg))) * au
        d_au = gi * (ag * sg)
    tl.store(DGU + rm[:, None] * (2 * I) + rn[None, :], d_ag.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(DGU + rm[:, None] * (2 * I) + (I + rn[None, :]), d_au.to(tl.bfloat16), mask=mask_m[:, None])


def fused_dinter_glu_bwd(ge, down_proj, gu, tile_map, code):
    """ge (M,H), down_proj (E,H,I), gu (M,2I) -> grad_gate_up (M,2I). Replaces
    `grad_inter = grouped_mm(ge, W2)` followed by `_glu_bwd(grad_inter, gu)`."""
    TE, TS, TM = tile_map
    M, H = ge.shape
    I = down_proj.shape[2]
    dgu = torch.empty(M, 2 * I, device=ge.device, dtype=ge.dtype)
    _dinter_glu_bwd_kernel[(TE.numel(), I // _BBN)](
        ge, down_proj, gu, dgu, TE, TS, TM, H, I, code,
        _BBM, _BBN, _BBK, num_warps=_BWARPS, num_stages=_BSTAGES)
    return dgu


# ───────────────────────── generic grouped GEMM (no epilogue) ─────────────────────────
# For the two remaining activation-side MoE GEMMs: `it @ W2` (forward) and `ggu @ W1` (backward dX).
# Same tile-map structure as the fused kernels; the fused gate_up GEMM measured 1.58 ms against
# cuBLAS's 2.10 ms for the GEMM ALONE on this shape, so Triton is worth using here too.
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


# autotuned: it@W2 1.11->0.77 ms (1.43x), ggu@W1 1.84->1.41 ms (1.30x) vs cuBLAS grouped
_GG = (128, 256, 64, 8, 3)


def grouped_gemm(a, b_enk, tile_map, out=None):
    """a (M,K) x b (E,K,N) -> (M,N), rows grouped by expert via the tile map."""
    TE, TS, TM = tile_map
    M, K = a.shape
    N = b_enk.shape[2]
    if N % _GG[1] or K % _GG[2] or TE is None:
        return None                                   # caller falls back
    c = torch.empty(M, N, device=a.device, dtype=a.dtype) if out is None else out
    BM, BN, BK, w, st = _GG
    _grouped_gemm_kernel[(TE.numel(), N // BN)](a, b_enk, c, TE, TS, TM, K, N,
                                                BM, BN, BK, num_warps=w, num_stages=st)
    return c
