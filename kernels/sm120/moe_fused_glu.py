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


def build_tile_map(counts, counts_t, device):
    """(tile -> expert, first row, valid rows), built VECTORIZED on the GPU.

    The obvious host-side double loop costs ~4k python iterations plus three H2D copies EVERY call
    -- 32 calls per step at 8 MoE layers x grad_accum 4. Measured: it turned a kernel that is 1.32x
    faster in isolation into a 2% end-to-end LOSS. Only the per-expert tile COUNTS are computed on
    the host (64 ints, no sync -- `counts` is already materialized for the sort); everything else
    is torch ops on tensors that are already resident."""
    ntile = [(c + _BM - 1) // _BM for c in counts]
    total = sum(ntile)
    nt = torch.tensor(ntile, device=device, dtype=torch.int32)
    te = torch.repeat_interleave(torch.arange(len(counts), device=device, dtype=torch.int32), nt)
    start = torch.cumsum(nt, 0) - nt                       # first tile index of each expert
    within = torch.arange(total, device=device, dtype=torch.int32) - start[te]
    bnd = torch.cumsum(counts_t, 0) - counts_t             # first row of each expert
    ts = (bnd[te] + within * _BM).to(torch.int32)
    tm = torch.clamp(counts_t[te] - within * _BM, max=_BM).to(torch.int32)
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
