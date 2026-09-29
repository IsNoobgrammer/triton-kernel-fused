"""MoE experts with MXFP8 expert GEMMs: the same call and returns as moe_per_expert.

    moe_fp8(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)

Everything from the router weights to the weighted sum:
  fwd    xq = MXFP8(hidden)                       (unsorted tokens, blocks along H)
         GU = xq[st] @ Wgu^T                      F1, fp8, row gather inside the GEMM
         inter = radial(G) * U  (+ MXFP8 copy)    fused: the act kernel writes bf16 AND e4m3 + scales
         EO = inter_q @ Wdn^T                     F3, fp8
         out = sum_k w_k * EO                     fp32 combine (router weights fp32), deterministic
  bwd    dO, dw (+ MXFP8 copy of dO)              fused into combine_bwd
         d_inter = dO_q @ Wdn                     B3, fp8
         dGU, dtheta (+ MXFP8 copy of dGU)        fused into the radial backward
         dx rows = dGU_q @ Wgu                    B6, fp8, then the deterministic k-way sum
         dWdn = dO^T @ inter, dWgu = dGU^T @ x    B2 / B5: bf16 grouped wgrad (phase 2 = fp8)
Weights: MXFP8 from the fp32 master, 2D 32x32 blocks, cached per optimizer step (mxfp8.quant_weight).

FUSED = False falls back to separate quantize passes (and the generic activation kernels).
STATS: set moe_fp8.STATS = {} to collect (flushed %, saturated %) of every fp8 operand per call.
"""
import importlib

import torch
import triton
import triton.language as tl

# importlib, not "from kernels.sm75 import moe": the package re-exports a FUNCTION named moe
K75 = importlib.import_module("kernels.sm75.moe")
MX = importlib.import_module("kernels.sm120.mxfp8")
FG = importlib.import_module("kernels.sm120.moe_fused_glu")

STATS = None
FUSED = True
DX_ROWS = torch.bfloat16  # B6 row buffer before the k-way sum: bf16 = -0.55 ms/layer (fp32 sum over k)
_EPS = K75._NS_EPS


def _stat(tag, x, q, s):
    if STATS is not None:
        STATS.setdefault(tag, []).append(MX.qstats(x, q, s))


WGRAD8 = False       # B2 / B5 in MXFP8 (token-axis quant inside the GEMM). OFF: correct (dW err 5.4e-2 -> 6.6e-2) but the layer
#                      goes 16.8 -> 29.4 ms -- fp8 MMA needs K(tokens)-major operands and ours are token-ROW-major


def _wgrad(a, b, offs, counts_t, acc, b_rows=None):
    """K75._wgrad contract: acc = the fp32 master Parameter -> accumulate into acc.grad, return None."""
    if not WGRAD8:
        return K75._wgrad(a, b, offs, acc=acc, b_rows=b_rows)
    if acc is not None:
        fresh = acc.grad is None
        buf = torch.empty_like(acc) if fresh else acc.grad
        MX.grouped_wgrad(a, b, counts_t, out=buf, accumulate=not fresh, b_rows=b_rows)
        if fresh:
            acc.grad = buf
        return None
    return MX.grouped_wgrad(a, b, counts_t, b_rows=b_rows)


def _q(x, tag):
    q, s = MX.quant_rows(x.contiguous())
    _stat(tag, x, q, s)
    return q, s


def _wstats(w, tag, qs):
    if STATS is not None:
        q, s = qs
        _stat(tag, w.reshape(-1, w.shape[-1]).float(), q.reshape(-1, q.shape[-1]), s.reshape(-1, s.shape[-1]))


# ------------------------------------------------------------------ fused producers (fp8 epilogues)
@triton.jit
def _q32(v, NB: tl.constexpr):
    """(NB*32,) fp32 -> e4m3 values + e8m0 scale per 32 (rounded up: nothing exceeds 448)."""
    vb = tl.reshape(v, (NB, 32))
    amax = tl.maximum(tl.max(tl.abs(vb), axis=1), 1e-30)
    e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
    q = tl.reshape(vb * tl.exp2(-e)[:, None], (NB * 32,))
    return q.to(tl.float8e4nv), (e + 127.0).to(tl.uint8)


@triton.jit
def _radial_fwd_q_kernel(GU, ACT, ALPHA, OUT, Q, S, I, EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    """radial r^p * SiLU(g/r) * u for codes 8 / 10 (K75._glu_fwd_rowloop_kernel math), writing the
    bf16 row AND its MXFP8 copy (blocks of 32 along I): the F3 input with no separate quant pass."""
    row = tl.program_id(0)
    at = tl.load(ACT + row)
    aa = tl.load(ALPHA + row).to(tl.float32)
    base = GU + row.to(tl.int64) * 2 * I
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        g = tl.load(base + i0 + tl.arange(0, BLOCK_I)).to(tl.float32)
        acc += g * g
    r = tl.sqrt(tl.sum(acc) / I + EPS)
    p8 = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    rp = tl.exp(p8 * tl.log(r))
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        g = tl.load(base + offs).to(tl.float32)
        u = tl.load(base + I + offs).to(tl.float32)
        z = g / r
        vb = (rp * (z * (1.0 / (1.0 + tl.exp(-z)))) * u).to(OUT.dtype.element_ty)
        tl.store(OUT + row.to(tl.int64) * I + offs, vb)
        q, sc = _q32(vb.to(tl.float32), BLOCK_I // 32)          # quantize exactly what is stored
        tl.store(Q + row.to(tl.int64) * I + offs, q)
        tl.store(S + row.to(tl.int64) * (I // 32) + i0 // 32 + tl.arange(0, BLOCK_I // 32), sc)


@triton.jit
def _radial_bwd_q_kernel(GO, GU, ACT, ALPHA, GGU, GQ, GS, DA, I, EPS: tl.constexpr,
                         WANT_AP: tl.constexpr, BLOCK_I: tl.constexpr):
    """K75._glu_bwd_rowloop_kernel for codes 8 / 10, writing dGU (bf16) AND its MXFP8 copy along
    2I (the B6 input). Gate / up halves quantize separately; I % 32 == 0 so no block straddles."""
    row = tl.program_id(0)
    at = tl.load(ACT + row)
    aa = tl.load(ALPHA + row).to(tl.float32)
    gub = GU + row.to(tl.int64) * 2 * I
    gob = GO + row.to(tl.int64) * I
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        g = tl.load(gub + i0 + tl.arange(0, BLOCK_I)).to(tl.float32)
        acc += g * g
    r = tl.sqrt(tl.sum(acc) / I + EPS)
    p8 = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    lr8 = tl.log(r)
    rp = tl.exp(p8 * lr8)
    rpm1 = tl.exp((p8 - 1.0) * lr8)
    sa = tl.zeros([BLOCK_I], dtype=tl.float32)
    tt = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        go = tl.load(gob + offs).to(tl.float32)
        g = tl.load(gub + offs).to(tl.float32)
        u = tl.load(gub + I + offs).to(tl.float32)
        gn = g / r
        sig = 1.0 / (1.0 + tl.exp(-gn))
        f = gn * sig
        df = sig * (1.0 + gn * (1.0 - sig))
        gu_ = go * u
        sa += gu_ * df * gn
        tt += gu_ * f
    S_ = tl.sum(sa)
    T = tl.sum(tt)
    NB: tl.constexpr = BLOCK_I // 32
    ob = row.to(tl.int64) * 2 * I
    sb = row.to(tl.int64) * (2 * I // 32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        go = tl.load(gob + offs).to(tl.float32)
        g = tl.load(gub + offs).to(tl.float32)
        u = tl.load(gub + I + offs).to(tl.float32)
        gn = g / r
        sig = 1.0 / (1.0 + tl.exp(-gn))
        f = gn * sig
        df = sig * (1.0 + gn * (1.0 - sig))
        gu_ = go * u
        gg = (rpm1 * (gu_ * df - (gn / I) * (S_ - p8 * T))).to(GGU.dtype.element_ty)
        gup = (go * (rp * f)).to(GGU.dtype.element_ty)
        tl.store(GGU + ob + offs, gg)
        tl.store(GGU + ob + I + offs, gup)
        q, sc = _q32(gg.to(tl.float32), NB)
        tl.store(GQ + ob + offs, q)
        tl.store(GS + sb + i0 // 32 + tl.arange(0, NB), sc)
        q, sc = _q32(gup.to(tl.float32), NB)
        tl.store(GQ + ob + I + offs, q)
        tl.store(GS + sb + (I + i0) // 32 + tl.arange(0, NB), sc)
    if WANT_AP:
        tl.store(DA + row, tl.where(at == 10, 1.0 - p8 * p8, p8 * (1.0 - p8)) * rp * lr8 * T)


@triton.jit
def _combine_bwd_q_kernel(GO, EO, W, TOK, GEO, GQ, GS, GW, m, H: tl.constexpr, BLOCK_M: tl.constexpr):
    """K75._combine_bwd_kernel (dO = w * grad_out[token], dw = <grad_out, eo>) plus the MXFP8 copy
    of dO along H (the B3 input). H is a power of two, so one program row covers it whole."""
    offs_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_h = tl.arange(0, H)
    mm = offs_m < m
    tok = tl.load(TOK + offs_m, mask=mm, other=0)
    go = tl.load(GO + tok[:, None].to(tl.int64) * H + offs_h[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
    eo = tl.load(EO + offs_m[:, None].to(tl.int64) * H + offs_h[None, :], mask=mm[:, None], other=0.0).to(tl.float32)
    w = tl.load(W + offs_m, mask=mm, other=0.0).to(tl.float32)[:, None]
    ge = (go * w).to(GEO.dtype.element_ty)
    tl.store(GEO + offs_m[:, None].to(tl.int64) * H + offs_h[None, :], ge, mask=mm[:, None])
    tl.store(GW + offs_m, tl.sum(go * eo, axis=1), mask=mm)
    gb = tl.reshape(ge.to(tl.float32), (BLOCK_M, H // 32, 32))
    amax = tl.maximum(tl.max(tl.abs(gb), axis=2), 1e-30)
    e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
    q = tl.reshape(gb * tl.exp2(-e)[:, :, None], (BLOCK_M, H))
    tl.store(GQ + offs_m[:, None].to(tl.int64) * H + offs_h[None, :], q.to(tl.float8e4nv), mask=mm[:, None])
    tl.store(GS + offs_m[:, None].to(tl.int64) * (H // 32) + tl.arange(0, H // 32)[None, :],
             (e + 127.0).to(tl.uint8), mask=mm[:, None])


def _bi(I):
    return 256 if I % 256 == 0 else (64 if I % 64 == 0 else 32)


def radial_fwd_q(gu, row_act, row_alpha):
    M, I = gu.shape[0], gu.shape[1] // 2
    out = torch.empty(M, I, device=gu.device, dtype=gu.dtype)
    q = torch.empty(M, I, device=gu.device, dtype=MX.F8)
    s = torch.empty(M, I // 32, device=gu.device, dtype=torch.uint8)
    if M:
        _radial_fwd_q_kernel[(M,)](gu, row_act, row_alpha, out, q, s, I, _EPS, _bi(I), num_warps=4)
    return out, q, s


def radial_bwd_q(go, gu, row_act, row_alpha, want_ap):
    M, I = gu.shape[0], gu.shape[1] // 2
    ggu = torch.empty_like(gu)
    q = torch.empty(M, 2 * I, device=gu.device, dtype=MX.F8)
    s = torch.empty(M, 2 * I // 32, device=gu.device, dtype=torch.uint8)
    da = torch.empty(M, device=gu.device, dtype=torch.float32) if want_ap else ggu
    if M:
        _radial_bwd_q_kernel[(M,)](go.contiguous(), gu, row_act, row_alpha, ggu, q, s, da, I, _EPS,
                                   want_ap, _bi(I), num_warps=4)
    return ggu, q, s, (da if want_ap else None)


def combine_bwd_q(grad_out, eo, w, tok):
    m, H = eo.shape
    ge = torch.empty_like(eo)
    q = torch.empty(m, H, device=eo.device, dtype=MX.F8)
    s = torch.empty(m, H // 32, device=eo.device, dtype=torch.uint8)
    gw = torch.empty(m, device=eo.device, dtype=torch.float32)
    _combine_bwd_q_kernel[(triton.cdiv(m, 16),)](grad_out.contiguous(), eo, w, tok, ge, q, s, gw, m, H, 16,
                                                  num_warps=4)
    return ge, gw, q, s


# ------------------------------------------------------------------ autograd
class _MoEFP8(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, idx, wt, gate_up_proj, down_proj, act_codes, act_params=None):
        ctx.acc = (K75._acc_target(gate_up_proj), K75._acc_target(down_proj))
        wgu = MX.quant_weight(gate_up_proj)                 # from the fp32 MASTER, before any cast
        wdn = MX.quant_weight(down_proj)
        _wstats(gate_up_proj, "W gate_up", wgu["rc"]); _wstats(down_proj, "W down", wdn["rc"])
        hidden, = K75._amp_cast(hidden)
        hidden = hidden.contiguous()
        wt = wt.float()                     # router weights stay fp32 through the combine (bf16 path casts them)
        N, H = hidden.shape
        E = act_codes.shape[0]
        top_k = idx.shape[1]
        dev = hidden.device
        st, sw, order, _, _, counts_t = K75._sort_by_expert(idx, wt, E, host=False)
        M = idx.numel()
        offs = counts_t.cumsum(0).to(torch.int32)
        row_act = torch.repeat_interleave(act_codes, counts_t, output_size=M).to(torch.int32)
        row_alpha = row_expert = ap_shape = None
        if act_params is not None:
            ap32 = act_params.float().contiguous()
            ap_shape = ap32.shape
            if ap32.ndim == 1:
                ap32 = ap32[:, None].contiguous()
            row_alpha = torch.repeat_interleave(ap32[:, 0].contiguous(), counts_t, output_size=M)
            row_expert = torch.repeat_interleave(torch.arange(E, device=dev), counts_t, output_size=M)
        codes = K75._codes_list(act_codes)
        hint = codes[0] if len(set(codes)) == 1 else None
        fused = (FUSED and row_alpha is not None and all(c in (8, 10) for c in codes)
                 and (H & (H - 1)) == 0 and hidden.dtype == torch.bfloat16)

        xq, xs = _q(hidden, "F1 in (x)")
        gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=st)            # (M, 2I) bf16
        if fused:
            inter, iq, is_ = radial_fwd_q(gu, row_act, row_alpha)
            _stat("F3 in (act*up)", inter, iq, is_)
        else:
            inter = K75._glu_fwd(gu, row_act, code_hint=hint, row_alpha=row_alpha)
            iq, is_ = _q(inter, "F3 in (act*up)")
        eo = MX.grouped_gemm(iq, is_, *wdn["rc"], counts_t, M)                     # (M, H) bf16
        inv = FG.inverse_order(order)
        out = FG.combine_gather(eo, inv, N, top_k, w=sw, out_dtype=hidden.dtype)

        ctx.save_for_backward(hidden, st, sw, order, row_act, gu, inter, eo)
        ctx.inv, ctx.offs, ctx.counts_t, ctx.fused = inv, offs, counts_t, fused
        ctx.wq = (wgu, wdn)
        ctx.shapes = (N, H, top_k, E, M)
        ctx.row_alpha, ctx.row_expert, ctx.ap_shape, ctx.hint = row_alpha, row_expert, ap_shape, hint
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, st, sw, order, row_act, gu, inter, eo = ctx.saved_tensors
        N, H, top_k, E, M = ctx.shapes
        wgu, wdn = ctx.wq
        grad_out = grad_out.contiguous()
        if ctx.fused:
            ge, gw, gq, gs = combine_bwd_q(grad_out, eo, sw, st)
            _stat("B3 in (dO)", ge, gq, gs)
        else:
            ge, gw = K75._combine_bwd(grad_out, eo, sw, st)
            gq, gs = _q(ge, "B3 in (dO)")
        grad_down = _wgrad(ge, inter, ctx.offs, ctx.counts_t, ctx.acc[1])            # B2
        d_inter = MX.grouped_gemm(gq, gs, *wdn["cr"], ctx.counts_t, M)             # B3 (M, I)
        grad_ap = None
        want_ap = ctx.row_alpha is not None and ctx.needs_input_grad[6]
        if ctx.fused:
            dgu, dq, ds, da = radial_bwd_q(d_inter, gu, row_act, ctx.row_alpha, want_ap)
            _stat("B6 in (dGU)", dgu, dq, ds)
        else:
            res = K75._glu_bwd(d_inter, gu, row_act, code_hint=ctx.hint, row_alpha=ctx.row_alpha,
                               want_act_grads=want_ap)
            dgu, da = res if want_ap else (res, None)
            dq, ds = _q(dgu, "B6 in (dGU)")
        if want_ap:
            grad_ap = K75._ap_grad_from_rows(da, ctx.row_expert, E, ctx.ap_shape, grad_out.device)
        grad_gu = _wgrad(dgu, hidden, ctx.offs, ctx.counts_t, ctx.acc[0], b_rows=st)  # B5
        dx_rows = MX.grouped_gemm(dq, ds, *wgu["cr"], ctx.counts_t, M, out_dtype=DX_ROWS)  # B6
        grad_hidden = FG.combine_gather(dx_rows, ctx.inv, N, top_k, out_dtype=grad_out.dtype)
        grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = gw.to(grad_out.dtype)
        return grad_hidden, None, grad_wt.view(N, top_k), grad_gu, grad_down, None, grad_ap


def moe_fp8_phase1(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    return _MoEFP8.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)


# ================================================================== FULL fp8: both layouts, no bf16 intermediates
# Every producer runs on 32-row tiles that never cross an expert (build_tile_map bm=32) and writes
#   row copy   (M, C) e4m3 + (M, C/32) scales      blocks of 32 along C   -> fwd / dgrad GEMMs
#   token copy (C, Mp) e4m3 + (C, Mp/32) scales    blocks of 32 TOKENS    -> wgrad GEMMs (K-major)
# The tokens of expert e sit at columns [PST[e], PST[e] + PCNT[e]) of the token copy, PCNT = counts
# rounded up to 32 (the pad is written as zeros by the tile that owns it). Mp = M + 32 E is a static
# bound, so nothing is read back to the host.
FULL = True
TILE_ROWS = 32      # rows per producer tile (a multiple of 32 dividing KPAD)


@triton.jit
def _qrow_store(v, rows, cols, mr, Q, S, LD, BR: tl.constexpr, BC: tl.constexpr):
    """row quant of a (BR, BC) tile along its columns: blocks of 32 inside one row."""
    vb = tl.reshape(v, (BR, BC // 32, 32))
    ex = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(vb), axis=2), 1e-30) / 448.0)), -127.0), 127.0)
    q = tl.reshape(vb * tl.exp2(-ex)[:, :, None], (BR, BC))
    tl.store(Q + rows[:, None].to(tl.int64) * LD + cols[None, :], q.to(tl.float8e4nv), mask=mr[:, None])
    sc = tl.min(cols, axis=0) // 32 + tl.arange(0, BC // 32)
    tl.store(S + rows[:, None].to(tl.int64) * (LD // 32) + sc[None, :], (ex + 127.0).to(tl.uint8), mask=mr[:, None])


@triton.jit
def _qtok_store(v, cols, col0, QT, ST, Mp, C, TR: tl.constexpr = 32):
    """token quant of a (TR, BC) tile (TR tokens of ONE expert, masked rows already 0): one scale per
    column per 32 tokens, stored TRANSPOSED at QT[col, col0 + j]; scales token-block-major."""
    BC: tl.constexpr = v.shape[1]
    vt = tl.reshape(v, (TR // 32, 32, BC))
    ex = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(vt), axis=1), 1e-30) / 448.0)), -127.0), 127.0)
    q = tl.reshape(vt * tl.exp2(-ex)[:, None, :], (TR, BC))
    tl.store(QT + cols[:, None].to(tl.int64) * Mp + (col0 + tl.arange(0, TR))[None, :], tl.trans(q).to(tl.float8e4nv))
    # scales TOKEN-BLOCK-major (Mp/32, C): one K step of the wgrad reads a contiguous row of them
    tl.store(ST + (col0 // 32 + tl.arange(0, TR // 32))[:, None].to(tl.int64) * C + cols[None, :], (ex + 127.0).to(tl.uint8))


@triton.jit
def _ld_gu(GU, GUS, rows, cols, mr, TWO_I, GU8: tl.constexpr, BC: tl.constexpr):
    """a (32, BC) tile of GU as fp32: bf16 directly, or fp8 with its per-32 scales dequantized."""
    v = tl.load(GU + rows[:, None].to(tl.int64) * TWO_I + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if GU8:
        sb = tl.min(cols, axis=0) // 32 + tl.arange(0, BC // 32)
        sc = tl.load(GUS + rows[:, None].to(tl.int64) * (TWO_I // 32) + sb[None, :], mask=mr[:, None], other=127)
        v = tl.reshape(tl.reshape(v, (v.shape[0], BC // 32, 32)) * tl.exp2(sc.to(tl.float32) - 127.0)[:, :, None], (v.shape[0], BC))
    return v


@triton.jit
def _x_tok_kernel(X, SRT, TE, TS, TM, START, PST, QT, STS, Mp, H: tl.constexpr, BC: tl.constexpr,
                  QR=None, SR=None, ROW: tl.constexpr = False, TR: tl.constexpr = 32):
    """token copy of x in EXPERT order (gathered through the sort): the B5 right operand; with ROW
    also the row-quantized copy in expert order, so F1 reads contiguously (no gather in the GEMM)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, TR)
    mr = tl.arange(0, TR) < mm
    tok = tl.load(SRT + rows, mask=mr, other=0)
    for c0 in range(0, H, BC):
        cols = c0 + tl.arange(0, BC)
        x = tl.load(X + tok[:, None].to(tl.int64) * H + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        _qtok_store(x, cols, col0, QT, STS, Mp, H, TR)
        if ROW:
            _qrow_store(x, rows, cols, mr, QR, SR, H, TR, BC)


@triton.jit
def _radial_fwd_tile_kernel(GU, GUS, ACT, ALPHA, TE, TS, TM, START, PST, QR, SR, QT, STS, RSS, ROUT, Mp,
                            I: tl.constexpr, EPS: tl.constexpr, BC: tl.constexpr,
                            NP: tl.constexpr, NPP: tl.constexpr, TOK: tl.constexpr = True,
                            GU8: tl.constexpr = False, TR: tl.constexpr = 32):
    """inter = r^p SiLU(g/r) u (codes 8/10) on 32 rows: row copy (F3 input) + token copy (B2 input)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, TR)
    mr = tl.arange(0, TR) < mm
    at = tl.load(ACT + e)                              # per EXPERT: a tile never crosses one
    aa = tl.load(ALPHA + e).to(tl.float32)
    base = GU + rows[:, None].to(tl.int64) * (2 * I)
    if NP > 0:
        jj = tl.arange(0, NPP)
        ss = tl.load(RSS + rows[:, None].to(tl.int64) * NP + jj[None, :], mask=mr[:, None] & (jj < NP)[None, :], other=0.0)
        r = tl.sqrt(tl.sum(ss, axis=1) / I + EPS)    # sum of squares from the F1 epilogue: no extra pass
    else:
        acc = tl.zeros((TR,), tl.float32)
        for c0 in range(0, I, BC):
            g = _ld_gu(GU, GUS, rows, c0 + tl.arange(0, BC), mr, 2 * I, GU8, BC)
            acc += tl.sum(g * g, axis=1)
        r = tl.sqrt(acc / I + EPS)
    tl.store(ROUT + rows, r, mask=mr)
    p = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    rp = tl.exp(p * tl.log(r))
    rinv = 1.0 / r
    for c0 in range(0, I, BC):
        cols = c0 + tl.arange(0, BC)
        g = _ld_gu(GU, GUS, rows, cols, mr, 2 * I, GU8, BC)
        u = _ld_gu(GU, GUS, rows, I + cols, mr, 2 * I, GU8, BC)
        z = g * rinv[:, None]
        v = tl.where(mr[:, None], rp[:, None] * (z * tl.sigmoid(z)) * u, 0.0)
        _qrow_store(v, rows, cols, mr, QR, SR, I, TR, BC)
        if TOK:
            _qtok_store(v, cols, col0, QT, STS, Mp, I, TR)


@triton.jit
def _radial_bwd_tile_kernel(GO, GOS, GU, GUS, ACT, ALPHA, TE, TS, TM, START, PST, QR, SR, QT, STS, DA, RIN, PART,
                            TW, TGW, Mp,
                            I: tl.constexpr, EPS: tl.constexpr, WANT_AP: tl.constexpr, BC: tl.constexpr,
                            NP: tl.constexpr, NPP: tl.constexpr, TOK: tl.constexpr = True,
                            GU8: tl.constexpr = False, GO8: tl.constexpr = False, TR: tl.constexpr = 32,
                            ROWST: tl.constexpr = True):
    """dGU of radial on 32 rows: row copy along 2I (B6 input) + token copy (B5 left operand)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, TR)
    mr = tl.arange(0, TR) < mm
    at = tl.load(ACT + e)                              # per EXPERT: a tile never crosses one
    aa = tl.load(ALPHA + e).to(tl.float32)
    gub = GU + rows[:, None].to(tl.int64) * (2 * I)
    gob = GO + rows[:, None].to(tl.int64) * I
    r = tl.load(RIN + rows, mask=mr, other=1.0)             # saved by the forward
    p = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    lr = tl.log(r)
    rp = tl.exp(p * lr)
    rpm1 = tl.exp((p - 1.0) * lr)
    rinv = 1.0 / r
    if NP > 0:
        jj = tl.arange(0, NPP)
        pm = mr[:, None] & (jj < NP)[None, :]
        sa = tl.sum(tl.load(PART + rows[:, None].to(tl.int64) * NP + jj[None, :], mask=pm, other=0.0), axis=1)
        # T = sum go*u*f = <d_inter, inter> / r^p = <dO, eo> / r^p = w * gw / r^p   (gw = <grad_out, eo>,
        # already computed by the combine backward): no pass over the row at all
        tt = tl.load(TW + rows, mask=mr, other=0.0) * tl.load(TGW + rows, mask=mr, other=0.0) / rp
    else:
        sa = tl.zeros((TR,), tl.float32)
        for c0 in range(0, I, BC):
            cols = c0 + tl.arange(0, BC)
            go = _ld_gu(GO, GOS, rows, cols, mr, I, GO8, BC)
            g = _ld_gu(GU, GUS, rows, cols, mr, 2 * I, GU8, BC)
            u = _ld_gu(GU, GUS, rows, I + cols, mr, 2 * I, GU8, BC)
            gn = g * rinv[:, None]
            sig = tl.sigmoid(gn)
            sa += tl.sum(go * u * sig * (1.0 + gn * (1.0 - sig)) * gn, axis=1)
        tt = tl.load(TW + rows, mask=mr, other=0.0) * tl.load(TGW + rows, mask=mr, other=0.0) / rp
    for c0 in range(0, I, BC):
        cols = c0 + tl.arange(0, BC)
        go = _ld_gu(GO, GOS, rows, cols, mr, I, GO8, BC)
        g = _ld_gu(GU, GUS, rows, cols, mr, 2 * I, GU8, BC)
        u = _ld_gu(GU, GUS, rows, I + cols, mr, 2 * I, GU8, BC)
        gn = g * rinv[:, None]
        sig = tl.sigmoid(gn)
        f = gn * sig
        df = sig * (1.0 + gn * (1.0 - sig))
        gu_ = go * u
        gg = tl.where(mr[:, None], rpm1[:, None] * (gu_ * df - (gn * (1.0 / I)) * (sa - p * tt)[:, None]), 0.0)
        gup = tl.where(mr[:, None], go * (rp[:, None] * f), 0.0)
        if ROWST:
            _qrow_store(gg, rows, cols, mr, QR, SR, 2 * I, TR, BC)
            _qrow_store(gup, rows, I + cols, mr, QR, SR, 2 * I, TR, BC)
        if TOK:
            _qtok_store(gg, cols, col0, QT, STS, Mp, 2 * I, TR)
            _qtok_store(gup, I + cols, col0, QT, STS, Mp, 2 * I, TR)
    if WANT_AP:                                        # d(theta) summed over the tile: one value per tile
        da = tl.where(mr, tl.where(at == 10, 1.0 - p * p, p * (1.0 - p)) * rp * lr * tt, 0.0)
        tl.store(DA + t, tl.sum(da, axis=0))


@triton.jit
def _combine_bwd_tile_kernel(GO, EO, W, SRT, TE, TS, TM, START, PST, QR, SR, QT, STS, GW, Mp,
                             H: tl.constexpr, BC: tl.constexpr, EOS=None, EO8: tl.constexpr = False, TR: tl.constexpr = 32):
    """dO = w * grad_out[token], dw = <grad_out[token], eo> on 32 rows: row copy (B3 input) + token
    copy (B2 left operand)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, TR)
    mr = tl.arange(0, TR) < mm
    tok = tl.load(SRT + rows, mask=mr, other=0)
    w = tl.load(W + rows, mask=mr, other=0.0).to(tl.float32)
    gw = tl.zeros((TR,), tl.float32)
    for c0 in range(0, H, BC):
        cols = c0 + tl.arange(0, BC)
        go = tl.load(GO + tok[:, None].to(tl.int64) * H + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        eo = _ld_gu(EO, EOS, rows, cols, mr, H, EO8, BC)
        gw += tl.sum(go * eo, axis=1)
        ge = go * w[:, None]
        _qrow_store(ge, rows, cols, mr, QR, SR, H, TR, BC)
        _qtok_store(ge, cols, col0, QT, STS, Mp, H, TR)
    tl.store(GW + rows, gw, mask=mr)


EPI_FUSE = False   # (measured a wash) S partials in the B3 epilogue (+ T from the combine grad) -> one-pass radial bwd
# (history: with bf16 GU and both S and T in the epilogue it was SLOWER; row reductions in the F1/B3 epilogues: measured SLOWER (B3 +0.9 ms reading G/U,
#                    radial bwd unchanged at 2.55 ms) -- the radial kernels are not pass-bound
RADIAL_WARPS, RADIAL_BC = 4, None   # tuning knobs (None = _bc)
RADIAL_BWD_WARPS, RADIAL_BWD_BC = 4, 64      # radial bwd: swept, 2.33 -> 2.13 ms (fwd keeps BC 128)
GU_FP8 = True                       # F1 output GU cached in MXFP8 (DeepSeek-V3: SwiGLU input in fp8)
X_SORTED = False                    # True: x row copy in expert order (+0.22 ms x_tok, -0.12 ms F1: net loss)
EO_FP8 = True                       # F3 output EO in MXFP8 (combine fwd + combine bwd read it)
DI_FP8 = True                       # B3 output d_inter in MXFP8 (the radial bwd reads it twice)
_DEBUG_NO_TOK = False
_DEBUG_NO_ROW = False               # TIMING ONLY: skip the radial-bwd row stores               # TIMING ONLY: skip the token-copy stores (wgrads then read garbage)
KPAD = 128    # expert token ranges in the token copies are padded to this: the wgrad K loop runs UNMASKED
#               (masked fp8 K loads were 2.8x slower than bf16; unmasked is 1.9-2.4x FASTER)


@triton.jit
def _combine_gather_q_kernel(RQ, RS, W, INV, OUT, NT, H: tl.constexpr, K: tl.constexpr,
                             BT: tl.constexpr, BH: tl.constexpr):
    """out[t] = sum_j w[r] * dequant(rows[r]), r = inv[t*K + j], j in order (deterministic)."""
    t = tl.program_id(0) * BT + tl.arange(0, BT)
    h = tl.program_id(1) * BH + tl.arange(0, BH)
    mt = t < NT
    acc = tl.zeros((BT, BH), tl.float32)
    for j in tl.static_range(K):
        r = tl.load(INV + t.to(tl.int64) * K + j, mask=mt, other=0)
        x = tl.load(RQ + r[:, None].to(tl.int64) * H + h[None, :], mask=mt[:, None], other=0.0).to(tl.float32)
        sc = tl.load(RS + r[:, None].to(tl.int64) * (H // 32) + (tl.program_id(1) * (BH // 32) + tl.arange(0, BH // 32))[None, :],
                     mask=mt[:, None], other=127)
        x = tl.reshape(tl.reshape(x, (BT, BH // 32, 32)) * tl.exp2(sc.to(tl.float32) - 127.0)[:, :, None], (BT, BH))
        acc += x * tl.load(W + r, mask=mt, other=0.0).to(tl.float32)[:, None]
    tl.store(OUT + t.to(tl.int64)[:, None] * H + h[None, :], acc.to(OUT.dtype.element_ty), mask=mt[:, None])


def combine_gather_q(rq, rs, inv, n_tok, k, w, out_dtype):
    H = rq.shape[1]
    out = torch.empty(n_tok, H, device=rq.device, dtype=out_dtype)
    _combine_gather_q_kernel[(triton.cdiv(n_tok, 32), H // 128)](rq, rs, w, inv, out, n_tok, H, k, 32, 128,
                                                                 num_warps=4)
    return out


def _tiles(counts_t, M, dev):
    TE, TS, TM = MX.tile_map(counts_t, M, TILE_ROWS)
    c = counts_t.to(torch.int32)
    start = (torch.cumsum(c, 0) - c).to(torch.int32)
    c32 = ((c + 31) // 32) * 32
    pcnt = ((c + KPAD - 1) // KPAD) * KPAD
    pst = (torch.cumsum(pcnt, 0) - pcnt).to(torch.int32)
    Mp = M + KPAD * counts_t.numel()
    return (TE, TS, TM, start, pst), pcnt.to(torch.int32), Mp, (pst + c32).to(torch.int32)


@triton.jit
def _pad_kernel(QT, STS, PST, PADS, PCNT, Mp, C: tl.constexpr, BC: tl.constexpr):
    """zero the columns [PADS[e], PST[e] + PCNT[e]) the 32-row tiles never write, with scale 2^0.
    (0xFF is NaN in e8m0, so the scales must be written too: 0 * NaN = NaN.)"""
    e = tl.program_id(0)
    cols = tl.program_id(1) * BC + tl.arange(0, BC)
    j0 = tl.load(PADS + e)
    j1 = tl.load(PST + e) + tl.load(PCNT + e)
    for j in range(j0, j1, 32):
        tl.store(QT + cols[:, None].to(tl.int64) * Mp + (j + tl.arange(0, 32))[None, :],
                 tl.zeros((BC, 32), tl.float8e4nv))
        tl.store(STS + (j // 32).to(tl.int64) * C + cols, tl.full((BC,), 127, tl.uint8))


def _pad(qt, sts, pst, pads, pcnt, Mp):
    C = qt.shape[0]
    _pad_kernel[(pst.numel(), C // _bc(C))](qt, sts, pst, pads, pcnt, Mp, C, _bc(C), num_warps=4)


def _row_buf(M, C, dev):
    return torch.empty(M, C, device=dev, dtype=MX.F8), torch.empty(M, C // 32, device=dev, dtype=torch.uint8)


def _tok_buf(C, Mp, dev):
    return torch.empty(C, Mp, device=dev, dtype=MX.F8), torch.empty(Mp // 32, C, device=dev, dtype=torch.uint8)


def _bc(C):
    return 128 if C % 128 == 0 else (64 if C % 64 == 0 else 32)


def _acc_wgrad(acc, fn):
    """accumulate into the fp32 master .grad (and return None to autograd), else return the tensor."""
    if acc is None:
        return fn(None, False)
    fresh = acc.grad is None
    buf = torch.empty_like(acc) if fresh else acc.grad
    fn(buf, not fresh)
    if fresh:
        acc.grad = buf
    return None


class _MoEFP8Full(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, idx, wt, gate_up_proj, down_proj, act_codes, act_params):
        ctx.acc = (K75._acc_target(gate_up_proj), K75._acc_target(down_proj))
        wgu, wdn = MX.quant_weight(gate_up_proj), MX.quant_weight(down_proj)
        _wstats(gate_up_proj, "W gate_up", wgu["rc"]); _wstats(down_proj, "W down", wdn["rc"])
        hidden, = K75._amp_cast(hidden)
        hidden = hidden.contiguous()
        wt = wt.float()
        N, H = hidden.shape
        E = act_codes.shape[0]
        top_k = idx.shape[1]
        dev = hidden.device
        st, sw, order, _, _, counts_t = K75._sort_by_expert(idx, wt, E, host=False)
        M = idx.numel()
        I = gate_up_proj.shape[1] // 2
        ap32 = act_params.float().contiguous()
        ap_shape = ap32.shape
        if ap32.ndim == 1:
            ap32 = ap32[:, None].contiguous()
        row_act = act_codes.to(torch.int32).contiguous()          # per EXPERT (names kept for the kernels)
        row_alpha = ap32[:, 0].contiguous()
        tiles, pcnt, Mp, pads = _tiles(counts_t, M, dev)
        nt = tiles[0].numel()

        xT, xTs = _tok_buf(H, Mp, dev)
        if X_SORTED:                                      # x row copy in EXPERT order (F1 without gather)
            xq, xs = _row_buf(M, H, dev)
            _x_tok_kernel[(nt,)](hidden, st, *tiles, xT, xTs, Mp, H, _bc(H), xq, xs, True, TR=TILE_ROWS, num_warps=4)
            rows_f1 = None
        else:                                             # quantize the N unsorted tokens once, F1 gathers
            xq, xs = _q(hidden, "F1 in (x)")
            _x_tok_kernel[(nt,)](hidden, st, *tiles, xT, xTs, Mp, H, _bc(H), TR=TILE_ROWS, num_warps=4)
            rows_f1 = st
        _pad(xT, xTs, tiles[4], pads, pcnt, Mp)
        np1 = I // MX.gemm_bn(H, 2 * I) if (EPI_FUSE or GU_FP8) else 0
        rss = torch.empty(M, max(np1, 1), device=dev, dtype=torch.float32) if np1 else row_alpha
        if GU_FP8:
            gus = torch.empty(M, 2 * I // 32, device=dev, dtype=torch.uint8)
            gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=rows_f1, epi=3, x1=gus, x2=rss,
                                 np_=np1)                     # F1 -> GU fp8 + per-row sum(g^2) partials
        else:
            gus = row_alpha
            gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=rows_f1, epi=1 if EPI_FUSE else 0,
                                 x1=rss if EPI_FUSE else None, np_=max(np1, 1))       # F1 (+ sum g^2)
        iq, is_ = _row_buf(M, I, dev)
        iT, iTs = _tok_buf(I, Mp, dev)
        r = torch.empty(M, device=dev, dtype=torch.float32)
        _radial_fwd_tile_kernel[(nt,)](gu, gus, row_act, row_alpha, *tiles, iq, is_, iT, iTs, rss, r, Mp, I, _EPS,
                                       RADIAL_BC or _bc(I), np1, triton.next_power_of_2(max(np1, 1)),
                                       TR=TILE_ROWS, num_warps=RADIAL_WARPS, TOK=not _DEBUG_NO_TOK, GU8=GU_FP8)
        _pad(iT, iTs, tiles[4], pads, pcnt, Mp)
        inv = FG.inverse_order(order)
        if EO_FP8:
            eos = torch.empty(M, H // 32, device=dev, dtype=torch.uint8)
            eo = MX.grouped_gemm(iq, is_, *wdn["rc"], counts_t, M, epi=3, x1=eos)    # F3 -> fp8
            out = combine_gather_q(eo, eos, inv, N, top_k, sw, hidden.dtype)
        else:
            eos = sw
            eo = MX.grouped_gemm(iq, is_, *wdn["rc"], counts_t, M)                 # F3 -> (M, H) bf16
            out = FG.combine_gather(eo, inv, N, top_k, w=sw, out_dtype=hidden.dtype)
        if STATS is not None:
            STATS.setdefault("F3 in (act*up): exact-0 %", []).append(
                (100 * (iq.float() == 0).float().mean().item(), 0.0))

        ctx.save_for_backward(st, sw, order, row_act, row_alpha, gu, eo, xT, xTs, iT, iTs, pcnt,
                              pads, r, gus, eos, *tiles)
        ctx.inv, ctx.counts_t, ctx.Mp, ctx.wq = inv, counts_t, Mp, (wgu, wdn)
        ctx.shapes = (N, H, top_k, E, M, I)
        ctx.ap_shape = ap_shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (st, sw, order, row_act, row_alpha, gu, eo, xT, xTs, iT, iTs, pcnt, pads, r, gus, eos, *tiles) = ctx.saved_tensors
        N, H, top_k, E, M, I = ctx.shapes
        wgu, wdn = ctx.wq
        Mp, dev = ctx.Mp, grad_out.device
        nt = tiles[0].numel()
        pst = tiles[4]
        grad_out = grad_out.contiguous()
        gq, gs = _row_buf(M, H, dev)
        gT, gTs = _tok_buf(H, Mp, dev)
        gw = torch.empty(M, device=dev, dtype=torch.float32)
        _combine_bwd_tile_kernel[(nt,)](grad_out, eo, sw, st, *tiles, gq, gs, gT, gTs, gw, Mp, H, _bc(H),
                                        eos, EO_FP8, TR=TILE_ROWS, num_warps=4)
        _pad(gT, gTs, pst, pads, pcnt, Mp)
        grad_down = _acc_wgrad(ctx.acc[1], lambda out, a: MX.wgrad_kmajor(gT, gTs, iT, iTs, pst, pcnt, E,
                                                                          out=out, accumulate=a, even=True))       # B2
        fuse_st = EPI_FUSE and GU_FP8
        np3 = I // MX.gemm_bn(H, I) if fuse_st else 0
        part = torch.empty(M, max(np3, 1), device=dev, dtype=torch.float32) if fuse_st else r
        if DI_FP8 and not fuse_st:
            dis = torch.empty(M, I // 32, device=dev, dtype=torch.uint8)
            d_inter = MX.grouped_gemm(gq, gs, *wdn["cr"], ctx.counts_t, M, epi=3, x1=dis)   # B3 -> fp8
        else:
            dis = r
            d_inter = MX.grouped_gemm(gq, gs, *wdn["cr"], ctx.counts_t, M, epi=2 if fuse_st else 0,
                                      x1=part if fuse_st else None, x2=gu, x3=r, x4=gus, i2=I,
                                      np_=max(np3, 1))                              # B3 (+ S partials)
        want_ap = ctx.needs_input_grad[6]
        dq, ds = _row_buf(M, 2 * I, dev)
        dT, dTs = _tok_buf(2 * I, Mp, dev)
        da = torch.zeros(nt, device=dev, dtype=torch.float32) if want_ap else gw    # per TILE
        _radial_bwd_tile_kernel[(nt,)](d_inter, dis, gu, gus, row_act, row_alpha, *tiles, dq, ds, dT, dTs, da, r, part,
                                       sw, gw, Mp,
                                       I, _EPS, want_ap, RADIAL_BWD_BC or RADIAL_BC or _bc(I), np3,
                                       triton.next_power_of_2(max(np3, 1)), TR=TILE_ROWS, num_warps=RADIAL_BWD_WARPS or RADIAL_WARPS,
                                       TOK=not _DEBUG_NO_TOK, GU8=GU_FP8, ROWST=not _DEBUG_NO_ROW, GO8=DI_FP8 and not fuse_st)
        _pad(dT, dTs, pst, pads, pcnt, Mp)
        grad_ap = _ap_grad_from_tiles(da, ctx.counts_t, E, ctx.ap_shape, bm=TILE_ROWS) if want_ap else None
        grad_gu = _acc_wgrad(ctx.acc[0], lambda out, a: MX.wgrad_kmajor(dT, dTs, xT, xTs, pst, pcnt, E,
                                                                        out=out, accumulate=a, even=True))         # B5
        dx_rows = MX.grouped_gemm(dq, ds, *wgu["cr"], ctx.counts_t, M, out_dtype=DX_ROWS)      # B6
        grad_hidden = FG.combine_gather(dx_rows, ctx.inv, N, top_k, out_dtype=grad_out.dtype)
        grad_wt = torch.zeros(N * top_k, device=dev, dtype=grad_out.dtype)
        grad_wt[order] = gw.to(grad_out.dtype)
        return grad_hidden, None, grad_wt.view(N, top_k), grad_gu, grad_down, None, grad_ap


def _ap_grad_from_tiles(da_t, counts_t, E, ap_shape, bm=32):
    """per-TILE d(theta) -> per EXPERT. Tiles are expert-sorted (32-row, never crossing an expert), so
    each expert owns a contiguous run of ceil(count / 32) tiles: fixed-order fp64 prefix sums
    differenced at the run ends -- deterministic, no atomics."""
    nt_e = (counts_t + bm - 1) // bm
    end = torch.cumsum(nt_e, 0)
    cs = torch.cat([da_t.new_zeros(1, dtype=torch.float64), da_t.double().cumsum(0)])
    per_e = (cs[end] - cs[end - nt_e]).float()
    if len(ap_shape) == 1:
        return per_e
    g = torch.zeros(E, 2, device=da_t.device, dtype=torch.float32)
    g[:, 0] = per_e
    return g[:, :ap_shape[1]]


def _full_ok(hidden, act_codes, act_params, gate_up_proj):
    if not FULL or act_params is None:
        return False
    codes = K75._codes_list(act_codes)
    return (all(c in (8, 10) for c in codes) and hidden.shape[1] % 128 == 0
            and (gate_up_proj.shape[1] // 2) % 32 == 0)


def moe_fp8_full(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    if _full_ok(hidden, act_codes, act_params, gate_up_proj):
        return _MoEFP8Full.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
    return _MoEFP8.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)


# ================================================================== FUSE_ACT: radial inside the GEMM prologues
# The radial activation needs a row-wide RMS, so it cannot live in the F1 EPILOGUE (a tile sees 128 of
# 768 gate columns). It can live in the PROLOGUE of the GEMM that CONSUMES it: F3 loads GU (fp8) and r
# (from the F1 epilogue partials), computes act*up in registers, quantizes along K and feeds the MMA.
# Backward: B3 writes d_inter in fp8 plus S partials; B6 computes dGU in its prologue from d_inter,
# GU, r, S and T (= w * gw / r^p). Neither the radial fwd nor the radial bwd kernel runs, and neither
# act*up nor dGU is ever materialized as a row copy. The pid_n == 0 program of each row tile also
# writes the token-major copy (B2 / B5 inputs): tiles are 128-row, expert-aligned (KPAD = 128), so a
# tile covers exactly its own padded token range, zeros included.
FUSE_ACT = False   # measured SLOWER: f3r 2.17 vs 1.67 ms, b6r 4.5 vs 3.58 ms (prologue ALU stalls the MMA pipeline)


@triton.jit
def _dq_tile(Q, S, rows, cols, mr, LD, BR: tl.constexpr, BC: tl.constexpr):
    """(BR, BC) fp32 from an MXFP8 row-quantized matrix (Q (., LD) e4m3, S (., LD/32) e8m0)."""
    q = tl.load(Q + rows[:, None].to(tl.int64) * LD + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    sb = tl.min(cols, axis=0) // 32 + tl.arange(0, BC // 32)
    s = tl.load(S + rows[:, None].to(tl.int64) * (LD // 32) + sb[None, :], mask=mr[:, None], other=127)
    return tl.reshape(tl.reshape(q, (BR, BC // 32, 32)) * tl.exp2(s.to(tl.float32) - 127.0)[:, :, None], (BR, BC))


@triton.jit
def _qk(v, BR: tl.constexpr, BC: tl.constexpr):
    """row quant along the K (column) axis: e4m3 values + (BR, BC/32) e8m0 scales, rounded up."""
    vb = tl.reshape(v, (BR, BC // 32, 32))
    ex = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(vb), axis=2), 1e-30) / 448.0)), -127.0), 127.0)
    return tl.reshape(vb * tl.exp2(-ex)[:, :, None], (BR, BC)).to(tl.float8e4nv), (ex + 127.0).to(tl.uint8)


@triton.jit
def _tok_tile(v, kk, col0, QT, STS, Mp: tl.constexpr, C: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    """token-major copy of a (BR, BC) tile (BR tokens of one expert, masked rows already 0): blocks of
    32 tokens per column, stored transposed at QT[kk, col0 + j], scales at STS[(col0 + j) / 32, kk]."""
    vt = tl.reshape(v, (BR // 32, 32, BC))
    et = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(vt), axis=1), 1e-30) / 448.0)), -127.0), 127.0)
    qt = tl.reshape(vt * tl.exp2(-et)[:, None, :], (BR, BC))
    tl.store(QT + kk[:, None].to(tl.int64) * Mp + (col0 + tl.arange(0, BR))[None, :], tl.trans(qt).to(tl.float8e4nv))
    tl.store(STS + (col0 // 32 + tl.arange(0, BR // 32))[:, None].to(tl.int64) * C + kk[None, :], (et + 127.0).to(tl.uint8))


@triton.jit
def _f3r_kernel(GU, GUS, RSS, ACT, ALPHA, B, BS, C, TE, TS, TM, START, PST, QT, STS, ROUT,
                Mp: tl.constexpr, I: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, NPP: tl.constexpr,
                EPS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """EO = radial(GU) @ Wdn^T with the activation computed in the prologue (see FUSE_ACT)."""
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    rm = r0 + tl.arange(0, BM)
    mr = tl.arange(0, BM) < mm
    rn = pid_n * BN + tl.arange(0, BN)
    jj = tl.arange(0, NPP)
    ss = tl.load(RSS + rm[:, None].to(tl.int64) * NP + jj[None, :], mask=mr[:, None] & (jj < NP)[None, :], other=0.0)
    r = tl.sqrt(tl.sum(ss, axis=1) / I + EPS)
    at = tl.load(ACT + e)
    aa = tl.load(ALPHA + e).to(tl.float32)
    p = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    rp = tl.exp(p * tl.log(r))
    tok = pid_n == 0
    if tok:
        tl.store(ROUT + rm, r, mask=mr)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    KS: tl.constexpr = I // 32
    Bb = B + e.to(tl.int64) * (N * I)
    BSb = BS + e.to(tl.int64) * (N * KS)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, I, BK):
        kk = k0 + tl.arange(0, BK)
        g = _dq_tile(GU, GUS, rm, kk, mr, 2 * I, BM, BK)
        u = _dq_tile(GU, GUS, rm, I + kk, mr, 2 * I, BM, BK)
        z = g / r[:, None]
        v = tl.where(mr[:, None], rp[:, None] * (z * (1.0 / (1.0 + tl.exp(-z)))) * u, 0.0)
        a, a_s = _qk(v, BM, BK)
        b = tl.load(Bb + rn[None, :] * I + kk[:, None])
        b_s = tl.load(BSb + rn[:, None] * KS + (k0 // 32 + tl.arange(0, BK // 32))[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
        if tok:
            _tok_tile(v, kk, col0, QT, STS, Mp, I, BM, BK)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mr[:, None])


@triton.jit
def _b3s_kernel(A, AS, B, BS, CQ, CS, GU, GUS, R, SP, TE, TS, TM,
                K: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr,
                BK: tl.constexpr):
    """d_inter = dO @ Wdn stored as MXFP8, plus per-row partials of S = sum go*u*df*gn over this tile
    (go = the DEQUANTIZED stored value, so B6 sees exactly what S was computed from)."""
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    rm = r0 + tl.arange(0, BM)
    mr = tl.arange(0, BM) < mm
    rn = pid_n * BN + tl.arange(0, BN)
    KS: tl.constexpr = K // 32
    Bb = B + e.to(tl.int64) * (N * K)
    BSb = BS + e.to(tl.int64) * (N * KS)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        rs = k0 // 32 + tl.arange(0, BK // 32)
        a = tl.load(A + rm[:, None].to(tl.int64) * K + rk[None, :], mask=mr[:, None], other=0.0)
        a_s = tl.load(AS + rm[:, None].to(tl.int64) * KS + rs[None, :], mask=mr[:, None], other=127)
        b = tl.load(Bb + rn[None, :] * K + rk[:, None])
        b_s = tl.load(BSb + rn[:, None] * KS + rs[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
    vb = tl.reshape(acc, (BM, BN // 32, 32))
    ex = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(vb), axis=2), 1e-30) / 448.0)), -127.0), 127.0)
    q = tl.reshape(vb * tl.exp2(-ex)[:, :, None], (BM, BN)).to(tl.float8e4nv)
    tl.store(CQ + rm[:, None].to(tl.int64) * N + rn[None, :], q, mask=mr[:, None])
    tl.store(CS + rm[:, None].to(tl.int64) * (N // 32) + (pid_n * (BN // 32) + tl.arange(0, BN // 32))[None, :],
             (ex + 127.0).to(tl.uint8), mask=mr[:, None])
    go = tl.reshape(tl.reshape(q.to(tl.float32), (BM, BN // 32, 32)) * tl.exp2(ex)[:, :, None], (BM, BN))
    g = _dq_tile(GU, GUS, rm, rn, mr, 2 * N, BM, BN)
    u = _dq_tile(GU, GUS, rm, N + rn, mr, 2 * N, BM, BN)
    r = tl.load(R + rm, mask=mr, other=1.0)
    gn = g / r[:, None]
    sig = 1.0 / (1.0 + tl.exp(-gn))
    tl.store(SP + rm.to(tl.int64) * NP + pid_n, tl.sum(go * u * sig * (1.0 + gn * (1.0 - sig)) * gn, axis=1), mask=mr)


@triton.jit
def _b6r_kernel(GO, GOS, GU, GUS, R, SP, TW, TGW, ACT, ALPHA, B, BS, C, TE, TS, TM, START, PST, QT, STS, DA,
                Mp: tl.constexpr, I: tl.constexpr, N: tl.constexpr, NP: tl.constexpr, NPP: tl.constexpr,
                WANT_AP: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """dx rows = dGU @ Wgu with dGU (radial backward) computed in the prologue; pid_n == 0 also writes
    dGU token-major (B5 input) and the tile d(theta)."""
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    rm = r0 + tl.arange(0, BM)
    mr = tl.arange(0, BM) < mm
    rn = pid_n * BN + tl.arange(0, BN)
    r = tl.load(R + rm, mask=mr, other=1.0)
    at = tl.load(ACT + e)
    aa = tl.load(ALPHA + e).to(tl.float32)
    p = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    lr = tl.log(r)
    rp = tl.exp(p * lr)
    rpm1 = tl.exp((p - 1.0) * lr)
    jj = tl.arange(0, NPP)
    S_ = tl.sum(tl.load(SP + rm[:, None].to(tl.int64) * NP + jj[None, :], mask=mr[:, None] & (jj < NP)[None, :], other=0.0), axis=1)
    T_ = tl.load(TW + rm, mask=mr, other=0.0) * tl.load(TGW + rm, mask=mr, other=0.0) / rp
    tok = pid_n == 0
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    K2: tl.constexpr = 2 * I
    KS: tl.constexpr = K2 // 32
    Bb = B + e.to(tl.int64) * (N * K2)
    BSb = BS + e.to(tl.int64) * (N * KS)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, I, BK):                                   # gate half of dGU
        kk = k0 + tl.arange(0, BK)
        go = _dq_tile(GO, GOS, rm, kk, mr, I, BM, BK)
        g = _dq_tile(GU, GUS, rm, kk, mr, K2, BM, BK)
        u = _dq_tile(GU, GUS, rm, I + kk, mr, K2, BM, BK)
        gn = g / r[:, None]
        sig = 1.0 / (1.0 + tl.exp(-gn))
        df = sig * (1.0 + gn * (1.0 - sig))
        v = tl.where(mr[:, None], rpm1[:, None] * (go * u * df - (gn / I) * (S_ - p * T_)[:, None]), 0.0)
        a, a_s = _qk(v, BM, BK)
        b = tl.load(Bb + rn[None, :] * K2 + kk[:, None])
        b_s = tl.load(BSb + rn[:, None] * KS + (k0 // 32 + tl.arange(0, BK // 32))[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
        if tok:
            _tok_tile(v, kk, col0, QT, STS, Mp, K2, BM, BK)
    for k0 in range(0, I, BK):                                   # up half of dGU
        kk = k0 + tl.arange(0, BK)
        go = _dq_tile(GO, GOS, rm, kk, mr, I, BM, BK)
        g = _dq_tile(GU, GUS, rm, kk, mr, K2, BM, BK)
        gn = g / r[:, None]
        v = tl.where(mr[:, None], go * (rp[:, None] * (gn * (1.0 / (1.0 + tl.exp(-gn))))), 0.0)
        a, a_s = _qk(v, BM, BK)
        b = tl.load(Bb + rn[None, :] * K2 + (I + kk)[:, None])
        b_s = tl.load(BSb + rn[:, None] * KS + ((I + k0) // 32 + tl.arange(0, BK // 32))[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
        if tok:
            _tok_tile(v, I + kk, col0, QT, STS, Mp, K2, BM, BK)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mr[:, None])
    if WANT_AP:
        if tok:
            da = tl.where(mr, tl.where(at == 10, 1.0 - p * p, p * (1.0 - p)) * rp * lr * T_, 0.0)
            tl.store(DA + t, tl.sum(da, axis=0))


_FA_BM, _FA_BN, _FA_BK, _FA_W, _FA_ST = 128, 128, 64, 8, 3
# (BM, BN, BK, warps, stages) for the prologue-fused GEMMs: a WIDE BN means fewer programs recompute
# the same rows' activation (N / BN of them per row tile). BM <= KPAD, and BM < KPAD needs the pad kernel.
FA_CFG = {"f3": (128, 128, 64, 8, 3), "b6": (128, 128, 64, 8, 3)}


class _MoEFP8Act(torch.autograd.Function):
    """FULL fp8 with the radial activation fused into the F3 / B6 prologues (FUSE_ACT)."""

    @staticmethod
    def forward(ctx, hidden, idx, wt, gate_up_proj, down_proj, act_codes, act_params):
        ctx.acc = (K75._acc_target(gate_up_proj), K75._acc_target(down_proj))
        wgu, wdn = MX.quant_weight(gate_up_proj), MX.quant_weight(down_proj)
        hidden, = K75._amp_cast(hidden)
        hidden = hidden.contiguous()
        wt = wt.float()
        N, H = hidden.shape
        E = act_codes.shape[0]
        top_k = idx.shape[1]
        dev = hidden.device
        st, sw, order, _, _, counts_t = K75._sort_by_expert(idx, wt, E, host=False)
        M = idx.numel()
        I = gate_up_proj.shape[1] // 2
        ap32 = act_params.float().contiguous()
        ap_shape = ap32.shape
        if ap32.ndim == 1:
            ap32 = ap32[:, None].contiguous()
        act_e = act_codes.to(torch.int32).contiguous()
        alpha_e = ap32[:, 0].contiguous()
        tiles, pcnt, Mp, pads = _tiles(counts_t, M, dev)
        nt = tiles[0].numel()
        tiles128 = MX.tile_map(counts_t, M, FA_CFG["f3"][0])
        c = counts_t.to(torch.int32)
        t128 = (*tiles128, tiles[3], tiles[4])                     # (TE, TS, TM, START, PST) at 128 rows

        xq, xs = _q(hidden, "F1 in (x)")
        xT, xTs = _tok_buf(H, Mp, dev)
        _x_tok_kernel[(nt,)](hidden, st, *tiles, xT, xTs, Mp, H, _bc(H), num_warps=4)
        _pad(xT, xTs, tiles[4], pads, pcnt, Mp)
        np1 = I // MX.gemm_bn(H, 2 * I)
        rss = torch.empty(M, np1, device=dev, dtype=torch.float32)
        gus = torch.empty(M, 2 * I // 32, device=dev, dtype=torch.uint8)
        gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=st, epi=3, x1=gus, x2=rss, np_=np1)
        iT, iTs = _tok_buf(I, Mp, dev)
        r = torch.empty(M, device=dev, dtype=torch.float32)
        eo = torch.empty(M, H, device=dev, dtype=hidden.dtype)
        BM, BN, BK, W, ST = FA_CFG["f3"]
        _f3r_kernel[(tiles128[0].numel() * (H // BN),)](
            gu, gus, rss, act_e, alpha_e, *wdn["rc"], eo, *t128, iT, iTs, r, Mp, I, H, np1,
            triton.next_power_of_2(np1), _EPS, BM, BN, BK, num_warps=W, num_stages=ST)
        if BM < KPAD:
            _pad(iT, iTs, tiles[4], pads, pcnt, Mp)
        inv = FG.inverse_order(order)
        out = FG.combine_gather(eo, inv, N, top_k, w=sw, out_dtype=hidden.dtype)
        ctx.save_for_backward(st, sw, order, act_e, alpha_e, gu, gus, eo, xT, xTs, iT, iTs, pcnt, pads, r,
                              *tiles, *tiles128)
        ctx.inv, ctx.counts_t, ctx.Mp, ctx.wq = inv, counts_t, Mp, (wgu, wdn)
        ctx.shapes = (N, H, top_k, E, M, I)
        ctx.ap_shape = ap_shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (st, sw, order, act_e, alpha_e, gu, gus, eo, xT, xTs, iT, iTs, pcnt, pads, r, *tt) = ctx.saved_tensors
        tiles, tiles128 = tuple(tt[:5]), tuple(tt[5:])
        N, H, top_k, E, M, I = ctx.shapes
        wgu, wdn = ctx.wq
        Mp, dev = ctx.Mp, grad_out.device
        nt = tiles[0].numel()
        pst = tiles[4]
        t128 = (*tiles128, tiles[3], tiles[4])
        grad_out = grad_out.contiguous()
        gq, gs = _row_buf(M, H, dev)
        gT, gTs = _tok_buf(H, Mp, dev)
        gw = torch.empty(M, device=dev, dtype=torch.float32)
        _combine_bwd_tile_kernel[(nt,)](grad_out, eo, sw, st, *tiles, gq, gs, gT, gTs, gw, Mp, H, _bc(H),
                                        num_warps=4)
        _pad(gT, gTs, pst, pads, pcnt, Mp)
        grad_down = _acc_wgrad(ctx.acc[1], lambda out, a: MX.wgrad_kmajor(gT, gTs, iT, iTs, pst, pcnt, E,
                                                                          out=out, accumulate=a, even=True))  # B2
        np3 = I // 128
        sp = torch.empty(M, np3, device=dev, dtype=torch.float32)
        diq, dis = _row_buf(M, I, dev)
        tb3 = MX.tile_map(ctx.counts_t, M, 128)
        _b3s_kernel[(tb3[0].numel() * (I // 128),)](
            gq, gs, *wdn["cr"], diq, dis, gu, gus, r, sp, *tb3, H, I, np3, 128, 128, 128,
            num_warps=8, num_stages=3)                                              # B3 -> fp8 + S partials
        want_ap = ctx.needs_input_grad[6]
        dT, dTs = _tok_buf(2 * I, Mp, dev)
        BM, BN, BK, W, ST = FA_CFG["b6"]
        tb6 = MX.tile_map(ctx.counts_t, M, BM)
        t6 = (*tb6, tiles[3], tiles[4])
        nt6 = tb6[0].numel()
        da = torch.zeros(nt6, device=dev, dtype=torch.float32) if want_ap else gw
        dx_rows = torch.empty(M, H, device=dev, dtype=DX_ROWS)
        _b6r_kernel[(nt6 * (H // BN),)](
            diq, dis, gu, gus, r, sp, sw, gw, act_e, alpha_e, *wgu["cr"], dx_rows, *t6, dT, dTs, da,
            Mp, I, H, np3, triton.next_power_of_2(np3), want_ap, BM, BN, BK,
            num_warps=W, num_stages=ST)                                             # B6 (radial bwd inside)
        if BM < KPAD:
            _pad(dT, dTs, pst, pads, pcnt, Mp)
        grad_ap = _ap_grad_from_tiles(da, ctx.counts_t, E, ctx.ap_shape, bm=BM) if want_ap else None
        grad_gu = _acc_wgrad(ctx.acc[0], lambda out, a: MX.wgrad_kmajor(dT, dTs, xT, xTs, pst, pcnt, E,
                                                                        out=out, accumulate=a, even=True))    # B5
        grad_hidden = FG.combine_gather(dx_rows, ctx.inv, N, top_k, out_dtype=grad_out.dtype)
        grad_wt = torch.zeros(N * top_k, device=dev, dtype=grad_out.dtype)
        grad_wt[order] = gw.to(grad_out.dtype)
        return grad_hidden, None, grad_wt.view(N, top_k), grad_gu, grad_down, None, grad_ap


def moe_fp8(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    if _full_ok(hidden, act_codes, act_params, gate_up_proj):
        I = gate_up_proj.shape[1] // 2
        if (FUSE_ACT and I % 128 == 0 and hidden.shape[1] % FA_CFG["f3"][1] == 0
                and hidden.shape[1] % FA_CFG["b6"][1] == 0 and KPAD % FA_CFG["f3"][0] == 0
                and KPAD % FA_CFG["b6"][0] == 0):
            return _MoEFP8Act.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
                                    act_params)
        return _MoEFP8Full.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
    return _MoEFP8.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
