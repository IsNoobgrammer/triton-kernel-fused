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
def _qtok_store(v, cols, col0, QT, ST, Mp, C):
    """token quant of a (32, BC) tile (32 tokens of ONE expert, masked rows already 0): one scale per
    column over the 32 tokens, stored TRANSPOSED at QT[col, col0 + j]."""
    ex = tl.minimum(tl.maximum(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(v), axis=0), 1e-30) / 448.0)), -127.0), 127.0)
    q = v * tl.exp2(-ex)[None, :]
    tl.store(QT + cols[:, None].to(tl.int64) * Mp + (col0 + tl.arange(0, 32))[None, :], tl.trans(q).to(tl.float8e4nv))
    # scales TOKEN-BLOCK-major (Mp/32, C): one K step of the wgrad reads a contiguous row of them
    tl.store(ST + (col0 // 32).to(tl.int64) * C + cols, (ex + 127.0).to(tl.uint8))


@triton.jit
def _ld_gu(GU, GUS, rows, cols, mr, TWO_I, GU8: tl.constexpr, BC: tl.constexpr):
    """a (32, BC) tile of GU as fp32: bf16 directly, or fp8 with its per-32 scales dequantized."""
    v = tl.load(GU + rows[:, None].to(tl.int64) * TWO_I + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
    if GU8:
        sb = tl.min(cols, axis=0) // 32 + tl.arange(0, BC // 32)
        sc = tl.load(GUS + rows[:, None].to(tl.int64) * (TWO_I // 32) + sb[None, :], mask=mr[:, None], other=127)
        v = tl.reshape(tl.reshape(v, (32, BC // 32, 32)) * tl.exp2(sc.to(tl.float32) - 127.0)[:, :, None], (32, BC))
    return v


@triton.jit
def _x_tok_kernel(X, SRT, TE, TS, TM, START, PST, QT, STS, Mp, H: tl.constexpr, BC: tl.constexpr):
    """token copy of x in EXPERT order (gathered through the sort): the B5 right operand."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, 32)
    mr = tl.arange(0, 32) < mm
    tok = tl.load(SRT + rows, mask=mr, other=0)
    for c0 in range(0, H, BC):
        cols = c0 + tl.arange(0, BC)
        x = tl.load(X + tok[:, None].to(tl.int64) * H + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        _qtok_store(x, cols, col0, QT, STS, Mp, H)


@triton.jit
def _radial_fwd_tile_kernel(GU, GUS, ACT, ALPHA, TE, TS, TM, START, PST, QR, SR, QT, STS, RSS, ROUT, Mp,
                            I: tl.constexpr, EPS: tl.constexpr, BC: tl.constexpr,
                            NP: tl.constexpr, NPP: tl.constexpr, TOK: tl.constexpr = True,
                            GU8: tl.constexpr = False):
    """inter = r^p SiLU(g/r) u (codes 8/10) on 32 rows: row copy (F3 input) + token copy (B2 input)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, 32)
    mr = tl.arange(0, 32) < mm
    at = tl.load(ACT + e)                              # per EXPERT: a tile never crosses one
    aa = tl.load(ALPHA + e).to(tl.float32)
    base = GU + rows[:, None].to(tl.int64) * (2 * I)
    if NP > 0:
        jj = tl.arange(0, NPP)
        ss = tl.load(RSS + rows[:, None].to(tl.int64) * NP + jj[None, :], mask=mr[:, None] & (jj < NP)[None, :], other=0.0)
        r = tl.sqrt(tl.sum(ss, axis=1) / I + EPS)    # sum of squares from the F1 epilogue: no extra pass
    else:
        acc = tl.zeros((32,), tl.float32)
        for c0 in range(0, I, BC):
            g = _ld_gu(GU, GUS, rows, c0 + tl.arange(0, BC), mr, 2 * I, GU8, BC)
            acc += tl.sum(g * g, axis=1)
        r = tl.sqrt(acc / I + EPS)
    tl.store(ROUT + rows, r, mask=mr)
    p = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    rp = tl.exp(p * tl.log(r))
    for c0 in range(0, I, BC):
        cols = c0 + tl.arange(0, BC)
        g = _ld_gu(GU, GUS, rows, cols, mr, 2 * I, GU8, BC)
        u = _ld_gu(GU, GUS, rows, I + cols, mr, 2 * I, GU8, BC)
        z = g / r[:, None]
        v = tl.where(mr[:, None], rp[:, None] * (z * (1.0 / (1.0 + tl.exp(-z)))) * u, 0.0)
        _qrow_store(v, rows, cols, mr, QR, SR, I, 32, BC)
        if TOK:
            _qtok_store(v, cols, col0, QT, STS, Mp, I)


@triton.jit
def _radial_bwd_tile_kernel(GO, GOS, GU, GUS, ACT, ALPHA, TE, TS, TM, START, PST, QR, SR, QT, STS, DA, RIN, PART,
                            TW, TGW, Mp,
                            I: tl.constexpr, EPS: tl.constexpr, WANT_AP: tl.constexpr, BC: tl.constexpr,
                            NP: tl.constexpr, NPP: tl.constexpr, TOK: tl.constexpr = True,
                            GU8: tl.constexpr = False, GO8: tl.constexpr = False):
    """dGU of radial on 32 rows: row copy along 2I (B6 input) + token copy (B5 left operand)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, 32)
    mr = tl.arange(0, 32) < mm
    at = tl.load(ACT + e)                              # per EXPERT: a tile never crosses one
    aa = tl.load(ALPHA + e).to(tl.float32)
    gub = GU + rows[:, None].to(tl.int64) * (2 * I)
    gob = GO + rows[:, None].to(tl.int64) * I
    r = tl.load(RIN + rows, mask=mr, other=1.0)             # saved by the forward
    p = tl.where(at == 10, 2.0 / (1.0 + tl.exp(-2.0 * aa)) - 1.0, 1.0 / (1.0 + tl.exp(-aa)))
    lr = tl.log(r)
    rp = tl.exp(p * lr)
    rpm1 = tl.exp((p - 1.0) * lr)
    if NP > 0:
        jj = tl.arange(0, NPP)
        pm = mr[:, None] & (jj < NP)[None, :]
        sa = tl.sum(tl.load(PART + rows[:, None].to(tl.int64) * NP + jj[None, :], mask=pm, other=0.0), axis=1)
        # T = sum go*u*f = <d_inter, inter> / r^p = <dO, eo> / r^p = w * gw / r^p   (gw = <grad_out, eo>,
        # already computed by the combine backward): no pass over the row at all
        tt = tl.load(TW + rows, mask=mr, other=0.0) * tl.load(TGW + rows, mask=mr, other=0.0) / rp
    else:
        sa = tl.zeros((32,), tl.float32)
        for c0 in range(0, I, BC):
            cols = c0 + tl.arange(0, BC)
            go = _ld_gu(GO, GOS, rows, cols, mr, I, GO8, BC)
            g = _ld_gu(GU, GUS, rows, cols, mr, 2 * I, GU8, BC)
            u = _ld_gu(GU, GUS, rows, I + cols, mr, 2 * I, GU8, BC)
            gn = g / r[:, None]
            sig = 1.0 / (1.0 + tl.exp(-gn))
            sa += tl.sum(go * u * sig * (1.0 + gn * (1.0 - sig)) * gn, axis=1)
        tt = tl.load(TW + rows, mask=mr, other=0.0) * tl.load(TGW + rows, mask=mr, other=0.0) / rp
    for c0 in range(0, I, BC):
        cols = c0 + tl.arange(0, BC)
        go = _ld_gu(GO, GOS, rows, cols, mr, I, GO8, BC)
        g = _ld_gu(GU, GUS, rows, cols, mr, 2 * I, GU8, BC)
        u = _ld_gu(GU, GUS, rows, I + cols, mr, 2 * I, GU8, BC)
        gn = g / r[:, None]
        sig = 1.0 / (1.0 + tl.exp(-gn))
        f = gn * sig
        df = sig * (1.0 + gn * (1.0 - sig))
        gu_ = go * u
        gg = tl.where(mr[:, None], rpm1[:, None] * (gu_ * df - (gn / I) * (sa - p * tt)[:, None]), 0.0)
        gup = tl.where(mr[:, None], go * (rp[:, None] * f), 0.0)
        _qrow_store(gg, rows, cols, mr, QR, SR, 2 * I, 32, BC)
        _qrow_store(gup, rows, I + cols, mr, QR, SR, 2 * I, 32, BC)
        if TOK:
            _qtok_store(gg, cols, col0, QT, STS, Mp, 2 * I)
            _qtok_store(gup, I + cols, col0, QT, STS, Mp, 2 * I)
    if WANT_AP:                                        # d(theta) summed over the tile: one value per tile
        da = tl.where(mr, tl.where(at == 10, 1.0 - p * p, p * (1.0 - p)) * rp * lr * tt, 0.0)
        tl.store(DA + t, tl.sum(da, axis=0))


@triton.jit
def _combine_bwd_tile_kernel(GO, EO, W, SRT, TE, TS, TM, START, PST, QR, SR, QT, STS, GW, Mp,
                             H: tl.constexpr, BC: tl.constexpr):
    """dO = w * grad_out[token], dw = <grad_out[token], eo> on 32 rows: row copy (B3 input) + token
    copy (B2 left operand)."""
    t = tl.program_id(0)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    col0 = tl.load(PST + e) + (r0 - tl.load(START + e))
    rows = r0 + tl.arange(0, 32)
    mr = tl.arange(0, 32) < mm
    tok = tl.load(SRT + rows, mask=mr, other=0)
    w = tl.load(W + rows, mask=mr, other=0.0).to(tl.float32)
    gw = tl.zeros((32,), tl.float32)
    for c0 in range(0, H, BC):
        cols = c0 + tl.arange(0, BC)
        go = tl.load(GO + tok[:, None].to(tl.int64) * H + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        eo = tl.load(EO + rows[:, None].to(tl.int64) * H + cols[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        gw += tl.sum(go * eo, axis=1)
        ge = go * w[:, None]
        _qrow_store(ge, rows, cols, mr, QR, SR, H, 32, BC)
        _qtok_store(ge, cols, col0, QT, STS, Mp, H)
    tl.store(GW + rows, gw, mask=mr)


EPI_FUSE = False   # (measured a wash) S partials in the B3 epilogue (+ T from the combine grad) -> one-pass radial bwd
# (history: with bf16 GU and both S and T in the epilogue it was SLOWER; row reductions in the F1/B3 epilogues: measured SLOWER (B3 +0.9 ms reading G/U,
#                    radial bwd unchanged at 2.55 ms) -- the radial kernels are not pass-bound
RADIAL_WARPS, RADIAL_BC = 4, None   # tuning knobs (None = _bc)
GU_FP8 = True                       # F1 output GU cached in MXFP8 (DeepSeek-V3: SwiGLU input in fp8)
DI_FP8 = True                       # B3 output d_inter in MXFP8 (the radial bwd reads it twice)
_DEBUG_NO_TOK = False               # TIMING ONLY: skip the token-copy stores (wgrads then read garbage)
KPAD = 128    # expert token ranges in the token copies are padded to this: the wgrad K loop runs UNMASKED
#               (masked fp8 K loads were 2.8x slower than bf16; unmasked is 1.9-2.4x FASTER)


def _tiles(counts_t, M, dev):
    TE, TS, TM = FG.build_tile_map(None, counts_t, dev, bm=32, m_rows=M)
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

        xq, xs = _q(hidden, "F1 in (x)")
        xT, xTs = _tok_buf(H, Mp, dev)
        _x_tok_kernel[(nt,)](hidden, st, *tiles, xT, xTs, Mp, H, _bc(H), num_warps=4)
        _pad(xT, xTs, tiles[4], pads, pcnt, Mp)
        np1 = I // MX.gemm_bn(H, 2 * I) if (EPI_FUSE or GU_FP8) else 0
        rss = torch.empty(M, max(np1, 1), device=dev, dtype=torch.float32) if np1 else row_alpha
        if GU_FP8:
            gus = torch.empty(M, 2 * I // 32, device=dev, dtype=torch.uint8)
            gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=st, epi=3, x1=gus, x2=rss,
                                 np_=np1)                     # F1 -> GU fp8 + per-row sum(g^2) partials
        else:
            gus = row_alpha
            gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=st, epi=1 if EPI_FUSE else 0,
                                 x1=rss if EPI_FUSE else None, np_=max(np1, 1))       # F1 (+ sum g^2)
        iq, is_ = _row_buf(M, I, dev)
        iT, iTs = _tok_buf(I, Mp, dev)
        r = torch.empty(M, device=dev, dtype=torch.float32)
        _radial_fwd_tile_kernel[(nt,)](gu, gus, row_act, row_alpha, *tiles, iq, is_, iT, iTs, rss, r, Mp, I, _EPS,
                                       RADIAL_BC or _bc(I), np1, triton.next_power_of_2(max(np1, 1)),
                                       num_warps=RADIAL_WARPS, TOK=not _DEBUG_NO_TOK, GU8=GU_FP8)
        _pad(iT, iTs, tiles[4], pads, pcnt, Mp)
        eo = MX.grouped_gemm(iq, is_, *wdn["rc"], counts_t, M)                     # F3 -> (M, H) bf16
        inv = FG.inverse_order(order)
        out = FG.combine_gather(eo, inv, N, top_k, w=sw, out_dtype=hidden.dtype)
        if STATS is not None:
            STATS.setdefault("F3 in (act*up): exact-0 %", []).append(
                (100 * (iq.float() == 0).float().mean().item(), 0.0))

        ctx.save_for_backward(st, sw, order, row_act, row_alpha, gu, eo, xT, xTs, iT, iTs, pcnt,
                              pads, r, gus, *tiles)
        ctx.inv, ctx.counts_t, ctx.Mp, ctx.wq = inv, counts_t, Mp, (wgu, wdn)
        ctx.shapes = (N, H, top_k, E, M, I)
        ctx.ap_shape = ap_shape
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (st, sw, order, row_act, row_alpha, gu, eo, xT, xTs, iT, iTs, pcnt, pads, r, gus, *tiles) = ctx.saved_tensors
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
                                        num_warps=4)
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
                                       I, _EPS, want_ap, RADIAL_BC or _bc(I), np3,
                                       triton.next_power_of_2(max(np3, 1)), num_warps=RADIAL_WARPS,
                                       TOK=not _DEBUG_NO_TOK, GU8=GU_FP8, GO8=DI_FP8 and not fuse_st)
        _pad(dT, dTs, pst, pads, pcnt, Mp)
        grad_ap = _ap_grad_from_tiles(da, ctx.counts_t, E, ctx.ap_shape) if want_ap else None
        grad_gu = _acc_wgrad(ctx.acc[0], lambda out, a: MX.wgrad_kmajor(dT, dTs, xT, xTs, pst, pcnt, E,
                                                                        out=out, accumulate=a, even=True))         # B5
        dx_rows = MX.grouped_gemm(dq, ds, *wgu["cr"], ctx.counts_t, M, out_dtype=DX_ROWS)      # B6
        grad_hidden = FG.combine_gather(dx_rows, ctx.inv, N, top_k, out_dtype=grad_out.dtype)
        grad_wt = torch.zeros(N * top_k, device=dev, dtype=grad_out.dtype)
        grad_wt[order] = gw.to(grad_out.dtype)
        return grad_hidden, None, grad_wt.view(N, top_k), grad_gu, grad_down, None, grad_ap


def _ap_grad_from_tiles(da_t, counts_t, E, ap_shape):
    """per-TILE d(theta) -> per EXPERT. Tiles are expert-sorted (32-row, never crossing an expert), so
    each expert owns a contiguous run of ceil(count / 32) tiles: fixed-order fp64 prefix sums
    differenced at the run ends -- deterministic, no atomics."""
    nt_e = (counts_t + 31) // 32
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


def moe_fp8(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    if _full_ok(hidden, act_codes, act_params, gate_up_proj):
        return _MoEFP8Full.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
    return _MoEFP8.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
