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
DX_ROWS = torch.float32   # B6 row buffer before the k-way sum (bf16 halves its traffic)
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


def moe_fp8(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    return _MoEFP8.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
