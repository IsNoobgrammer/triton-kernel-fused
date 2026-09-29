"""MoE experts with MXFP8 expert GEMMs (phase 1): the same call and returns as moe_per_expert.

    moe_fp8(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)

Everything from the router weights to the weighted sum:
  fwd    xq = MXFP8(hidden)                       (unsorted tokens, blocks along H)
         GU = xq[st] @ Wgu^T                      F1, fp8, row gather inside the GEMM
         inter = radial(G) * U                    bf16 act kernel (existing _glu_fwd, per-row theta)
         EO = MXFP8(inter) @ Wdn^T                F3, fp8
         out = sum_k w_k * EO                     fp32 combine (router weights fp32), deterministic
  bwd    dO, dw = combine_bwd                     fp32 math, existing kernel
         d_inter = MXFP8(dO) @ Wdn                B3, fp8
         dGU, dtheta = radial bwd                 existing _glu_bwd
         dx rows = MXFP8(dGU) @ Wgu               B6, fp8, then the deterministic k-way sum
         dWdn = dO^T @ inter, dWgu = dGU^T @ x    B2 / B5: bf16 grouped wgrad (phase 2 = fp8)
Weights: MXFP8 from the fp32 master, 2D 32x32 blocks, cached per optimizer step (mxfp8.quant_weight).

STATS: set moe_fp8.STATS = {} to collect (flushed %, saturated %) of every fp8 operand per call.
"""
import importlib

import torch

# importlib, not "from kernels.sm75 import moe": the package re-exports a FUNCTION named moe
K75 = importlib.import_module("kernels.sm75.moe")
MX = importlib.import_module("kernels.sm120.mxfp8")
FG = importlib.import_module("kernels.sm120.moe_fused_glu")

STATS = None


def _q(x, tag):
    q, s = MX.quant_rows(x.contiguous())
    if STATS is not None:
        STATS.setdefault(tag, []).append(MX.qstats(x, q, s))
    return q, s


def _wstats(w, tag, qs):
    if STATS is not None:
        q, s = qs
        STATS.setdefault(tag, []).append(MX.qstats(w.reshape(-1, w.shape[-1]).float(),
                                                   q.reshape(-1, q.shape[-1]), s.reshape(-1, s.shape[-1])))


class _MoEFP8(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, idx, wt, gate_up_proj, down_proj, act_codes, act_params=None):
        ctx.acc = (K75._acc_target(gate_up_proj), K75._acc_target(down_proj))
        wgu = MX.quant_weight(gate_up_proj)                 # from the fp32 MASTER, before any cast
        wdn = MX.quant_weight(down_proj)
        _wstats(gate_up_proj, "W gate_up", wgu["rc"]); _wstats(down_proj, "W down", wdn["rc"])
        hidden, wt, gate_up_bf, down_bf = K75._amp_cast(hidden, wt, gate_up_proj, down_proj)
        hidden = hidden.contiguous()
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

        xq, xs = _q(hidden, "F1 in (x)")
        gu = MX.grouped_gemm(xq, xs, *wgu["rc"], counts_t, M, rows=st)            # (M, 2I) bf16
        inter = K75._glu_fwd(gu, row_act, code_hint=hint, row_alpha=row_alpha)
        iq, is_ = _q(inter, "F3 in (act*up)")
        eo = MX.grouped_gemm(iq, is_, *wdn["rc"], counts_t, M)                     # (M, H) bf16
        inv = FG.inverse_order(order)
        out = FG.combine_gather(eo, inv, N, top_k, w=sw, out_dtype=hidden.dtype)

        ctx.save_for_backward(hidden, st, sw, order, row_act, gu, inter, eo, gate_up_bf, down_bf)
        ctx.inv, ctx.offs, ctx.counts_t = inv, offs, counts_t
        ctx.wq = (wgu, wdn)
        ctx.shapes = (N, H, top_k, E, M)
        ctx.row_alpha, ctx.row_expert, ctx.ap_shape, ctx.hint = row_alpha, row_expert, ap_shape, hint
        return out

    @staticmethod
    def backward(ctx, grad_out):
        hidden, st, sw, order, row_act, gu, inter, eo, gate_up_bf, down_bf = ctx.saved_tensors
        N, H, top_k, E, M = ctx.shapes
        wgu, wdn = ctx.wq
        grad_out = grad_out.contiguous()
        ge, gw = K75._combine_bwd(grad_out, eo, sw, st)                             # dO (M, H), d w
        grad_down = K75._wgrad(ge, inter, ctx.offs, acc=ctx.acc[1])                 # B2 (bf16, phase 1)
        gq, gs = _q(ge, "B3 in (dO)")
        d_inter = MX.grouped_gemm(gq, gs, *wdn["cr"], ctx.counts_t, M)             # B3 (M, I)
        grad_ap = None
        if ctx.row_alpha is not None and ctx.needs_input_grad[6]:
            dgu, da = K75._glu_bwd(d_inter, gu, row_act, code_hint=ctx.hint, row_alpha=ctx.row_alpha,
                                   want_act_grads=True)
            grad_ap = K75._ap_grad_from_rows(da, ctx.row_expert, E, ctx.ap_shape, grad_out.device)
        else:
            dgu = K75._glu_bwd(d_inter, gu, row_act, code_hint=ctx.hint, row_alpha=ctx.row_alpha)
        grad_gu = K75._wgrad(dgu, hidden, ctx.offs, acc=ctx.acc[0], b_rows=st)      # B5 (bf16, phase 1)
        dq, ds = _q(dgu, "B6 in (dGU)")
        dx_rows = MX.grouped_gemm(dq, ds, *wgu["cr"], ctx.counts_t, M, out_dtype=torch.float32)  # B6
        grad_hidden = FG.combine_gather(dx_rows, ctx.inv, N, top_k, out_dtype=grad_out.dtype)
        grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = gw.to(grad_out.dtype)
        return grad_hidden, None, grad_wt.view(N, top_k), grad_gu, grad_down, None, grad_ap


def moe_fp8(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    return _MoEFP8.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params)
