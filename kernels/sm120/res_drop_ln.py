"""Fused conformer sublayer boundary (sm120): r' = residual + dropout(x) * factor;  y = LayerNorm(r').

NeMo's ConformerLayer does this 4 times per layer (ff1 -> attn -> conv -> ff2 -> out), eagerly: dropout, mul, add,
layer_norm forward, and in the backward layer_norm's input grad, its GammaBeta weight-grad kernel, the add / mul /
dropout grads. Here: one forward kernel (reads residual and x once, writes r' and y, keeps mean / rstd per row) and
one backward kernel (grad from the residual stream + the LN grad -> d_residual and dx, with the dropout mask
regenerated from the seed; LN weight / bias grads as per-block partial column sums, summed in a fixed order).

Rounding follows eager exactly: dropout's output is rounded to x's dtype (bf16), times factor in that dtype, the add
in the residual's dtype (fp32 under autocast: LayerNorm outputs fp32 there; bf16 with --bf16_master). The LN runs in
fp32 and writes y in the residual's dtype, as torch's layer_norm does for these inputs. Backward: dx goes through the
same casts (bf16(bf16(bf16(dr) * factor) * mask * scale)), as autograd's dtype-promotion and bf16 nodes do.

    r, y = res_dropout_layernorm(residual, x, ln.weight, ln.bias, ln.eps, p=0.1, factor=0.5, seed=seed)
    y = res_dropout_layernorm(residual, None, ...)[1]                 # LN alone (a layer's first norm)
"""
import torch
import triton
import triton.language as tl

__all__ = ["res_dropout_layernorm"]


@triton.jit
def _fwd(RES, X, W, B, ROUT, Y, MEAN, RSTD, R, C, eps, seed, p, scale, factor,
         BR: tl.constexpr, BC: tl.constexpr, HAS_X: tl.constexpr, DROP: tl.constexpr, SCALE_F: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    cols = tl.arange(0, BC)
    mr = rows < R
    mc = cols < C
    m2 = mr[:, None] & mc[None, :]
    off = rows[:, None].to(tl.int64) * C + cols[None, :]
    r = tl.load(RES + off, mask=m2, other=0.0).to(tl.float32)
    if HAS_X:
        x = tl.load(X + off, mask=m2, other=0.0)
        xt = x.dtype
        xf = x.to(tl.float32)
        if DROP:
            keep = tl.rand(seed, (rows[:, None] * C + cols[None, :]).to(tl.int32)) >= p
            xf = tl.where(keep, (xf * scale).to(xt).to(tl.float32), 0.0)
        if SCALE_F:
            xf = (xf * factor).to(xt).to(tl.float32)
        r = (r + xf).to(ROUT.dtype.element_ty).to(tl.float32)
        tl.store(ROUT + off, r.to(ROUT.dtype.element_ty), mask=m2)
    mean = tl.sum(r, axis=1) / C
    d = tl.where(m2, r - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=1) / C + eps)
    w = tl.load(W + cols, mask=mc, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=mc, other=0.0).to(tl.float32)
    tl.store(Y + off, (d * rstd[:, None] * w[None, :] + b[None, :]).to(Y.dtype.element_ty), mask=m2)
    tl.store(MEAN + rows, mean, mask=mr)
    tl.store(RSTD + rows, rstd, mask=mr)


@triton.jit
def _bwd(DROUT, DY, RIN, W, MEAN, RSTD, DRES, DX, DWP, DBP, R, C, seed, p, scale, factor,
         BR: tl.constexpr, BC: tl.constexpr, HAS_X: tl.constexpr, HAS_DROUT: tl.constexpr, DROP: tl.constexpr,
         SCALE_F: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BR + tl.arange(0, BR)
    cols = tl.arange(0, BC)
    mr = rows < R
    mc = cols < C
    m2 = mr[:, None] & mc[None, :]
    off = rows[:, None].to(tl.int64) * C + cols[None, :]
    mean = tl.load(MEAN + rows, mask=mr, other=0.0)
    rstd = tl.load(RSTD + rows, mask=mr, other=0.0)
    xhat = tl.where(m2, (tl.load(RIN + off, mask=m2, other=0.0).to(tl.float32) - mean[:, None]) * rstd[:, None], 0.0)
    dy = tl.load(DY + off, mask=m2, other=0.0).to(tl.float32)
    w = tl.load(W + cols, mask=mc, other=0.0).to(tl.float32)
    g = dy * w[None, :]
    c1 = tl.sum(g * xhat, axis=1) / C
    c2 = tl.sum(g, axis=1) / C
    # eager rounds twice in bf16: layer_norm's input grad is written in the input dtype, then autograd's
    # accumulation adds the residual-stream grad in that dtype. One rounding would be MORE accurate = a model change.
    dr = (rstd[:, None] * (g - xhat * c1[:, None] - c2[:, None])).to(DRES.dtype.element_ty)
    if HAS_DROUT:
        dr = (dr.to(tl.float32) + tl.load(DROUT + off, mask=m2, other=0.0).to(tl.float32)).to(DRES.dtype.element_ty)
    tl.store(DRES + off, dr, mask=m2)
    if HAS_X:
        xt = DX.dtype.element_ty
        dx = dr.to(xt).to(tl.float32)
        if SCALE_F:
            dx = (dx * factor).to(xt).to(tl.float32)
        if DROP:
            keep = tl.rand(seed, (rows[:, None] * C + cols[None, :]).to(tl.int32)) >= p
            dx = tl.where(keep, dx * scale, 0.0)
        tl.store(DX + off, dx.to(xt), mask=m2)
    tl.store(DWP + pid * C + cols, tl.sum(dy * xhat, axis=0), mask=mc)
    tl.store(DBP + pid * C + cols, tl.sum(dy, axis=0), mask=mc)


_CFG = {"fwd": None, "bwd": None}                 # (rows per program, warps) override for bench_res_drop_ln.py


def _cfg(C, which):
    BC = triton.next_power_of_2(C)
    if _CFG[which]:
        return _CFG[which][0], BC, _CFG[which][1]
    # swept at 15000 x 512 (bench_res_drop_ln.py): fwd 1 row / 4 warps 83 us, bwd 8 rows / 4 warps 114 us (eager 122 / 208)
    return (1, BC, 4) if which == "fwd" else (max(1, min(8, 4096 // BC)), BC, 4)


class _ResDropLN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, res, x, w, b, eps, p, factor, seed, ydt):
        shp = res.shape
        C = shp[-1]
        res2 = res.reshape(-1, C).contiguous()
        R = res2.shape[0]
        x2 = x.reshape(-1, C).contiguous() if x is not None else res2
        rout = torch.empty_like(res2) if x is not None else res2.clone()      # an output, never the input itself
        y = torch.empty(res2.shape, device=res.device, dtype=ydt or res.dtype)
        mean = torch.empty(R, device=res.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        BR, BC, NW = _cfg(C, "fwd")
        drop = x is not None and p > 0
        _fwd[(triton.cdiv(R, BR),)](res2, x2, w, b, rout, y, mean, rstd, R, C, eps, seed, p, 1.0 / (1.0 - p),
                                    factor, BR=BR, BC=BC, HAS_X=x is not None, DROP=drop, SCALE_F=factor != 1.0,
                                    num_warps=NW)
        ctx.save_for_backward(rout, w, mean, rstd)
        ctx.cfg = (shp, x is not None, x.dtype if x is not None else None, p, factor, seed, w.dtype, b.dtype)
        return rout.view(shp), y.view(shp)

    @staticmethod
    def backward(ctx, drout, dy):
        rout, w, mean, rstd = ctx.saved_tensors
        shp, has_x, xdt, p, factor, seed, wdt, bdt = ctx.cfg
        R, C = rout.shape
        BR, BC, NW = _cfg(C, "bwd")
        nb = triton.cdiv(R, BR)
        dy = dy.reshape(-1, C).contiguous() if dy is not None else torch.zeros_like(rout)
        has_dr = drout is not None
        dr_in = drout.reshape(-1, C).contiguous() if has_dr else dy
        dres = torch.empty_like(rout)
        dx = torch.empty(R, C, device=rout.device, dtype=xdt) if has_x else dres
        dwp = torch.empty(nb, C, device=rout.device, dtype=torch.float32)
        dbp = torch.empty_like(dwp)
        _bwd[(nb,)](dr_in, dy, rout, w, mean, rstd, dres, dx, dwp, dbp, R, C, seed, p, 1.0 / (1.0 - p), factor,
                    BR=BR, BC=BC, HAS_X=has_x, HAS_DROUT=has_dr, DROP=has_x and p > 0, SCALE_F=factor != 1.0,
                    num_warps=NW)
        return (dres.view(shp), dx.view(shp) if has_x else None, dwp.sum(0).to(wdt), dbp.sum(0).to(bdt),
                None, None, None, None, None)


def res_dropout_layernorm(residual, x, weight, bias, eps, p=0.0, factor=1.0, seed=None, y_dtype=None):
    """(residual + dropout(x) * factor, LayerNorm(that)); x=None: (residual, LayerNorm(residual)). p = the caller's
    (0 outside training). Rows = all leading dims, LN over the last. y_dtype: the LN output dtype (fp32 under autocast,
    whatever the residual's; default = the residual's)."""
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))
    return _ResDropLN.apply(residual, x, weight, bias, float(eps), float(p), float(factor), seed, y_dtype)
