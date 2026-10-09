"""Fused conformer convolution module (sm120), between its two pointwise linears, channel-last:

    y = Swish(LayerNorm(DepthwiseConv1d(masked_fill(GLU(g), pad, 0))))       g = pointwise_conv1 output (B, T, 2C)

NeMo's ConformerConvolution (conv_norm_type layer_norm) runs this as glu, transpose + masked_fill copy, F.pad copy,
cuDNN depthwise conv (channel-first), transpose + fp32 cast copy for the LN, LN, sigmoid, mul (and as many backward
kernels). Here: one forward kernel (reads g once per tap from L2, writes z = the conv output and y) and two backward
kernels (dz with the LN / Swish grads; then dh from dz's taps, the GLU grad -> dg). Weight grads are per-program
partial column sums, summed in a fixed order (deterministic).

Rounding follows eager under bf16 autocast exactly where eager rounds: GLU output in g's dtype, the conv in g's dtype
(bf16 weights / bias, fp32 accumulate, output rounded), LN + Swish in fp32, y written in out_dtype. Backward: the LN
input grad rounded to g's dtype (autocast's cast node), the conv grads in g's dtype (dW / db rounded after the full
sum, as cuDNN's bf16 weight grad then the cast back), the GLU grads rounded. fp32 g: no rounding anywhere.

    y = conv_module(g, pad_mask, dw.weight, dw.bias, ln.weight, ln.bias, ln.eps, left_pad, out_dtype=torch.bfloat16)
"""
import torch
import triton
import triton.language as tl

__all__ = ["conv_module"]


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _cm_fwd(G, PAD, CW, CB, LW, LB, Z, Y, MEAN, RSTD, T, C, eps,
            LEFT: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, HAS_PAD: tl.constexpr):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    t = tb * BT + tl.arange(0, BT)
    c = tl.arange(0, BC)
    mc = c < C
    dt = G.dtype.element_ty
    gb = G + b.to(tl.int64) * T * 2 * C
    acc = tl.zeros((BT, BC), tl.float32)
    for k in tl.static_range(K):
        ts = t - LEFT + k
        ok = (ts >= 0) & (ts < T)
        if HAS_PAD:
            ok &= tl.load(PAD + b * T + ts, mask=ok, other=1) == 0
        m2 = ok[:, None] & mc[None, :]
        a = tl.load(gb + ts[:, None] * 2 * C + c[None, :], mask=m2, other=0.0).to(tl.float32)
        s = tl.load(gb + ts[:, None] * 2 * C + C + c[None, :], mask=m2, other=0.0).to(tl.float32)
        h = (a * _sig(s)).to(dt).to(tl.float32)                     # glu in fp32, rounded once
        w = tl.load(CW + c * K + k, mask=mc, other=0.0).to(dt).to(tl.float32)
        acc += tl.where(m2, h, 0.0) * w[None, :]
    mt = t < T
    m2 = mt[:, None] & mc[None, :]
    z = (acc + tl.load(CB + c, mask=mc, other=0.0).to(dt).to(tl.float32)[None, :]).to(dt)
    off = (b.to(tl.int64) * T + t)[:, None] * C + c[None, :]
    tl.store(Z + off, z, mask=m2)
    zf = tl.where(m2, z.to(tl.float32), 0.0)
    mean = tl.sum(zf, axis=1) / C
    d = tl.where(m2, zf - mean[:, None], 0.0)
    rstd = 1.0 / tl.sqrt(tl.sum(d * d, axis=1) / C + eps)
    n = d * rstd[:, None] * tl.load(LW + c, mask=mc, other=0.0)[None, :] + tl.load(LB + c, mask=mc, other=0.0)[None, :]
    tl.store(Y + off, (n * _sig(n)).to(Y.dtype.element_ty), mask=m2)
    tl.store(MEAN + b * T + t, mean, mask=mt)
    tl.store(RSTD + b * T + t, rstd, mask=mt)


@triton.jit
def _cm_bwd_z(DY, Z, LW, LB, MEAN, RSTD, DZ, LWP, LBP, CBP, T, C,
              BT: tl.constexpr, BC: tl.constexpr):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    pid = b * tl.num_programs(0) + tb
    t = tb * BT + tl.arange(0, BT)
    c = tl.arange(0, BC)
    mt = t < T
    mc = c < C
    m2 = mt[:, None] & mc[None, :]
    off = (b.to(tl.int64) * T + t)[:, None] * C + c[None, :]
    mean = tl.load(MEAN + b * T + t, mask=mt, other=0.0)
    rstd = tl.load(RSTD + b * T + t, mask=mt, other=0.0)
    xhat = tl.where(m2, (tl.load(Z + off, mask=m2, other=0.0).to(tl.float32) - mean[:, None]) * rstd[:, None], 0.0)
    w = tl.load(LW + c, mask=mc, other=0.0)
    n = xhat * w[None, :] + tl.load(LB + c, mask=mc, other=0.0)[None, :]
    s = _sig(n)
    dy = tl.load(DY + off, mask=m2, other=0.0).to(tl.float32)
    dn = tl.where(m2, dy * s * (1.0 + n * (1.0 - s)), 0.0)          # silu_backward (NeMo Swish = nn.SiLU)
    g = dn * w[None, :]
    c1 = tl.sum(g * xhat, axis=1) / C
    c2 = tl.sum(g, axis=1) / C
    dz = (rstd[:, None] * (g - xhat * c1[:, None] - c2[:, None])).to(DZ.dtype.element_ty)
    tl.store(DZ + off, dz, mask=m2)
    tl.store(LWP + pid * C + c, tl.sum(dn * xhat, axis=0), mask=mc)
    tl.store(LBP + pid * C + c, tl.sum(dn, axis=0), mask=mc)
    tl.store(CBP + pid * C + c, tl.sum(tl.where(m2, dz.to(tl.float32), 0.0), axis=0), mask=mc)


@triton.jit
def _cm_bwd_g(G, PAD, CW, DZ, DG, CWP, T, C,
              LEFT: tl.constexpr, K: tl.constexpr, BT: tl.constexpr, BC: tl.constexpr, HAS_PAD: tl.constexpr):
    tb = tl.program_id(0)
    b = tl.program_id(1)
    pid = b * tl.num_programs(0) + tb
    t = tb * BT + tl.arange(0, BT)
    c = tl.arange(0, BC)
    mc = c < C
    ok = t < T
    if HAS_PAD:
        ok &= tl.load(PAD + b * T + t, mask=t < T, other=1) == 0
    m2 = ok[:, None] & mc[None, :]
    dt = G.dtype.element_ty
    goff = (b.to(tl.int64) * T + t)[:, None] * 2 * C + c[None, :]
    a = tl.load(G + goff, mask=m2, other=0.0).to(tl.float32)
    s = _sig(tl.load(G + goff + C, mask=m2, other=0.0).to(tl.float32))
    h = tl.where(m2, (a * s).to(dt).to(tl.float32), 0.0)
    dh = tl.zeros((BT, BC), tl.float32)
    dzb = DZ + b.to(tl.int64) * T * C
    for k in tl.static_range(K):
        to = t + LEFT - k                                           # the output row that read row t through tap k
        mo = (to >= 0) & (to < T)
        dz = tl.load(dzb + to[:, None] * C + c[None, :], mask=mo[:, None] & mc[None, :], other=0.0).to(tl.float32)
        w = tl.load(CW + c * K + k, mask=mc, other=0.0).to(dt).to(tl.float32)
        dh += dz * w[None, :]
        tl.store(CWP + (pid * K + k) * C + c, tl.sum(h * dz, axis=0), mask=mc)
    dh = tl.where(m2, dh.to(dt).to(tl.float32), 0.0)                # conv input grad in g's dtype; masked_fill bwd
    mg = (t < T)[:, None] & mc[None, :]
    tl.store(DG + goff, (dh * s).to(DG.dtype.element_ty), mask=mg)
    tl.store(DG + goff + C, (dh * a * ((1.0 - s) * s)).to(DG.dtype.element_ty), mask=mg)


_CFG = {"fwd": None, "bwd": None}                 # (rows per program, warps) override for bench_conv_module.py


def _cfg(C, which):
    BC = triton.next_power_of_2(C)
    if _CFG[which]:
        return (BC,) + _CFG[which]
    # swept at 30400 frames x 512 (bench_conv_module.py --sweep): fwd 4 rows / 8 warps, bwd 16 rows / 8 warps
    return (BC, max(1, 2048 // BC), 8) if which == "fwd" else (BC, max(1, 8192 // BC), 8)


class _ConvModule(torch.autograd.Function):
    @staticmethod
    def forward(ctx, g, pad, cw, cb, lw, lb, eps, left, odt):
        B, T, C2 = g.shape
        C, K = C2 // 2, cw.shape[-1]
        g = g.contiguous()
        cw2 = cw.reshape(C, K).contiguous()
        z = torch.empty(B, T, C, device=g.device, dtype=g.dtype)
        y = torch.empty(B, T, C, device=g.device, dtype=odt or g.dtype)
        mean = torch.empty(B * T, device=g.device, dtype=torch.float32)
        rstd = torch.empty_like(mean)
        BC, BT, NW = _cfg(C, "fwd")
        has_pad = pad is not None
        pad8 = pad.contiguous().view(torch.uint8) if has_pad else g
        _cm_fwd[(triton.cdiv(T, BT), B)](g, pad8, cw2, cb, lw, lb, z, y, mean, rstd, T, C, eps,
                                         LEFT=left, K=K, BT=BT, BC=BC, HAS_PAD=has_pad, num_warps=NW)
        ctx.save_for_backward(g, pad8, cw2, lw, lb, z, mean, rstd)
        ctx.cfg = (has_pad, left, cw.shape, cw.dtype, cb.dtype, lw.dtype, lb.dtype)
        return y

    @staticmethod
    def backward(ctx, dy):
        g, pad8, cw2, lw, lb, z, mean, rstd = ctx.saved_tensors
        has_pad, left, cws, cwdt, cbdt, lwdt, lbdt = ctx.cfg
        B, T, C = z.shape
        K = cw2.shape[1]
        BC, BT, NW = _cfg(C, "bwd")
        grid = (triton.cdiv(T, BT), B)
        np_ = grid[0] * B
        dz = torch.empty_like(z)
        lwp = torch.empty(np_, C, device=g.device, dtype=torch.float32)
        lbp, cbp = torch.empty_like(lwp), torch.empty_like(lwp)
        _cm_bwd_z[grid](dy.contiguous(), z, lw, lb, mean, rstd, dz, lwp, lbp, cbp, T, C, BT=BT, BC=BC, num_warps=NW)
        dg = torch.empty_like(g)
        cwp = torch.empty(np_, K, C, device=g.device, dtype=torch.float32)
        _cm_bwd_g[grid](g, pad8, cw2, dz, dg, cwp, T, C, LEFT=left, K=K, BT=BT, BC=BC, HAS_PAD=has_pad, num_warps=NW)
        dt = g.dtype                                                # cuDNN's weight grads in g's dtype, then cast back
        dcw = cwp.sum(0).t().to(dt).to(cwdt).reshape(cws)
        dcb = cbp.sum(0).to(dt).to(cbdt)
        return dg, None, dcw, dcb, lwp.sum(0).to(lwdt), lbp.sum(0).to(lbdt), None, None, None


def conv_module(g, pad_mask, conv_weight, conv_bias, ln_weight, ln_bias, eps, left_pad, out_dtype=None):
    """g: (B, T, 2C) pointwise_conv1 output; pad_mask: (B, T) bool, True = padding (or None); conv_weight (C, 1, K);
    left_pad: the causal conv's left padding (right = K - 1 - left). -> (B, T, C) in out_dtype (default g's)."""
    assert conv_bias is not None and conv_weight.shape[-1] - 1 - left_pad >= 0
    return _ConvModule.apply(g, pad_mask, conv_weight, conv_bias, ln_weight, ln_bias, float(eps), int(left_pad),
                             out_dtype)
