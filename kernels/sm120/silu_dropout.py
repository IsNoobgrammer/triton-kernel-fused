"""SiLU + dropout in one pass (sm120): NeMo ConformerFeedForward's `dropout(Swish(linear1(x)))`.

Eager runs silu, dropout (writes the output and a bool mask) forward and dropout grad, silu_backward backward, saving
both the linear1 output and the mask. Here one kernel each way, the mask regenerated from a seed (Triton RNG), only
the linear1 output saved. Rounding as eager: silu in fp32 rounded to x's dtype, then * mask * 1/(1-p) rounded;
backward dropout grad rounded, then silu_backward (fp32, rounded once).

    y = silu_dropout(linear1_out, p, seed)
"""
import torch
import triton
import triton.language as tl

__all__ = ["silu_dropout"]


@triton.jit
def _sd_fwd(X, Y, N, seed, p, scale, BLOCK: tl.constexpr, DROP: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0.0)
    dt = x.dtype
    xf = x.to(tl.float32)
    a = (xf / (1.0 + tl.exp(-xf))).to(dt).to(tl.float32)
    if DROP:
        a = tl.where(tl.rand(seed, off) >= p, a * scale, 0.0)
    tl.store(Y + off, a.to(dt), mask=m)


@triton.jit
def _sd_bwd(X, DY, DX, N, seed, p, scale, BLOCK: tl.constexpr, DROP: tl.constexpr):
    off = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = off < N
    x = tl.load(X + off, mask=m, other=0.0)
    dt = x.dtype
    xf = x.to(tl.float32)
    d = tl.load(DY + off, mask=m, other=0.0).to(tl.float32)
    if DROP:
        d = tl.where(tl.rand(seed, off) >= p, (d * scale).to(dt).to(tl.float32), 0.0)
    s = 1.0 / (1.0 + tl.exp(-xf))
    tl.store(DX + off, (d * s * (1.0 + xf * (1.0 - s))).to(dt), mask=m)


_BLOCK = 4096


class _SiluDropout(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p, seed):
        x = x.contiguous()
        N = x.numel()
        assert N < 2 ** 31
        y = torch.empty_like(x)
        _sd_fwd[(triton.cdiv(N, _BLOCK),)](x, y, N, seed, p, 1.0 / (1.0 - p), BLOCK=_BLOCK, DROP=p > 0, num_warps=8)
        ctx.save_for_backward(x)
        ctx.cfg = (p, seed)
        return y

    @staticmethod
    def backward(ctx, dy):
        (x,) = ctx.saved_tensors
        p, seed = ctx.cfg
        dx = torch.empty_like(x)
        N = x.numel()
        _sd_bwd[(triton.cdiv(N, _BLOCK),)](x, dy.contiguous(), dx, N, seed, p, 1.0 / (1.0 - p), BLOCK=_BLOCK,
                                           DROP=p > 0, num_warps=8)
        return dx, None, None


def silu_dropout(x, p=0.0, seed=None):
    """dropout(silu(x)), p = the caller's (0 outside training)."""
    if seed is None:
        seed = int(torch.randint(0, 2 ** 31 - 1, ()))
    return _SiluDropout.apply(x, float(p), seed)
