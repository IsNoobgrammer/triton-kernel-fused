"""Whole MLP block as one autograd.Function: norm -> router -> experts -> weighted combine.

Candidate 1 is deliberately conservative. The fused norm+router kernel replaces liger+eager, and
the expert path reuses `moe_per_expert` unchanged. The backward RECOMPUTES the norm and router in
PyTorch from the saved input rather than saving their activations.

Recompute is correct but not free, and the reason it is correct is subtle: `rstd` depends on `x`,
so reusing the saved `rstd` in the backward would drop d(rstd)/dx and produce a silently wrong
gradient. The recomputed graph carries that term.

What the backward must NEVER do is re-run the ROUTER's selection: the load-balancing bias is
mutated by `.add_()` outside the optimizer and must fire exactly once per step. The saved top-k
indices are reused, and only the differentiable path (scores -> gathered weights) is rebuilt.
"""
import torch

from .norm_router import norm_router_forward


class _NormRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nw, rw, bias, top_k, eps):
        hn, idx, wgt, rstd, counts = norm_router_forward(
            x, nw, rw, bias, top_k, eps, write_hn=True)
        ctx.save_for_backward(x, nw, rw, idx)
        ctx.top_k, ctx.eps = top_k, eps
        ctx.mark_non_differentiable(idx)
        return hn, idx, wgt

    @staticmethod
    def backward(ctx, g_hn, g_idx, g_wgt):
        x, nw, rw, idx = ctx.saved_tensors
        with torch.enable_grad():
            xd = x.detach().requires_grad_(True)
            nwd = nw.detach().requires_grad_(True)
            rwd = rw.detach().requires_grad_(True)
            f = xd.float()
            # full recompute, INCLUDING rstd: it depends on x, and reusing the saved value would
            # drop d(rstd)/dx and give a wrong gradient that no shape check would catch
            hn = (f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + ctx.eps)).to(x.dtype) * nwd
            scores = torch.sigmoid((hn @ rwd).float())
            # idx is REUSED, never recomputed -- re-running the selection would apply the
            # load-balancing bias a second time
            w = scores.gather(-1, idx.long())
            w = w / (w.sum(-1, keepdim=True) + 1e-20)
            gx, gnw, grw = torch.autograd.grad(
                [hn, w], [xd, nwd, rwd], [g_hn, g_wgt], allow_unused=True)
        return gx, gnw, grw, None, None, None


def megakernel_block(x, w, codes, top_k=6, eps=1e-6):
    """Signature matches bench.eval_mlp_block.baseline_block so the frozen eval can score both."""
    from kernels.sm120.moe import moe_per_expert
    hn, idx, wgt = _NormRouter.apply(x, w["nw"], w["rw"], w["bias"], top_k, eps)
    return moe_per_expert(hn, idx.long(), wgt.float(), w["gu"], w["dn"], codes)
