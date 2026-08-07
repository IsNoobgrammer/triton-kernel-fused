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

from .norm_router import norm_router_forward, rmsnorm_backward


class _NormRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nw, rw, bias, top_k, eps):
        hn, idx, wgt, rstd, counts, ssum = norm_router_forward(
            x, nw, rw, bias, top_k, eps, write_hn=True)
        # hn and rstd are SAVED, not recomputed. Candidate 1 rebuilt the whole norm+router graph
        # in PyTorch and paid +3.1 ms for it -- the forward was a tie and the backward regressed by
        # exactly that amount, which is what identified the recompute as the regression.
        ctx.save_for_backward(x, nw, rw, idx, rstd, hn, wgt, ssum)
        ctx.eps = eps
        ctx.mark_non_differentiable(idx)
        return hn, idx, wgt

    @staticmethod
    def backward(ctx, g_hn, g_idx, g_wgt):
        x, nw, rw, idx, rstd, hn, wgt, ssum = ctx.saved_tensors
        H = x.shape[-1]
        idxl = idx.long()

        # ---- router. w_k = s_k / S, so ds_k = (g_k - sum_j g_j w_j) / S. The sum over the
        # SELECTED experts only; scores at unselected experts get no gradient because they never
        # entered the output.
        s_sel = wgt.float()                                   # already normalised: w_j
        gw = g_wgt.float()
        dot = (gw * s_sel).sum(-1, keepdim=True)              # sum_j g_j w_j
        # s_k = w_k * S. S is saved by the forward, so this costs a multiply instead of the
        # [T,H]x[H,E] matmul + sigmoid that recomputing the scores would need.
        S = ssum[:, None] + 1e-20
        scores_sel = s_sel * S
        d_s = (gw - dot) / S                                  # d/d scores_sel
        d_logit_sel = d_s * scores_sel * (1.0 - scores_sel)   # through the sigmoid
        d_logits = torch.zeros(x.shape[0], rw.shape[1], device=x.device, dtype=torch.float32)
        d_logits.scatter_(1, idxl, d_logit_sel)

        dl = d_logits.to(x.dtype)
        d_hn = g_hn + dl @ rw.t()                             # router's contribution to d hn
        d_rw = hn.t() @ dl

        # ---- rmsnorm backward, FUSED. The PyTorch form built five [T,H] fp32 temporaries
        # (~670 MB of traffic at T=65536) against liger's single 0.280 ms kernel -- that was the
        # whole remaining regression after the recompute was removed.
        d_x, d_nw = rmsnorm_backward(x, d_hn, nw, rstd)
        return d_x, d_nw, d_rw, None, None, None


def megakernel_block(x, w, codes, top_k=6, eps=1e-6):
    """Signature matches bench.eval_mlp_block.baseline_block so the frozen eval can score both."""
    from kernels.sm120.moe import moe_per_expert
    hn, idx, wgt = _NormRouter.apply(x, w["nw"], w["rw"], w["bias"], top_k, eps)
    return moe_per_expert(hn, idx.long(), wgt.float(), w["gu"], w["dn"], codes)
