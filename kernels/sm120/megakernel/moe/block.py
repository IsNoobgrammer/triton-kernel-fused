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
import os

import torch

from .norm_router import norm_router_forward, rmsnorm_backward


class _NormRouter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, nw, rw, bias, top_k, eps, want_gap=False):
        hn, idx, wgt, rstd, counts, ssum, gap = norm_router_forward(
            x, nw, rw, bias, top_k, eps, write_hn=True, write_gap=want_gap)
        # hn and rstd are SAVED, not recomputed. Candidate 1 rebuilt the whole norm+router graph
        # in PyTorch and paid +3.1 ms for it -- the forward was a tie and the backward regressed by
        # exactly that amount, which is what identified the recompute as the regression.
        ctx.save_for_backward(x, nw, rw, idx, rstd, hn, wgt, ssum)
        ctx.eps = eps
        ctx.mark_non_differentiable(idx)
        if gap is not None:
            ctx.mark_non_differentiable(gap)
        return hn, idx, wgt, gap

    @staticmethod
    def backward(ctx, g_hn, g_idx, g_wgt, g_gap):
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
        # rw is bf16 in the bench but fp32 in a real model, which keeps master weights in fp32 and
        # relies on autocast to narrow them. Both matmul operands must agree, and the gradient
        # handed back to autograd must match the PARAMETER's dtype, not the activation's -- writing
        # this against the bench's dtype is what made the forward pass and the backward die.
        # Contracting in the activation dtype (rather than promoting to fp32) is deliberate: it is
        # what eager does under autocast, and it is the arrangement the fp64 grader scored at
        # 55-68x closer than eager, so matching it preserves the measured accuracy.
        d_hn = g_hn + dl @ rw.to(dl.dtype).t()                # router's contribution to d hn
        d_rw = (hn.t() @ dl).to(rw.dtype)

        # ---- rmsnorm backward, FUSED. The PyTorch form built five [T,H] fp32 temporaries
        # (~670 MB of traffic at T=65536) against liger's single 0.280 ms kernel -- that was the
        # whole remaining regression after the recompute was removed.
        d_x, d_nw = rmsnorm_backward(x, d_hn, nw, rstd)
        return d_x, d_nw, d_rw, None, None, None, None


def megakernel_block(x, w, codes, top_k=6, eps=1e-6, act_params=None, return_routing=False,
                     want_gap=False):
    """Signature matches bench.eval_mlp_block.baseline_block so the frozen eval can score both.

    `act_params` carries radial's exponent logit (`radial_theta`). Act code 8 RAISES without it, so
    a training patch running radial MUST pass it -- the frozen eval never did, which is why this
    argument did not exist until the kernel was wired into a real model.

    `return_routing=True` also hands back (idx, weights, gap). Both the load-balancing bias update
    and the router diagnostics (top1 weight, router entropy, balance entropy, boundary gap) are
    driven from these and they are produced HERE and nowhere else, so dropping them silently
    disables balancing and blanks the diagnostics -- changes that surface as a quality regression,
    not an error. `gap` is None unless `want_gap=True`.

    All default off, so the frozen eval's single-tensor contract is untouched.
    """
    # moe() dispatches to the GROUPED path when it can; moe_per_expert always takes the
    # per-expert loop, whose backward is ~119 separate cuBLAS dW launches plus 73 split-K
    # reductions -- 3.96 ms, 23% of the lap, and the largest remaining inefficiency.
    from kernels.sm120.moe import moe, moe_per_expert
    hn, idx, wgt, gap = _NormRouter.apply(x, w["nw"], w["rw"], w["bias"], top_k, eps, want_gap)
    # PER_EXPERT BY DEFAULT, matching BIBO_MOE_DISPATCH and therefore the radial baseline.
    #
    # This used to default to grouped, and the condition below is `act_params is None` -- which is
    # true exactly when the activation is SiLU. So flipping --act radial -> silu silently also
    # flipped the expert kernel, and an activation A/B would have measured two changes at once with
    # throughput moving for the wrong reason. Grouped is a measured WASH on real steps anyway
    # (205.1k vs 205.6k tok/s), so there is nothing to give up. MK_GROUPED=1 still opts in.
    grouped = os.environ.get("MK_GROUPED", "0") == "1" and act_params is None
    # the grouped path index_add_s into a bf16 buffer and rejects fp32 weights;
    # per_expert takes fp32, which is what the model's router emits
    tw = wgt.to(hn.dtype) if grouped else wgt.float()
    out = (moe(hn, idx.long(), tw, w["gu"], w["dn"], codes) if grouped else
           moe_per_expert(hn, idx.long(), tw, w["gu"], w["dn"], codes, act_params=act_params))
    return (out, idx, tw, gap) if return_routing else out
