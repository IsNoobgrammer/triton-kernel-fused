from kernels.sm75.cross_entropy import fused_linear_cross_entropy as _sm75_flce

__all__ = ["fused_linear_cross_entropy"]

_BWD_LOGITS_BUDGET = 1024 * 1024 * 1024


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None):
    return _sm75_flce(hidden, weight, labels, ignore_index,
                      _BWD_LOGITS_BUDGET if bwd_logits_budget is None else bwd_logits_budget)
