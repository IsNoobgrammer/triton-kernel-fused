"""sm120 cross-entropy = the sm75 kernel, with a Blackwell-tuned chunk budget.

Same kernel/math as sm75 (byte-identical results). Only the default (chunk, V) transient
budget differs: 1024 MiB here vs 192 MiB on sm75. On Blackwell, CE latency is monotone-
decreasing with budget (fewer cuBLAS launches), the opposite of the T4 knee. Measured on
RTX PRO 6000 (V=81920, N=65536, bf16, fwd+bwd): ~85ms @ 192MB -> ~74ms @ 1024MB (~13%),
peak 1.15 -> 2.8 GB (trivial on 96 GB). Pass bwd_logits_budget to override per-call.
"""
from kernels.sm75.cross_entropy import fused_linear_cross_entropy as _sm75_flce

__all__ = ["fused_linear_cross_entropy"]

_BWD_LOGITS_BUDGET = 1024 * 1024 * 1024  # Blackwell default (sm75 uses 192 MiB)


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None):
    return _sm75_flce(hidden, weight, labels, ignore_index,
                      _BWD_LOGITS_BUDGET if bwd_logits_budget is None else bwd_logits_budget)
