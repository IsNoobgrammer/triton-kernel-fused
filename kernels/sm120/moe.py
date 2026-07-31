from kernels.sm75.moe import (
    moe_per_expert, moe_eager, moe_grouped, moe_grouped_cublas, GROUPED_MIN_TOKENS, _code_max,
)
from .moe_grouped import (
    moe_grouped_cublas_polyglu, grouped_supported, prefer_grouped, GROUPED_TOKENS_PER_EXPERT_MAX,
)

__all__ = [
    "moe", "moe_per_expert", "moe_eager", "moe_grouped", "moe_grouped_cublas",
    "moe_grouped_cublas_polyglu", "grouped_supported", "prefer_grouped",
    "GROUPED_MIN_TOKENS", "GROUPED_TOKENS_PER_EXPERT_MAX",
]


def moe(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    if (grouped_supported(hidden, gate_up_proj, down_proj)
            and prefer_grouped(top_k_indices, gate_up_proj)
            and _code_max(act_codes) <= 4):
        return moe_grouped_cublas_polyglu(hidden, top_k_indices, top_k_weights,
                                          gate_up_proj, down_proj, act_codes)
    return moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
                          act_params)
