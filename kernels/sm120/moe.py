"""sm120 MoE = sm75 implementation + a Blackwell-only grouped dispatch.

per-expert is the proven Blackwell win at LOW expert counts (each expert's cuBLAS GEMM is large enough
to saturate the tensor cores). But at higher expert counts / lower tokens-per-expert — the real-MoE
shape — the per-expert loop degenerates into many tiny GEMMs + a DtoD copy per expert, and a single
cuBLAS *grouped* GEMM wins. `moe_grouped_cublas_polyglu` (this arch, `.moe_grouped`) is that path: a
`torch._grouped_mm` grouped GEMM that IS correct on the special-experts stack (Identity/Zero on the
sorted tail) and supports PolyGLU. Measured 1.6-3.0x over per-expert at <= or ~equal memory when
tokens-per-expert is low; see kernels/sm120/moe_grouped.py and .autoresearch/moe_grouped_findings.

`moe()` here dispatches: grouped when it is supported (torch._grouped_mm, sm_80+, bf16/fp16, 16B-aligned)
AND inside its win regime (tokens-per-expert <= GROUPED_TOKENS_PER_EXPERT_MAX); otherwise per-expert.
per-expert remains the correct, memory-frugal fallback on every shape/arch.

The old GLU-only `moe_grouped` (sm75, tl.dot) is still re-exported for parity benches but is NOT used by
`moe()` here — it has no special-expert handling and the cuBLAS grouped path supersedes it on Blackwell.
"""
from kernels.sm75.moe import (  # noqa: F401
    moe_per_expert, moe_eager, moe_grouped, moe_grouped_cublas, GROUPED_MIN_TOKENS,
)
from .moe_grouped import (
    moe_grouped_cublas_polyglu, grouped_supported, prefer_grouped, GROUPED_TOKENS_PER_EXPERT_MAX,
)

__all__ = [
    "moe", "moe_per_expert", "moe_eager", "moe_grouped", "moe_grouped_cublas",
    "moe_grouped_cublas_polyglu", "grouped_supported", "prefer_grouped",
    "GROUPED_MIN_TOKENS", "GROUPED_TOKENS_PER_EXPERT_MAX",
]


def moe(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Auto-dispatch on Blackwell: cuBLAS grouped when supported AND tokens-per-expert is low (its win
    regime, where it beats per-expert on time at <= memory); else the per-expert champion.

    The grouped path handles Identity (3) / Zero (4) special experts directly, so unlike the old
    sm75 grouped there is no `glu_only` restriction. per-expert is the fallback whenever grouped is
    unsupported (no torch._grouped_mm / sm_<80 / non-bf16-fp16 / unaligned shapes) or tokens-per-expert
    is high enough that the per-expert GEMMs are already efficient.
    """
    if grouped_supported(hidden, gate_up_proj, down_proj) and prefer_grouped(top_k_indices, gate_up_proj):
        return moe_grouped_cublas_polyglu(hidden, top_k_indices, top_k_weights,
                                          gate_up_proj, down_proj, act_codes)
    return moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)
