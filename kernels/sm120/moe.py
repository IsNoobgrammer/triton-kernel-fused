"""sm120 MoE = the sm75 implementation, reused verbatim.

per-expert is the proven Blackwell win (~4x fwd+bwd, correct). The auto-dispatch `moe()` and its
special-expert guard are arch-independent and reused as-is. The grouped tl.dot path is faster on
Blackwell but only correct for pure-GLU stacks (it has no Identity/Zero handling) — fixing that is a
real divergence that would justify forking this module; until then we reuse sm75 unchanged.
"""
from kernels.sm75.moe import (  # noqa: F401
    moe, moe_per_expert, moe_eager, moe_grouped, moe_grouped_cublas, GROUPED_MIN_TOKENS,
)
