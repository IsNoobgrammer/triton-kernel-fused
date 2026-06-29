"""Blackwell (`sm_120`, RTX PRO 6000 / GB202) fused kernels.

    from kernels.sm120 import fused_linear_cross_entropy, fused_xsa, fused_router, moe, FusedMuon

REUSE + OVERRIDE: the kernel implementations are **identical to sm75** and are re-exported from there
(single source of truth — a fix in `kernels/sm75/` lands here automatically). Only the bits that are
measurably tuned differently on Blackwell are overridden in this package; fork a whole module into
`kernels/sm120/` only once a kernel genuinely diverges on this arch.

Blackwell deltas measured so far (see docs / .autoresearch):
  - Muon `ns_batch_elems` default 4M -> 8M (the Blackwell mem-gated knee: ~2.42-2.47x at peak<=baseline,
    vs 4M's ~2.0x).
  - Conv `fused_router` FORKED to a single-launch fused Triton conv (no cuDNN) — recovers the T4 win that
    the cuDNN path lost on Blackwell to a layout-copy tax (~1.45x fwd+bwd, bwd 1.62x). The MLP router and
    CE/XSA/MoE are byte-identical to sm75.
"""
from .cross_entropy import fused_linear_cross_entropy
from .xsa import fused_xsa, FusedXSA
from .router import fused_router, mlp_router, router_bias_update, FusedConvRouterCuDNN, FusedMLPRouter
from .moe import moe, moe_per_expert, moe_eager
from .muon import FusedMuon, DistributedMuon, newton_schulz

__all__ = [
    "fused_linear_cross_entropy",
    "fused_xsa", "FusedXSA",
    "fused_router", "mlp_router", "router_bias_update", "FusedConvRouterCuDNN", "FusedMLPRouter",
    "moe", "moe_per_expert", "moe_eager",
    "FusedMuon", "DistributedMuon", "newton_schulz",
]
