"""MoE megakernel for sm120.

Phase 1 (here): fused RMSNorm + router.
Phase 2 (next): dispatch + grouped GEMM + activation + weighted combine, sharing one custom
autograd Function with phase 1 so the backward can recompute intermediates instead of saving them.

Design constraints fixed up front:

* ONE activation per build, chosen at compile time (radial NormSiLU or SiLU, never both, no
  polyglu). A constexpr specialization, not a per-expert act_code lookup, so the backward hardcodes
  one derivative rather than branching.
* The router's load-balancing bias update is a SIDE EFFECT that must fire exactly once per step.
  The backward reuses the saved top-k indices and never re-runs the router; recomputing it would
  apply the balancing twice.
* Graded against an fp64 eager reference, and reported alongside the existing kernel and bf16
  eager, so a numerics change is visible rather than assumed.
"""
from .norm_router import norm_router_forward, norm_router_reference
from .block import megakernel_block

__all__ = ["norm_router_forward", "norm_router_reference", "megakernel_block"]
