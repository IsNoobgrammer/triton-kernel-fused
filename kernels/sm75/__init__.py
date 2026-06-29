"""Turing / Tesla T4 (`sm_75`) fused Triton kernels — forward + backward.

    from kernels.sm75 import fused_linear_cross_entropy, fused_xsa, fused_router, moe, FusedMuon

Each kernel is a tested replacement for an eager PyTorch block, with a custom backward. They target the
cases where torch.compile leaves performance on the table: data-dependent routing, terminal-loss fusion,
read-once reductions, and native-op seams (e.g. top-k) the compiler must keep as library calls. Every
number in the docs is measured on `sm_75` — see other `kernels/<arch>/` packages for other GPU classes.
"""
from .cross_entropy import fused_linear_cross_entropy
from .xsa import fused_xsa, FusedXSA
from .router import fused_router, router_bias_update, FusedConvRouterCuDNN
from .moe import moe, moe_per_expert, moe_eager
from .muon import FusedMuon, DistributedMuon, newton_schulz

__all__ = [
    "fused_linear_cross_entropy",
    "fused_xsa", "FusedXSA",
    "fused_router", "router_bias_update", "FusedConvRouterCuDNN",
    "moe", "moe_per_expert", "moe_eager",
    "FusedMuon", "DistributedMuon", "newton_schulz",
]
