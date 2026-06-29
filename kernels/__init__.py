"""triton-kernel-fused — drop-in fused Triton kernels (forward + backward) for Turing/T4.

    from kernels import fused_linear_cross_entropy, fused_xsa, fused_router, moe

Each kernel is a tested replacement for an eager PyTorch block, with a custom backward. They target
the cases where torch.compile leaves performance on the table: data-dependent routing, terminal-loss
fusion, read-once reductions, and native-op seams (e.g. top-k) the compiler must keep as library calls.
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
