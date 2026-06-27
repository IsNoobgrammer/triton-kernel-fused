"""triton-kernel-fused — drop-in fused Triton kernels (forward + backward).

    from kernels import fused_swiglu, fused_linear_cross_entropy, fused_xsa, causal_conv1d_router
"""
from .swiglu import fused_swiglu, FusedSwiGLUMLP, FusedSwiGLU
from .cross_entropy import fused_linear_cross_entropy
from .xsa import fused_xsa, FusedXSA
from .causal_conv1d_router import causal_conv1d_router, CausalConv1dRouter
from .moe import moe, moe_per_expert, moe_grouped, moe_eager, GROUPED_MIN_TOKENS

__all__ = [
    "fused_swiglu", "FusedSwiGLUMLP", "FusedSwiGLU",
    "fused_linear_cross_entropy",
    "fused_xsa", "FusedXSA",
    "causal_conv1d_router", "CausalConv1dRouter",
    "moe", "moe_per_expert", "moe_grouped", "moe_eager", "GROUPED_MIN_TOKENS",
]
