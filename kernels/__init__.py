"""triton-kernel-fused — drop-in fused Triton kernels (forward + backward).

    from kernels import fused_linear_cross_entropy, fused_xsa, causal_conv1d_router

SwiGLU is intentionally NOT here — torch.compile's lifted SiLU-mul kernel ties/beats a hand-written
one (it was noise), so we leave dense-MLP activation to the compiler.
"""
from .cross_entropy import fused_linear_cross_entropy
from .xsa import fused_xsa, FusedXSA
from .causal_conv1d_router import causal_conv1d_router, CausalConv1dRouter
from .router import fused_router, router_bias_update, FusedConvRouterCuDNN
from .moe import moe, moe_per_expert, moe_grouped, moe_eager, GROUPED_MIN_TOKENS

__all__ = [
    "fused_linear_cross_entropy",
    "fused_xsa", "FusedXSA",
    "causal_conv1d_router", "CausalConv1dRouter",
    "fused_router", "router_bias_update", "FusedConvRouterCuDNN",
    "moe", "moe_per_expert", "moe_grouped", "moe_eager", "GROUPED_MIN_TOKENS",
]
