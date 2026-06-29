"""sm120 conv router = the sm75 implementation, reused verbatim (no Blackwell-specific divergence yet)."""
from kernels.sm75.router import (  # noqa: F401
    fused_router, router_bias_update, FusedConvRouterCuDNN, _count_experts,
)
