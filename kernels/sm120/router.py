"""sm120 conv router = the sm75 implementation, reused verbatim (champion).

The sm75 `cudnn` router is reused unchanged on Blackwell (0.96× compiled fwd+bwd). It is a REGRESSION
vs compiled here (cuDNN's nchwToNhwc/nhwcToNchw layout transposes are unavoidable), but it is the best
correct backend we have — see `.autoresearch/router_sm120_results.jsonl` for the optimization ledger.

REFUTED candidate (do not retry as-is): a transpose-free conv expressed as K torch matmuls
(`FusedConvRouterGEMM`). It removed cuDNN/transposes but EXPLODED launches+copies (Blackwell profile:
161 launches/iter, 240 `aten::mm`, 760 `aten::copy_`, 180 `aten::add_`) → **0.54× fwd+bwd, WORSE than
cuDNN**. Confirms the T4 finding (K-GEMM conv = launch/copy-bound) holds on Blackwell too: any multi-torch-
op decomposition loses on this tiny op. The ONLY path that can win is a SINGLE fused Triton conv kernel
(one launch, read x once, sigmoid+top-k+gather in-register) — that is the open candidate #2.
"""
from kernels.sm75.router import (  # noqa: F401
    fused_router, router_bias_update, FusedConvRouterCuDNN, _count_experts,
)
