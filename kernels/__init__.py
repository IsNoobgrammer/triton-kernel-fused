"""triton-kernel-fused — drop-in fused Triton kernels (forward + backward) for transformer training.

Kernels are organized by **GPU architecture** (CUDA compute capability). Import from the package that
matches the hardware you run on — performance is architecture-specific and the kernels are tuned per arch:

    from kernels.sm75 import fused_linear_cross_entropy, fused_xsa, fused_router, moe, FusedMuon

Available architectures:
  - `kernels.sm75` — Turing / Tesla T4 (the reference arch; everything is tuned and verified here).

To contribute kernels for another architecture (e.g. Ampere `sm_80`, Hopper `sm_90`), add a
`kernels/sm<XX>/` package — see CONTRIBUTING.md.
"""

# Compute-capability -> package-name mapping for the architectures shipped in this repo. This is a
# directory of what exists; it does NOT auto-import (callers pick their arch explicitly).
AVAILABLE_ARCHS = ("sm75",)

__all__ = ["AVAILABLE_ARCHS"]
