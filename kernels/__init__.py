"""triton-kernel-fused — drop-in fused Triton kernels (forward + backward) for transformer training.

Kernels are organized by **GPU architecture** (CUDA compute capability). Import from the package that
matches the hardware you run on — performance is architecture-specific and the kernels are tuned per arch:

    from kernels.sm75 import fused_linear_cross_entropy, fused_xsa, fused_router, moe, FusedMuon

Available architectures:
  - `kernels.sm75`  — Turing / Tesla T4 (the reference arch; every kernel is implemented and tuned here).
  - `kernels.sm120` — Blackwell (RTX PRO 6000 / GB202). Reuses the sm75 implementations and overrides
    only what is measurably tuned differently on Blackwell (today: Muon `ns_batch_elems` default 8M).

To contribute kernels for another architecture (e.g. Ampere `sm_80`, Hopper `sm_90`), add a
`kernels/sm<XX>/` package — see CONTRIBUTING.md.
"""

# Compute-capability -> package-name mapping for the architectures shipped in this repo. This is a
# directory of what exists; it does NOT auto-import (callers pick their arch explicitly, e.g. bench.py
# resolves the best entry <= the device's capability via `arch_for_capability`).
AVAILABLE_ARCHS = ("sm75", "sm120")


def arch_for_capability(major, minor):
    """Return the kernels.<arch> package name to use for a CUDA compute capability (major, minor):
    the highest shipped arch whose capability is <= the device's. e.g. sm_75 -> 'sm75', sm_89 -> 'sm75',
    sm_90 -> 'sm75', sm_120 -> 'sm120'. Falls back to the lowest shipped arch below all of them."""
    cap = major * 10 + minor
    avail = sorted(int(a[2:]) for a in AVAILABLE_ARCHS)          # ("sm75","sm120") -> [75, 120]
    pick = max([a for a in avail if a <= cap], default=min(avail))
    return f"sm{pick}"


__all__ = ["AVAILABLE_ARCHS", "arch_for_capability"]

__all__ = ["AVAILABLE_ARCHS"]
