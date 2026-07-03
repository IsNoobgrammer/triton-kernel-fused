"""Shared, arch-independent Muon building blocks (used by kernels/sm75 and kernels/sm120)."""
from .muon_scaling import (
    ALL_MODES, SCALAR_MODES, PERROW_MODES, DEFAULT_MODE,
    is_perrow, validate, scalar_scale, perrow_state, apply_perrow,
)

__all__ = [
    "ALL_MODES", "SCALAR_MODES", "PERROW_MODES", "DEFAULT_MODE",
    "is_perrow", "validate", "scalar_scale", "perrow_state", "apply_perrow",
]
