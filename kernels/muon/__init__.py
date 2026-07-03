"""Shared, arch-independent Muon building blocks (used by kernels/sm75 and kernels/sm120)."""
from .muon_scaling import (
    ALL_MODES, SCALAR_MODES, PERROW_MODES, AURORA_MODES, DEFAULT_MODE, RMS_TARGET, SPECTRAL_GAIN,
    AURORA_K,
    is_perrow, is_aurora, validate, scalar_scale, perrow_state, apply_perrow, aurora_update,
)

__all__ = [
    "ALL_MODES", "SCALAR_MODES", "PERROW_MODES", "AURORA_MODES", "DEFAULT_MODE",
    "RMS_TARGET", "SPECTRAL_GAIN", "AURORA_K",
    "is_perrow", "is_aurora", "validate", "scalar_scale", "perrow_state",
    "apply_perrow", "aurora_update",
]
