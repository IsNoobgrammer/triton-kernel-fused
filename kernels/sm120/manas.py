"""sm120 ManasOptimizer — the sm75 rolling probe riding the Blackwell gram-NS Muon base.

The probe machinery is architecture-independent (elementwise buffers + a few GEMMs); the only
thing that has to change on Blackwell is the BASE step, which sm120's FusedMuon overrides with
gram NS -> symmul -> cuBLAS shape gating. Cooperative multiple inheritance does exactly that:

    MRO: ManasOptimizer -> _Manas75 -> _FusedMuon120 -> _FusedMuon75

so `super().step()` inside _Manas75.step resolves to the sm120 step, `_polar`/`_ns` come from
sm120, and `use_gram` / `gram_restarts` pass through _Manas75's **kw untouched.

WHY THIS SHIM EXISTS AT ALL: kernels.sm75.manas.ManasOptimizer subclasses the sm75 FusedMuon
directly, so importing it on Blackwell would silently swap the NS backend along with the
optimizer — an A/B against an sm120 Muon baseline would then be confounded by the NS change,
not just the probe. Use THIS class whenever the control arm is kernels.sm120.muon.FusedMuon.
"""
import torch  # noqa: F401  (re-exported symmetry with the sm75 module)

from kernels.sm75.manas import ManasOptimizer as _Manas75, NS8_COEFFS
from kernels.sm120.muon import FusedMuon as _FusedMuon120

__all__ = ["ManasOptimizer", "NS8_COEFFS"]


class ManasOptimizer(_Manas75, _FusedMuon120):
    __doc__ = (_Manas75.__doc__ or "") + "\n\nsm120: base step is gram-NS Muon (see module docstring)."
