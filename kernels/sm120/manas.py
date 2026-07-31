from kernels.sm75.manas import ManasOptimizer as _Manas75, NS8_COEFFS
from kernels.sm120.muon import FusedMuon as _FusedMuon120

__all__ = ["ManasOptimizer", "NS8_COEFFS"]


class ManasOptimizer(_Manas75, _FusedMuon120):
    pass
