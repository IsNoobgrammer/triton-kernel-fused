"""sm120 Muon = the sm75 Polar-Express implementation, with the Blackwell-tuned default.

Only the default `ns_batch_elems` changes: 8M is the measured Blackwell mem-gated knee (~2.42-2.47x at
peak<=baseline on both 75M and 302M param sets), vs sm75's 4M (which would leave ~18% on the table here).
The Newton-Schulz math, the foreach+baddbmm fusion, and `ns_dtype=fp16` are all reused unchanged — fp16
stays the right NS dtype even on Blackwell (more mantissa than bf16 -> tighter orthogonalization, and
Blackwell runs fp16 on full-rate tensor cores; see the bf16 note in the docs).
"""
from kernels.sm75.muon import newton_schulz, _PE_COEFFS  # noqa: F401
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75

# Blackwell mem-gated knee (peak<=baseline at both sizes); sm75 uses 4M. Callers can still override.
NS_BATCH_ELEMS = 8 * 1024 * 1024


class FusedMuon(_FusedMuon75):
    """sm75 FusedMuon with `ns_batch_elems` defaulting to the Blackwell knee (8M)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)


class DistributedMuon(_DistributedMuon75):
    """sm75 DistributedMuon with `ns_batch_elems` defaulting to the Blackwell knee (8M)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)
