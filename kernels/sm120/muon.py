"""sm120 Muon for Blackwell — two interchangeable optimizers:

  FusedMuon       : the champion. foreach launch-collapse + batched same-shape state + baddbmm fold,
                    pure cuBLAS Newton-Schulz. No Triton dependency. The git-versioned reference.
  AmalgamatedMuon : FusedMuon's exact step + levers, with the two SYMMETRIC NS GEMMs (X X^T, A A)
                    done by the Triton symmul kernel (compute one triangle, mirror it -> ~half the
                    GEMM FLOPs). Measured on RTX PRO 6000: 1.28-1.43x faster than FusedMuon on large
                    matrices (gram >= 2048), 1.31-3.49x vs torch.compile, beats flash-muon's exact
                    impl 1.10-1.17x, and uses <= compiled memory. Scale-invariant 1B -> 2.6B params.

PRECISION IS NOT A TRADEOFF. AmalgamatedMuon runs the IDENTICAL fp16 NS math with fp32 accumulate as
FusedMuon; the symmetric kernel only changes float rounding ORDER (triangle+mirror vs full bmm),
~1e-4..1e-3 -> parity vs FusedMuon is 5.9e-3, well inside the NS tolerance (2e-2), and orthogonalization
quality is the same (SV ~0.98). It also self-gates: below the gram knee (min(rows,cols) < SYMMUL_MIN_DIM)
it calls the EXACT FusedMuon NS, so small matrices are bit-for-bit the champion. => AmalgamatedMuon can
REPLACE FusedMuon outright on Blackwell. Keep FusedMuon for: Triton-free environments, or as the
pure-cuBLAS reference. Only the default `ns_batch_elems` (8M) differs from sm75 (the Blackwell mem knee).
"""
import torch

from kernels.sm75.muon import newton_schulz, _PE_COEFFS  # noqa: F401
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul

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


class AmalgamatedMuon(FusedMuon):
    """FusedMuon with the symmetric-matmul ("symmul") Newton-Schulz — faster on large matrices, same
    precision (see module docstring). Drop-in: same constructor/state_dict as FusedMuon; the only
    change is the NS function (`newton_schulz_symmul`, which gates to the champion NS below the knee).

    The eager `step` is FusedMuon's step verbatim except `newton_schulz(...)` -> `newton_schulz_symmul`.
    """

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            plan = self._plan(group, params)
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:
                    mom_c = mom[start:start + crows]
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c
                    out = newton_schulz_symmul(u, self.coeffs, self.ns_dtype)   # <-- the only change
                    torch._foreach_add_([p for p, _, _ in members],
                                        [out[o:o + n].reshape(p.shape) for p, o, n in members], alpha=alpha)
        return loss
