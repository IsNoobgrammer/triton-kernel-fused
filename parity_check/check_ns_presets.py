"""Every NS_PRESETS schedule, as a scalar map on the singular values of X / ||X||_F (what the NS kernels apply).

    python -m parity_check.check_ns_presets

slope0 = d/dx at 0 (product of the a's): how hard the smallest directions are lifted. sv@x = output at sigma = x.
The gate: every schedule keeps sigma bounded and POSITIVE on (0, 1.03] (bf16 rounding reaches ~1.03 on a near-rank-1
momentum; pe8 without its margin went NaN in 60 steps) and lifts sigma >= 0.05 to at least 0.6.
"""
from parity_check import _paths  # noqa: F401
import math

import numpy as np

from kernels.muon.muon_scaling import NS_PRESETS

np.seterr(all="ignore")
x = np.concatenate([np.logspace(-4, 0, 4001), np.linspace(1.0, 1.03, 301)[1:]])
print(f"{'preset':9} {'steps':>5} {'slope0':>9} {'max':>6} {'min>=1e-2':>9} {'min>=5e-2':>9}  sv@1e-3 sv@1e-2 sv@0.1")
for name, coeffs in NS_PRESETS.items():
    y = x.copy()
    for a, b, c in coeffs:
        y = a * y + b * y ** 3 + c * y ** 5
    slope0 = math.prod(a for a, _, _ in coeffs)
    at = lambda v: y[np.searchsorted(x, v)]
    print(f"{name:9} {len(coeffs):5d} {slope0:9.3g} {y.max():6.3f} {y[(x >= 1e-2) & (x <= 1)].min():9.3f} {y[(x >= 5e-2) & (x <= 1)].min():9.3f}"
          f"  {at(1e-3):7.3f} {at(1e-2):7.3f} {at(0.1):6.3f}")
    assert np.isfinite(y).all() and 0 < y.min() and y.max() < 1.35, f"{name}: leaves (0, 1.35] for sigma in (0, 1.03]"
    assert y[(x >= 5e-2) & (x <= 1)].min() > 0.6, f"{name}: sigma >= 0.05 not lifted to 0.6"
print("NS_PRESETS PASS")
