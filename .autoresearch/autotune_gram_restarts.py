"""Autotune the gram-NS restart placement for PE-8, ns8 and ns10 at bf16 vs fp16.

Answers "is the bf16-optimal restart the same as the fp16 tuning?" for each schedule, and gives the
bf16 restart to hardcode as the sm120 default. Run on the sm120 box (RTX PRO 6000):

    PYTHONPATH=. python .autoresearch/autotune_gram_restarts.py

Context: GRAM_RESTART_AT was tuned in FP16 -- known fp16 winners: ns10 (_DSV4_COEFFS) -> [4,6],
PE-8 (_PE_COEFFS) -> [3]. The sm120 default is bf16. fp16's restarts were partly overflow/NaN
avoidance (max 65504); bf16 has fp32 range so that driver is gone -> prediction: bf16 wants
FEWER/later restarts. This confirms it and sets the ns8 (our new default) restart.
"""
import torch

from kernels.sm120.newton_schulz_gram import autotune_restarts
from kernels.sm75.muon import _PE_COEFFS, _DSV4_COEFFS   # PE-8 (8 iters); dsv4 = ns10 (8 KJ + 2 pin)

KJ = (3.4445, -4.7750, 2.0315)      # Keller-Jordan quintic (aggressor)
PIN = (2.0, -1.5, 0.5)              # pinned tail (finisher; fixed point at 1)
NS8 = (KJ,) * 6 + (PIN,) * 2        # 6 KJ + 2 pin = 8 iters (dsv4 coeffs, our target default)

SCHEDULES = {"pe8": _PE_COEFFS, "ns8": NS8, "ns10": _DSV4_COEFFS}
summary = []
for name, coeffs in SCHEDULES.items():
    for dt, dn in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
        for nr in (1, 2):
            print(f"\n===== {name} ({len(coeffs)} iters)  dtype={dn}  num_restarts={nr} =====", flush=True)
            try:
                best = autotune_restarts(coeffs, num_restarts=nr, ns_dtype=dt, bench=False)
                print(f">>> {name} {dn} nr={nr}  BEST restart(s): {best}", flush=True)
                summary.append((name, dn, nr, best))
            except Exception as e:
                import traceback
                print(f"FAILED ({name}, {dn}, nr={nr}): {e}", flush=True)
                traceback.print_exc()
                summary.append((name, dn, nr, "FAILED"))

print("\n================ SUMMARY (best restart placement) ================", flush=True)
for name, dn, nr, best in summary:
    print(f"  {name:5s} {dn}  nr={nr}: {best}", flush=True)
print("Known fp16 refs: ns10 -> [4,6], pe8 -> [3]. Compare bf16 vs fp16 per schedule.", flush=True)
