"""Autotune the gram-NS restart placement for the ns8 schedule (6 KJ + 2 pin = 8 iters, dsv4 coeffs)
at bf16 vs fp16 -- answers "is the bf16-optimal restart the same as the fp16 tuning?"

Run on the sm120 box (RTX PRO 6000):  PYTHONPATH=. python .autoresearch/autotune_gram_ns8.py

Context: GRAM_RESTART_AT ([4,6]) was tuned in FP16 for the 10-iter _DSV4_COEFFS. The sm120 default
is bf16 and we want ns8 (6+2) as the default schedule -> the restart is doubly-untuned. fp16 needed
2 restarts partly to dodge overflow/NaN (max 65504); bf16 has fp32 range so that driver is gone ->
prediction: bf16 wants FEWER/later restarts. This settles it empirically.
"""
import torch

from kernels.sm120.newton_schulz_gram import autotune_restarts

KJ = (3.4445, -4.7750, 2.0315)      # Keller-Jordan quintic (aggressor)
PIN = (2.0, -1.5, 0.5)              # pinned tail (finisher, fixed point at 1)
NS8 = (KJ,) * 6 + (PIN,) * 2        # 6 KJ + 2 pin = 8 iters (dsv4 coeffs)

for dt, name in ((torch.bfloat16, "bf16"), (torch.float16, "fp16")):
    for nr in (1, 2):
        print(f"\n===== ns8 (6+2)  dtype={name}  num_restarts={nr} =====", flush=True)
        try:
            best = autotune_restarts(NS8, num_restarts=nr, ns_dtype=dt, bench=False)
            print(f">>> ns8 {name} num_restarts={nr}  BEST restart(s): {best}", flush=True)
        except Exception as e:
            import traceback
            print(f"FAILED ({name}, nr={nr}): {e}", flush=True)
            traceback.print_exc()

print("\nCompare the bf16 vs fp16 BEST lines above. If bf16's best (esp. num_restarts=1) reaches "
      "worst-ratio ~1.0 while fp16's single restart drifts/NaNs, bf16 is cheaper -> set the ns8 "
      "default gram restart to the bf16 winner.", flush=True)
