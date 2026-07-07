# Manas round report

Frozen eval: `.autoresearch/manas/eval_manas.py` (MNIST-1D heterogeneous sources, paired
seeds, cummin-frontier matching; opt seeds 0-4, held-out 5-7). 22 logged runs, 9 waves,
5 promotions, all on the local RTX 3050.

## Champion

ManasOptimizer, gamma 0.08, rho 0.98, rank 8, refresh 200 (probe state ~3% of a weights
copy). Held-out, paired vs base Muon:
- OOD acc at matched train loss: +0.0255 +/- 0.0016
- peak OOD acc: neutral (-0.0013); full-d variant (same gamma/rho): +0.0245 frontier,
  +0.0073 peak
- trains slightly faster than base (min-train parity negative throughout the dose curve)

Shipped defaults updated in kernels/sm75/manas.py: probe_gamma 1e-3 -> 0.08,
probe_rho 0.85 -> 0.98 (old defaults sat in a measurably dead zone). gamma is the
scale-sensitive knob; re-sweep per model scale.

## Full comparison (paired deltas vs base Muon, same lr/wd)

| arm                                   | frontier  | peak OOD acc | min train |
|---------------------------------------|-----------|--------------|-----------|
| AdamW (accum K=4)                     | -0.028    | -0.036       | +0.213    |
| AdamW + Nexus clone (K=4)             | -0.030    | -0.022       | +0.126    |
| Muon + Manas full-d (g.08, rho.98)    | +0.0245 H | +0.0073 H    | -0.018    |
| Muon + Manas rank-8 (g.08, rho.98)    | +0.0255 H | -0.0013 H    | -0.002    |

H = held-out. Ordering: Muon+Manas > Muon >> AdamW+Nexus > AdamW. Muon supplies ~10x the
probe's increment over the AdamW family; Manas adds a separately-attributed increment on
top. Nexus clone DOES work on its own base (vs accum-AdamW: +0.0136 peak acc, 5/5 seeds,
faster training) — the positive control that validated the testbed.

## Mechanism verdict (the science)

The Nexus pairwise-alignment theory is NOT what powers Manas. Measured (a7_mechanism.py):
the alignment force injects 10x more cleanly at rho=0 (cos +0.24) than at winning rho
(+0.02), yet rho=0 is downstream-inert; trajectory motion is 20-160x ||d|| (quasi-static
assumption dead). Ablation-pinned requirements for the gain:
1. LONG memory — rho 0.98 (~50 batches); saturates above; rho=0 pure extragradient inert.
2. DESCENT sign — SAM-direction probe inert.
3. Momentum-FREE normalized-gradient direction — probing along Muon's own momentum inert
   (Nesterov already covers it); per-increment normalization is load-bearing.
Working model: trajectory-level lookahead along a variance-reduced descent direction
distinct from the update direction. Composition with Muon survives because the probe only
rotates the gradient direction (polar-safe), unlike Nexus's magnitude-carrying pseudo-grad.

Failed variants (all held to the same gate): EMA reference point (2.3x slower training),
momentum-direction probe (inert), sign flip (inert), trust cap (tie), per-matrix norm (tie).
Self-refutations logged: B1 downstream impact, B2 refresh mistuning, B4 spikes.

## Open issues (for the next round / user)

- A1 (USER): reframed — staleness appears to BE the mechanism, not the bug; mid-training
  gating intuition still untested (C11 LR-schedule arm is the vehicle).
- C8: lr/wd-tuned AdamW baseline before any public superiority table.
- C9: second testbed (grokking two-source) to check the effect isn't protocol-specific.
- D2 residual: Muon+Nexus-clone arm (reproduce their incompatibility on our turf).
- Long tail: A3, A5, A6, A8, B3, B5-B10, C10-C12.
- The real test: BiBo-scale LM run with the champion config (gamma re-sweep mandatory).
