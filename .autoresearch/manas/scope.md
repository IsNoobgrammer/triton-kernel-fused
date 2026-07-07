# Manas round — scope contract

## Real goal
Establish whether Manas (probe-based gradient-alignment on Muon) delivers the Nexus claim
(better OOD/downstream at matched train loss) ON TOP of Muon, and position it against
AdamW+Nexus. The metric is a proxy for "same pretraining loss, better downstream" at LM
scale; the toy round's job is mechanism attribution + protocol validity, not the final LM claim.

## Artifact under optimization
`kernels/sm75/manas.py` (ManasOptimizer: probe rule, hyperparams gamma/rho/rank/refresh,
and candidate mechanism variants). Arm configs live in the eval harness; code changes to
manas.py are candidates like any other.

## Frozen eval
`.autoresearch/manas/eval_manas.py` — MNIST-1D heterogeneous-sources protocol with the
C1-C5 fixes baked in:
- 3 train regimes + 1 OOD regime, sources SHUFFLED per step (seeded) [C5]
- paired seeds: same seed => same init + same data + same batch/source order across arms [C4]
- primary metric: OOD accuracy at matched train loss along the CUMMIN frontier [C2/C3],
  reported as the mean over a fixed frontier grid; secondary: best-OOD-acc checkpoint,
  final OOD CE
- optimization seeds {0,1,2,3,4}; HELD-OUT seeds {5,6,7} — held-out consulted only at
  promotion [statistics.md]
- score(arm) = mean over seeds of paired delta vs the base-Muon arm run under the
  identical seed. Noise floor = std of paired deltas / sqrt(n_seeds); promote only > 2x.
FROZEN as of run-0. Any change to this file invalidates comparability and is forbidden
without explicit user sign-off (log the version hash in results.jsonl).

## Objective
AMENDED (user free-hand sign-off, wave 3): CO-PRIMARY metrics, both required positive for a
superiority claim, promotion on either if the other doesn't regress past noise:
  (a) delta_frontier — OOD acc at matched train loss (the axis Manas moves; held-out
      validated +0.0032 at 4 sigma for gamma 1e-2 rho 0.94)
  (b) delta_best_ood_acc — peak OOD accuracy (the axis the Nexus clone moves; positive
      control 5/5 seeds +0.0136)
The eval protocol file itself stays frozen (metrics both already computed by it).
Experimental mechanism variants live in exp_manas.py (sandbox, free to mutate);
kernels/sm75/manas.py is only updated for promoted, held-out-validated changes.
Stop target: a champion arm with held-out paired delta > 2 sigma AND no regression on
standing slices (final train loss parity within noise; best-OOD-acc not worse). Secondary
program goal: Muon+Manas >= AdamW+Nexus(clone) on the same protocol [C7].

## Constraints / invariants
- Never modify eval_manas.py after freeze; never tune on held-out seeds.
- Base Muon arm always re-run under the same harness version as the candidate (paired).
- A1 (trajectory-motion fix) is USER-owned: loop measures, does not redesign the probe
  reference point. Everything else in issues.md is fair game.
- Local GPU: RTX 3050 Laptop 4GB (BiBo venv, torch 2.6 cu124). Keep per-wave GPU time
  under ~30 min; batch arms into waves.
- No emoji anywhere. Commit results to master per workflow memory.

## In-scope knobs
gamma, rho, rank/refresh, probe sign, per-layer vs global normalization, probe-grad vs
clean-grad momentum split, d dtype, gamma scheduling, guard/trust-ratio logic, Nexus-clone
reference arms, second testbed (grokking) if mnist1d is inside noise.

## Out of scope
The eval protocol post-freeze; the held-out seeds; Muon internals (NS coeffs, aurora,
scale_mode — settled in previous rounds); the A1 reference-point redesign (user);
real-LM (BiBo) runs this round unless the user asks.

## Prior art (do not re-tread)
- manas_mnist.py: MNIST toy saturated -> blind; adamw outer arms uninformative there.
- manas_mnist1d_results.json: INVALID (all-NaN conclusive arrays, pre-sort-fix run).
- Previous rounds: wd is the dominant generalization knob (perf-per-flop round); coeff
  axis dead at matched wd; dither demoted (kappa round). Single-seed sweeps are noise.
- rank-8 low-rank capture 45%, dcos 0.49 (manas_mnist shadow measurement).

## Definition of done
Either: (a) a promoted champion (>2 sigma held-out, slices clean) + updated issues.md +
reflections, ready for the user's A1 method and an eventual BiBo-scale test; or (b) a
clean negative: mechanism-attribution verdict (A2/A7/A10) + protocol-valid null result +
recommendation, with all P0-P1 issues resolved or explicitly blocked-on-user.

## Budget / stopping
Max ~25 iterations this round; patience 3 (no promotion) -> plateau-break once -> stop;
watchdog wakeup every ~30 min re-reads this file + state.json and re-grounds.
