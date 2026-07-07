# Scope: lowest kappa at r=1 per NS-iteration budget (Round: kappa-pareto)

## Real goal
Square (r=1) weight matrices are the main case (attention QKVO). Current default (dsv4_10 + aurora_k1,
10 NS iters) leaves kappa ~40-470 at r=1. Exact kappa 1 costs 20 iters (dsv4_10 + aurora_k2). Goal:
push the kappa-vs-iterations Pareto front — the lowest kappa at r=1 for the fewest NS iterations.

## Frozen eval
.autoresearch/eval_kappa.py — deterministic. Input matrices: n=2048 square Gaussian, row-skew
decay in {0, 2}, OPTIMIZATION seeds {0,1,2}, HELD-OUT seeds {10,11,12} (promotion only).
Metric: kappa = smax/smin via fp64 Gram eigh of the fp16-pipeline output (prod dtype).
SCORE = geomean kappa at r=1 decay=2. Cost = total NS iterations (quintic step = 1; O(n^2) row/col
ops = free). Standing slice: r=2 decay=2 seed 0 must keep kappa <= 1.05, dead% = 0.

## Definition of done
kappa <= 3 at budget <= 12 iters on held-out (beats today: budget-16 pe8+k2 = 3.45), r=2 slice clean.
Stretch: kappa <= 1.1 at budget <= 12. Hard stop: 25 loop iterations or user stop.

## In-scope knobs
- Coefficient schedules (incl. a minimax/PE-style per-iteration solver for the TRUE square floor
  l0 ~ 2.5e-6..5e-5, or the post-prescale floor ~5e-4).
- Prescale / interstage row ops (aurora-like), sinkhorn pre-norm, polar splits (K polars, different
  per-polar schedules — restart effect real: 2x10 -> 1.00 where 1x20-tail -> 6.6).
- Anything O(n^2) between polars is free.

## Out of scope
- Touching the eval. Promoting to repo default before the T4/VM training verdict.
- fp32-only tricks (prod is fp16 NS). Full fp64 SVD anywhere in the loop (3050: fp64 = 1/64 rate).

## Prior art (do NOT re-tread)
- sink alone: not an orthogonalizer (r=1 floor ~2070, decay-invariant; L>2 useless). sink on top of
  aurora/prescale: redundant. sink+pe8 ~= plain pe8 at r=1.
- Constant-tuple sweeps lose to schedules. KJ f(1)=0.70 -> band [0.68,1.13]; pinned (2,-1.5,0.5)
  locks it. Best pinned constant (2.5,-2.5,1.0) reached only 8.28@15it.
- PE-8 (l0=1e-3) r=1 decay2 fp32: 5160. dsv4_10: 446. +aurora_k1: 40 (fp16 36). +aurora_k2 (20it): 1.00.
- Root cause: hard edge x0 ~ 4.8e-5 (decay0) / 2.5e-6 (decay2); each iter lifts smin <= a.
- fp32 not a stabilizer. gram backend == cuBLAS numerics (restarts [4,6]); eval uses plain cuBLAS NS.
- aurora rownorm prescale lifts the polar-input floor to ~4.8e-4 at r=1.

## Baselines (opt seeds, fp16, to beat at each budget)
Fill from run 0: dsv4_10+rownorm (B=10), pe8+rownorm x2 (B=16), dsv4_10 x2 rownorm (B=20), dsv4_12.

## Resources
Local RTX 3050. BiBo venv via ../BiBo/.venv. Eval ~1 min. Watchdog: ScheduleWakeup ~30 min.
