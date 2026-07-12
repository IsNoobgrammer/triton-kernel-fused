# The Manas Journey

From a paper skim to an optimizer that beats LR-tuned Muon by -0.12 to -0.145
train loss at 8 votes/step, with every hyperparameter computed, none tuned.
This is the record of how it got there - every redesign, what motivated it,
what it measured, and what died along the way.

A recurring pattern worth naming up front: every structural leap in this
design started as a user intuition of the form "this doesn't sit right",
usually against the current implementation's own logic - and every one of
them measured out. The step clock, the coefficient-1 fold, the memory-LR
decoupling, the consensus window, the unit-vote union, the QR-every-boundary
collapse: all user calls, all confirmed by tables.

---

## 0. Origin: the Nexus paper (arXiv 2604.09258)

The seed was a question about a paper:

> "can you read the nexus paper and there they say that their optimizer
> optimizes to the true common minima instead of summed losses common
> minima; are we able to implement that??"

Nexus finds common minima across heterogeneous data sources via gradient
similarity, using an inner-model walk - a two-model scheme. The user's crux,
stated at the start and never relaxed:

1. as simple as possible - no two-model / inner-loop complexity;
2. barely more memory or FLOPs than base (aurora) Muon;
3. viable at frontier scale (100M-token batches), not just toys.

## 1. v0: the rolling probe

The adaptation that met the crux: keep a small offset d, run every
forward/backward at theta + d, hand those gradients to the ordinary Muon
step, restore theta exactly.

    d <- rho * d - (gamma / ||g||) * g        (normalized, momentum-free)
    grad measured at theta + d; theta restored before step()

Validated on an MNIST-1D heterogeneous-sources protocol: +0.0245 OOD
accuracy at matched train loss (5 sigma), paired seeds.

## 2. The mechanism autopsy (Nexus theory refuted)

Direct measurement (a7_mechanism.py) killed the original story. The
pairwise-alignment force the Nexus mapping predicted injects most cleanly at
rho=0 - which is downstream-inert - and the trajectory moves 20-160x farther
than ||d|| over the memory window. What actually carries the win:

- LONG memory (rho ~0.98; rho=0 pure extragradient does nothing);
- the DESCENT sign (+probe inert);
- the momentum-FREE normalized-gradient direction (probing along Muon's own
  momentum is inert - Nesterov already prices that direction in).

Working model ever since: trajectory-level lookahead along a variance-reduced
consensus direction that is distinct from both the gradient and the momentum.
Manas pays in proportion to the information it has that momentum does not.

## 3. The graveyard (knobs killed by measurement)

- u buffer (EMA of applied updates added to the probe): tie at toy AND BiBo
  scale; ~= post-polar momentum direction, a measured inert control. Removed.
  (Notably the same object modded-nanogpt's EMA-Nesterov uses - it helps
  there because nothing else in that stack provides a lookahead.)
- rgd_tau (loss-surprise vote weighting): no-op; batch-mean loss spread too
  small at LM batch sizes. Removed.
- cos_beta (geometry vote weighting): sharpen neutral, novelty LOSES half
  the gain - direct proof the mechanism is variance reduction. EQUAL VOTE IS
  THE MECHANISM. Removed.
- nexus_gamma walker (in-step undecayed offset): linearly harmful (~0.6/unit)
  as an additive feature; later understood as the gamma_i > gamma branch of
  the ratio law - the mechanism was real but the dose was wrong by design.

## 4. Micro-voting: votes from gradient accumulation for free

With gradient accumulation, d gets one vote per MICRO-batch, recovered as
rank-space deltas of the accumulating p.grad (telescoping difference - no
model-sized snapshot). Discovery: at fixed global batch and identical FLOPs,
slicing 64x1 -> 32x2 -> 16x4 moved the edge -0.027 -> -0.057 -> -0.065 while
muon stayed invariant (it only sees the sum). Votes are free at scale, where
accumulation is forced anyway. The direction-count law replaced the original
rho-batch law: outcome tracks the number of independent directions in the
consensus.

## 5. The two-clock decay (user: "we need a step rho and a ga rho")

The per-vote rho clock failed at both regime edges: at ga128 it kept ~40 of
128 votes and forgot prior steps; at ga1 it stretched to a stale 50-step
memory and LOST to muon at b64/b128. User called it: separate the clocks.

- probe_rho_step: the only decay - memory in STEPS (N = 1/(1-rho)),
  invariant to accumulation count;
- within a step, all votes weigh equally (micro order is arbitrary; in-step
  recency is ordering noise).

Fixed ga1 on the spot (manas beat muon in all 9 cells at b64ga1).

## 6. Raw storage and use-time doses (user: "the fold coeff should be 1")

Two iterations of user pushback on the fold arithmetic ended with ALL stored
state raw:

    block   = sum of unit votes (coefficient 1, always)
    history = rho_step * history + block          (only rho ever touches storage)
    probe:  d = Q @ (gamma * history + gamma_i * block)

Both doses applied at probe time, nothing baked into buffers - either can be
scheduled mid-run and rescales its buffer retroactively. gamma_i defaults to
tracking gamma live.

The gamma_i grid then produced the RATIO LAW: winners sit on the diagonal
gamma_i = gamma; muting fresh votes (ratio < 1) is catastrophic (+0.07),
boosting them mildly harmful (~0.5/unit) - which retroactively explained the
walker's death as the ratio > 1 branch.

## 7. The dose law: gamma computed, not tuned

Assembled from grids at every slicing and batch:

    gamma = 0.08 * sqrt(lr / 3e-4) * k / sqrt(m)
    (k = votes/step, m = micro batch size; rho_step = 0.96 universal)

- per-vote gamma is the invariant (0.02/vote at ga1 == 0.1/step at ga4);
- k/sqrt(m): votes x per-vote gradient noise (validated 6/6 blind across six
  configs, including two never-measured cells predicted correctly);
- sqrt(lr): the transfer sweep's geometric-mid win; memory is LR-INVARIANT
  (user call: "memory is information, gamma is the strength dial" -
  rho 0.96 vs 0.90 tied to four decimals at the corrected dose);
- the overdose cliff RECEDES with k: at k=1 a 3x overdose flips sign; at k=8
  a 2.5x overdose costs ~0.01. All measured errors were benign under-dose.
- known boundary: at k=1 against muon at its exact tuned optimum, the edge
  compresses to ~0 - one vote per step means no consensus to sell. The
  answer is slicing (FLOPs-free), not tuning. At multi-GPU scale, DP ranks
  are votes (per-device consensus design), and ga1 stops existing.
  FINAL RECOMMENDATION (measured): ga1 -> plain Muon (edge ~0-0.02, probe
  overhead 5-10% tps: wall-clock negative); ga>=2 -> Manas. The window gate
  was retested under QR-every-boundary dynamics (3 cells, union vs controls):
  inconclusive-trending-positive but under the +-0.018 replicate bar - gate
  kept. Named future project for true single-backward ga1: GHOST VOTES
  (per-example rank-space votes from the outer-product structure of linear
  grads, Opacus-style hooks - k = batch size votes from ONE backward;
  decouples votes from accumulation everywhere).

Free-vote scaling at fixed global batch 256 vs LR-TUNED muon:
128x2 -0.038, 64x4 -0.077, 16x8 -0.145. Near-linear to k=8, ~6% tps cost.

## 8. Verification on real tasks

MNIST + MNIST-1D, muon LR tuned per dataset, manas at the formula constants
transferred verbatim from 137M LM sweeps, zero tuning: manas lower tail loss
6/6 paired seeds (~1.2-1.25x steps-to-target), test accuracy neutral at
saturation. The reproducible laptop demo (speed_mnist.py, ~5 min on an
RTX 3050). The 2D toy honestly TIES (8 seeds) - the consensus needs
dimensionality; the mechanism illustration and the win live at different
scales.

## 9. The window era (user: "something with Q doesn't sit right")

Q - the rank-8 basis everything lives in - was the last non-consensus object
in the design: rebuilt from ONE boundary gradient's randomized range. Three
user-driven redesigns in 24 hours:

1. EMA sketch (Y = rho_q*Y + G@omega, Q = QR(Y) at refresh): window aimed by
   consensus instead of a snapshot. Measured +50% edge at k=4
   (-0.080 -> -0.120). Aim noise had been costing ~0.04 all along.
2. Two-clock window (user: "shouldn't Q be a buffer of per-step -> per-ga?"):
   per-vote UNIT-NORMALIZED sketch increments into a raw pad Y_now, boundary
   fold Y = rho_q*Y + Y_now. The normalization is the point - raw deltas
   telescope back to the boundary sum (spread destroyed); unit increments
   give each micro DIRECTION equal voice. Union-vs-mean measured 3x at k=2.
3. The collapse (user: "reduce the slop, you have overengineered"): mode zoo
   deleted, snapshot deleted except the ga1 gate fallback, QR developed
   EVERY boundary - "refresh" stopped existing as a concept. Net -25 lines.
   Under continuous develops the rho_q sensitivity DISSOLVED (0.80-0.96
   indistinguishable at k=2 and k=4): the simplification deleted a tuning
   dimension that the complexity had itself created.

THE WINDOW, one sentence: Q is the QR of an EMA of unit vote directions.

GA1 gate (measured): a step with < 2 votes has no spread to cover - its
window falls back to snapshot cadence. The info cap at one vote/step is
real and unit-normalization does not rescue it (min_votes=1 arm: null).

## 10. Where it stands

The whole optimizer in four lines:

    votes:  unit micro directions; block raw; history = rho_step*history + block
    probe:  d = Q @ (gamma*history + gamma_i*block); theta restored exactly
    window: Q = QR(rho_q*Y + sum of unit vote directions), every boundary
    doses:  gamma = 0.08*sqrt(lr/3e-4)*k/sqrt(m); rho_step 0.96; all computed

Measured, all within-session anchors, 300-step BiBo harness at 137M:
-0.038 / -0.077 / -0.122 to -0.145 vs LR-tuned muon at 2 / 4 / 8 votes,
growing with ga, ~6-9% tps cost. Two-level consensus - what to think (C) and
where to listen (Q) - each built from equal-voice unit directions on its own
clock.

Still owed before any public claim: EMA-Nesterov baseline arm (prior-art
obligation), the 300M-token bpb/ICL A/B gate, the k=16/32 ladder (gamma cap,
vote saturation, rank stress - the frontier extrapolation), and the
union-vs-off deconfound at small m on the current build.

## Appendix: the discipline that made the numbers trustworthy

- Within-session muon anchors only; cross-session comparisons banned (anchor
  drift ~0.006-0.010, worse at high LR).
- Measured noise floors before reading effects: +-0.002-0.004 per arm at
  lr 3e-4, +-0.012 at k<=2 / lr 3e-3 (the ga1 gi-triples doubled as free
  replicate sets).
- Preregistered predictions before decisive sweeps (the 6/6 blind dose
  validation, the vote-clock discrimination, the transfer-law arms).
- Effects smaller than the cell's noise envelope are jitter until they
  replicate - several exciting patterns (quarter-power LR exponent at k=1,
  gi orderings at ga1) died correctly by this rule.
- Hierarchical sweeps (winner-first), never cross-products; knobs closed by
  measurement stay closed (u buffer, vote weighting, rho scans).
