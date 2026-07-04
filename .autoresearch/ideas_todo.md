# Ideas ledger — perf-per-flop round (optimizer downstream perf vs compute)

Status: TODO (untested) | RUNNING | ACCEPTED (kept, evidence) | REJECTED (evidence) | BLOCKED (needs X first).
Cost accounting: the unit is GEMMs, not NS iterations. One quintic NS iter = 3 GEMMs
(A = X^T X, A2 = A@A, X @ (aI + bA + cA2)). dsv4_10 = 30 GEMMs. ns8 = 24. ns6 = 18.
Gram-space NS replaces some full GEMMs with symmetric rank-k updates (~1.4x faster overall).
Muon NS total ~ 9% of step wall-clock at 137M (k2 measurement) => ~0.9%/iter, ~0.3%/GEMM.

## A. Coefficient / schedule axis

- [RUNNING] ns8 = KJ x6 + pin x2 (24 GEMMs). L0: matches dsv4_10 on ALL tested spectra incl.
  power-law smin/smax 2e-3. Predicted loss-neutral, +1.8% tps. VM wave 1.
- [RUNNING] ns6 = KJ x4 + pin x2 (18 GEMMs). L0: under-converges on decayed spectra (11% Fro err
  at p=1). Fidelity trade; screen decides if loss cares. VM wave 1.
- [REJECTED] mmx6 greedy per-step minimax at cap 1.125: solver stalls at KJ-clone fixed point
  [0.45, 1.11] after step 3 — greedy + tight cap cannot contract the endgame. No VM run needed.
- [REJECTED, prior round] Cap ballooning (umax 3.0): finds a=8.14 but interval blows to 2.82,
  net floor-lift 2.89/iter < KJ's 3.05. Worst-case coeff axis is exhausted PER-STEP.
- [TODO next] JOINT 6-step solve: optimize all 18 coeffs at once through the composed map,
  objective = min over [l0,1] of composite (or max |composite-1|), constraint = intermediate
  overshoot cap for fp16 safety. Greedy is only optimal unconstrained — joint can beat it under
  caps. Solve at l0 = 2e-3 (decayed momentum target) and 0.05. -> arm jns6.
- [TODO] Degree-7/9 polynomials: one extra GEMM per iter buys a steeper ramp. Compare at EQUAL
  GEMM budget (e.g. 4 iters of degree-7 = 16 GEMMs... deg-7 iter = 4 GEMMs vs quintic 3). Solve
  jointly; only worth it if floor-lift per GEMM beats quintic (PE paper hints quintic ~ sweet
  spot, but under OUR fp16 cap + decayed spectra the answer may differ).
- [TODO] Spectral-norm input normalization: replace Frobenius normalize with 2-3 power-iteration
  smax estimate (matvecs, negligible). Frobenius shrinks smax to ~1/sqrt(n) on flat spectra ->
  first iters waste lift recovering scale. Tighter [l0,1] start => fewer iters or better floor.
  Retune schedule after. -> arm specnorm6/8.
- [TODO] Per-group adaptive iteration count: cheap smin/smax estimate per shape-group (power
  iteration on the batched momentum), pick 6 vs 10 iters from a lookup. Saves iters on
  well-conditioned groups only. Complexity moderate; try after jns6/specnorm verdicts.

## B. Backend / kernel axis (wall-clock at fixed math)

- [RUNNING->wave2] gram arm: dsv4_10 through sm120 gram-space NS restart@(4,6) (Blackwell-
  validated: 5.70ms vs 8.17ms symmul, vs-truth 1.15e-3). Pure tps lever, ~2-3% step time.
- [TODO] gram + ns8 composition (needs restart re-autotune for the 8-coeff schedule; autotune is
  hardware-agnostic and runs in minutes locally).
- [TODO] Fold the aurora prescale row-norm into the first NS GEMM epilogue (kernel fusion; saves
  one O(n^2) pass — small, do only if a winner ships).

## C. Scale-mode / update-rule axis

- [RUNNING] normuon arm: per-row 2nd-moment EMA rescale after polar (paper NorMuon). Aurora paper
  measured it WORSE than Muon at 1.1B (2.33 vs 2.31); our aurora_k1 already covers the row-balance
  need. Low prior, cheap test. VM wave 1.
- [ACCEPTED baseline] aurora_k1 prescale (current default; Aurora 1.1B: 2.26 vs Muon 2.31).
- [TIE, prior round] aurora_k2 (k2 arm): best final loss (3.8900 vs 3.8923) but +9% wall-clock —
  per-GPU-hour loser at 137M. Revisit only at bigger scale or with gram backend cutting its cost.
- [TODO] k2-lite: aurora_k=2 with ns8 base (16 it, 48 GEMMs vs k2's 60). Only if ns8 confirms
  loss-neutral AND we want the perf-axis arm cheaper.
- [TODO] Momentum value sweep (0.9 / 0.95 / 0.98) under whd schedule: not optimizer-structure but
  in-scope "better config". One 2x2 factorial wave if structure axis dries up.

## D. Preconditioning / alternative optimizers

- [REJECTED x3, FINAL] Sinkhorn as NS preconditioner / prescale (SinkGD import): sinkhorn
  equalizes row/col norms NOT singular values. L0: power-law spectra pass through with smin/smax
  unchanged (2.0e-3 -> 2.1e-3), output kappa slightly worse. Family closed (sink2 prescale
  refuted twice before on kappa, once here on spectra).
- [REJECTED, math] Polar warm-start from cached previous Q (low-rank/EMA flavor): NS convergence
  depends ONLY on the input singular values, which rotating by Q_{t-1} does not change —
  polar(Q^T M) = Q^T polar(M), same spectrum, zero iterations saved. Newton/QDWH variants that do
  benefit from warm starts need inverses/QR per iter — GEMM-unfriendly on tensor cores.
- [TODO, user idea, reshaped] Low-rank EMA preconditioner: maintain EMA of top-r singular
  subspace (r~32, subspace iteration piggybacked on NS output, cheap); use it to (a) deflate the
  head before NS so the tail gets lifted in fewer iters, or (b) SOAP-lite rotate. Version (a) is
  the honest test of the user's "EMA of low-rank approx" idea: M' = M - (1-eps)*P_r M, NS(M'),
  add back rotated head. Risky (changes update direction — champ lesson), needs careful L0 first:
  does deflation actually reduce iters-to-converge on power-law spectra?
- [REJECTED, prior rounds] Signed-perm dither (champ): kappa-metric champion, training loser,
  dose-response penalty 0.035->0.097. Perturbing the input to fake kappa=1 costs more than it buys.
- [TODO, low prior] SinkGD/LEO as full optimizer arms (row/col norm only, no NS): far cheaper per
  step (0 GEMMs) but papers show quality gap vs Muon; only worth a run if we pivot to the
  "same perf, MUCH less compute" end of the Pareto (137M screen would need loss ~parity).

## E. Eval / methodology notes (frozen — never edit to flatter a candidate)

- Screen: exp_perf.yaml, 600 steps, ~4.2 min/run. Baseline twins: base1 5.0617 @155308 tps.
  base2 = noise floor (pending pickup). Confirm: 2000-step v2 config, noise floor 0.0016.
- Champ lesson (standing): kappa/orthogonality metrics do NOT transfer to loss — only the
  frozen training eval promotes. L0 spectral checks are for REJECTING (can't-converge) only.
- Toy tasks (sorting; MNIST-class): measured INSENSITIVE to optimizer-structure deltas at our
  scale — do not use for promotion decisions (prior round NULL result).
- MNIST-1D (arXiv 2602.13348 benchmarks it): considered 2026-07-04 as ablation data — REJECTED
  as primary (discriminates architectures not optimizers; no op/task labels so MI specialization
  metric dies; sample test set vs our exhaustive held-out). DEFERRED as post-hoc generality check
  for a promoted winner only (~4k examples, minutes on T4). For a second MoE task axis, prefer
  mixing task families (mod-arith + sorting + copy) — keeps labels exhaustive and skewable.

## F. MoE specialization round (tangential imports; opened 2026-07-04, user-driven)

Framing: the joint landscape has no single global minimum per module; the operational target is
FUNCTIONAL DIVERSITY between experts (traffic fairness is already owned by BiBo's DSv3
selection-bias heuristic, b += 0.01*sign(mean-load) every ~300k tokens - slow global correction,
free short-run specialization). The optimizer-side gap = function diversity, not load diversity.
Testbed: train_grok_moe.py - 4-op mod-97 grok task with SKEWED train op mix (40/30/20/10),
BiBo-semantics router, stacked (E,din,dout) expert params (FusedMuon batches ndim==3 natively),
metrics = heldout acc + grok_step + MI(top1 expert, op) per layer + load entropy.
Potato preset: --p 61 --batch 1024 --steps 4000 (local 3050).

### Wave 1 induction (T4 x2, 2026-07-04, results.jsonl)
- Muon groks the MoE task (acc ~1.0 @ 1800-2200); AdamW @ awd 1.0 stuck at 0.23-0.29 by 3000 but
  CLIMBING (curve shape = pre-grok memorization) - budget-truncated, NOT a fair loss for AdamW
  yet. Also awd 1.0 applied to ALL params (emb/norms/router) is likely too high (user call).
  Wave 2: adamw_wd sweep {0.1, 0.3, 1.0} @ 6000 steps before claiming the Muon-2x headline here.
- MI concentrates in the LAST layer only (L0=L1=0.00 always; L2 ~1.0-1.5 bits) - CONFIRMS the
  dense-early/MoE-late arch decision a second time, now at full grok.
- EXPERT COLLAPSE post-grok: MI peaks ~1.48 @ 2000 then decays to exactly 1.00, minload -> 0.000
  (dead experts). Reading: wd 2.0 prunes experts the routing stopped using; the 300k-token bias
  balancer is too slow to resurrect them. 1.00 bit = a stable 2-way functional split survives.
  Motivates #2 per-module wd: lower wd on expert stacks (e.g. 0.5) vs attn 2.0 - does it keep
  more experts alive / MI above 1.0 without slowing grok?
- AdamW shows nonzero mid-layer MI (0.11-0.13) where Muon shows 0.00 - memorization-phase
  routing artifact, not specialization; do not over-read.

### Wave 2 induction (T4 x2, 2026-07-04)
- FAIR ADAMW CONTROL DONE: awd sweep monotone (0.1 = memorized-dead acc 0.037 w/ train loss
  0.009; 0.3 = crawl 0.136; 1.0 = GROKS @5000, acc 0.9996). wd law holds for AdamW same as
  Muon. Headline now honest: Muon groks 1800-2200 vs AdamW 5000 at each one's workable wd
  -> ~2.4x on MoE, matching the dense prior. "awd 1.0 too high" hypothesis REFUTED.
- expert_wd 0.5 (idea #2, LOW direction) REJECTED: slows grok (0.46/0.88 @3000, no grok, vs
  baseline 1800-2200) AND lowers MI (0.37/0.48). Keeps experts alive (minload 0.05-0.08) but
  alive-and-undifferentiated: low wd lets experts retain memorization. Rate-distortion says
  experts want MORE compression -> wave 3 tests ewd 3.0/4.0 (HIGH direction).
- Post-grok "collapse" is STABLE CONVERGENCE, not decay: default @6000 pins MI at exactly
  1.00 bits and acc 0.9995 from 3000-6000. Minimal 2-way expert split is the equilibrium
  under wd 2.0 + slow balancer. Acc unaffected -> MI is diagnostic, not target; do not chase
  MI for its own sake (fitness-sharing lesson generalizes).
- adamw awd0.3 mid-layer MI hits 0.65 while acc 0.14 - more evidence mid-layer MI is a
  memorization-routing artifact, inversely related to generalization if anything.
- STANDING DIAGNOSTIC (2026-07-04): mid-layer MI = memorization marker. Evidence gradient:
  awd0.3 never-groks 0.65 > awd0.1 0.28 > grokked-adamw scar 0.57-vestigial > muon 0.00.
  Hypothesis: memorized per-op lookup features are op-separable early (router splits);
  the generalizing shared Fourier circuit is op-shared until readout (specialize last layer
  only - both grokked arms agree, L_last MI 1.00). AdamW groks ON TOP of its 5000-step
  memorization scaffold (scar persists); Muon transitions 2.4x faster + spectral spreading
  -> scar never consolidates. USE: rising mid-layer MI in a new arm = memorizing, not
  generalizing. Untested causal check (not queued): uniform-routing L1 in grokked adamw
  should be acc-neutral (vestigial); L_last should hurt in both.
- Wave 3 (pushed): decor {0.5, 1.0} x2 seeds (idea #3 grad-space variant: g_e -= decor*
  mean_E g before step) + ewd {3.0, 4.0}. NOTE decor acts on grads pre-momentum/pre-polar,
  not on the post-NS update (FusedMuon internals untouched) - screen-grade approximation.

### Wave 3 induction (T4 x2, 2026-07-04) - DOUBLE NULL, round converging
- decor {0.5, 1.0} and ewd {3.0, 4.0}: ALL arms grok 1800-2200 (baseline band), acc ~1.0,
  MI 1.00, minload -> 0. Nothing separates from seed noise (+-200-400 steps).
- ROUND PATTERN after 3 waves: state repulsion HARMFUL, update decorrelation NULL, per-module
  expert wd NULL both directions. No optimizer-side specialization intervention beats plain
  Muon + uniform wd 2.0 + BiBo balancer. Specialization self-organizes (last layer, 1 bit,
  exactly enough) and acc saturates at 1.0.
- STANDING CAVEAT: this task saturates post-grok - there is NO HEADROOM for specialization
  to pay off in acc. A discriminating test needs a task family where acc does NOT saturate
  (mixed mod-arith + sorting + copy under skew, per section E note). Remaining queue items
  (annealing, Shapley, tournament) are heavy and would inherit the same saturation problem -
  do not run them on this testbed.
- Round wins to keep: Muon 2.4x AdamW on MoE-grok; wd law extends to AdamW (monotone 0.1->
  1.0); mid-layer-MI memorization diagnostic; dense-early/MoE-late confirmed twice.

### Algorithmic paradigm imports (wave 4, opened 2026-07-04 - user directive: mechanisms
### like repulsion, NOT hp tuning; wd axis closed as diagnosis-only)
- [NULL 2026-07-04] Grokfast x Muon: lam=2 = parity (grok ~2000/~2400 vs baseline 1800/2200,
  both seeds inside noise), lam=5 = SLOWER (grok ~2600-2800, acc 0.984 @3000). Dose-response
  toward harm. MECHANISM READING: grokfast's 50x on Adam comes from MAGNITUDE amplification
  of the slow grad component; Muon's polar DISCARDS magnitude (all singular values -> 1) and
  its momentum already low-pass filters direction. Muon structurally CONTAINS grokfast's
  benefit - the composition is redundant, and at high lam the stale EMA direction actively
  fights fresh signal. Transfers: do not stack EMA-amplification tricks on orthogonalizing
  optimizers.
- [REJECTED 2026-07-04] Lookahead x Muon (k=5, beta=0.5): strongly harmful - no grok by 3000,
  acc 0.25-0.27, curves look like baseline at ~half speed. slow.lerp_(fast, 0.5) every 5
  steps is an effective-lr halving; in a grok regime where escape time scales with lr this
  is pure slowdown. Flat-minima averaging buys nothing here. Combo gf2+la5 = 0.90 @3000
  (grokfast partially rescues lookahead's lr cut, still worse than plain baseline).
- SECONDARY FINDING (real, not promotable): grokfast STEERS SPECIALIZATION LOCATION. gf2 s1
  put the 1-bit expert split in the MIDDLE layer (MI 1.00 L1, 0.32 L2) at acc 0.997; gf5
  ended with TWO specialized layers (0.99/1.00); combo likewise (1.04 L1). First mechanism
  in 4 waves to move specialization off the last layer while grokked. CAVEAT to the
  mid-layer-MI diagnostic: it means memorization only when acc is FLAT/LOW; in a grokked
  net a mid-layer split can be a legitimate generalizing configuration.
- [QUEUED] Dion-style low-rank orthogonalization (orthogonalize only top-r subspace):
  compute-side win candidate for LM phase, pairs with the perf-per-flop goal.
- ROUND STATUS after wave 4: FOUR waves, every mechanism <= baseline Muon + wd 2.0
  (repulsion harmful, decor null, per-module wd null, grokfast null/redundant, lookahead
  harmful). Testbed-close call OVERRULED by user: work the ledger backlog first (wave 5).

### Wave 5 (RUNNING, 10 arms - ledger backlog + user combos; user: retry repulsion at
### VERY low beta = aux-loss regime, and test combos rep+load+grokfast)
- micro-repulsion beta {1e-4, 1e-5}: compounding drift (1+b)^3000 = 1.35x / 1.03x - the
  blowup that killed 1e-2 does not apply; acts as a weak diversity prior in the update.
- grad_rep 0.5: grad-space repulsion, g_e += b*(g_e - mean_E g) - the sign-flip of decor
  (which was null); amplifies each expert's deviation BEFORE momentum+polar.
- xorth (x2 seeds): cross-expert grad whitening along E axis (E x E gram inverse-sqrt via
  eigh, E=8 ~free) - section "Muon-native #3", genuinely ours, first run.
- niche 0.5: fitness-sharing lr - expert grads scaled (1/(E*load_frac))^0.5 in [0.5,2];
  load read FREE from the router's own load buffer.
- scap 2.0 @ wd 0.1: sigma-cap as wd SUBSTITUTE - persistent-power-iteration smax estimate,
  clip only the top singular value post-step. First test of the LM-ranked idea (2) on grok:
  can targeted spectral compression replace uniform decay for generalization?
- cautious 2.0: sign-masked decay (decay only where the step already shrinks |w|; FusedMuon
  wd=0, manual post-step). LM-ranked idea (1) screened on grok.
- combos (user ask): rep1e-4 + gf2; rep1e-4 + niche0.5 + gf2.
- Still in backlog after this wave: specialization annealing (needs per-expert entropy
  plumbing), MI-guided bias (router-side), Shapley (heavy), tournament (heavy), low-rank
  EMA deflation (needs L0 first), degree-7 polys / specnorm / per-group iters (coeff axis,
  moot on grok - LM only), SinkGD/LEO full-optimizer arm (scale calibration needed),
  momentum sweep (hp - excluded by user directive), mixed task families (harness change).

### Wave 5 induction (T4 x2, 2026-07-04) - backlog screened, no promotions
- micro-repulsion 1e-4/1e-5: NULL, clean parity (grok ~2000, acc 0.999). User's aux-loss
  regime confirmed harmless at these doses - but buys nothing. Dose ladder now complete:
  1e-5 null, 1e-4 null, 1e-3 -1000 steps, 1e-2 divergence. Family CLOSED with full curve.
- grad_rep 0.5: NULL (grok ~2000, parity). Same story as decor: routing already
  decorrelates expert grads; amplifying deviation changes nothing measurable.
- xorth: NULL, seeds straddle baseline (s1 grok <=2000 vs baseline 2200 = faster; s0 ~2200
  vs 1800 = slower). No harm from E-axis whitening, no win. Our pre-batched-expert
  advantage produces no grok-side signal; keep as LM-phase candidate only if free.
- niche 0.5: NULL (grok ~2000-2200). Load-proportional lr neither helps nor hurts -
  consistent with bias-balancer redundancy prediction. Family closed.
- scap 2.0 @ wd 0.1: FAILED to grok (acc 0.15 @3000, mid-MI rising = memorizing).
  CAVEAT: smax was not logged - unknown whether the cap ever bound; cap 2.0 may be a
  no-op on these weights. But the mechanism-level reading stands: generalization pressure
  must act on the WHOLE spectrum (wd shrinks every direction; the memorized solution's
  components are not confined to the top singular direction). Top-sv clipping alone is
  not a wd substitute on grok. LM candidacy demoted until an smax-logged rerun says the
  cap binds.
- cautious 2.0: SLOWER (acc 0.39 @3000, climbing; mid-MI 0.54 rising). Sign-masking
  halves effective compression pressure -> behaves like wd ~1.0 per the dose law.
  On grok, weaker pressure = slower escape, exactly as predicted. Its LM promise
  (compression without signal tax) is UNTOUCHED by this - grok punishes weak pressure,
  LM rewards low tax. Still LM-ranked (1), now with a calibration note: match effective
  decay dose, not nominal.
- combos: rep1e-4+gf2 NULL at grok point but 2.4x baseline acc at step 1000 (0.34 vs
  0.14) - grokfast+repulsion accelerates PRE-grok progress, converges to same grok step.
  rep+ni+gf NULL (transient L1 MI 1.15 @2000, collapses after grok).
- FIVE-WAVE VERDICT: 13 mechanisms screened, zero beat Muon + wd 2.0 on grok. The polar
  + full-spectrum decay pair is the frontier on this testbed. Remaining discriminating
  power lives in the LM regime.

## G. ONLINE LM-EMULATOR (opened 2026-07-04, user pivot - THE testbed going forward)

User reframe, accepted: grok = memorize-then-generalize = WRONG regime for LM. New eval
= single-epoch online stream: fresh compositional mod-97 chains every step (depth 1-4,
Zipf mix 0.4/0.3/0.2/0.1, left-fold eval), val held out by key, sample space >> stream
(no repeats ever) -> memorization impossible, all progress = compression. Emulates the
compute-bound LM regime at toy cost. Harness: ablate_muon/olm.py + run_olm.py
(`bash ablate_muon/run.sh olm`), 4-layer model, T4-ready.
COMPRESSION CALIBRATION: LM 81k vocab starts at ln(81920)=11.3 nats CE, strong models
land ~1.0 nat => frac ~0.09 of initial entropy remains. Ours: init ln(97)=4.575 nats,
LM-matched target ~0.41-0.5 nats at 6000x768 budget. Metric `frac` = CE/ln(97).
Task difficulty is TUNABLE (depth mix / max depth / p) if wave 1 lands too easy/hard.
- [UPDATED same day, user] v2: +5% label noise (train AND val) -> irreducible CE floor
  0.4229 nats = frac 0.0924, deliberately == LM residual 0.09: the race is toward an
  entropy floor, never zero, exactly like text. Metric of record = gap (CE - floor).
  Depth extended to 6 (Zipf mix 0.30/0.25/0.20/0.125/0.075/0.05). Default arch =
  dense_first=1 (layer 1 dense, 2-4 MoE, user call); all-MoE df0 = contrast arm.
- [RUNNING olm wave 1] default df1 wd0.1 x2 seeds, default wd2.0 (regime check:
  grok-optimal wd predicted to HURT online), adamw x2 seeds, df0 all-MoE contrast.
- [olm wave v2 DONE - too-hard task, NO WSD; superseded by v3 but regime checks stand]
  Three clean regime confirmations at 6000 steps: (a) Muon wd2.0 DEAD (frac 0.995, flat,
  MI 0) - grok-optimal wd kills online learning, regime flip REAL; (b) AdamW DEAD both
  seeds (frac 0.995) - BUT confounded by zero warmup (Adam cold-start second moments in a
  one-pass stream), NOT yet a fair "adamw fails" claim -> v3 warmup arm decides; (c) Muon
  wd0.1 LEARNS (frac 0.82-0.88 df1/df0). Task too hard (best 0.82 vs target 0.3-0.5) ->
  v3 recalibrated. NEW: online MI spreads across MULTIPLE MoE layers (0.51/0.32/0.34) and
  appears LATE w/ learning onset - unlike grok's last-layer-only; regime-specific, read as
  early-learning MI given low acc.
- [olm wave v3 DONE - recalibrated task + WSD + warmup, UNIFORM eval metric (pre-dist-fix);
  dist-matched acc recomputed by hand from per-depth]. KEY RESULTS:
  1. WARMUP RESCUES ADAMW: v2 adamw dead@chance -> v3 adamw learns (d1 0.087/0.218).
     Confirms v2 flatline = Adam cold-start second moments, not regime. Fair test valid.
  2. HEADLINE (fair: matched WSD+warmup+lr, online/compute-bound regime): Muon CRUSHES
     AdamW. Dist-matched acc Muon ~0.46/0.41 vs AdamW ~0.05/0.11 (4-9x); depth-1
     ~0.88 vs ~0.15; Muon cracks depth-2 (0.10-0.16) AdamW does not (0.02-0.03).
     Strongest perf-per-flop result of the round - LM-like regime, everything matched.
  3. WARMUP ROBUSTNESS ASYMMETRY (new): Muon-no-warmup LEARNS (d1 0.574, faster early
     0.478@2k vs warmed 0.260, plateaus lower) vs AdamW-no-warmup COLLAPSES (v2 chance).
     Muon warmup-robust, AdamW warmup-dependent. Refines "muon needs no warmup".
  4. df0 all-MoE matched/beat df1 dense-first, reached deepest (d3=0.110, dist-acc ~0.43)
     - CONTRADICTS grok dense-early finding BUT confounded (df0 = 4 MoE layers vs df1 3 =
     more capacity; 1 seed). Needs equal-param rerun before believing.
  Seed variance high (adamw 0.087 vs 0.218) but Muon>AdamW gap >> seed spread = robust.
  NEXT olm run reports dist-matched frac/acc directly (eval fix pushed a58f5a4).
- [PROXY VALIDATED 2026-07-04 - olm wave v4] olm REPRODUCES the 137M-LM ordering:
  final frac (lower=better): normuon 0.527/0.561 (2 seeds, NO overlap with default) <
  ns8 0.560 ~ default 0.566/0.592 ~ k2 0.568. normuon WON (matches LM won), ns8 & k2
  TIED inside default band (matches LM tied). normuon depth-2 0.313 vs default 0.156 =
  2x on the composition zone; transitions earlier AND converges lower (both signals
  agree). DECISIVE: normuon was grok-HARMFUL (sign flip) - olm lands it on the LM side,
  i.e. olm DISAGREES WITH GROK exactly where grok lied. Not mere correlation; captures
  the compute-bound regime. Caveats: n=2 (normuon/default), n=1 (ns8/k2); proxy keeps
  ORDERING not magnitude (normuon edge ~6% frac here vs 0.026 nats at 137M).
  => olm is now the CHEAP SCREEN for the mechanism backlog. Rescreen candidates here
  before spending 120M-token LM runs. normuon promoted: LM candidate + BiBo.
- NS8-DEFAULT FLOOR (olm, deterministic): seed0 0.560 / seed1 0.556 - tighter than and
  slightly under the dsv4_10 floor (0.566/0.592), reconfirming ns8 tied-or-better + cheaper.
  Do NOT re-run default each wave (user); compare mechanisms (seed 0) to 0.560.
- [olm v5 RUNNING] mechanism re-bench on validated proxy. Default now ns8 (6 KJ) aurora_k1
  (user call: normuon is a candidate for real LM/BiBo, NOT the toy default; ns8 = tied +
  cheaper). Mechanisms extracted to mech.py (shared; grok_moe keeps its inline copies).
  Arms: default x2 (floor), cautious2.0, scap2.0 (smax logged), repulse1e-3, grad_rep0.5,
  xorth, grokfast2.0. Bar: clear the 2-seed default spread (v4: 0.566-0.592 frac).
  Watch: cautious (LM-predicted-good) and scap (wd substitute) - grok said slow/failed but
  grok punishes weak/targeted compression, LM should reward it. If scap smax >> 2.0 the
  cap binds; if ~2.0 it is a near-no-op.
- [olm v5 DONE] mechanism re-bench vs ns8 floor 0.556-0.560 (final frac, seed 0):
  xorth 0.565 [CORRECTED from NULL - see utilization note] | grad_rep 0.562 NULL |
  scap2.0 0.563 NULL-loss (may be non-binding) |
  cautious2.0 0.598 MILD HARM (slower, same as grok - weak decay = slow escape even
  online at this budget; LM-good prediction REFUTED on proxy) | repulse1e-3 0.692 HARM
  (never learns depth-2, MI~0, regresses - weight repulsion blocks the composition
  circuit) | grokfast2.0 0.721 HARM+UNSTABLE (acc DEGRADES 0.12->0.07 end; online fresh
  data makes the grad EMA average over DIFFERENT samples so amplifying the "slow
  component" amplifies staleness - flips from grok-null to olm-harmful).
  VERDICT (loss/compression): zero mechanism beats the floor. Only normuon (v4) wins on loss.
  [XORTH RECLASSIFIED 2026-07-04, user caught it]: on the MoE-SPECIALIZATION+UTILIZATION
  axis (the round's original objective, not compression), xorth (our cross-expert grad
  whitening) is the WINNER and acc-neutral. At step 6000: MI 0.32/0.23/0.48 = specialized
  in ALL 3 sparse layers (default leaves layer-1 dead at MI 0.00), minload 0.039 = healthy
  utilization no collapse, acc 0.46 = at floor. scap specializes too (0.48/0.16/0.49) but
  COLLAPSES load (minload 0.001); xorth gets specialization AND utilization AND acc. This
  is exactly the "acc-neutral functional diversity" bar the MoE round set. NOT yet a loss
  win because this task saturates at floor with 2-3 experts -> extra utilization is free
  but UNREWARDED. To convert to a quality win need a CAPACITY-BOUND regime (more experts /
  harder-multitask data / narrower experts) where the task actually needs all E experts.
  -> xorth reopened as a candidate FOR THE UTILIZATION/CAPACITY axis; design a capacity-
  bound olm variant to test if its utilization edge becomes a loss edge.
- [olm v6 RUNNING] CHEAP NS-FREE OPTIMIZERS ('same quality, less compute' arm), alt_opt.py:
  * SinkGD (2502.06742): Sinkhorn alternating row/col RMS normalize of momentum, 0 GEMMs.
  * LEO (github vukrosic/leo-optimizer): Lion double-EMA + one-shot row/col normalize to
    align_const, element-wise, 0 GEMMs. Own lr (0.01 / 3e-3 arms).
  * Dion (2504.05295): low-rank orthonormalization, amortized power iter + error feedback,
    rank_frac {0.25, 0.5, 1.0}. rf1.0 = sanity (should ~ Muon floor 0.560).
  Arms: leo lr{1e-2,3e-3}, sinkgd lr{1e-3,3e-3}, dion rf{0.25,0.5,1.0}, dion rf0.5 s1.
  Bar: reach ~0.560 floor => a cheaper optimizer at Muon quality = big perf-per-flop win.
  Note SinkGD-as-NS-PRECOND was refuted 3x on kappa; this tests it as a FULL optimizer
  (different claim - row/col norm as the whole update, not a Muon prescale).
- [olm v6 DONE - CHEAP NS-FREE OPTIMIZERS ALL REJECTED] LEO, SinkGD, Dion all land far
  above the Muon ns8 floor 0.56 (LEO worst, ~0.99 dead; SinkGD bad; Dion best-of-the-three
  but still very bad). Even Dion rf1.0 (full-rank sanity) did not reach Muon. Verdict: the
  NS orthogonalization does real work that row/col-norm (SinkGD/LEO) and low-rank power
  iteration (Dion) cannot cheaply replace at this scale/regime. Consistent with the earlier
  'sinkhorn != orthogonalizer' finding, now confirmed for FULL optimizers not just preconds.
  CODE REMOVED (alt_opt.py deleted, arms unwired) - family closed. 'same perf, less compute'
  via cheaper-than-NS optimizer = dead end; the compute win stays ns8 (fewer NS iters).
- [olm v7 QUEUED - new instrument + winners together + combo + capacity-bound]
  INSTRUMENT UPGRADE (user-directed): bias balancer every 10 STEPS (was ~every 391 steps
  = 300k tok / 768 samples -> only ~15 updates/run = collapse cause); pad-masked load, MI,
  utilization (loss already pad-free); soft top-2 WEIGHTED MI (both selected experts by
  combine weight, not top-1); metrics.py: effective-experts exp(H(load)) + spec-fraction
  MI/ceiling replace noisy minload. bias_factor 0.01 (600 updates/run). mult knob for
  narrow/capacity-bound experts. Gauge: 768 samples/step, ~6.2 real tok/sample = ~4750
  real tok/step, 14 positions incl pad.
  Arms: default s0/s1 (RE-baseline under new bias), normuon, xorth, normuon+xorth combo
  (s0/s1), capacity-bound mult=1 default + combo. Tests: combo = loss+utilization both?
  capacity-bound = does xorth utilization edge become a loss edge when task needs all E?
- [olm v7 BIG EARLY RESULT] The bias-cadence fix (every 10 steps) improved LOSS, not just
  the utilization metric: DEFAULT s1 frac 0.556 (broken bias) -> 0.451 (fixed), eff 7.3-7.6/8
  (was collapsed), depth-2 acc 0.16->0.42, depth-3 0.04->0.18. Dead experts were a REAL
  capacity leak; reviving them unlocked deeper composition. => (a) FLOOR RESETS to ~0.45;
  ALL prior verdicts (normuon win, mechanism nulls) were under broken-bias/collapsed regime
  and must be re-read under healthy utilization; (b) my "xorth utilization is free-but-
  unrewarded" call was WRONG - utilization IS rewarded here, the broken bias masked it, so
  the healthy baseline now captures most of it and xorth must ADD on top (higher bar, cleaner
  test). Await full v7 table for per-arm verdicts.
- [olm v7 SEED VARIANCE CAVEAT] default s0 0.535 vs s1 0.451 = 0.084 frac spread (phase-
  transition timing: s1 cracked depth-2 to 0.42, s0 to 0.22). This spread > the arm-to-arm
  deltas we chase -> single-seed arms (normuon/xorth/combo/capacity in v7) are NOT separable
  from noise. Both seeds beat old floor 0.556 so bias fix is robust, but per-arm verdicts
  need MULTI-SEED. v8 = 3 seeds each on top 2-3 configs, compare means; or longer budget to
  converge the transition tail. Emergence variance is intrinsic (sharp seed-dependent jumps).
- [xorth WALL-CLOCK TAX ~20%] xorth = 5k steps when muon/normuon done 6k. Cause: 6 eigh
  calls/step (w1,w2 x 3 MoE layers) on 8x8 grams - trivial math, high launch overhead,
  36k tiny eigh/run. xorth case now WEAKENED twice: (1) bias fix gives healthy eff ~7.5/8
  FREE on default, so its utilization win is mostly already delivered; (2) +20% compute for
  perf-per-flop. Survives only if v8 multi-seed shows an ADDITIVE frac win over healthy
  baseline. Fix if kept: batch 6 grams -> one (6,8,8) eigh, and/or run every-N-steps
  (amortize). Else drop.
- [olm v7 FULL TABLE - REVERSAL] Under FIXED bias (healthy eff ~7.5/8 everywhere), same-seed
  final frac: default s0 0.535/s1 0.451 = BEST; normuon s0 0.549 (+0.014 WORSE); xorth s0
  0.559 (+0.024 worse, no spec advantage: 0.13/0.24/0.15 ~ default 0.13/0.26/0.20); combo
  normuon+xo 0.558/0.559 (+0.023/+0.108 worse). Plain Muon+ns8 beats every variant same-seed,
  AND default s1 cracked deepest composition (d2 0.42, d3 0.18).
  * NORMUON MoE-WIN was a BROKEN-BIAS ARTIFACT: v4 (broken) normuon +0.035 better -> v7
    (fixed) -0.014 worse. Sign flipped with the balancer. RESOLVES Aurora tension (Aurora:
    normuon worse @1.1B) - healthy MoE regime now agrees. RETRACT normuon MoE promotion.
    (Dense 137M normuon win is separate - no MoE balancer there - still open for real-LM.)
  * XORTH REJECTED: utilization now free (bias fix), no spec edge, +20% compute. Dead.
  * COMBO = variance-damper not winner: consistent 0.558/0.559 but KILLS the upside
    (default's lucky deep-transition seed 0.451 never happens under combo).
  * mult=1 CRIPPLED the model (plain acc 0.096, degrading) - not a clean capacity test; BUT
    combo stabilized it (0.096->0.443 acc, rising) = normuon/xorth are STABILIZERS in
    degenerate regimes (matches normuon's Moonlight bf16-guardrail origin), not healthy-regime
    loss wins.
  THROUGH-LINE: v4/v5 normuon+xorth wins were collapsed-expert artifacts. Fix the balancer ->
  plain Muon+ns8 is the frontier; the mechanisms only help when something else is broken.
  Awaiting user's inductive biases on normuon/xo.
- [olm v7 COMPOSITIONAL + SPEC re-analysis (user prompt)] Re-ranked on the axes I skipped:
  * DEPTH-2 (compositional gen) same-seed @6000: default 0.225/0.421 BEST; normuon 0.180;
    xorth 0.138; combo 0.152/0.141. Variants LOSE HARDER on composition than on aggregate
    frac (aggregate cushioned by depth-1 which all nail). depth-3 same story. Reinforces
    default-wins, does not rescue variants. NOTE depth-2 is TRAINED (25%) = IID gen not pure
    OOD.
  * SPEC: utilization maxed ~7.5/8 everywhere (non-differentiating). Healthy-regime spec
    modest+undifferentiated; xorth does NOT specialize more than default (0.13/0.24/0.15 vs
    0.13/0.26/0.20) - its spec edge was ALSO a broken-bias artifact. spec is CAPACITY-driven
    (mult=1 arms highest spec 0.39-0.47) not optimizer-driven, and ANTI-correlated with loss
    (capacity-starved = more spec, worse loss).
  * OOD-eval idea DROPPED (user): depth-4 already ~chance, so held-out depth-5/6
    extrapolation would measure nothing. No headroom for a length-OOD test at this scale.
- [OLM = THE KEY DELIVERABLE OF THE ROUND] validated single-GPU LM emulator: tracks real-LM
  optimizer ordering, exposes LM-only phenomena (emergence, phase transitions, per-depth
  compositional learning), rich cheap metrics (frac/per-depth acc/eff/spec). Reusable
  instrument - every downstream verdict depends on it.
- [olm v8 RESULT - NS ITER COUNT MATTERS, cheap wins] 8-iter (ns_kj=6, 6KJ+2pin, DEFAULT)
  BEATS 10-iter (ns_kj=8, dsv4_10 = DeepSeek): frac 0.535/0.451 vs 0.592/0.609, both seeds
  (default's unlucky 0.535 still beats both 10-iter seeds). 10-iter s0 stuck at depth-2
  CHANCE (0.022) while 8-iter cracks it (0.42) - extra 2 KJ iters HURT (likely quintic
  overshoot the 2 pinned iters cannot clean). CONTRADICTS grok "iter axis dead" (were equal
  on grok) - olm is more discriminating. => cutting NS iters 10->8 is beneficial here, not
  just free. TAG FIXED: was "_ns8" (=ns_kj=8=10 iters, CONFUSING) -> now "_it10" (total
  iters). k2 arm still pending.
  SCALE CAVEAT (do NOT over-claim): NS iters-to-converge scale with matrix size/spectrum.
  Our expert mats tiny (128x512, well-conditioned) -> converge in few iters, so 8 suffices
  and dsv4_10's extra KJ iters overshoot/hurt. DeepSeek's much larger mats may genuinely
  need 10 -> right at THEIR scale. Optimal iter count is SCALE-DEPENDENT, not universal.
  "8 beats 10" is an OLM-small-scale result; don't extrapolate to big models. What transfers:
  iter count is a live knob olm CAN resolve (grok could not); at small scale go cheaper.
- [olm v8 COMPLETE - 2 seeds each, NS-config verdict] final frac (mean) / AUC:
  8-iter (ns8): 0.493 / 0.676 | k2: 0.527 / 0.701 | 10-iter (dsv4_10): 0.601 / 0.727.
  PAIRED (same-seed) ranking CLEAN both seeds: 8-iter < k2 < 10-iter (s0: .535/.558/.592;
  s1: .451/.495/.609). Neither more NS iters (10) nor more aurora passes (k2) beats the
  cheap 8-iter aurora_k1 - both are WORSE. Compositional: 8-iter cracks depth-2 (s1 0.42)
  and depth-3 (s1 0.18); 10-iter FAILS depth-2 (chance ~0.02, both seeds) with high spec
  (0.5, collapse-in-disguise); k2 cracks depth-2 (s1 0.39) but not depth-3 (0.03).
  AUC seed-spread ~3x tighter than final (8-iter 0.023 vs 0.084) - noise-robust ranking
  = same. VERDICT: 8-iter aurora_k1 (= current default) wins on quality AND compute at this
  scale. Coeff/aurora axis: no gain to be had by spending more here. Plots ->
  .autoresearch/plots/emergence/ (v8).
- [olm v9 DONE - coeff-family / cheaper-iters, vs 8-iter KJ floor (0.535/0.451)]
  * 6-iter (ns_4, KJx4+2pin, 18 GEMMs): tracks but WORSE both seeds - s0 0.542 / s1 0.484
    (vs 0.535/0.451). Dropping 2 iters is NOT free at this scale; 8-iter stays the floor.
  * PE-8 (Polar-Express minimax 8-tuple, different coeff FAMILY): REJECTED. s0 NaN'd @<250
    steps (eff collapsed 2.0/8), s1 emerged LATE and WORST (frac 0.580, depth-2 0.138) with
    the HIGHEST spec (collapse-in-disguise). The PE minimax schedule assumes a singular-value
    range our tiny d=128 fp16 updates fall outside (overshoot->NaN or underdrive->no learning).
  VERDICT: coeff/iter axis CLOSED - KJ 8-iter beats 6/10-iter, k2, and PE-8. Dashboard
  re-pointed to v9 (8it vs 6it vs pe8). Plots -> .autoresearch/plots/emergence/ (v9).
- [olm v10 DONE - WEIGHT-DECAY sweep, THE dominant knob, first time swept on olm] Fixed
  8-iter KJ aurora_k1, vary muon wd, 2 seeds, rank on AUC. Final frac (mean) / depth-2:
    wd 0.01: 0.556/0.466 (0.511) | 0.03: 0.554/0.566* (0.560) | 0.05: 0.557/0.548 (0.553) |
    wd 0.1 (anchor): 0.535/0.451 (0.493) | wd 0.2: 0.427/0.463 (0.445) = NEW CHAMP.
  HIGHER wd IS BETTER: monotone improvement 0.05->0.1->0.2; wd=0.2 s0 hits frac 0.427 with
  depth-2 0.541 / depth-3 0.272 / depth-4 0.130 = deepest composition seen (vs 0.1's d2
  0.225/0.421). Higher compression pressure -> stronger sparse-signal emergence. This
  VALIDATES the grok wd finding on olm AND the direction transfers (higher=better).
  * PREDICTION (a) "wd optimum flips SMALL online" = REFUTED. Optimum drops from grok's 2.0
    but stays WELL ABOVE conventional 0.1 (>=0.2). We have been running everything at the
    SUBOPTIMAL wd=0.1 - the whole coeff/scale-mode axis was benched under-decayed.
  * Ceiling: wd=2.0 was DEAD (v2 regime check) -> optimum in (0.2, ~1.0). NEXT: probe up
    0.3/0.5/0.7 to bracket the peak.
  * (*wd=0.03 s1 = outlier bad-timing seed, depth-2 collapsed to 0.08 while 0.01/0.05 s1 fine
    - ignore; AUC smooths it.)
  * [MECHANISM, user-spotted] wd0.2 s1 has WORSE depth-1 (0.637 vs wd0.1 s1's 0.948) but
    BETTER depth-2/3/4 (0.499/0.297/0.109 vs 0.421/0.177/0.068) at ~same frac (0.463 vs
    0.451). This is the grokking MEMORIZE->GENERALIZE transition, wd = the driving pressure:
    depth-1 is a finite lookup table (97x97/op) that nails d1 but does NOT compose; the
    compact Fourier/circular-arithmetic circuit (Nanda et al.) is slightly less pinpoint on
    d1 but COMPOSES -> d2/3/4 unlock. Higher wd makes the table expensive to store and pushes
    the model off lookup onto the composable algorithm. The depth-1 DROP is the signature of
    abandoning the table for the circuit = generalization, not overfit. Frac hides it; only
    per-depth reveals it. SEED NUANCE: wd0.2 s0 got BOTH (d1 0.946 + d2 0.541 = clean
    transition); s1 is caught MID-reallocation (d1 plateaued ~0.64, d2 still climbing at
    step 6000 log: 0.096->0.264->0.473->0.499 over last 1500) - transient of a hard/late
    transition, not inherent to high wd; longer budget likely recovers d1. => OLM reproduces
    the compression->composition story that makes wd dominant; watch d1-vs-d2 tradeoff in the
    v12 upward sweep (does d1 keep eroding at wd 0.3/0.5, marking a composition-vs-precision
    optimum?).
- [olm v11 DONE - SCALE-MODE axis CLOSED, all at wd 0.1 / 8-iter KJ] final frac (mean) /
  depth-2 (mean):
    aurora_k1 0.493/0.323 = WINS | aurora_k2 0.527/0.271 | normuon 0.555/0.169 |
    polar (base muon) 0.561/0.137 = WORST.
  aurora_k1 wins on BOTH frac and composition. normuon LOSES (confirms fixed-bias v7:
  broken-bias normuon "win" was an artifact). polar/base-muon = never tested on olm before,
  now confirmed WEAKEST - vindicates the "base muon less probable" prior. util (eff ~7.5)
  and spec FLAT across all four - the entire difference is composition quality.
- [olm v12+v13 DONE - DEVICE-MATCHED wd curve on RTX 6000, peak CONFIRMED at wd 0.2]
  Re-ran on RTX 6000 (v13: 0.1/0.2; v12: 0.3/0.5/0.7/1.0). TWO findings:
  * DEVICE SHIFT is real: RTX 6000 frac runs ~+0.03-0.06 vs T4 (wd0.1 0.526 vs 0.493;
    wd0.2 0.507 vs 0.445). fp16 NS / cublas differ across hardware -> ALWAYS device-match a
    sweep before comparing points. (Why the re-run mattered.)
  * PEAK SURVIVES at wd 0.2 on-device: mean frac 0.2=0.507 < 0.1=0.526 < 0.5=0.544 <
    1.0=0.568 < 0.3=0.573 < 0.7=0.627. wd 0.2 wins; 0.1 close; then downslope 0.3-1.0 toward
    the dead 2.0, with RISING SEED FRAGILITY (wd0.7 s1 NEVER emerged: frac 0.687, d1 0.277).
    wd0.2 s1 = standout (d2 0.492, d3 0.262, d4 0.073). Ordering matches T4 -> device shift
    raises the floor but preserves the wd ranking.
  * SCHEDULE INSIGHT (from the WSD overlay): the biggest depth-2 emergence jumps fire during
    the DECAY phase (steps 4800-6000, LR 1.0->0.1) - LR decay triggers/accelerates the final
    composition. dashboard_wd.png rebuilt device-matched (T4 points dropped). Low side
    (0.007-0.05, v14) pending -> completes the curve left of the peak.
  MECHANISM (matches theory): all four saturate depth-1 (~0.946); they split on depth-2.
  polar (scalar scale, rows NOT uniform) = worst; normuon (uniform rows, breaks orthogonality)
  = mid; aurora_k1 (uniform rows AND re-orthogonalized) = best. BOTH uniformity and
  orthogonality matter for the compositional circuit; only aurora has both. Scale-mode axis
  EXHAUSTED: aurora_k1 is the frontier, cheaper modes lose. Dashboard: dashboard_scale.png.
  CAVEAT: comparison at wd 0.1; re-confirm aurora_k1 dominance at the v10 optimum (wd 0.2)
  only if a reason appears - low prior it flips.
  * [SECONDARY, user-spotted] polar also has the LOWEST expert util (eff mean 7.40 vs
    normuon 7.6 / aurora 7.47), concentrated in the LAST MoE layer (s0 7.1) - small
    (~0.2/8, near noise, mostly s1) but SAME root cause as its depth-2 loss: scale mode acts
    on EXPERT weights (router is AdamW-side); polar's non-uniform row norms let experts drift
    to uneven output magnitudes, the selection-only bias balancer can't cancel a persistent
    magnitude asymmetry -> load concentrates. aurora/normuon uniform per-row RMS keeps
    experts balanced. One coherent weakness (row non-uniformity), not two - secondary
    confirmation, not worth chasing.
  3 configs x 3 seeds (beat 0.084 seed spread, compare MEANS): aurora_k1 8-iter (ns_kj=6,
  current default) vs aurora_k1 10-iter (ns_kj=8, dsv4_10) vs aurora_k2 8-iter (k2).
  Q: does more NS fidelity (10it) or aurora k2 beat cheap ns8 on the validated proxy, or is
  the iter/coeff axis DEAD here too (as on grok) -> ns8 = cheapest-tied, perf-per-flop win.
- NEXT: real LM for any olm survivor. Also still open: Dion low-rank
  (compute-side), param/compute-matched df0/df1 (dense-compute MoE = 8x FLOPs of dense
  layer, so as-is NOT compute-matched - decide sparse-compute or equal-FLOP first).
- PREDICTIONS ON RECORD: (a) wd optimum flips small in the online regime; (b) Muon>AdamW
  gap shrinks vs grok's 2.4x but stays positive (matches 137M LM screen ~2x); (c) deep
  depths (3-4) learn slowest = the discriminating tail; (d) frac plateaus per-depth like
  an LM scaling curve.
- Old grok harness retained for mechanism-vs-regime contrasts (a mechanism that wins
  ONLINE but not on grok = LM specialist, e.g. normuon pattern).
- MNIST-1D: fallback if this task cannot discriminate optimizers (user); arch-bias
  objection stands, revisit only on olm failure.
- [DOWNRANKED] SAM-style sharpness aware: 2x grad cost fails perf-per-flop by construction;
  only if a 1-extra-forward variant appears.

### Swarm / evolutionary imports
- [REJECTED 2026-07-04] Expert weight repulsion (PSO anti-averaging): W_e += beta*(W_e - mean_E W)
  after step. T4 wave 1 (ablate_muon, 3000 steps): beta=1e-3 DELAYS grok 1800->2800 with ZERO MI
  gain (1.00 bits, identical to baseline); beta=1e-2 destroys training outright (acc 0.01, loss
  diverges - exponential blowup, (1+beta)^3000 ~ 1e13 on any weight component off the mean).
  Dose-response harm, no benefit at any dose. Champ lesson holds again: unstructured state
  perturbation taxes loss. Anti-averaging pressure on WEIGHTS is the wrong lever; if anything
  survives in this family it must act on UPDATES (see decorrelation #3), not state.
- [TODO] Fitness sharing / niching lr (lr_e ~ inverse recent load, load inferred FREE from
  momentum Frobenius norm - no router plumbing). DOWNGRADED: mostly redundant with BiBo's bias
  balancer in steady state (equal traffic -> equal momentum norms). Revisit only under skew if
  repulsion shows signal.
- [TODO] Tournament/island selection across seeds: train E experts as independent lineages,
  periodically clone-and-perturb the best into the worst (FunSearch-style). Heavy; only if
  repulsion works.

### Game theory imports
- [NOTED] BiBo bias balancer IS congestion pricing (selection-only tax, sign updates). Already
  shipped; do not duplicate at optimizer level.
- [NULL 2026-07-04] Correlated-equilibrium / decorrelated updates (g_e -= gamma*mean_E g,
  grad-space screen): wave 3, gamma {0.5, 1.0} x2 seeds - grok 2000-2200 vs baseline
  1800-2200, acc/MI identical, even FULL removal of the common component (gamma=1) changes
  nothing. Reading: top-k routing already decorrelates expert grads (non-selected experts get
  ~zero grad per token), so the shared component is tiny - the "all experts learn the same
  thing" failure mode does not exist under a working router. Post-polar E x E gram variant
  not worth building given the grad-space null. CLOSED at this scale.
- [TODO, heavy] Shapley-style credit: scale expert lr by marginal-contribution estimate
  (leave-one-out routing on a probe batch). O(E) extra forwards; only if MI metric shows
  specialization stalls and we need credit assignment to explain why.

### Information theory imports
- [CLOSED-NULL 2026-07-04] Per-module weight decay, expert-vs-attn axis: ewd 0.5 REJECTED
  (slows grok, lowers MI, wave 2); ewd {3.0, 4.0} NULL (grok/acc/MI == baseline wd 2.0,
  wave 3; ewd4 slight end-of-run acc wobble = top of plateau). Expert stacks sit on the same
  wd 2-4 plateau as everything else - no per-module ratio to extract on THIS task. Keep
  uniform hidden wd. The emb/norm/router axes were not swept (AdamW-side); only revisit if
  an LM-side signal appears.
- [DEMOTED 2026-07-04, same day] LM wd sweep: user call, accepted - wd is a HYPERPARAMETER,
  not a method; in LM it must stay small or it eats signal. Sweeping it is not the round's
  deliverable. Keep ONE thin control arm (wd 0.3) to bound the axis; the wd findings stay as
  DIAGNOSIS (they identify compression pressure as the active ingredient), not as the knob.
- [REFRAMED 2026-07-04] LM phase = ALGORITHM arms at fixed conventional wd 0.1, ranked:
  (1) cautious / sign-aligned decay (2510.12402) - decay only where it does not fight the
      update; structural fix for "compression eats signal", not a compromise scalar.
  (2) sigma-cap: cap/soft-threshold only runaway TOP singular values of W post-step -
      targeted compression on the directions wd exists to police, zero pressure on the
      rest. Muon-native (reuse NS machinery / power iter), unexplored as wd replacement.
  (3) normuon at 121M - best prior LM-screen arm (-0.026 @1200, growing) despite being
      grok-harmful: benefit lives exactly in the signal-rich LM regime. Sign-flip now
      EXPECTED under the regime framing, not anomalous.
  (4) light low-rank EMA preconditioner (original round-1 ask, never run): rank-k EMA of
      grad subspace preconditioning Muon input - curvature memory. Only if budget remains.
- [TODO] Specialization annealing: per-expert lr ~ routing entropy (diffuse expert -> explore
  with high lr; specialized expert -> consolidate with low lr / higher wd). Needs router stats
  in the optimizer (harness-level plumbing) - after repulsion/decorrelation verdicts.
- [TODO] MI-guided bias: replace sign(mean-load) with sign nudges that MAXIMIZE MI(expert, op)
  estimate instead of load uniformity - changes the balancer's objective from fairness to
  specialization. Router-side (not optimizer), flag for BiBo discussion; conflicts with strict
  fairness at serving time.

### Muon-native (our own) ideas
- [TODO #3] Joint cross-expert orthogonalization: experts already stacked (E,r,c) in FusedMuon;
  whiten across the E axis before/after the polar so per-expert update DIRECTIONS are mutually
  decorrelated (E x E gram + Cholesky/Newton inverse-sqrt, E<=64 so negligible cost). The
  optimizer enforces "different experts learn different subspaces" with zero router involvement.
  Nobody else's optimizer has experts pre-batched like this - genuinely ours.
- [NOTED, standing observation] Muon is a partial load-equalizer already: orthogonalization pins
  every expert's update spectrum to ~1 regardless of tokens routed -> rare experts take
  full-size steps (AdamW does this per-coordinate, Muon spectrally). Muon x MoE routing dynamics
  = under-studied interaction; worth a writeup after the round.
- [TODO] Per-module NS budget: attn/dense get ns8; expert stacks (tiny 128x512 slices at test
  scale, well-conditioned) may need fewer iters - shape-aware schedule from the (dead) coeff
  axis, revived only as a wall-clock lever at LM scale.

### Physics / ecology (parked)
- [PARKED] Per-module Langevin temperature (replica-symmetry-breaking flavor: noise annealed
  differently per module). Champ lesson says injected noise costs loss; only revisit with
  strong motivation.
- [PARKED] Competitive exclusion pressure = fitness sharing, already covered above.

ARCH DECISION (user, 2026-07-04): BiBo mlp_only_layers -> FIRST TWO layers dense, LAST layer MoE
(was [0, 9] = dense first+last). Grounds: toy MI shows specialization lives in the last layer
(1.0 bit vs 0.0 in early layers) so dense-last forfeits the most specialization-hungry slot;
dense-early matches MiMo/DeepSeek practice (shared low-level features, routing stability).
Keep as we scale. Budget note: grok-MoE runs use --steps 3000 (grok completes ~2400).

Wave order: baseline (default vs adamw, skew) -> repulsion sweep (#1) -> per-module wd (#2) ->
update decorrelation / joint orthogonalization (#3). Promotion gate: MI + grok_step must both
beat baseline noise (3 seeds), and any winner must hold at uniform op mix too (no regression).

## Decision log

- 2026-07-04: round opened. Wave 1 = ns6/ns8/normuon queued behind baseline twins.
- 2026-07-04: mmx6 dropped pre-run (greedy solver fixed point); sinkhorn closed (L0); warm-start
  closed (math). Next candidates: joint 6-step solve (jns6), specnorm, gram (wave 2).
- 2026-07-04 (close): ROUND CLOSED, full story in perf_round_report.md. Status updates:
  ns8 ACCEPTED-pending-121M-A/B (LM tie + grok tie + 1.9% tps). ns6 grok-clean at every scale
  but -3 sigma on 600-step LM; aggressive option. jns6 TESTED: == everything at matched wd
  (joint solver ~ KJx4+PINx2 + last-step upscale; KJ per-step optimal even jointly).
  normuon SIGN FLIP: LM screen winner (-0.026@1200, growing) but grok-harmful - LM specialist
  only, finale PENDING. gram REJECTED at 137M shapes (no tps win on 512x512). k2 == ns6 on
  grokking (fidelity beyond 6 it buys nothing). NEW DOMINANT FINDING: muon hidden-wd
  (0.1 -> 2-4 optimal on grok, 7x faster generalization; wd-matched muon 2x adamw) - the wd
  axis is the top untested LM knob. Degree-7 polys, specnorm, per-group adaptive iters,
  low-rank deflation: UNTESTED (moot for grokking given the tie; revisit only if an LM eval
  ever shows coeff sensitivity). molab sandbox died mid-finale; 121M A/B is the open item.
