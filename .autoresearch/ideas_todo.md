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
