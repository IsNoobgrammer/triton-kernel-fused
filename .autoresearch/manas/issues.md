# Manas issues ledger

Goal context: make Manas (rolling-probe gradient-alignment on top of aurora-K1 Muon,
`kernels/sm75/manas.py`) a demonstrably superior optimizer to base Muon and AdamW+Nexus,
using orthogonalization + common-minima probing. Reference: Nexus (arXiv:2604.09258).

Severity scale:
- P0 — invalidates any conclusion until fixed; nothing can be claimed while open.
- P1 — blocks the core scientific claim (common-minima benefit exists and composes with Muon).
- P2 — materially distorts results or stability; must fix before scaling past toys.
- P3 — weakens the claim or the engineering; fix during the round.
- P4 — footgun / polish; fix opportunistically.
- P5 — cosmetic / documentation.

Status: OPEN | USER (user is designing the fix) | FIXED(run-N) | REFUTED(run-N) | WONTFIX(reason).
All issues start OPEN; statuses are updated only after a logged run in results.jsonl.

---

## A. Theory / mechanism

### A1. Moving-weights confound (the trajectory-motion problem) — P0 — USER (REFRAMED by run-11)
Measurements delivered: motion/||d|| = 22x at lag 1, 160x at lag 8 (rho 0.94, gamma 5e-3) —
the quasi-static assumption is numerically dead. BUT the effect survives it (champion
held-out-replicated), and the clean same-point alignment force (rho=0) is useless — so the
moving-weights "problem" may not need fixing at all: the operative mechanism appears to be
smoothed-direction lookahead, which WANTS the memory despite staleness. The exp_ema
reference-point fix slowed training 2.3x (lagged outer gradient) and was rejected.
User's mid-training gating intuition remains untested (C11 LR-schedule arm is the vehicle).
Nexus's theorem pairs K microbatch gradients all measured at the SAME theta_t (inner clone,
reset every accumulation window). Manas pairs gradients across ~1/(1-rho) ~ 7 OUTER Muon
steps of a moving trajectory: d_t sums normalized grads from theta_{t-1}, theta_{t-2}, ...
while theta itself moves at update-RMS 0.2 per step. The docstring's symmetric-pair argument
("(t-k, t) picks up its half at t+k") needs H and g quasi-static over the rho-window; Muon's
motion over 7 steps is not small next to ||d|| <= gamma/(1-rho) ~ 1e-2 global. What Manas
optimizes is alignment of the current batch with gradients of RECENT PAST ITERATES — a
different (extragradient-flavored) objective from the paper's same-point cross-batch
alignment. User note: hypothesis that mid-training (CE ~4-5, smaller relative weight motion)
is where the quasi-static assumption starts holding — implies a warmup gate or a
weight-motion-normalized gamma. User is designing the fix; the loop provides measurement
support (A7 mechanism check, plus a ||theta_t - theta_{t-k}|| vs ||d|| ratio tracker over
training) but does not redesign the probe reference point unilaterally.

### A2. No rho=0 ablation — cannot attribute gains to "common minima" — P1 — FIXED(run-7: attribution CONFIRMED for memory)
Resolved across waves 0-3. At inactive gamma (1.5e-3) everything is flat. At active gamma
(1e-2): rho=0.94 gives frontier +0.0032 (held-out replicated, 4 sigma) while rho=0 at the
SAME gamma gives -0.0008 +/- 0.0005 (nothing). Pure extragradient contributes zero; the
multi-batch memory IS the mechanism. The common-minima story survives; Manas is NOT
"extragradient wearing the Nexus name". Also weakens A1's staleness worry at toy scale —
long memory (0.94) beats short (0.85 was in a dead zone at low gamma). Dose curve still
rising at gamma 2e-2 (+0.0045 opt, held-out pending run-8).

### A3. Objective mixture is set by the data loader, not by the optimizer — P2 — OPEN
With rho=0.85 the probe holds lags 1..~7 with weights rho^(k-1). Which lags are
CROSS-source vs SAME-source depends entirely on the batch schedule. mnist1d cycles sources
deterministically with period 3, so lags 3 and 6 (weights 0.72, 0.44) are same-source
self-alignment (~40% of the force) — within-source flatness, not the cross-source
common-minima force being tested. The optimizer's effective objective changes when the
loader schedule changes. Fix: shuffle source order per step in the eval; longer-term,
consider source-aware probe accounting or per-source d buffers as a candidate.

### A4. Probe-to-gradient scale ratio grows over training — P2 — FIXED(run-20: dose curve mapped; normalized scale WORKS, optimum gamma 0.08-0.12 at rho 0.98, curve turns after 1.2e-1; no schedule needed at toy scale — re-sweep gamma per model scale)
Increments are normalized (gamma/||g||)*g, so ||d|| stays ~gamma/(1-rho) forever, while
true gradient norms shrink as loss falls. Late in training the probe displacement becomes
arbitrarily large in gradient-relative terms — the Taylor expansion g(theta+d) ~ g + Hd
that carries the whole mechanism degrades exactly when the user wants Manas strongest
(mid/late training). Paper ties gamma to the base LR schedule; Manas has no schedule.
Candidates: gamma schedule tied to outer LR; unnormalized increments; trust-ratio cap
(||d|| <= c*||theta_update_ema||). Interacts with A1 (both are "probe vs trajectory scale").

### A5. Double temporal filtering through Muon momentum — unanalyzed — P2 — OPEN
The probe-point gradient feeds Muon's Nesterov momentum (beta 0.95, ~20-step memory). The
alignment perturbation Hd is therefore filtered/amplified through a second EMA on top of
the rho-window. Effective alignment force ~ gamma * sum over both windows — could be ~20x
the naive estimate, and stale (momentum replays old alignment forces at new weights).
No experiment separates "probe signal" from "probe signal x momentum amplification".
Candidate: probe-grad vs clean-grad split (feed momentum the clean grad, add the Hd term
outside momentum) as an ablation arm.

### A6. NS polar can noise-amplify the probe perturbation in degenerate subspaces — P3 — OPEN
The composition argument (polar only rotates) is why Manas composes with Muon, but polar
rotation is ill-conditioned where singular values are nearly degenerate: a small additive
perturbation to the momentum matrix produces an O(perturbation/gap) rotation of the
orthogonal factor. The probe signal is deliberately small; in near-degenerate subspaces NS
amplifies it (signal or noise alike). Measure: rotation angle of NS output with probe on
vs off, correlated with singular-gap structure. This is also (per the paper's own
hypothesis) the suspected cause of THEIR Muon incompatibility — we should verify we
actually escaped it rather than assert it.

### A7. No direct mechanistic validation of the alignment force — P1 — FIXED(run-11: measured, story INVERTED)
a7_mechanism.py, gamma 5e-3: cos(-dg, F_align) = +0.24 at rho=0 but only +0.02 at rho=0.94
— the paper-faithful pairwise-alignment force is injected 10x more cleanly WITHOUT memory,
yet rho=0 is downstream-inert and rho=0.94 wins (run-7). Conclusion: Manas's gain is NOT
Nexus's alignment mechanism. Working hypothesis (wave-5 test): trajectory-level lookahead
along a variance-reduced descent direction — the smoothing is the active ingredient, not
the pairwise force. Also: motion/||d|| = 22x (k=1) to 160x (k=8): the quasi-static
assumption is numerically dead, yet the effect survives — see A1 reframe.

### A8. Partial displacement: probe shifts only 2D/3D matrices — P3 — OPEN
Nexus displaces the whole inner model; Manas shifts matrices only (embeddings, norms,
biases, convs stay at theta). Gradients are therefore evaluated at a MIXED point — matrix
subspace displaced, everything else not. The alignment force is only assembled in the
matrix subspace and is measured against a network whose other parameters didn't move.
Probably second-order, but untested; a cheap arm that also shifts 1D params (held in a
side buffer with AdamW outer) would bound the effect.

### A9. "Muon lacks memory" framing is wrong in the docs and the positioning — P5 — FIXED(consolidation: manas.py docstring rewritten to the measured mechanism — trajectory-level lookahead along a variance-reduced momentum-free direction; Nexus-force theory explicitly marked refuted-by-measurement)

### A10. Descent-direction probe vs SAM sign — sensitivity unexplored — P3 — FIXED(run-10: sign=+1 arm INERT (-0.0002); effect is descent-direction-specific, not generic perturbation smoothing)
Manas probes DOWNHILL (d = -sum of normalized grads), evaluating g(theta - eps*ghat_avg);
SAM probes uphill. Nexus's theorem says descent cross-terms yield the +alignment force, and
the paper's inner loop also descends, so the sign is faithful — but the same-batch diagonal
terms (lag-k same-source pairs, see A3) then REDUCE the per-source gradient penalty
(anti-SAM within source) while cross terms align across sources. Whether the within-source
component helps, hurts, or washes is unknown. The +probe (SAM-sign) arm is a one-line
ablation that separates it.

## B. Implementation (kernels/sm75/manas.py)

### B1. Low-rank probe loses half the signal — P1 — REFUTED-DOWNSTREAM(run-21)
The shadow measurement (rank-8 captures ~45% energy, dcos 0.49) is real but does NOT cost
downstream performance at champion config: r8 arms score +0.0150-0.0189 vs full-d +0.0159
on the frontier metric (opt seeds). The smoothed direction survives lossy projection.
Memory pitch restored: rank-8 probe state is ~3% of a weights copy.

### B2. probe_refresh=200 is ~28x longer than the rho-window — P1 — REFUTED(run-21)
The diagnosis failed its own toggle test: at champion config (rho 0.98, window ~50),
refresh 200 scored +0.0189 vs refresh 16's +0.0150 — the stale basis does not hurt and may
mildly help (a stabler projection of the smoothed direction). Shipped default stands.

### B3. One-sided, single-snapshot basis — P2 — OPEN
Q is the randomized range of ONE gradient at refresh time. The probe content is a mixture
over the window; a basis accumulated over the window (frequent-directions sketch of the
last ~7 grads, or oversampled range of the momentum matrix) will capture strictly more.
Also only the column space is compressed — row-space structure is discarded by
construction. Try: two-sided (Q_l, Q_r) Tucker-style sketch at the same memory.

### B4. Train-loss spikes are more frequent/larger in Manas arms — P2 — REFUTED(waves 0-8: under the fixed protocol, parity and min-train stay clean up to gamma 1.6e-1 — 100x the originally suspect dose; the old 3-seed spike eyeballing was pre-protocol noise)
Observed (mnist1d): manas arms spike to 0.83/0.85/0.5 train loss vs base's worst ~0.18.
A probe that lands in a high-loss region injects a large bad gradient into BOTH the Muon
step and (normalized) into d. On an fp16 LM this is a divergence seed, not a spike.
Caveat: the observation is 3-seed eyeballing; first quantify (spike rate per arm over many
paired seeds, plus directly measure loss(theta+d) - loss(theta) per step), then guard only
if real (trust-ratio cap on ||d||, or skip probe-update when the probe loss blows out).

### B5. Full-mode memory = +1x weights fp32 — kills the footprint pitch — P3 — OPEN
manas_d is one fp32 model-shaped buffer per matrix: same as Adam's exp_avg, ON TOP of
Muon's fp16 momentum. "Muon memory footprint" claims require low-rank to actually work
(B1-B3) or an fp16/bf16 d (probably fine — d is bounded by construction; try it).

### B6. Low-rank shift materializes Q@C twice per step — P3 — OPEN
apply_probe and remove_probe each recompute Q@C per param (2 GEMMs/param/step overhead,
plus exact-inverse determinism dependence on cuBLAS run-to-run determinism). Same-stream
same-shape cuBLAS GEMMs are deterministic in practice, but a saved dense d per step
(materialize once, subtract the saved tensor) is both faster and unconditionally exact.
Costs transient memory only during the probe window.

### B7. Basis refresh consumes global CUDA RNG — P4 — OPEN
The refresh omega (manas.py:189) draws from the global generator, shifting all downstream
random ops (dropout, data order if on-device) vs a base run. Paired comparisons stay valid
only because toys use CPU generators for data. Fix: dedicated torch.Generator seeded from
param shape, like the init already does.

### B8. _probe_updates counter not in state_dict — refresh phase resets on resume — P4 — OPEN
Checkpoint-resume shifts the refresh schedule (and the "refresh fires on first update"
trick re-fires, discarding the loaded basis alignment). One-line fix via state_dict/
load_state_dict override.

### B9. Stateful-layer contamination footgun — P4 — OPEN
A probe forward through BatchNorm updates running stats at theta+d (contaminates eval
statistics). Current models have no BN, LM has none either, but the class is general-use.
Fix: document loudly, or assert no BN in _probe_params' modules (can't see modules from
params — document only).

### B10. Global-norm coupling across layers — P3 — OPEN
One layer with a transiently huge gradient suppresses the probe increment for ALL layers
(single global gn). Faithful to the paper (full-vector normalized SGD) so not a bug, but
per-layer normalization is a natural variant worth one arm — Muon itself is per-matrix,
and per-matrix probes would match its geometry.

## C. Experimental methodology

### C1. The "conclusive" result is all-NaN — no valid verdict exists — P0 — FIXED(run-0: protocol rebuilt as manas/eval_manas.py, frozen; valid baseline established)
manas_mnist1d_results.json has base_ood/manas_ood arrays that are 100% NaN, and its grid
(19 pts, 0.9-1.8) doesn't match the current script (15 pts, computed range): the saved run
predates the np.interp ascending-sort fix at manas_mnist1d.py:136 (descending xp made
interp return the NaN right-fill everywhere). The headline comparison has literally never
been computed on valid output. Fix: re-run under the corrected protocol (C2-C5 first so we
only pay for one re-run).

### C2. Sorting a spiky non-monotone train-loss series breaks matched-loss comparison — P1 — FIXED(run-0: cummin frontier in eval_manas.py)
ood_at_matched_train sorts by train loss; a LATE spike checkpoint (train 0.83 after the
model drifted OOD-worse) gets compared against the other arm's EARLY checkpoint at the
same train loss. Systematically biases against the spikier arm — currently Manas (B4).
Fix: compare along the running-minimum (cummin) frontier of train loss, or a monotone
isotonic fit; never raw-sorted.

### C3. OOD cross-entropy is calibration-dominated — wrong primary metric — P1 — FIXED(run-0: primary = OOD acc at matched frontier; best-OOD-acc + CE secondary)
OOD CE rises 2.0 -> 4+ across training while OOD accuracy stays flat ~0.42: the loss
trajectory measures growing confidence miscalibration, not transfer. The paper's claim is
downstream task performance. Primary metric should be OOD accuracy at matched train loss
(report CE secondary). Also evaluate at the best-OOD-acc checkpoint per arm (early-stopped
comparison), which is what a practitioner gets.

### C4. Statistical power is hopeless for the expected effect size — P1 — FIXED(run-0: paired-by-seed design, 5 opt + 3 held-out seeds; measured paired sem ~0.001 on the frontier metric — resolvable effects now ~0.002+)
Nexus's headline OOD-loss effect at 3B is 0.012; per-eval OOD noise on the toy is ~0.3 and
we ran 3 seeds. Orders of magnitude underpowered — "no detectable difference" was the only
possible outcome. Fixes that don't need a bigger model: (a) PAIRED design — same seed =
same init + same data order across arms, analyze per-seed deltas (arms already share
seeds; analysis must pair, not average separately); (b) 8 seeds (SEEDS8 convention);
(c) average OOD eval over the last K checkpoints and over multiple OOD batches;
(d) report the paired noise floor and refuse conclusions inside it.

### C5. Deterministic source cycling aliases with the rho-window — P2 — FIXED(run-0: source shuffled per step, seeded, in eval_manas.py)
Period-3 source rotation makes fixed lags same-source (see A3). Shuffle source per step
with a seeded generator so the lag-source structure is random and the measured effect is
schedule-independent.

### C6. Single-seed hyperparameter sweeps — P2 — OPEN
gamma and rank arms ran on seed 0 only; with per-seed OOD noise ~ the expected effect,
single-seed sweep rankings are noise. Every sweep arm needs the paired multi-seed
treatment (cheap at this scale: 1500 steps x 8 seeds x arm = minutes on the 3050).

### C7. No Nexus reference arm — "better than Nexus" is unfalsifiable as run — P1 — FIXED(run-3: NexusCloneTrainer + AccumAdamWTrainer in run_wave.py; clone beats accum-AdamW on best_ood_acc 5/5 seeds +0.0136 and trains faster)
Positive control PASSED — the testbed detects the paper's mechanism. Remaining: Muon+Nexus
(clone) incompatibility reproduction, and the final protocol-matched Manas-vs-Nexus
comparison once a Manas champion exists.

### C13. Frozen primary metric fails its own positive control — P1 — OPEN (needs USER sign-off)
Run-3: the known-good treatment (Nexus clone) scores NEGATIVE on OOD-acc-at-matched-frontier
(-0.0017 +/- 0.0006) while clearly working (best OOD acc +0.0136 on 5/5 seeds, min train
loss 0.087 lower). On this testbed the Nexus effect manifests as FASTER OPTIMIZATION plus a
HIGHER OOD PEAK, not as better OOD at matched train loss. The frozen primary cannot detect
the effect class we are hunting. Amending the eval is forbidden without user sign-off
(scope.md). Proposal: primary becomes delta_best_ood_acc (control-validated, 5/5 separation)
with delta_frontier and min-train kept as secondary/slices. Until signed off, waves report
both; no promotion happens on a metric the control fails.

### C8. No LR/wd retune for the champion claim — P2 — OPEN
Paired same-hparams comparison is right for detecting the mechanism, but the memory
(perf-per-flop round) says wd is the dominant knob and coeff wins evaporated at matched
wd. A "superior optimizer" claim needs the BASELINE tuned (lr x wd grid for base Muon and
base AdamW) before the final comparison. Do this once, at the end, for the champion only.

### C9. Toy may be structurally unable to show the effect — no second testbed — P2 — OPEN
MNIST-1D OOD with a 200k-param GLU net may just not have the "common minima" structure
that 130M+ LMs on heterogeneous mixtures have. The repo already has grokking
(train_grok.py) and MoE-pipeline testbeds; grokking directly measures the
same-train-loss/different-generalization gap, which is exactly the paper's claim shape.
Add ONE second testbed (grokking with 2 modular tasks as "sources") before believing any
negative result from mnist1d.

### C10. No cost accounting in comparisons — P3 — OPEN
Probe adds no extra fwd/bwd (its gradient IS the training gradient — the key win over
Nexus's clone, worth stating), but does add _shift traffic + d update + (low-rank) QR
every refresh. Report wall-clock/step and peak memory per arm alongside quality so
"superior" is cost-honest. Nexus's clone costs a full weights copy + K normalized-SGD
updates; quantify our advantage explicitly.

### C11. probe_gamma / probe scale untested against LR schedules — P3 — OPEN
Toys run constant LR. Real training has warmup/decay; the paper ties gamma to the LR
schedule. Interaction (A4) untested even on the toy — add a cosine-LR arm once the
protocol is fixed.

### C12. Eval never asserts probe-off at eval time — P4 — OPEN
Eval forwards run outside the probe context (correct), but nothing ASSERTS the offset is
off — a future refactor that evals inside the context silently reports theta+d metrics.
Cheap: assert not opt._probe_on at eval, or expose opt.probe_active property.

## D. Positioning / claims

### D1. Docstring oversells the theory — P5 — FIXED(consolidation: header now reports the A7 measurement and the ablation-pinned requirements — long memory, descent sign, momentum-free direction — instead of the refuted alignment-assembly claim; validated numbers and defaults gamma 0.08 / rho 0.98 recorded)

### D2. "Better than AdamW" scope is undefined — P3 — OPEN
Superior in what regime — same LR/wd budget? same memory? same wall-clock? Muon already
beats AdamW 2x on perf-per-flop (memory: perf-per-flop round); Manas inherits that. The
NEW claim must be Manas > Muon (paired) and Manas-on-Muon > Nexus-on-AdamW (protocol-
matched). Freeze this scope before the final comparison or the goalposts will drift.

---

## Priority queue (what the loop attacks, in order)

1. C1+C2+C3+C4+C5 as ONE fixed eval protocol (the frozen eval for this round) — everything
   else is unmeasurable until this exists. [P0/P1 cluster]
2. A2 (rho=0 arm) + A10 (+probe sign arm) — mechanism attribution, cheapest decisive tests. [P1]
3. A7 (mechanistic force-correlation measurement) + C7 (Nexus clone reference arms). [P1]
4. B2 (refresh cadence) then B1/B3 (low-rank quality) — only after full-d verdict exists. [P1/P2]
5. B4 (spike diagnosis + guard), A4 (gamma scaling), A3/C5 residue, A5 (momentum split). [P2]
6. C9 (grokking second testbed) if mnist1d verdict is negative or inside noise. [P2]
7. Long tail: B5-B10, C8, C10-C12, A6, A8, D1, D2. [P3-P5]

A1 stays USER-owned (weight-motion fix); the loop feeds it measurements (A7, and the
||theta motion|| / ||d|| ratio tracker) but does not redesign the probe reference point
without the user's method.
