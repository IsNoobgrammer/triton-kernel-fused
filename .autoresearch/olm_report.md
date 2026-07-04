# Online LM-Emulator (olm) — full round report

Goal of the whole program: find the optimizer that gives the best downstream performance
per unit compute (better perf at same FLOPs, or same perf at fewer FLOPs). The grok testbed
was the wrong regime (memorize-then-generalize); this report covers the pivot to an online,
single-epoch emulator of language-model training and everything it decided.

## 1. What olm is, and why we built it

Grokking = memorize the train set, then generalize. LM pretraining is the opposite:
compute-bound, every token seen once, all progress is compression. An optimizer can look
great on grok and useless on LM (or vice-versa). So olm emulates the LM regime at toy cost:

- **Task**: left-fold chains of modular arithmetic mod 97 — `v0 op1 v1 op2 v2 ... = ?`,
  composition depth 1..6, Zipf mix (0.45/0.25/0.15/0.08/0.045/0.025). Division is a
  depth-1-only skill (deep chains use +,-,* — see calibration below).
- **Online / one epoch**: fresh samples every step, val excluded by key, sample space
  (~5.6e9 at depth 3 alone) >> stream (~4.6M samples at 6000x768). Memorization is
  impossible; all val progress is generalization/compression.
- **5% label noise** on stream AND val -> irreducible CE floor 0.4229 nats = frac 0.0924,
  deliberately equal to LM's residual ~0.09. The task races toward a floor it can never
  cross, exactly like text.
- **Model**: 4 layers, d=128, default arch = dense first layer + MoE layers 2-4 (BiBo-style
  router, stacked experts, 3.58M params).
- **Schedule**: LM-standard WSD (linear warmup 500 -> stable -> cosine decay to 0.1x over
  last 20%), IDENTICAL for AdamW and Muon.

## 2. The metric

- `frac = val_CE / ln(97)` = fraction of initial entropy remaining. 1.0 = untrained,
  0.092 = perfect (noise floor). LOWER = more compression = better. This is the nats
  analog of bits-per-byte.
- `gap = val_CE - floor` = distance above the entropy floor.
- Eval is **distribution-matched** (Zipf-weighted over depth, like LM in-distribution eval)
  — critical fix; uniform-over-depth pinned the metric on an unlearnable deep tail.
- Also tracked: per-depth accuracy (the learning hierarchy), MI(expert, depth) per layer,
  min expert load, and (for scap) the power-iteration smax.

## 3. Calibration journey (v1 -> v5)

| ver | change | outcome |
|---|---|---|
| v1 | depth 1-4, no noise, dense-first-2 | ran; too easy to reason about |
| v2 | depth 1-6, 5% noise, dense-first-1 | frac stuck 0.82 - too hard |
| v3 | div -> depth-1 only, shallower mix | depth-1 cracked (0.02->0.92) but... |
| v3+ | distribution-matched eval | ...frac was pinned by the deep tail; fixed, lands ~0.46-0.56 |
| v4/v5 | WSD + warmup default; ns8 default | calibrated; floor band tight |

Two calibration traps, both fixed: (a) division inside deep chains poisons them until div
is learned -> restrict div to depth-1; (b) uniform-over-depth eval over-weights depths 3-6
which never learn at this budget -> weight val by the train (Zipf) distribution.

Residual known limitation: only depth-1 fully learns and depth-2 is the live "emergence"
zone; depths 3-6 stay near chance at 3.58M params / 6000 steps. Discrimination therefore
happens on depth-1 transition speed + depth-2 emergence. Deeper composition needs a bigger
model / longer budget (not yet run).

## 4. Regime findings (the emulator captures LM, not grok)

- **wd flips**: grok wanted wd 2-4; online wd 2.0 is DEAD (frac 0.995, flat, no learning).
  Small wd (0.1) is correct online. Direct empirical confirmation of memorization-bound vs
  compute-bound.
- **AdamW vs Muon (fair: matched WSD, warmup, lr)**: Muon crushes AdamW. Distribution-
  matched acc Muon ~0.46/0.41 vs AdamW ~0.05/0.11 (4-9x); depth-1 ~0.88 vs ~0.15; Muon
  cracks depth-2 (0.10-0.16), AdamW does not (0.02-0.03). Strongest perf-per-flop evidence
  of the round, in the correct regime.
- **Warmup asymmetry**: AdamW without warmup COLLAPSES to chance (cold-start second
  moments in a one-pass stream); Muon without warmup LEARNS (just plateaus a bit lower,
  and learns faster early). Muon is warmup-robust; AdamW is warmup-dependent.
- **Emergence not grokking**: sharp capability phase transitions on FRESH data (e.g. a seed
  jumping depth-1 0.26->0.61 over 500 steps), with seed-variable timing. This is the LM
  "emergent ability" phenomenon, not memorize-then-generalize. Timing variance means fixed-
  budget comparisons need the noise floor + ideally transition-onset tracking.

## 5. Proxy validation (the milestone)

Ran the three optimizer variants for which we already own 137M-LM ground truth:

| variant | olm frac (final) | LM ground truth |
|---|---|---|
| normuon | 0.527 / 0.561 (2 seeds, no overlap w/ default) | WON (-0.026 @1200) |
| ns8 (6 KJ + 2 pin) | 0.560 | tied |
| k2 (aurora_k=2) | 0.568 | tied |
| default (dsv4_10) | 0.566 / 0.592 | - |

olm reproduces the LM ordering exactly: normuon wins, ns8/k2 tie. **Decisive**: normuon was
grok-HARMFUL (sign flip) — olm lands it on the LM side, i.e. olm DISAGREES WITH GROK exactly
where grok lied. Not mere correlation; it captures the compute-bound regime. Caveats: n=2
(normuon/default), n=1 (ns8/k2); a proxy preserves ORDERING not magnitude.

Consequence: olm is a validated, minutes-on-a-T4 screen for optimizer changes. The mechanism
backlog can be screened here before spending 120M-token LM runs.

## 6. Mechanism re-bench on the validated proxy (v5)

Default is now ns8 + aurora_k1 (tied + cheaper: 24 vs 30 GEMMs). ns8 floor: seed0 0.560,
seed1 0.556. Every mechanism tested at seed 0; must clear ~0.560 to count.

| mechanism | frac | verdict | why |
|---|---|---|---|
| grad_rep 0.5 | 0.562 | NULL | routing already decorrelates experts; nothing to add |
| scap 2.0 | 0.563 | NULL | top-sv clip; likely non-binding (smax ~1.5 at init) |
| xorth | 0.565 | NULL on loss / WIN on utilization | see note below |
| cautious 2.0 | 0.598 | MILD HARM | sign-masking halves effective decay -> slower escape; LM-good prediction REFUTED at this budget |
| repulse 1e-3 | 0.692 | HARM | blocks the depth-2 composition circuit (d2 stuck 0.02, MI~0, regresses) |
| grokfast 2.0 | 0.721 | HARM + UNSTABLE | online fresh data -> grad EMA averages over DIFFERENT samples -> amplifying the "slow component" amplifies staleness; acc degrades 0.12->0.07 |

Zero mechanism beats the floor ON LOSS. Only normuon (a scaling refinement) wins on loss.

**xorth utilization finding (v5, on the MoE-specialization axis the round originally targeted):**
Grading v5 on frac alone missed this. On expert utilization + per-layer specialization,
xorth (our cross-expert gradient whitening along the E axis) is the winner and acc-neutral:

| arm | frac | acc | minload | MI (3 sparse layers) |
|---|---|---|---|---|
| default | 0.560 | 0.47 | 0.031 | 0.00 / 0.24 / 0.44 |
| xorth | 0.565 | 0.46 | 0.039 | 0.32 / 0.23 / 0.48 |
| scap | 0.563 | 0.47 | 0.001 | 0.48 / 0.16 / 0.49 |

xorth specializes ALL three sparse layers (default's first is dead at MI 0.00) AND keeps
load healthy (no collapse), AND holds accuracy at the floor — the exact "acc-neutral
functional diversity" bar the MoE round set. scap also specializes but collapses load
(minload 0.001). NOT yet a loss win: the task saturates at floor with 2-3 experts, so the
extra utilization is free-but-unrewarded. Convert-to-win test = a CAPACITY-BOUND regime
(more experts / harder multi-task data / narrower experts) where all E are genuinely needed.

## 7. Conclusions and intuitions

1. **Muon's polar + weight decay is the frontier.** Across grok (13 mechanisms) and olm
   (6 more), the ONLY thing that beats plain Muon is normuon — a per-row second-moment
   rescale of the SCALING, not an added mechanism. Every "bolt something on" idea is null
   or harmful. Intuition: the polar already bundles direction filtering (via momentum),
   magnitude normalization (singular values -> 1), and with wd, compression pressure. The
   mechanisms either re-supply what the polar has, or fight the task structure.

2. **Top-k routing already decorrelates experts.** Every diversity mechanism (weight
   repulsion, grad decorrelation, grad repulsion, cross-expert orthogonalization,
   fitness-sharing) is null-to-harmful because non-selected experts get ~zero gradient per
   token — the "experts aren't diverse enough" premise is a non-problem under a working
   router. Aggressive versions (grad_rep, repulse) inject noise and block learning.

3. **Regime is everything, and olm proves it.** olm reversed grok's normuon verdict to match
   LM, and flipped grokfast from grok-null to olm-harmful. The gradient-EMA family
   (grokfast) is specifically anti-useful online: fresh data every step means the EMA
   averages over different samples, so its "slow direction" is stale.

4. **Compression pressure knob (wd) is regime-dependent.** grok optimum 2-4; online optimum
   small; wd 2.0 online = dead. cautious decay (weaken/mask the pressure) is slower in BOTH
   regimes at this budget — the online task still rewards adequate pressure, not less.

5. **Muon's compute-bound advantage over AdamW is real and large** (4-9x here, matched
   schedule) and it needs no warmup to avoid collapse. This is the core perf-per-flop thesis,
   now shown in the LM-faithful regime.

6. **Architecture (MoE placement)**: specialization (MI) spreads across layers online (vs
   grok's last-layer-only). df0 (all-MoE) vs df1 (dense-first) is inconclusive — confounded
   because dense-compute MoE is ~8x the FLOPs of a dense layer, so the pair is not
   compute-matched. Needs a compute-matched rerun before it can revise the dense-early call.

## 8. Shippable recommendation and open items

- **Recommended config**: Muon with **ns8** (6 KJ + 2 pin; 24 GEMMs, ~20% fewer than
  dsv4_10, quality-tied on olm AND LM) + **small wd** + WSD. This is a clean perf-per-flop
  win: same quality, less NS compute.
- **normuon**: validated win on olm AND the 137M LM screen. BUT the Aurora paper measured
  normuon WORSE than Muon at 1.1B (2.33 vs 2.31). Tension = likely scale- or
  convergence-dependent (our wins are 137M-and-below, mid-training). Do NOT ship to large
  BiBo without a to-convergence A/B at larger scale. Promote as a candidate, not a default.
- **Rejected (do not pursue)**: weight/grad repulsion, decorrelation, cross-expert
  orthogonalization, fitness-sharing, grokfast, lookahead, sigma-cap (as tested), cautious
  decay. All null-or-harmful in the correct regime.
- **Still open**: Dion-style low-rank orthogonalization (compute-side lever, untested);
  compute-matched df0/df1 arch study; deeper-composition olm (bigger model/budget) to get
  depths 3-6 into the learnable range and widen the discrimination window; scap with a
  binding (lower) cap + smax logging to confirm the non-binding hypothesis.
