# Perf-per-flop round — final report (2026-07-04, closed at watchdog 09:23)

Goal: best downstream performance per unit compute from the Muon optimizer config.
Two win modes: better perf at same compute, or same perf at lower compute.

## Phase 1 — LM screen (closed early by user directive)

Frozen eval: BiBo 137M square-heavy v2 config, 600-step screen (~4.2 min/run, RTX PRO 6000),
twin noise floor 0.0021. Results:

| arm | loss@600 | tps | verdict |
|---|---|---|---|
| base1 / base2 | 5.0617 / 5.0638 | 155.1k / 155.7k | baseline + noise floor |
| ns6 (KJx4+pinx2) | 5.0684 | 162.0k (+4.4%) | -3 sigma loss, big tps win |
| ns8 (KJx6+pinx2) | 5.0622 | 158.1k (+1.9%) | LOSS TIE, free compression |
| normuon | 5.0453 | 156.4k | -8 sigma WIN (confirm at 1200: -0.026, growing) |
| gram backend | 5.0693 | 155.7k | no tps win at 512x512 shapes, REJECTED here |

## Phase 2 — synthetic screen (multi-op modular arithmetic, p=97, grokking regime)

Harness: .autoresearch/train_grok.py (frozen split seed, exhaustive held-out, curve output).
Calibration: frac 0.45 transition point; mid-transition acc noisy (twin 3.7 pts) -> rank by
grok_step (first eval >= 90%) + acc@budget curve; 3 seeds.

Grid-1 (arm x 3 seeds, muon wd 0.1): all NS variants tie inside seed noise; adamw (wd 1.0)
AHEAD of muon (wd 0.1) -> wd confound discovered. wd probe: muon wd 0.1/0.3/1.0 ->
grok ~5500/3800/1600.

Grid-2 (all arms at wd 1.0): default = ns6 = ns8 = jns6 = k2 EXACTLY per-seed (1400/1600/1600).
normuon consistently worse (1600-2000, ragged transition, 2/3 seeds sub-75% at step 1600).
wd 2.0 -> grok@1000.

Grid-3 (scaling: {default, ns6} x d{128,256,512} x frac{0.35,0.55} x 2 seeds, wd 2.0):
ns6 == default at ALL 12 cells (per-seed grok step identical to within one eval interval).
wd 3.0/4.0 probes: grok@800 -> wd optimum plateaus at ~2-4. Bigger model and more data both
grok FASTER (d512/f55: 600 steps).

## Phase 3 — LM finale (121M tokens, default/ns8/normuon): LOST

molab sandbox terminated mid-run (HTTP 404 at 09:23); results unrecoverable. PENDING on next GPU.

## Durable conclusions

1. COEFFICIENT/FIDELITY AXIS IS DEAD at matched weight decay, at every scale tested:
   ns6 (18 GEMMs) == dsv4_10 (30) == k2 (60) for generalization dynamics. NS fidelity beyond
   ~6 iterations buys nothing the task can see. Combined with the LM screen: ns8 is the SAFE
   compression (tie on BOTH evals, +1.9% tps); ns6 is the aggressive one (grok-clean,
   -3 sigma on 600-step LM loss but +4.4% tps -> likely per-wall-clock winner; unconfirmed).
2. OPTIMIZER FAMILY MATTERS, COEFFS DON'T: wd-matched Muon groks ~2x faster than AdamW
   (1600 vs 2600-3800). The Muon advantage is the orthogonalization itself, not its precision.
3. WEIGHT DECAY DOMINATES generalization speed: muon hidden-wd 0.1 -> 3.0 = grok 5500 -> 800
   (7x), dwarfing every coefficient/scale-mode delta. Our LM default (wd 0.1 on hidden) is
   probably UNDER-REGULARIZED for generalization-bound regimes; the wd axis transfers to LM
   as the highest-priority untested knob.
4. NORMUON IS TASK-DEPENDENT (sign flip): best LM screen arm (-0.016@600, -0.026@1200,
   growing) but consistently HARMFUL for grokking. Do not promote to a global default;
   it is an LM-specialist candidate pending the 121M finale.
5. jns6 (joint-solved 6-step schedule, band [0.96,1.12] from l0=2e-3): mathematically real
   improvement over greedy-6, empirically indistinguishable from everything else at matched wd.
   The joint solver converged to ~KJ x4 + PIN x2 + final upscale, confirming KJ is per-step
   optimal even jointly, and that the last-step gain is the only free knob at 6 steps.

## Rejected this round (with evidence)
- sinkhorn-as-NS-preconditioner (3rd refutation: equalizes row/col norms, not singular values)
- polar warm-start from cached Q (math: NS depends only on singular values)
- greedy 6-step minimax (fixed-point stall) and cap ballooning (prior round)
- gram backend at 137M-class shapes (no tps win on batched 512x512)

## Recommended actions
- Adopt ns8 as the FusedMuon default schedule after one clean 121M A/B (default vs ns8 vs
  normuon) on the next GPU session - the runs are one command each via bench/exp_kappa.py
  (arms already committed to BiBo main, config exp_final.yaml recipe in run_final.sh pattern).
- Sweep muon hidden-wd upward (0.1 -> 0.3/1.0) in the next LM round; the grok evidence says
  this is the highest-leverage untested LM knob.
- VM budget used: 11 short LM runs + 2 partial confirms + ~74 grok jobs + 3 solver jobs.
