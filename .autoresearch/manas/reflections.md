
## Wave 0 (runs 0-1)
Primary frontier metric is flat for every arm at gamma 1.5e-3 (all |delta| < 0.001 ~ sem).
Either the probe does nothing at this gamma, or the frontier metric averages over the OOD-acc
plateau and is insensitive. rho=0.94 improved BEST OOD acc on 5/5 paired seeds (+0.012,
~3.5 sigma) - long memory beat short memory (rho0/rho05 ~ +0.004, rho085 +0.0006). This is
the OPPOSITE of the staleness prediction (A1/A2). Do not metric-shop: best_acc stays a
slice/secondary; verify rho094 on held-out before any belief. Next: held-out rho094; extend
rho to 0.97; gamma dose at rho094 (primary may need bigger gamma to move).

## Wave 1 (run 2)
rho094's 5/5 best_acc signal did NOT survive held-out (+0.0013 vs +0.0122): winner's curse,
caught by the gate. Everything remains inside noise on the primary. The real gap: we have no
POSITIVE CONTROL - if AdamW+Nexus(clone) itself cannot move this testbed, no Manas variant
result here means anything. Wave 2 = Nexus clone (K=4, faithful Alg-3) vs accum-AdamW base,
plus gamma 1e-2 escalation to bracket the probe's active range. If Nexus-clone is also flat
-> testbed insensitive -> pivot to grokking two-source testbed (C9). Slice archive entry for
rho094 downgraded (held-out fail).

## Wave 2 (runs 3-4)
Positive control verdict: the testbed CAN detect the Nexus mechanism, but on best_ood_acc
(clone: 5/5 seeds, +0.0136) and training speed (-0.087 min-train), NOT on the frozen
frontier metric (clone scores NEGATIVE there, 2.8 sigma). Frontier-at-matched-loss fails
its own positive control -> flag to user for metric amendment; do not silently switch.
Manas gamma escalation: frontier delta rises monotonically with gamma, g1e-2 first 2-sigma
arm. Two live threads: (a) is g1e-2 real on held-out; (b) at active gamma, does rho matter
(A2 re-test, g10_rho0 arm)? If rho0 == rho094 at g1e-2, memory is irrelevant and the win is
extragradient; if rho094 > rho0, common-minima memory earns its keep.

## Wave 3 partial (run 5) - FIRST PROMOTION
g10_rho094 held-out: frontier +0.0032 +/- 0.0008, matching opt seeds exactly. Champion
promoted. C13 resolution via user free-hand: co-primary metrics (frontier for the Manas
axis, best_ood_acc for the Nexus-clone axis); a future superior-optimizer claim wants BOTH
positive. Remaining wave-3 arms (g20 dose, g10_rho0 attribution) still running. Next:
exp_manas sandbox (ref_mode EMA for A1, per-matrix norm B10, sign flip A10, motion trust
A4) tested against BOTH base and the new champion.

## Wave 3 complete (runs 6-7) - ATTRIBUTION LANDED
rho=0 at gamma 1e-2: NOTHING (-0.0008 +/- 0.0005). rho=0.94 same gamma: +0.0032 held-out
4 sigma. Memory is the mechanism, extragradient alone is inert -> A2 closed in favor of the
common-minima story. Gamma dose still rising at 2e-2 (+0.0045 opt). Open: where does the
dose curve turn (stability edge)? Does rho want to go higher still (0.98 arm in wave 4)?
Do any exp_manas mechanism variants (ema-ref, permat, sign+, trust) beat the plain champion?
A7 mechanism measurement chained after wave 4 on the GPU.

## Wave 4 (runs 9-11) - MECHANISM REFRAME
g20_rho094 promoted (held-out 2.2 sigma). Sign flip inert -> descent-specific. exp_ema
rejected (slows training 2.3x - lagged outer gradient; its frontier "win" is a range
artifact; lesson: ANY variant that changes training speed makes the frontier metric
treacherous - always check train_parity slice first). A7 inversion: clean alignment-force
injection (rho=0) is useless; noisy long-memory probe wins -> the gain is NOT the paper's
mechanism. Working hypothesis: variance-reduced trajectory lookahead. If exp_momdir
(probe along Muon momentum, zero extra memory) matches the champion, Manas collapses to a
2-line Muon patch with no probe state at all - the strongest possible engineering outcome.

## Wave 5 (runs 12-14)
g10_rho098 promoted (held-out +0.0077, both co-primaries positive). momdir DEAD at both
doses -> the mechanism needs the momentum-free normalized-grad EMA specifically, not just
any smoothed direction; window length (rho 0.98 ~ 50 steps) beats momentum beta window.
Purity story partially back. Meta-pattern across 5 waves: EVERY winning move so far has
been "more smoothing of the probe direction" (rho up) or "bigger probe" (gamma up), and
every mechanism-swap variant (ema-ref, sign, momdir) died -> the shipped probe rule is
architecturally right at toy scale; the knobs were just set an order of magnitude too
timid (defaults gamma 1e-3 rho 0.85 vs champion gamma 1e-2+ rho 0.98).

## Wave 8 (runs 19-20) - DOSE CURVE CLOSED
g80_rho098 promoted at held-out +0.0245 (largest, cleanest result of the round). Curve
turns after g120; optimum plateau gamma 0.08-0.12, rho 0.98. Both original default knobs
were order(s) of magnitude too small. No instability anywhere on the curve (parity/min-train
clean up to g160 - B4 spikes never materialized at high gamma under the fixed protocol).
Remaining: low-rank at champion config (wave 9, incl. refresh 16 vs 200 A/B for B2), then
consolidation: manas.py defaults + docstring reposition, final comparison table, commit.

## User hypothesis (C14): rho is the batch/noise-coupled knob, gamma the geometry knob
User theory: rho-window x batch ~ const (memory in TOKENS is the invariant) - long memory
for small/noisy batches (our toy), short memory for big-batch pretraining. Gamma = the
curvature-signal magnitude; couples to weight-space geometry (LR/weight norms), so its
dimensionless form (fraction of step motion - trust-ratio) should transfer, raw value
re-swept per scale. Diagnostic bs_rho.py launched: BS {32, 512} x rho {.90, .98, .995} at
gamma .08, paired within BS. Prediction: rho* decreases with BS. If confirmed, ship a
"tokens-of-memory" parameterization (user-facing knob = effective samples, rho derived).

## Post-round (run 23) - C14 CONFIRMED
User's rho-batch law holds with a tight invariant: effective window x batch ~= 5-6k samples
at every batch size tested (32: rho .995; 128: rho .98; 512: rho .90). rho documented as
memory-in-samples, not a free knob. Gamma remains the geometry-coupled dose. Implication
for BiBo/pretraining: derive rho from tokens-per-step (expect rho << 0.9 at LM batch
sizes), sweep only gamma.

## Wave 10 (runs 24-25) - user update-history idea, verdict
comp+1 (extend probe along rho-decayed realized updates): frontier-equal-or-hair-better
than champion (held-out +0.0266 vs +0.0255, overlapping), peak-acc neutral after its
opt-seed 5/5 died at the gate (SECOND best_acc winner's curse - that axis needs more
held-out seeds than 3 to resolve 0.01-class effects; treat any 5/5 opt best_acc as noise
until gated). Cost: +1x weights fp32 (u buffer) - so r8 champion keeps the title on
cost-adjusted quality. Mechanism note: back-off sign (user's original guess) neutral,
extend sign carries whatever effect exists - direction of realized travel adds lookahead
depth, not double-counting correction. Worth a low-rank-u revisit at LM scale.

## Wave 11 (run 27) - all-low-rank validated
Low-rank u through d's shared r8 basis fully replicates champion-class frontier (+0.0235
held-out). Both memories compress to ~1% each with nothing measurable lost - strongest
evidence yet for the smoothing/denoising story: the mechanism lives in a tiny subspace.
u still separates from nothing at toy scale; its real test is heterogeneous large-batch
(LM). Production candidate lineup frozen: (a) champion r8-no-u; (b) exp_lrdu r8-d+r8-u
comp+1 - identical cost class, LM decides.

## Wave 12 (run 28) - PSO lens, honest tie
User framing: Manas = PSO whose swarm is past steps (consensus over data shards, harvested
from the trajectory, zero parallel cost). Tested the one falsifiable import - agreement-
weighted (social-reinforcement) votes: TIE with equal-vote at champion and 1.4x gamma.
So PSO is the right NAME for the mechanism, not a lever that improves it here. Equal-vote
(plain recency-decayed average) is as good as self-sharpening consensus at toy scale.
Lens kept for the writeup; vote gating closed. Larger unexplored PSO door = a gbest-style
attractor (pull toward best-consensus point seen) - u-buffer is a crude half-step to it;
full version is a bigger design, deferred pending user + the LM run.

## Wave 13 (run 29) - Shampoo ablation (friend request)
Grafted Kronecker Shampoo, tuned lr, on the frozen protocol: matched-loss frontier -0.0133
held-out (loses to Muon), small peak-acc edge +0.009, trains FASTER (real curvature). Same
family signature as Adam/Nexus: better optimization, worse same-loss generalization.
Operationally it needed eigh ridge-escalation + momentum norm-grafting just to not diverge -
fragility Muon/Manas never touch. Final board (held-out frontier vs Muon): Manas r8 +0.0255
> Muon 0 > Shampoo -0.013 > Adam(tuned) -0.024 > Adam+Nexus -0.030. Manas is the only
positive. C17 closed.

## Task #1 (divergence hunt) - hetero_dose CONTROL verdict: NEGATIVE
Seed-floor control (Muon<->Muon same-data-diff-init vs Muon<->Manas same-init-data):
excess divergence ~0 at alpha 0-0.25 (-0.4%, +0.1%, within noise). The monotone raw
functional-divergence rise (14->23%) is DATA SENSITIVITY, not Manas mechanism - two Muons
diverge as much as Muon-vs-Manas. On heterogeneous MNIST-1D, Manas is functionally
indistinguishable from Muon beyond seed noise. Loss-balance signal was also noise (earlier).
So: the +0.0255 matched-loss OOD win is real (gated) but does NOT manifest as a visible
functional/trajectory difference here - the effect is too small to SEE on this task, only to
measure statistically. Weight-distance metric was uninformative (symmetry-dominated, ~110%
at all alpha). Remaining divergence candidates: grokking (train_grok.py) same-loss/diff-
generalization; large-batch u-buffer separation (running). If those also null, the honest
conclusion is Manas's difference from Muon is real-but-small at toy scale and only a
LM-scale run could make it visible.

## Task #1 - U-buffer under heterogeneity: NO SEPARATION
hetero_u.py (Manas+u comp=1 vs Manas no-u, rank-8, 5 paired seeds, alpha grid): delta acc
~0 at every alpha (+0.23,-0.12,+0.16,-0.09,+0.09%, all within noise), no dose-response.
The "heterogeneity is where u earns its keep" hypothesis is NOT confirmed at toy scale -
u remains a free passenger. Faint ungated hint: spread(+u) < spread(no-u) at 4/5 alphas
(weak balance aid), not chased. u's value, if any, needs LM scale / large heterogeneous
batches. Keep u OFF by default (shipped); revisit only at BiBo scale.

## Task #1 CLOSE - grokking verdict (8 seeds) + overall
Heterogeneous grok (add+mul, 8 seeds, 6000 steps): grok_step Manas-Muon = -143 +/-61 steps
(6/7 grokked-seeds earlier, ~2.3 sigma) - Manas groks slightly earlier UNDER heterogeneity.
Best-acc difference WASHED OUT at 8 seeds (+0.02 at 3 seeds -> ~0 at 8; was 2-seed luck).
Single-op grok (homogeneous): Manas == Muon (766 vs 800, tie). The homogeneous-vs-hetero
contrast is the cleanest mechanism demonstration: no difference without disagreement, small
edge with it.

TASK #1 VERDICT: Manas's difference from Muon is REAL but SMALL at toy scale, everywhere.
4 testbeds: functional-divergence=noise-floor; u-buffer=no-sep; single-op-grok=identical;
hetero-grok=~3% earlier (2 sigma), acc wash. Direction always consistent (Manas >= Muon,
earlier grok under heterogeneity) - matches the gated +0.0255 OOD win - but never DRAMATIC.
No visibly-different-at-toy-scale regime exists; effect needs LM scale to become a visible
gap. Honest close. Lesson reaffirmed: 3-seed excitement (earlier AND +2% acc) did not
survive 8-seed gating - always gate before believing.
