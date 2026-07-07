# Manas: the complete inductive bias (post-round synthesis)

Everything below is grounded in the 25-run frozen-eval round (.autoresearch/manas/), the
A7 mechanism measurement, and the C14 scaling law. Beliefs are labeled MEASURED (survived
a held-out gate or a direct measurement), SUPPORTED (consistent evidence, not gated), or
INTUITION (untested).

## 1. What Manas is

FusedMuon (aurora-K1, NS-8) plus one move: evaluate the training gradient at theta + d
instead of theta, where

    d = -gamma * sum_k rho^(k-1) * g_{t-k} / ||g_{t-k}||_global

a momentum-free, per-batch-normalized, rho-decayed gradient memory, applied before the
forward/backward and removed before the step. Zero extra forward/backward passes — the
probe gradient IS the training gradient. Optional: low-rank d (Q@C, rank r), and a second
buffer u (rho-decayed sum of the actually-applied Muon updates) added to the probe offset.

## 2. Why it composes with Muon when Nexus does not (MEASURED-adjacent, load-bearing)

Nexus hands AdamW a displacement pseudo-gradient whose MAGNITUDE structure encodes the
signal; Muon's Newton-Schulz flattens all singular values to 1, destroying exactly that
structure — hence the paper's incompatibility. Manas moves the signal into the gradient's
DIRECTION by shifting the evaluation point; the polar can only rotate a direction, never
erase it. This argument motivated the design and nothing in 25 runs contradicted it.

## 3. The mechanism — what we thought vs what is true

THOUGHT (the paper's story): the probe assembles the pairwise cross-batch gradient-
alignment force, steering to a common minimum of recent batches.

MEASURED, and it kills that story twice over:
- The alignment force injects ~10x more cleanly at rho=0 (cos +0.24 with the true
  double-backward alignment gradient) than at the winning rho 0.98 (~+0.02). Yet rho=0 is
  downstream-INERT and rho 0.98 wins. The force is not the active ingredient.
- The trajectory moves 22x (lag 1) to 160x (lag 8) farther than ||d|| — the same-point
  assumption behind the theorem never holds during real training, and the effect does not
  care.

TRUE requirements, each pinned by an inert ablation:
- LONG memory: rho=0 (pure extragradient) does nothing; the window must smooth many
  batches. (MEASURED)
- DESCENT sign: the SAM-direction probe is inert. (MEASURED)
- MOMENTUM-FREE, PER-BATCH-NORMALIZED source: probing along Muon's own momentum is inert
  (Nesterov already covers that direction); the paper's purity requirement is real.
  (MEASURED)
- The equal-vote normalization is the surviving kernel of the "common minima" intuition:
  because every batch contributes a UNIT vector, d is the direction recent batches AGREE
  on when each gets one vote — a data-consensus direction, not a magnitude-weighted
  momentum. (SUPPORTED — this is the interpretation that fits all ablations)

Working model: **consensus-direction lookahead**. Manas evaluates the landscape slightly
ahead along the variance-reduced direction the data has been pulling, and the returned
gradient carries curvature information about that consensus locus into Muon. The
generalization win shows up as better OOD at MATCHED train loss (the paper's claim shape)
while training slightly FASTER — both axes, held-out gated.

## 4. The knobs and their laws

rho — NOT a free hyperparameter. It holds MEMORY IN SAMPLES (MEASURED, C14):
    window x batch_size ~= N_mem  (a task constant; ~6k samples on the reference task)
    => rho* ~= 1 - batch_size / N_mem
Confirmed across BS 32/128/512 (optima 0.995/0.98/0.90 — monotone exactly as predicted).
Consequences: at LM batch sizes rho is SMALL, which automatically shrinks staleness
exposure at scale; and the knob to expose to users is N_mem (samples of memory), with rho
derived. N_mem's task-dependence is the main unknown (INTUITION: grows with data
heterogeneity).

gamma — the geometry-coupled dose (MEASURED): couples to whatever sets the landscape's
local scale (LR, weight norms, parameterization, wd, loss). Toy optimum 0.08-0.12 — 80x
the naive default; the curve is smooth with a broad plateau and NO instability up to 2x
the optimum (the spike fear was refuted). gamma is the brightness of the torch, d is the
direction; per-scale re-sweep is mandatory, and a self-normalizing form (gamma as a
fraction of realized step motion — the trust machinery exists in exp_manas) is the right
production parameterization. (INTUITION for the trust form; the raw dose law is MEASURED)

probe mass / rank — the payload is a heavily smoothed direction, so it compresses
brutally well (MEASURED): rank-8 keeps the FULL effect (+0.0255 held-out, the round's
champion) at ~3% of a weights copy, even though rank-8 captures only ~45% of per-step
gradient energy. Basis staleness is a non-issue (refresh 200 >= refresh 16). Smoothing is
why: the low-rank projection loses the noise, not the signal.

u buffer (comp) — rho-decayed memory of APPLIED updates added to the probe offset
(user's idea). MEASURED so far: kappa=+1 (extend along realized travel) is frontier-tied
with the champion (held-out +0.0266, nominal best of the round, inside joint noise);
kappa<0 (back-off) adds nothing; dose turns at +1; the peak-acc signal did not survive
held-out. Cost: a second weights-sized buffer (low-rankable the same way d was).
INTUITION (user, plausible): u carries update-size/curvature information that should
matter MORE when batches differ substantially — on MNIST-1D all batches are near-alike,
so the toy may understate it. Status: archived, first-in-line for the large-batch and
LM-scale retests.

## 5. What is dead (do not revisit without new evidence)

EMA reference point (lags the outer gradient — 2.3x slower training); SAM-sign probe;
momentum-source probe; back-off comp; trust cap at toy scale (neutral); the refresh-
cadence worry (B2); the spike worry (B4); the low-rank-hurts worry (B1). Also dead as a
diagnosis method: trusting a 5/5 opt-seed peak-acc signal — it died at the held-out gate
twice; that axis needs more than 3 held-out seeds to resolve 0.01-class effects.

## 6. The two generalization axes

The protocol measures two distinct things and different mechanisms express on different
ones: OOD-at-matched-train-loss (Manas's axis — the paper's claim shape) and peak OOD
accuracy (the Nexus clone's axis — it wins there on its own AdamW base, our positive
control). A superiority claim should state its axis. Manas is held-out-positive on the
frontier axis at every gated config and neutral-to-positive on peak acc.

## 7. Standing order and costs (toy scale, same lr/wd; C8 tuned-AdamW sweep in flight)

Muon+Manas > Muon >> AdamW+Nexus > AdamW, with Muon supplying ~10x the probe's increment
over the AdamW family. Manas's marginal cost over Muon: ~3% of a weights copy (rank-8 d),
one probe apply/remove per step, zero extra fwd/bwd — vs the Nexus clone's full inner
model copy plus K inner optimizer steps per accumulation window.

## 8. Production prescription for LM training (current best guess)

1. Estimate N_mem on a short pilot (or default to memory-in-tokens ~ a few thousand
   sequences); derive rho from tokens-per-step. Expect rho << 0.9 at real batch sizes.
2. Sweep ONLY gamma (3-4 short runs); everything else has a law or a validated default.
3. rank-8 low-rank d is now the SHIPPED DEFAULT (probe_rank=8); refresh 200; fp16 d is an
   untested-but-likely-fine footprint win.
4. u buffer is SHIPPED as an optional knob (comp; default off, comp=+1 when on; shares
   d's r8 basis at ~zero marginal memory). Toy: free passenger (tie, no separation).
   LM A/B (champion no-u vs comp=1) decides; fractional kappa sweep {0.2..1} belongs there,
   not on the toy (0.2-steps are inside the noise floor). If it separates, it's already
   low-rank and ready.
5. Onset gating (start the probe mid-training) is the user's remaining untested
   intuition; the LR-schedule interaction (C11) is the vehicle to test it.
6. Grad accumulation: apply the probe per OPTIMIZER step (accumulate at theta+d with d
   frozen across the window) — matches the mechanism; per-microbatch reapplication buys
   nothing and costs shifts. (INTUITION — decide empirically at BiBo scale)

## 9. Open risks for scale-up

N_mem measured on one task only; frontier effect size at 100M+ unknown (toy effects of
this family have died at LM scale before in this repo — dither round); LR-schedule
interaction untested; the Muon+Nexus incompatibility reproduction still owed for the
writeup; peak-acc axis statistically underpowered everywhere.
