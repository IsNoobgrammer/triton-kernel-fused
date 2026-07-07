# Round: perf-per-flop (started 2026-07-04)

## Real goal
Best BiBo downstream performance per unit compute from the optimizer. Two win modes:
(a) better val loss at same wall-clock, or (b) same val loss at meaningfully lower wall-clock.
Metric is 2D Pareto: (val_loss, tok/s). A candidate must dominate the baseline on held-out.

## Baseline (current champion config)
FusedMuon dsv4_10 (_DSV4_COEFFS = KJ x8 + pinned x2) + aurora_k1, lr 3e-4, momentum 0.95,
ns fp16, AdamW-band RMS 0.2. Known v2 numbers (2000 steps, 65.5M tok, RTX PRO 6000):
default 3.8923 / default2 3.8939 (twin noise floor 0.0016) / b12 3.8942 / k2 3.8900 (+9% time) /
champ 3.9890 (demoted).

## EVAL PIVOT (2026-07-04, user directive)
LM loss screening SUSPENDED - "we cannot use language modeling as a loss currently because it is
not grokking properly". New primary eval = GROKKING synthetic: multi-op modular arithmetic
(+, -, *, / mod p=97), small vocab (~102), tiny transformer (d=256), held-out = all unseen (a,b)
pairs. Calibrate train-fraction so the DEFAULT arm lands ~93-94% held-out accuracy at budget
(deliberately below saturation so arms can separate). Metrics: held-out acc @ budget + grok step
(first eval >=90%). Multiple seeds per arm, paired data split (split seed FROZEN, model seed
varies). Arms run as PARALLEL processes on the VM (tiny models). LM confirm budget (100-120M tok,
bigger batch) reserved for ideas that win the synthetic screen. In-flight LM confirms (normuon,
ns8) will be collected and journaled but no longer drive promotion alone.
CAVEAT logged: the earlier toy sorting task was INSENSITIVE - if calibration shows twins
indistinguishable from arm deltas here too, report that to the user rather than fake a signal.

## Frozen eval (L2-LM, SUSPENDED - kept for finalist confirms)
BiBo bench/exp_kappa.py harness + exp_kappa_v2.yaml config with total_steps=600 (~19.7M tok),
same everything else, RTX PRO 6000 VM, python -u, seed fixed by harness. Score = val loss @600
+ mean tps. Noise floor at 600 steps measured by a twin pair (perf_base1 vs perf_base2).
HELD-OUT confirm = full 2000-step run (same config as v2) for finalists only (max 3).
Eval is FROZEN: never change yaml/data/seed to flatter a candidate.

## Cheap fidelities
L0 (local RTX 3050): kappa/spectral checks, minimax coeff solving (.autoresearch/solve_minimax.py),
NS-quality metrics vs exact polar. L1 (toy sorting harness train_synth.py): known INSENSITIVE to
kappa arms (NULL result) - use only as a sanity/crash screen, never for promotion.

## In-scope knobs
NS iteration count/coeffs (compress to 6-8 steps), custom minimax coeffs (spectrum-aware x0),
scale mode (aurora k, normuon-style second-moment on ortho update), sinkhorn/row-col
preconditioning, low-rank EMA preconditioner (user idea), gram backend on sm120 for NS speed.
Out: lr schedule, data, model arch, eval config, batch/seq, AdamW side.

## Prior art (do NOT retread)
- Signed-perm dither: kappa-metric champ, TRAINING LOSER (dose-response penalty). DEAD.
- sink2 prescale: refuted twice. DEAD as tested (plain 2-iter sinkhorn prescale).
- Coeff axis for WORST-CASE floor lift: exhausted, KJ ~ per-step optimal for x0->0.
  BUT: coeffs tuned for x0~0.1 (rect soft edge, ignore square hard edge) NOT tried - kappa@r=1
  proven loss-neutral at 137M, so we may not need worst-case coverage -> fewer iters.
- b12 (12 it) vs default (10 it): tie. k2 (20 it): +0.002-0.004 loss, -9% tps (per-hour loser).
- Toy sorting bench: insensitive to all Muon arm differences.

## Compute accounting
k2 showed +10 NS iters = ~9% wall-clock => ~0.9%/iter. 10->6 iters ~ +3.5% tps.
Gram backend (sm120): NS 1.4x faster (5.70 vs 8.17ms) => additional ~2-3% tps if wired in.

## Definition of done
Either (a) an arm beating baseline val loss @2000 by >2*0.0016 at >=baseline tps, or
(b) an arm tying baseline (within 1 sigma) at >=4% higher tps, or (c) 6 consecutive
discarded candidates / budget out. Budget: <=14 short runs + 3 confirm runs on VM.
Then: journal results, update memory, git commit any promoted default, TURN OFF THE VM.

## Operational rules (user-mandated)
- ALL VM python via python -u. No polling loops: one-shot cron wakeups sized to run duration;
  short things run in background or waited on directly.
- User is OFFLINE - fully autonomous; no questions.
- Shut down the VM when the round is finished.
