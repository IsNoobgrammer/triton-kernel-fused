# Scope contract — MoE kernel throughput

## Real goal
Wall-clock training throughput at the production config. tps is the proxy; the goal is
cheaper 1B-token runs so the activation/architecture rounds iterate faster.

## Artifact (what may change)
kernels/sm75/moe.py, kernels/sm120/moe*.py — dispatch gating, kernel fusion, scheduling,
tiling/autotune. NOT the model, NOT the data, NOT the optimizer.

## Frozen eval
40-step real training run, steps 20-39 steady-state mean tps.
  cfg: bibo_min, 64 experts (polyglu_mult 32), top_k 8, batch 64 x grad_accum 4 x seq 1024,
       bf16, muon_lr 1e-2, cautious off, seed 42069, --eval_every -1
Measured on the idle Blackwell box, one arm at a time (contention corrupts tps).

## Objective / stop target
  radial  158.2k -> >= 165k   (+4.3%)
  silu    164.2k -> >= 170k   (+3.5%)
Radial alone is acceptable per the user.

## INVARIANTS (hard, non-negotiable)
1. NUMERICS UNCHANGED. All parity suites stay green: parity_radial, parity_rowloop,
   parity_normed_tiles, parity_expert_alpha, parity_specials. A speedup that changes the
   math is a FAIL, not a tradeoff. This is the standing regression trip-wire.
2. Never edit the eval or the parity tests to make a number look better.
3. bf16 training path only; fp32 master weights untouched.

## Prior art (do not re-tread) — from moe-kernel-speed-round
  WON:      fused Triton GEMM+GLU (128.9k->148.9k), version-keyed bf16 weight-cast cache
  REJECTED: activation in the down-proj PROLOGUE (tune_glu_prologue.py: forces BN=N=512,
            caps BK at 16, crippled GEMM gives it all back — 1.636 vs 1.588 ms)
  REJECTED: chunk-experts-for-L2 + recompute gu in backward (0.79x, big regression)
  REJECTED: grouped dispatch as default (microbench said 1.24-1.58x, real training said no;
            crossover is ~40 experts — REVISIT, we are AT 64 experts now)
  NEVER trust a microbench for dispatch — measure end-to-end.

## Known structural facts (measured this session)
  - routed experts = 59% of active FLOPs; a perfect MoE block caps the model at ~1.7x
  - 133 TFLOPS achieved vs 418 measured-achievable = 32%
  - recorded cause: expert load imbalance (max/mean 7.95 early, 1.76-2.54 converged);
    in-model GEMMs ran 79 TFLOPS vs 305 achievable
  - radial is excluded from the `uniform` fast path by `and ap32 is None` (moe.py:866)
  - GLU kernels themselves are already at 0.97x of silu — the RMS is NOT the cost

## Out of scope
  - k=8 -> k=6 (a real ~15% win but it changes the MODEL, not the kernel; user asked for kernel)
  - anything that changes bpb
