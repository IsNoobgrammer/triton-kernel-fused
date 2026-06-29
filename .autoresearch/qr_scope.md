# Scope contract -- batched compact-Householder QR (GPU MODE leaderboard 774, `qr_v2`)

## Real goal
Submit a correct, fast `custom_kernel(A) -> (H, tau)` to GPU MODE leaderboard 774
(`problems/linalg/qr_v2`). Ranking = geometric mean of 12 benchmark cases' runtime
among passing submissions. Current public top board (μs geomean): gau.nernst 1227.7,
10billiontokens 1235.9, dhu.randhar 1279.3, michaelmelons 1504.2, nikhilbarhate99 1505.5,
Olek 1553.6. Goal: ship a correct baseline first, then `optmaxx` toward / past ~1227 us.

## The frozen eval (NEVER edit)
Pulled verbatim from gpu-mode/reference-kernels @ problems/linalg/qr_v2 into
`qr_challenge/`: `reference.py` (generate_input + check_implementation), `task.py`
(types), `eval.py` (scoring), `utils.py` (clear_l2_cache/set_seed). Local driver
`qr_challenge/run_local.py` reproduces the test gate + benchmark geomean. The artifact
under optimization is ONLY `qr_challenge/submission.py::custom_kernel`.

## Correctness gate (hard, FP64-measured, purely relative, no atol)
For each matrix, with Q = householder_product(H,tau), R = triu(H), in fp64:
- FACTOR: ||R - Q^T A||_1 <= 20*n*eps32 * ||A||_1   (per-matrix; eps32=1.19e-7)
- ORTH  : max_b ||Q^T Q - I||_1 <= 100*n*eps32 * ||I||_1
- H,tau must be FP32, exact shape (b,n,n)/(b,n), finite.
Tolerances are generous (n=512: factor rtol 1.2e-3, orth rtol 6.1e-3) -> internal
fp16/bf16/fp8 strategies are ALLOWED as long as returned fp32 factors pass.

## Constraints / invariants
- Output must follow torch.geqrf compact convention exactly (householder_product(H,tau)
  must reconstruct Q; triu(H) must be R). LAPACK sign/convention matters.
- NO whole-batch routing by sampling: benchmark includes `mixed` (heterogeneous
  per-matrix conditioning) + homogeneous ill-conditioned (rankdef/clustered/nearrank).
  Each matrix factored correctly on its own merits. The runtime of the accurate path on
  hard inputs IS part of the score (robustness is ranked, not just gated).
- Returned factors fp32; QR invariants must hold for rank-deficient / clustered-scale /
  near-collinear / banded / row-scaled / upper-tri / near-rank inputs.

## HARDWARE GAP (important)
Leaderboard GPU = **B200 (sm_100)**. Our dev box = **RTX PRO 6000 Blackwell (sm_120)**.
Same Blackwell family (tcgen05/fp8/large SMEM), but NOT identical -- B200 has HBM3e
~8TB/s + 2x the SMs; sm_120 is GDDR7. We optimize + gate correctness on sm_120; the
final μs ranking will be re-measured on B200. Per the repo's portability philosophy:
the METHOD transfers, the exact multiplier does not. Track our sm_120 geomean as the
proxy; do not over-fit to sm_120-specific quirks.

## In scope (knobs the loop may move)
`submission.py` only -- any internal strategy: cuSOLVER (torch.geqrf / linalg), batched
blocked Householder (WY rep -> tensor-core GEMMs), CholeskyQR / CholeskyQR2 / shifted
CholeskyQR3 with per-matrix Householder fallback, Triton kernels, low-bit internal
compute + fp32 correction, custom CUDA. Stream/graph/batching layout.

## Out of scope / off-limits
- Editing reference.py / task.py / eval.py / utils.py (the frozen eval). Cardinal sin.
- Anything that inspects a subset and routes the whole batch to a well-conditioned-only
  path (explicitly defeated by the mixed/ill-cond benchmark + recheck-every-iteration).
- Returning non-fp32 factors, or a convention that householder_product can't reconstruct.

## Prior art / known traps (fill as we learn)
- Baseline torch.geqrf = cuSOLVER batched geqrf. Establish its geomean as RUN 0.
- CholeskyQR family is the tensor-core-fast path but unstable/invalid on rank-deficient
  (exactly singular G=A^T A) and ill-conditioned -> needs fallback; the benchmark is
  built to punish naive whole-batch CholeskyQR.

## Definition of done
1. (today) Correct baseline `torch.geqrf` passes all 22 test specs + record geomean.
2. Iterate to beat the sm_120 baseline geomean substantially, targeting a geomean that
   would be competitive on the B200 board (~1227 us reference). Ship the champion
   submission.py + a full results ladder. Stop on: target met, user stop, budget, or
   patience/plateau per the autoresearcher stopping rules.
