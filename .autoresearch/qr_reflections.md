# QR (leaderboard 774, qr_v2) -- reflections

## RUN 0 -- baseline (eager cuSOLVER geqrf) = 111865 us geomean (sm_120)

- **The whole game is BATCHING, not the math.** `torch.geqrf` serializes the batch:
  a host-side loop calling one cuSOLVER `geqrf` per matrix. The 640x512 cases cost
  ~937 ms each; the 60x1024 cases ~214 ms. These 7 large-batch cases dominate the
  geomean. Leaderboard winners (~1227 us on B200) get a ~90-100x edge almost entirely
  from doing the batch as batched GPU work, not a serial loop.
- **torch.compile cannot help geqrf.** `aten::geqrf.default` has no Meta/fake kernel,
  so `torch.compile(fullgraph=True)` RAISES (`Unsupported: ... fake tensors`), and there
  is no Inductor lowering -- the only legal compiled form graph-breaks back to eager.
  => the honest baseline bar is eager geqrf. (Recorded for the user; the real bar is the
  leaderboard geomean.)
- **Numerical headroom is enormous.** Passing uses <0.5% of the factor-residual budget
  (scaled ~0.01-0.1 vs 20) and the orthogonality budget (scaled ~0.1-0.4 vs 100). The
  spec explicitly permits fp16/fp8/nvfp4 *internal* compute as long as returned factors
  are fp32 and pass -> tensor-core low-bit QR is on the table.
- **Output format forces Householder.** Checker reconstructs Q via
  householder_product(H,tau) and R=triu(H); we MUST emit compact Householder reflectors
  (LAPACK larfg convention). CholeskyQR/CholeskyQR2 give R + Q=A R^-1 but NOT the
  reflector encoding -> not directly usable for the output. So the method must be
  Householder-based (blocked Householder -> WY -> tensor-core GEMM trailing updates).
- **Robustness is ranked, not just gated.** benchmark includes mixed (per-matrix
  heterogeneous conditioning) + homogeneous rankdef/clustered/nearrank. Whole-batch
  routing by sampling conditioning is explicitly defeated (recheck every iter). Shape-based
  dispatch (by n/batch, NOT by conditioning) is legitimate.

## Plan / ladder
- RUN 1: batched **unblocked** Householder QR in torch (vectorized over batch) -- simplest
  correct batched form; expected big win on the large-batch cases, may lose on large-n
  small-batch (n=4096) due to n sequential steps. Measure where it wins/loses.
- RUN 2+: **blocked** Householder (compact-WY, batched bmm trailing updates -> tensor
  cores); tune panel width nb; consider fp16/bf16 internal with fp32 correction; Triton
  panel kernel; shape-based method dispatch (eager geqrf may still win small-batch-large-n).
