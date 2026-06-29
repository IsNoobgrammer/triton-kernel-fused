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

## RUN 1 -- batched unblocked Householder (DISCARD)
- Python per-column loop = launch-bound: b00 (n=32) = 10ms = 0.03x baseline. Rank-1 bmm
  trailing update = HBM-bound (re-reads trailing matrix every column, O(n^3) traffic):
  b03 (640x512) = 320ms (2.93x, still far off), b04 (60x1024) = 0.62x. Dead family.

## RUN 2 -- fused Triton Householder, 1 program/matrix (KEEP small-n)
- One launch for the whole batch; whole matrix tile carried in registers; internal masked
  column loop (no per-step launches). larfg convention -> householder_product reconstructs Q.
- WIN n<=128: n64 16.6x, n96 6.3x, n128 4.7x, b00(n32) 38us=7.6x. HARD crossover at n>=176:
  register spill (BN>=256 tile) -> 0.1x. So this is the small-n path only.
- Triton can't do mutable-array dynamic single-column indexing; the functional whole-tile
  +tl.where update sidesteps that but costs O(N*BN^2) regs -> only small N fits.

## Toolchain constraint (important)
- Box: triton 3.7 OK, but NO nvcc / NO ninja / CUDA_HOME unset -> torch.utils.cpp_extension
  load_inline FAILS. CUDA custom kernels are NOT available. Triton + torch.bmm only.

## Strategy / dispatch (shape-based by n, NOT conditioning -- legitimate)
- n<=128  -> qr_hh_fused (RUN2).
- n>=176  -> RUN3 blocked Householder: panel factor (nb cols, cheap) + compact-WY T +
  trailing update via 3 batched bmm (cuBLAS tensor cores). Blocking cuts the unblocked's
  HBM re-reads by ~nb and puts the heavy flops on tensor cores -> the big-case (b03/b04/
  b07-b11) lever. Panel micro-step launch overhead (n total) is the next thing to fuse.
- 4096/2048 small-batch cases (b05,b06): blocked may still lose to cuSOLVER's per-matrix
  blocked geqrf -> consider eager geqrf fallback there (shape-dispatch).
