# symmul-Muon reflections (running log)

## Prior art carried in (do not relearn)
- `tl.dot` NS lost ~3x to cuBLAS on T4 (Turing, no bf16 TC). That is Turing-specific. flash-muon
  proves a Triton symmetric matmul BEATS cuBLAS ~1.3-1.8x at dim>=2048 on A100/H800/4090; wash at
  dim<=1024. => the win is large-dim only -> shape dispatch is mandatory (SYMMUL_MIN_DIM=1024 in v1).
- Losing the baddbmm fold on the 2 symmetric terms adds an elementwise axpy (b*A + c*AA). On large
  matrices the halved GEMM should dominate; on small matrices it may erode the win -> watch the
  batched-small guard, and consider fusing the polynomial into the Triton epilogue if v1 is short.
- A = symmul(X) is EXACTLY symmetric (diagonal tiles computed in full, off-diagonal mirrored by
  transpose-copy) so symmul(A)=A@A^T=A@A is valid. The transpose-copy is the correctness risk
  (Laker Newhouse's ThunderKittens version had a transpose-store bug) -> parity is a HARD gate.

## Run log
(awaiting first bench paste from the RTX PRO 6000 box)
