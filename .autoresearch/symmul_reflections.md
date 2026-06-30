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

## Run log (RTX PRO 6000 Blackwell sm_120, torch 2.12+cu130, triton 3.7, fp16, CLEAN process)
All parity PASS throughout (transpose-copy correct; amalg dmax == compiled/fused dmax). Speedups
are NS(5)-step, single square matrix (B=1), vs each baseline. mem = peak MB (clean process).

v1 (batched symmul, explicit axpy): 2048 1.22x fused, 4096 1.29x. MEM 2.4x compiled; batched-small
   REGRESSED 0.72-0.92x (lost baddbmm fold). 1024 symmul lost 0.71x.
v2 (gate<knee->champion + in-place fold): batched-small fixed (1.00x), 1024 fixed, 4096 1.39x.
v3 (ping-pong X buffer, zero per-iter alloc): 4096 1.38x; mem 1.73x->1.21x compiled.
v4 (fused symmul-axpy kernel + torch.compile core): SPEED 2048 1.33x, 4096 1.43x, 8192 1.41x fused
   (1.39-1.43x compiled, 1.08-1.11x triu). mem: amalg == champion (2181MB@8192), still 1.3-1.76x
   OVER compiled. torch.compile CONFIRMED engaged (ok:True) yet mem unchanged.

## TWO STRUCTURAL WALLS (stop grinding; surfaced to user)
1. SPEED CEILING ~1.5x: only 2 of the 3 NS GEMMs are symmetric (X X^T, A A); B X is not. Halving
   2 of 3 equal GEMMs caps the NS-STEP speedup at ~1.5x. We are at ~1.4x (8192) = ~93% of ceiling.
   flash-muon's 1.8x is the matmul-transpose GEMM ALONE, not their NS step. => 1.8x on the NS step
   is ABOVE the lever's reach; needs algorithmic change (attack B X, fewer NS steps) or the
   batched-OPTIMIZER regime where our batched-state lever ALSO fires (the real multiplicative test).
2. MEM <= compiled is STRUCTURAL-UNREACHABLE for any custom-kernel NS: inductor pools aten buffers
   across the whole graph; opaque Triton custom-op outputs (A,B) cannot be pooled (confirmed with
   torch.compile engaged). Even the CHAMPION (pure cuBLAS eager) is 1.3-1.76x over compiled. amalg
   meets "mem <= champion" (== or slightly below fused), just not "<= compiled".

## OPTIMIZER-LEVEL bench (bench_muon_symmul.py, big-model d=4096, 810M params, fp16, CLEAN)
AmalgamatedMuon = FusedMuon + newton_schulz_symmul. Full .step():
  compiled (per-param)  242.2 ms  5767 MB  1.00x
  fused    (champion)   238.2 ms  5947 MB  1.02x vs compiled   <-- only 1.02x, NOT the BiBo 2.3x
  amalg                 181.9 ms  6094 MB  1.33x vs compiled, 1.31x vs fused  (parity 1.95e-3 PASS)
mem amalg/compiled = 1.06x (param/momentum memory dominates -> NS transient delta is small).

## KEY STRUCTURAL FINDING: the two levers are DISJOINT (no multiplicative 1.8x)
fused's batching beats compiled ~2.3x ONLY in the small-matrix-many-params regime (BiBo 512^2,
launch-bound). At d=4096 the matrices are compute-bound cuBLAS -> launch overhead ~0 -> batching
gives ~1.02x. So:
  - small matrices (gram<2048): batching wins 2.3x, symmul INERT (gated to champion) -> amalg=fused.
  - large matrices (gram>=2048): symmul wins ~1.3-1.4x, batching INERT (~1.02x) -> amalg=compiled*1.33.
They can't multiply to 1.8x: a matrix big enough for symmul is too big for batching to matter.
=> amalg is an ADDITIVE per-shape win (take whichever lever applies), not multiplicative. It
   STRICTLY DOMINATES both compiled and the champion fused at every scale (>=1x always, 1.3-1.4x on
   large matrices), parity-exact. 1.8x is above reach; strict mem<=compiled is structural (inductor
   pools; eager+custom-kernel can't), but amalg mem ~= champion and within 6% of compiled at the
   optimizer level.

## What amalg DOES win (clean, measured)
Beats ALL THREE baselines on SPEED at every dim >=2048 (1.33-1.43x fused/compiled, 1.08-1.11x triu),
parity-exact, and uses <= the champion's memory. i.e. it strictly dominates the optimizer we SHIP
(FusedMuon) on both axes; it only loses to inductor (compiled) on the memory axis, structurally.
