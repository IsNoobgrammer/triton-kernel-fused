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

## CORRECTED VERDICT — 4-way on bench_muon's REAL methodology, H-swept (bench_muon_4way.py)
The NS-micro and my first d=4096-only optimizer bench were both MISLEADING. The faithful sweep
(BiBo inventory: attn+dense MLP+3D experts, do_bench step, clean per-contender peak):
  H     compiled   fused(x cmp)   amalg(x cmp / x fused)   mem amalg/compiled   parity
  512   5.48ms     1.57 (3.49x)   1.57 (3.49x / 1.00x)     1.04x                5.9e-3
  1024  9.17ms     7.16 (1.28x)   7.02 (1.31x / 1.02x)     0.92x                5.9e-3
  2048  43.5ms    39.5 (1.10x)   33.0 (1.32x / 1.20x)     0.97x                5.9e-3
  4096  267ms     262  (1.02x)   204  (1.31x / 1.28x)     0.97x                3.9e-3

KEY CORRECTIONS to earlier (over-pessimistic) notes:
1. fused-vs-compiled is ~3.49x at H=512 (matches the user's recollection), FADING to 1.02x at
   H=4096. My earlier "fused==compiled" was H=4096-ONLY (compute-bound, batching inert) -- not the
   whole curve. Not an optimizer bug.
2. "mem <= compiled unreachable" was a NS-MICRO artifact (no optimizer state to amortize). At the
   OPTIMIZER level amalg uses LESS mem than compiled for H>=1024 (0.92-0.97x); 1.04x at H=512
   (==champion, negligible). The mem bar is MET in the real regime.
3. amalg BEATS compiled at EVERY H (1.31-3.49x) and is >= fused everywhere (1.00x small H gated,
   1.20-1.28x once gram>=2048 so symmul fires). Best-of-both envelope: batching where matrices are
   small, symmul where large. Parity PASS throughout.

## SCALE TEST 1B-2.6B params (both width H AND depth layers up) -- win is scale-invariant
  H3072 L4 (0.96B): compiled 253ms  fused 240 (1.06x)  amalg 186 (1.36x cmp / 1.29x fused)  mem a/c 0.98x
  H3584 L5 (1.64B): compiled 479ms  fused 463 (1.03x)  amalg 364 (1.32x cmp / 1.27x fused)  mem a/c 0.99x
  H4096 L6 (2.57B): compiled 809ms  fused 786 (1.03x)  amalg 613 (1.32x cmp / 1.28x fused)  mem a/c 0.99x
parity 5.9e-3 PASS throughout. amalg ~1.32x compiled / ~1.28x champion, mem < compiled, at every
scale. Depth doesn't change it (more layers = more matrices of the same sizes -> per-matrix symmul
win replicates). At these large matrices batching is inert (fused 1.03-1.06x cmp) so amalg's gain is
the symmul FLOP cut, stable from 1B to 2.6B+. SHIP-WORTHY: dominates compiled and champion on speed
AND memory across the full large-model range.

## HEAD-TO-HEAD vs flash-muon's EXACT code (nil0x9/flash-muon main, verified verbatim)
NS micro (single matrix), amalg (PE/fp16, batched symmul + fused-axpy + compile) vs their EXACT
fast_newtonschulz (Jordan/bf16, matmul_transpose_assign x2 + elementwise B):
  d=2048: amalg 0.787ms (SV .982) | flash 0.887ms (SV .897) | 1.13x
  d=4096: amalg 3.853ms (SV .978) | flash 4.382ms (SV .852) | 1.14x
  d=8192: amalg 30.85ms (SV .974) | flash 33.86ms (SV .837) | 1.10x
Optimizer step, dense 0.81B (H=4096 L=4): compiled 299.8 | flash 273.1 | fused 291.5 | amalg 233.6 ms
  -> amalg/flash 1.17x, amalg/compiled 1.28x, amalg/fused 1.25x.
NOTE: flash (per-param symmul) BEATS fused/compiled at large dense matrices (gets the FLOP cut full
cuBLAS doesn't), but amalg beats flash by batching the same-shape group into one symmul + fused-axpy
+ compile vs their per-param launches. We beat their exact impl at BOTH levels AND orthogonalize
tighter (SV .97-.98 vs .84-.90, PE coeffs). [VRAM cleared between runs: empty_cache freed ~40GB.]

## B@X INVESTIGATION (the non-symmetric NS GEMM) — pushed with profiler+roofline+fp8, it's at floor
torch.profiler, 10x NS(5) @ d=4096, CUDA self-time:
  cutlass f16 gemm (B@X)   18.04ms  46.2%   <- biggest kernel, irreducible
  _bmmt_kernel (X X^T)     10.03ms  25.7%
  _bmmt_axpy (bA+cA^2)      9.82ms  25.1%
  memcpy/norm/elementwise   ~1.2ms   ~3%
Roofline: B@X = 371 TFLOP/s (cutlass tensorop, near SoL for square fp16 on RTX PRO 6000).
Levers tried on B@X, ALL fail:
  - symmetry: B@X output not symmetric -> no FLOP cut.
  - reassociation B@X=b(A@X)+cA(A@X): MORE FLOPs (replaces cheap 0.5*M^3 A^2 with two full M^2K). Ruled out.
  - fp8 (_scaled_mm): 0.69-0.81x of fp16 (quantize overhead, not at peak) AND breaks orthogonalization
    (fp8 NS SV spread 0.001..2.2; 3 mantissa bits kill small singular values). Dead on both counts.
  - hand Triton GEMM: loses to cutlass (known: tl.dot < cuBLAS).
CONCLUSION: B@X is at its floor; the ~1.5x NS-step ceiling is FUNDAMENTAL. amalg ~1.43x is ~93-95%
of it. Remaining slivers: symmul kernels are ~0.55x of B@X each vs ideal 0.5x -> ~10% headroom on OUR
triton kernels (worth ~5% on the step) via autotune. Only structural way past 1.5x = fewer NS steps
(4 vs 5 = algorithm/convergence change, user-owned, orthogonal to the kernel work).

## LOCAL GPU — RTX 3050 Laptop (sm_86 Ampere, 4GB), torch 2.6+cu124, triton 3.7 (free, no VM)
Compile guard added: module-level torch.compile(_amalg_core) wrapped in try/except (the local env has a
torch2.6/triton3.7 mismatch -> inductor import errors). Falls back to EAGER symmul kernels (AMALG_COMPILE
auto-set). Box (torch2.12) still compiles. Symmul transfers to Ampere AND graph question answered:
  NS micro symmul vs cuBLAS: d2048 1.49x, d4096 1.51x (better than Blackwell's 1.43x — consumer cuBLAS
    leaves more room), parity exact (0.0-1.2e-4).
  Optimizer (mixed 18M, 3x2048^2): symmul-eager 71.2ms / cuBLAS-eager 98.4ms / cuBLAS-graph 97.9ms.
  -> symmul 1.38x cuBLAS-eager. CUDA-graph CAPTURED (graph_captured=True) but 1.00x vs cuBLAS-eager:
     compute-bound even on a slow-CPU laptop -> use_graph is a SPEED WASH everywhere (T4/Blackwell/3050),
     only saves memory (644->445MB). use_graph=True costs 1.37x (loses symmul, gains ~0 from graph).
     KEEP use_graph=False (the default). Repro: .autoresearch/bench_graph_local.py.

## What amalg DOES win (clean, measured)
Beats ALL THREE baselines on SPEED at every dim >=2048 (1.33-1.43x fused/compiled, 1.08-1.11x triu),
parity-exact, and uses <= the champion's memory. i.e. it strictly dominates the optimizer we SHIP
(FusedMuon) on both axes; it only loses to inductor (compiled) on the memory axis, structurally.
