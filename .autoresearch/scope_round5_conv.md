# Scope contract — Round 5: transpose-free bandwidth-optimal Triton conv (beat cudnn 1.12x)

Champion to beat = `cudnn` backend (Round 4 WIN: T4 fwd+bwd 1.12x, fwd 1.15x, bwd 1.16x, exact grads,
mem parity, BiBo-parity PASS). PROTECT it — every Round-5 candidate is a SEPARATE backend, A/B'd on T4;
never overwrite cudnn until a candidate beats it on the held-out T4 run.

## Real goal
Push the conv router past 1.12x by going TRANSPOSE-FREE: cudnn's ceiling is ~482us/iter of mandatory
nchwToNhwc/nhwcToNchw layout transposes (confirmed not removable via cuDNN — channels-last A/B refuted)
+ the cuDNN conv GEMMs (fwd 312us, bwd 613us). The op is BANDWIDTH-bound: x=16.8MB -> fwd floor ~52us,
bwd floor ~105us. cuDNN runs 6-16x off that floor (general kernel on a degenerate 11-channel conv).
A native-(B,S,H) Triton conv pays ZERO layout transposes; if its compute approaches the bandwidth
floor it beats cudnn on both phases.

## The wall (from Round 4 T4 profiles)
- Our existing tldot transpose-free FWD kernel = 818us (15x off the 52us floor) — skinny E=11->16 tl.dot
  + K separate dots. Already transpose-free but too slow.
- Our tldot dx/dw BWD kernels = 782+720 = 1502us (vs cuDNN convolution_backward 613us) — same problem.
- So transpose-free ALREADY avoids the 482us tax, but the Triton conv compute is the bottleneck. The
  whole round = make the transpose-free conv fast (fwd+bwd), not 15x off the floor.

## Frozen eval (only ground truth)
- **T4 verdict:** `python bench.py --compile router` (sweep ref + cudnn + the new candidate). Baseline =
  compiled eager. Promote a candidate ONLY if it beats cudnn fwd+bwd on T4 by > noise, grads PASS.
- **Local (3050, correctness only):** `python bench.py router` + `parity_bibo.py` must PASS
  (idx 1.0, grad_rel < 1.5e-2, bias update exact). Local perf is NOT T4-comparable (Ampere, uncompiled).
- BiBo parity (`parity_bibo.py`) must stay PASS for any candidate promoted to default.

## Objective
Maximize T4 fwd+bwd x-vs-compiled, SUBJECT TO: grad_rel < 1.5e-2, idx-agree 1.0, count==bincount,
NaN-free, mem ≤ ~1.15x compiled (memory headroom is AVAILABLE to spend — the CE kernel owns the memory
saving; here we may trade up to ~1.15x mem for speed: SRAM tiling, saved fwd intermediates, im2col).

## Candidate directions (ranked)
1. **iter1 — merged-contraction forward** (`tlconv` backend): fold the K taps into ONE (k,h)=K*H=2048
   contraction (im2col-in-loop) so tl.dot runs fatter/fewer dots over a contiguous 2048 dim instead of
   K=4 skinny dots of 512. Transpose-free, reuses tldot's backward. Tests: does a better-tiled Triton
   conv FWD beat cuDNN's 312us + transpose share?
2. backward retile: dx/dw as merged-contraction / 2D-grid, target < cuDNN 613us (the bigger prize).
3. read-once SRAM tile: load x[s-K+1:s+BLOCK_S, :] once into SRAM, reuse across taps (the bandwidth
   play; awkward sliding-window in Triton — the readonce trap was loop-reorder, this is explicit SRAM).
4. spend memory: save fwd im2col / pre-laid-out buffers for the backward.

## Constraints / invariants (hard)
- cudnn champion untouched; candidates are separate backends, A/B on T4.
- grad_rel < 1.5e-2; idx exact; BiBo parity PASS before any default swap.
- tl.dot output N is stuck at 16 (E=11) — the contraction (K-dim) is where MMA efficiency is winnable,
  not N. Don't chase N.

## Out of scope (decided / refuted)
- Beating cuDNN via channels-last layout hint (R4 refuted — cuDNN copies to its layout regardless).
- cublas K-GEMM conv (0.35x), readonce loop-reorder (0.33x fwd) — both T4-refuted.
- If after a few iters no transpose-free kernel beats cudnn 1.12x: SHIP cudnn, close the round honestly
  (the conv may just be cuDNN's to own even off-floor, due to Triton codegen overhead on T4 sm_75).

---
(R4 WIN shipped: cudnn = cuDNN conv padding=K-1 + fused top-k epilogue + merged manual backward.
BiBo-parity PASS @ E=11. Full history in reflections.md + state.json.)
