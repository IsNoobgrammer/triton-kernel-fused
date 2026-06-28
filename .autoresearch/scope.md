# Scope contract — Round 4: the WHOLE conv router vs torch.compile

Semi-manual loop. Artifact = `kernels/router.py` conv-router backends. Compiler dump (inductor's
generated Triton for the compiled eager router) is the STARTING POINT (marksaroufim tactic, cited in
bench.py). Local = RTX 3050 (BiBo `.venv`, Ampere sm_86) for CORRECTNESS + relative ordering only;
**T4 (Turing sm_75) + `--compile` is the only verdict** — local ≠ T4 on tl.dot/occupancy (proven
every prior round).

## Real goal
The conv router currently LOSES to torch.compile on T4: tldot 0.73x fwd+bwd, cublas 0.35x. Goal:
get a hand-written backend to **≥1.0x vs compiled eager on T4** — or prove it can't and ship the
honest reference. The repo thesis is "hand kernels beat compile where there's a structural edge"
(MoE routing 2.x, XSA read-once 1.15x); find whether the conv router has one.

## What the dump showed (diagnosis)
- Compiled forward = transpose+pad (Triton) -> **cuDNN `extern_kernels.convolution`** -> sigmoid+bias
  (Triton) -> native topk -> gather+sum+div (Triton). Backward = **cuDNN `convolution_backward`** + 3
  tiny Triton kernels. The compiler did NOT write a Triton conv — it called cuDNN. The op is tiny
  (369 MFLOP, ~16MB x); the 0.8ms fwd is LAUNCH/overhead-bound (5 kernels), not compute-bound.
- tldot loses on fwd (0.67x): one fused transpose-free kernel but K=4 taps reload x + tl.dot runs a
  skinny E=11->16 padded output. cublas catastrophic on bwd (0.26x): 4+8 Python-loop cuBLAS GEMMs
  RMW-ing fp32 buffers. cublas DROPPED from the sweep (dominated by tldot).

## Frozen eval (the only ground truth)
- **Held-out / verdict (T4):** `python bench.py --compile router` on Tesla T4. Baseline = compiled
  eager (= the dump). Reports ref / tldot / readonce: fwd, bwd, fwd+bwd x-vs-compiled, peak mem,
  grad_rel, idx-agree, count==bincount, NaN-free. User runs + pastes.
- **Local (3050, correctness only):** `python bench.py router` (uncompiled baseline). idx-agree must
  be 1.0000, count==bincount True, NaN-free True, grad PASS. Perf x-numbers here are NOT comparable
  to T4 (uncompiled baseline + Ampere). Never edit the eval to flatter numbers.

## Objective
Maximize fwd+bwd x-vs-compiled on T4, SUBJECT TO: grad_rel < 1.5e-2, idx-agree == 1.0,
count==bincount, NaN-free, peak mem not worse than compiled.

## In-scope (artifact = kernels/router.py + bench wiring)
- `readonce` fwd kernel internals: tiling/loop-order/MMA shape/configs for the fused conv+sigmoid.
- ITER 2 (gated on T4 data): fuse topk(top-2 of E=11) + gather + norm INTO the conv kernel epilogue
  (in-register, scores never round-trip HBM) — the glue the compiler CANNOT fuse (topk is native).
  This is the candidate structural edge. Build only if iter-1 shows the conv itself can ~tie cuDNN.
- Backward dx/dw kernel tiling (shared `_conv_router_grads`).

## Constraints / invariants (hard)
- tl.dot GEMMs are weak on Turing for big contractions — but here the conv is memory/launch-bound,
  not a big GEMM, so tl.dot is plausibly fine; cuBLAS-per-tap (cublas backend) is worse (launch +
  fp32 RMW). Keep the conv as ONE fused Triton kernel, not K GEMMs.
- grad_rel < 1.5e-2 vs fp32 eager, every candidate. idx exact.
- No host syncs (`.item()/.tolist()`) in the hot path.

## Out of scope (decided)
- The `cublas` backend as a speed play (dominated by tldot; kept in code, off the default sweep).
- Beating cuDNN's conv COMPUTE with tl.dot or K cuBLAS GEMMs head-on — the op is tiny; the only
  available edge is killing launches/HBM round-trips (transpose-free + glue fusion), not out-MACing
  cuDNN.

---
(Prior rounds: R0 baselines; R1 MoE per-expert 2.87x sole win; R2 CE latency-at-memory; R3 CE beat
Liger + XSA re-tile 1.15x — all CONVERGED/shipped. Full history in reflections.md + state.json.)
