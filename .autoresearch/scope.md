# Scope contract — Round 2: CE latency at fixed memory

Semi-manual + local-proxy loop. NEW autonomous capability: the baseline is now **inductor's CE
lifted to raw Triton** (`kernels/ce_compiled.py`), so it runs WITHOUT torch.compile → the loop can
iterate **locally on the RTX 3050 at a proxy shape** (optimization set) and confirm on **T4 at
ce_fit** (held-out). torch.compile stays broken locally; the lifted baseline sidesteps that.

## Real goal
Our chunked fused-linear CE (`kernels/cross_entropy.py`) is the MEMORY/ENABLING kernel: it's the
only CE that runs when the (N,V) logits don't fit (ce_oom, long-context). But at ce_fit it's ~2.3×
SLOWER than compiled CE. Goal: **cut that latency gap while KEEPING the memory saving** — make the
enabling kernel cheap enough that there's little penalty for using it, and a real win in the OOM
regime where it's the only option.

## Frozen eval (the only ground truth)
- **Optimization set (local, autonomous):** `.autoresearch/ce_eval_local.py` on the RTX 3050 at a
  4GB-fitting proxy shape (N=4096, H=512, V=32000). Reports fwd+bwd ms, peak MB, grad_rel — for
  BOTH `ce_compiled` (lifted baseline) and our kernel, as raw Triton (no compile).
- **Held-out (manual, T4):** `python bench.py --compile ce_fit` on a Tesla T4 (N=16384, V=81000).
  User runs + pastes. A local win must transfer here to be promoted.
- Never edit the eval to flatter numbers. grad_rel < 1.5e-2 is the hard gate.

## Baseline (frozen reference)
- **Lifted compiled CE** (`ce_compiled.py`): materializes (N,V) fp16 ONCE in fwd, saves it, NO
  backward recompute. This is exactly why compiled is fast + heavy. T4 ce_fit: ~198ms / 3072MB.
- **Our kernel today** (T4 ce_fit): 384MB-budget 455ms / 2331MB; 128MB-budget 537ms / 989MB (3.1×
  less mem). Local proxy baseline numbers recorded in state.json after first run.

## Objective
**Minimize our CE fwd+bwd latency, SUBJECT TO: peak_mem ≤ our current low-mem peak (keep the ≥3×
saving vs compiled) AND grad_rel < 1.5e-2.** Latency is the number to push; memory is a hard
constraint, not a free variable. Report latency as ratio vs lifted-compiled baseline.

## In-scope changes (artifact = kernels/cross_entropy.py)
- Fuse the forward: ONE online-softmax Triton kernel (read fp16, fp32-accumulate in registers,
  gather target in same pass) → kill the `.float()` (C,V) fp32 buffer + the 3 separate passes.
- Chunk-size / launch tuning within the memory budget (bigger chunks = fewer, larger cuBLAS GEMMs).
- Backward grad-logit kernel internals (already in-place Triton) — tiling/warps/eviction.
- A `recompute=False` "save-logits" fast mode = the baseline path, for when memory DOES fit
  (ce_fit) — gives one CE call that auto-picks fast-when-fits / chunked-when-not.

## Constraints / invariants (hard)
- The backward GEMM recompute is the PRICE of the memory saving — removing it (saving full logits)
  is allowed ONLY in the explicit fast/`recompute=False` mode, never in the low-mem path.
- grad_rel < 1.5e-2 vs eager F.cross_entropy, every candidate.
- GPU-resident: no new `.item()/.tolist()/.float()`-scalar host syncs in the hot loop.
- tl.dot GEMMs are DEAD on Turing (proven 0.10×) — keep all big GEMMs on cuBLAS; Triton only for
  the elementwise/reduction fusion. Do NOT propose a tl.dot streaming-logit kernel.

## Out of scope (decided, do not revisit)
- Beating compiled CE on SPEED when logits fit — impossible without spending its memory (the
  save-vs-recompute axis IS the memory axis). We compete on memory, tie-ish on latency at best.
- CE is NOT memory-bound — it's GEMM-dominated (LM head ~1.36 TFLOP fwd). SRAM/tiling/warps help
  only the softmax-reduction slice; don't expect them to touch the GEMM floor.
- SwiGLU/XSA/conv as speed plays (lose to compile); MoE (separate round, already 1.74–2.85×).
