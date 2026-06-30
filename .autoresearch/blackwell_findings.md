# RTX PRO 6000 Blackwell (sm_120) — measured results

**Host:** NVIDIA RTX PRO 6000 Blackwell Server Edition · `sm_120` (cc 12.0, 188 SMs, ~128 MB L2,
GDDR7 ~1.6–1.8 TB/s) · torch 2.12.0+cu130 · triton 3.7.0 · `compile=ON`. Both fp16 and bf16 validated.
Kernels run from `kernels.sm120` (auto-detected by `bench.py`; reuses sm75 + Muon 8M default).

Baseline = `torch.compile`'d eager (`fullgraph` where possible) — the same bar as T4. Every number below is
measured on this host with grad-parity checked in the same run. Contrast: on T4 the same kernels were
~1.0–1.15× (CE memory-only, XSA 1.15×, Muon 1.09×, MoE 2.87×) — Blackwell widens most of the wins.

---

## Muon (Polar-Express step) — the win scales hard off Turing

fp32 master + fp16 NS (mixed). Parity: fused-fp32 vs full-fp32 = **7–9.5e-7** (isolates the fusion, run
variance); fused-mixed(fp16-NS) vs full-fp32 = 2.44e-5; fp16-NS SV~1, NaN-free. **`ns_batch_elems` knee =
8M on Blackwell** (vs 4M on T4) — `kernels.sm120.FusedMuon` defaults to it. Hard gate: peak ≤ baseline.

**48 tensors / 75.5M params** — baseline-mixed 16.43 ms / 916 MB (default fused-mixed = 2.34× / 913 MB):

| ns_batch_elems | speed | peak MB | gate |
|---|---|---|---|
| 4M | 1.94× | 879 | PASS |
| **8M (sm120 default)** | **2.29×** | 914 | **PASS (knee)** |
| 16M | 2.57× | 993 | OVER |
| 64M | 2.30× | 1172 | OVER |

**192 tensors / 302M params** — baseline-mixed 67.14 ms / 3249 MB (default fused-mixed = 2.46× / 3235 MB):

| ns_batch_elems | speed | peak MB | gate |
|---|---|---|---|
| 4M | 2.05× | 3196 | PASS |
| **8M (sm120 default)** | **2.48×** | 3231 | **PASS (knee)** |
| 16M | 2.74× | 3310 | OVER |
| 64M | 2.23× | 3845 | OVER |

**WIN: ~2.3× (75M) / ~2.48× (302M) at peak ≤ baseline** (T4 was 1.05–1.09×). 16M would give 2.57–2.74×
but breaks the mem gate. fp16 NS stays the right choice (bf16 has fewer mantissa bits; Blackwell runs fp16
on full-rate tensor cores).

---

## MoE (PolyGLU, BiBo stack: 9 GLU + Identity + Zero, N=16384 rows·k=32768, H=512 I=768 E=11, k=2)

Baseline = compiled `moe_eager` (Qwen3MoE / HF mask+loop+scatter pattern).

| path | fwd | bwd | fwd+bwd | peak vs eager | grad |
|---|---|---|---|---|---|
| **per-expert** | 2.91× | 4.59× | **3.93×** | 1.08× less (622/670 MB) | rel 6.96e-3 **PASS** |
| grouped | 3.14× | 6.38× | 4.95× | 0.76× less (877/670 MB) | rel 1.03e+00 **CHECK** |

**WIN: per-expert ~3.9× fwd+bwd (bwd 4.6×), correct, lighter memory — the shippable Blackwell MoE.**
(Across runs per-expert fwd+bwd ranged 3.75–4.86×; ~3.9× is the clean steady value. T4 was 2.87×.)

**grouped is NOT a win — it's fast but WRONG.** rel 1.03e+00 = broken gradients: `_GroupedMoE` runs a GLU
over every expert and never implements the Identity/Zero specials, so its 4.95× is timed on incorrect math.
`moe()` is guarded to fall back to per-expert whenever a special expert is present (any arch). Reclaiming
grouped on mixed stacks (fix specials in fwd+bwd) is the one open "fork a module into sm120" job.

---

## Fused-linear cross-entropy (N=16384, V=81000, H=512) — memory win, and BEATS Liger

CE is a **memory/OOM-enabler, not a speedup** (same as T4): compiled eager peak ~3122 MB; ours bounds it to
762–1626 MB. The backward is ~**25–26× faster** (scalar scale, no recompute); the forward is the slow part
(0.23–0.25×, cuBLAS-chunked logits), so fwd+bwd lands < 1× but **fits where standard CE OOMs**. grad PASS
(loss Δ ~2.4e-6, rel ~9.5e-3).

Our chunk sweep (bf16, fwd+bwd vs compiled 15.58 ms / 3125 MB):

| budget | fwd+bwd | peak MB | vs compiled | mem |
|---|---|---|---|---|
| 128 MB | 24.58 ms | 820 | 0.63× | 3.81× less |
| 192 MB | 22.89 ms | 955 | 0.68× | 3.27× less |
| 256 MB | 21.09 ms | 1089 | 0.74× | 2.87× less |
| 512 MB | 18.99 ms | 1626 | **0.82× (saturates ~0.83–0.84×)** | 1.92× less |

**WIN vs Liger — we dominate the memory↔speed frontier.** Liger's fused-linear CE on the same grid:

| Liger chunk | fwd+bwd | peak MB | vs compiled | mem |
|---|---|---|---|---|
| 256 | 88.53 ms | 762 | 0.19× | 4.10× less |
| 512 | 46.69 ms | 804 | 0.35× | 3.89× less |
| 1024 | 31.20 ms | 886 | 0.53× | 3.53× less |
| 2048 | 21.38 ms | 1134 | 0.77× | 2.76× less |

At **equal-or-lower peak memory, ours is faster**: ours @128 MB (820 MB) = 0.63× beats Liger @1024 (886 MB)
= 0.53× *and* uses less memory. Ours saturates at 0.83–0.84×; Liger tops out at 0.77× (chunk 2048) and
degrades much faster as you cut memory (Liger 256 → 0.19× vs ours 128 MB → 0.63×). At aggressive memory
saving we **drop fewer FLOPs** than Liger — the chunked-fused-fwd+bwd recipe (one Triton kernel does lse +
in-place grad; no recompute GEMM) is the edge. Verified (grad PASS on every point).

---

## XSA (B=16, Hq=4, S=1024, D=128, Hkv=2) — warm, 5-run stable

Measured **warm** (no L2 flush): XSA's `Y` is the attention output produced the instant before, so it is
L2-resident in real use. (do_bench's cold-L2 default read fwd 0.79× — an unrepresentative worst case.)
Stable across 5 runs:

| phase | kernel | eager | speedup |
|---|---|---|---|
| forward | 0.028 ms | 0.036 ms | **1.29×** |
| backward | 0.131 ms | 0.163 ms | **1.25×** |
| **fwd+bwd** | 0.194 ms | 0.313 ms | **1.61×** |

peak 235/235 MB (1.00×). grad PASS (bf16 rel 9.03e-3; fp16 rel 1.22e-3).

**WIN: ~1.61× fwd+bwd, grad-exact, no extra peak** (T4 was 1.15×). Note: the *pure-kernel* forward is ~3×
on the GPU (profiler: `_xsa_fwd_kernel` 7.1 µs vs eager 22.5 µs), but the op is tiny enough that Python /
launch dispatch dominates wall-clock, so the end-to-end warm forward is 1.29×. The honest, representative
number is the **1.61× fwd+bwd** — don't quote the 3× (it's GPU-only, not what a step sees). The backward is
still at the structural roofline (one fused read-once kernel, ~67 MB essential traffic; inductor's backward
materializes a ~16.8 MB intermediate = ~34 MB round-trip tax we avoid).

---

## Router — WIN on Blackwell via a transpose-free fused Triton conv (the T4 regression, recovered)

Conv MoE router, B=16 S=1024 H=512 E=11 K=4 k=2, bf16. `kernels.sm120` uses `FusedConvRouterFused` — a
**single-launch fused Triton conv** (no cuDNN, no layout transpose): forward does the K-tap H-contraction
in-SRAM + sigmoid + top-k + gather in-register; backward is two fused Triton kernels (grad_x H-tiled,
grad_W split-K over N). grad PASS, mem parity-or-better.

| baseline | forward | backward | fwd+bwd |
|---|---|---|---|
| compiled eager | **1.29×** | **2.30×** | **1.86×** |

grad PASS (abs 4.26e-3, **rel 2.3e-4**), idx-agree 1.0000, count==bincount True, NaN-free. peak 273/308 MB
(**1.13× less**). Our CUDA total 2.32 ms vs eager 4.83 ms (**2.08× less work**); ~31 launches vs cuDNN's
~54 — zero layout-transpose copies, x read once.

**The last lever was fusing the weight tail.** `norm_topk_prob` (÷Σ) + `routed_scaling` (×c) were running in
eager (~17 un-fused div/sum/mul/fill launches the compiled baseline fuses away) — the dominant launch tax
once the kernels were fast. `_TopkNormalize` folds both into one Triton Function (fwd `O=c·w/(Σw+eps)`; bwd
the sum-to-1 Jacobian `grad_w=c/s·(go−⟨go,p⟩)`). That cut launches 47→31 and lifted fwd+bwd 1.45→**1.86×**
(backward 1.62→**2.30×**). grad still exact.

**The history (why this took 3 candidates):** the sm75 **cuDNN** champion (T4 win 1.11–1.17×) *regressed to
0.82× uncompiled / 0.91× compiled on Blackwell* — its `x.transpose(1,2).contiguous()` + manual
`convolution_backward` cost `aten::copy_` = 48% of CUDA (2.68 ms), and inductor's compiled path was already
lean on glue, so the top-k fusion no longer paid. The fix was the transpose-free fused Triton conv shelved
on T4 (64 KB SRAM); Blackwell's ~99 KB Triton SMEM + bf16 TC is the "revisit on better hardware" case.

- **#1 K-GEMM (K torch matmuls):** correct but **0.54×** — launch/copy explosion (161 launches, 240 mm,
  760 copy_). Lesson: any multi-torch-op conv is launch-bound here too. Only a single fused kernel wins.
- **#2 fused Triton conv:** won wall-clock (1.47×) but **failed the fp32 gate** — `tl.dot` defaults to
  **TF32** on fp32 inputs → ~1e-3 logit error flipped near-tie top-k picks → count≠bincount, grad rel 1e-1.
  And backward was dw-bound (`_conv_router_dw` 58% of CUDA, 145 µs/call: grid only K·(H/64)=32 programs
  serial-reducing all N on 188 SMs).
- **#3 (shipped):** `input_precision="ieee"` on every `tl.dot` (true fp32, no-op on the bf16 timed path)
  → grad PASS + count True. **split-K over N** in grad_W (3rd grid axis, SPLIT_N=16 → 512 programs,
  atomic-add into a zero-init fp32 accumulator) → dw 145→**23 µs/call**, backward 0.87→**1.62×**.

**Remaining soft spot (optional):** the forward kernel is now the largest single GPU cost (54 µs, fwd
1.18×) — a skinny (M=32, N=16, K=512) matmul + in-register top-k, 512 programs, `BLOCK_S` pinned at 32 by
the fp32-check SMEM budget. Halving it would push fwd+bwd toward ~1.6×. Not pursued — the scope target
(tie-or-win, grad PASS, mem parity) is exceeded.

---

## Scoreboard (sm_120 vs `torch.compile`)

| kernel | Blackwell result | vs T4 |
|---|---|---|
| Muon | **~2.3× (75M) / ~2.48× (302M)** (8M knee, peak ≤ baseline) | 1.05–1.09× |
| MoE per-expert | **~3.9× fwd+bwd**, correct | 2.87× |
| MoE grouped | 4.95× but grad WRONG (specials) — not shippable | — |
| CE | memory: up to 3.8× less peak; **beats Liger** at equal mem | memory-only |
| XSA | **~1.61× fwd+bwd** (warm, 5-run stable) | 1.15× |
| router | **~1.86× fwd+bwd** (fwd 1.29× / bwd 2.30×), grad PASS, 1.13× less mem — fused Triton conv + fused norm, sm120 | 1.11–1.17× |

## MoE grouped (sm_120) — cuBLAS grouped GEMM beats per-expert at scale (2026-06-30)

New path `kernels/sm120/moe_grouped.py::moe_grouped_cublas_polyglu` (commit b8a6660). One
`torch._grouped_mm` (cuBLAS, autograd-native) grouped GEMM over all routed GLU tokens + PolyGLU
activation + special experts (Identity/Zero) on the sorted tail. Correct on the BiBo special-experts
stack (grad PASS), unlike the old GLU-only tl.dot `moe_grouped` (grad rel ~5.7 WRONG there).

The win is governed by TOKENS-PER-EXPERT (= routed/E_glu), not N or E alone (vs per-expert, bf16, fwd+bwd):
  tok/exp ~496 -> 3.0x + LESS mem | ~963 -> 2.1-2.5x + <= mem | ~1820 -> 1.6x | ~3277 (BiBo E=9) -> ~1.17x
  high tok/exp -> per-expert GEMMs already saturate the tensor cores; per-expert wins (and is leaner).
Why: per-expert fires one tiny cuBLAS GEMM + a DtoD copy PER expert (~1313 launches/~510 copies at E=32);
grouped collapses that to a few scheduled grouped GEMMs (~851 launches/~35 copies).

Speed/memory is a Pareto frontier: v4 manual-backward = fastest (2.55x) but +18-35% mem; v3 checkpoint =
leanest (<= mem) but slower; shipped v2 (autograd-native, bf16 scatter) = balanced (2-3x at <=/~equal mem
for top_k=2). `moe()` dispatches to grouped only when supported (sm_80+, bf16/fp16, 16B-aligned,
torch._grouped_mm) AND tokens-per-expert <= 2048; else per-expert (the memory-frugal fallback). BiBo's own
E=9 stack stays on per-expert; grouped is the win for higher-expert-count MoEs on Blackwell.
Full ladder + per-run numbers: .autoresearch/moe_grouped_results.jsonl, _reflections.md, _scope.md.
