# triton-kernel-fused

Four standalone, drop-in **fused Triton kernels** (forward **and** backward), each a swap-in
replacement for its PyTorch-eager equivalent. No framework, no `pip install` — copy the
`kernels/` folder into your project and import.

```python
from kernels import fused_linear_cross_entropy, fused_xsa, causal_conv1d_router, moe
```

Every kernel is wrapped in a `torch.autograd.Function` (trains normally) and is
grad-equivalent to eager within fp16 tolerance (relative error < 1.5e-2, verified by `bench.py`).

> **Honest headline (measured on T4 vs `torch.compile`, not vs slow eager).** Three of these beat
> what the compiler gives you for free, for structural reasons compile can't touch:
> **`moe` (per-expert)** — ~2.9× faster, because `torch.compile` can't fuse data-dependent routing;
> **`fused_linear_cross_entropy`** — the memory play that lets large-vocab CE fit when it would
> otherwise OOM, and which now also **beats Liger's fused-linear CE** (the standard for this op):
> 260 ms vs 321 ms at 16k tok / vocab 81k on T4, with gradients *tighter* to fp32 than Liger's; and
> **`fused_xsa`** — **1.20× fwd / 1.15× fwd+bwd** vs inductor, because one fused kernel reads V once
> while inductor's two kernels read it twice. `causal_conv1d_router` still **ties or loses** under
> compile (inductor uses cuDNN), kept as a no-`torch.compile` fallback. (SwiGLU was dropped entirely:
> `torch.compile`'s lifted SiLU-mul kernel matches a hand-written one, so we leave it to the compiler.)
> The grouped-MoE `tl.dot` path is a disaster on Turing and is **auto-disabled on sm_<80**.
> See [Reality check](#reality-check-vs-torchcompile-on-t4).

## Getting started

**To use the kernels in your own project:** copy the `kernels/` folder in — no install needed. They
import with just a CUDA `torch` + `triton` already in your env.

**To clone this repo and run the bench / examples** (needs a CUDA GPU):

```bash
# Option A — uv (sets up everything from scratch, incl. CUDA torch + triton):
uv sync
uv run python bench.py                     # all kernels, eager baseline
uv run python bench.py --compile           # torch.compile steady-state (run on the target GPU)
uv run python examples/moe_usage.py

# Option B — you already have a CUDA torch + triton env: just run from the repo root
python bench.py
```

`uv sync` pulls CUDA (cu124) torch from the official PyTorch index (configured in `pyproject.toml`)
plus triton, and installs `kernels` as an importable package. On Windows, triton ships inside the
torch wheel; on Linux it's a separate wheel (declared as a dependency).

## The kernels

| Kernel | Replaces | What it fuses |
|---|---|---|
| `fused_linear_cross_entropy(hidden, weight, labels)` | `F.cross_entropy(hidden @ W.T, labels)` | LM-head GEMM + softmax-CE **without materializing the (N,V) logits** (cut-cross-entropy style); grad computed in the forward chunk loop, **no backward recompute**. Beats Liger. |
| `fused_xsa(attn_out, value_states)` | Exclusive Self Attention rejection `y − (y·v̂)v̂` | normalize + dot + reject + GQA broadcast, one kernel each. No repeat_kv copy. |
| `causal_conv1d_router(x, weight)` | `pad → F.conv1d → reshape` (causal) | causal conv1d projection in native (B,S,H) layout, transpose-free fwd + bwd. |
| `moe(hidden, idx, weights, gate_up_proj, down_proj, act_codes)` | a hand-written PolyGLU MoE expert loop | the expert pipeline: dispatch → ragged GEMM + per-expert PolyGLU activation → weighted combine. Per-expert (cuBLAS + fused-act) and grouped-GEMM paths. |

## Usage

**Fused-linear CE** — pass the pre-logit hidden states and the LM-head weight directly:
```python
loss = fused_linear_cross_entropy(hidden, lm_head.weight, labels)   # hidden (N,H), weight (V,H)
# bwd_logits_budget=192*1024*1024 (default) caps the (chunk,V) transient — the memory dial.
```

**XSA** — apply after value aggregation, before `o_proj`:
```python
attn_out = fused_xsa(attn_out, value_states)      # Y (B,H,S,D), V (B,Hkv,S,D); GQA handled in-kernel
```

**Causal-conv1d router** — projection only (apply your sigmoid/top-k in eager so autograd
handles them):
```python
logits = causal_conv1d_router(x, conv.weight)     # x (B,S,H), weight (E,H,K) -> (B*S, E)
```

**MoE (PolyGLU)** — the router stays in your model; pass its top-k indices/weights in. Expert
weights are stacked `(E, …)`; `act_codes` (E,) picks each expert's activation (0=SiLU, 1=ReLU²,
2=Tanh):
```python
from kernels import moe, moe_per_expert, moe_grouped
out = moe(hidden, top_k_idx, top_k_w, gate_up_proj, down_proj, act_codes)  # auto-dispatch
#   hidden (N,H)  idx/w (N,k)  gate_up_proj (E,2I,H)  down_proj (E,H,I)  act_codes (E,) int32
# force a path: moe_per_expert(...) (low token counts) / moe_grouped(...) (high token counts)
```

## What is XSA?

**XSA (Exclusive Self Attention)** — [arXiv:2603.09078](https://arxiv.org/abs/2603.09078) — is a
parameter-free, two-line post-processing step on the attention output. In standard self-attention
the output at position `i`, `y_i = Σ_j a_{i,j}·v_j`, always carries a persistent component along
the token's **own** value `v_i` (the diagonal `a_{i,i}` self-term — the "attention sink"). XSA
removes it by **vector rejection** of `y_i` from `v_i`:

```
z_i = y_i − (y_iᵀ · v̂_i) · v̂_i        # v̂ = v / ‖v‖₂  → z_i · v_i = 0 by construction
```

So each token's output is forced orthogonal to its own value: the residual stream is enriched only
by what *other* tokens bring along directions `v_i` doesn't already span. It touches neither the
softmax, the logits, nor the KV cache — pure post-processing on the attention output, `O(B·H·S·D)`
(negligible vs the attention matmuls). This kernel fuses the whole thing (normalize + dot + reject
+ GQA broadcast) into one fwd and one bwd kernel, broadcasting V across the GQA group in-kernel
(no `repeat_kv` copy, no normalized-V written to HBM).

**Backing** (from the source model's verification + ablation):
- **Formula-exact**: `max|z − (y − (y·v)v/‖v‖²)| = 2.4e-7` (fp32) vs the paper formula.
- **Orthogonality holds**: `max|z·v| = 3.8e-6` (≈0) — output is provably orthogonal to the self-value.
- **Length-generalization neutral & safe**: on a synthetic passkey probe (train @128 tok, eval to
  32× length, 3 seeds), SSMax-only `0.96` vs SSMax+XSA `0.94` — within seed noise, no extrapolation
  penalty. (Its intended benefit is *representational*, not retrieval — this ablation establishes it
  is safe to keep, not that it lifts a retrieval metric.)
- The fused kernel itself: ~2.6× fwd / ~2.5× fwd+bwd and ~25% less peak memory vs the materialized
  `repeat_kv` eager path (16384 tok, H32/Hkv8 shapes; see the bench table below).

## MoE (PolyGLU) — why it's the hard one

The other four kernels are *dense*: every row does identical work, so one kernel with a fixed grid
covers it. An MoE is *data-dependent* — the router sends each token to a runtime-chosen subset of
experts, so the work is a ragged collection of per-expert GEMMs whose sizes aren't known until the
router fires. That breaks single-kernel fusion at three points — "from weights, to dispatch, to the
final summed tensor":

1. **Dispatch (gather)** — tokens for expert `e` are scattered across the batch; you must gather
   them into a contiguous block before any GEMM can touch them.
2. **Ragged GEMM** — expert `e` gets `count[e]` tokens, different every step. A plain batched GEMM
   needs equal sizes; here each "batch" is a different M. You either loop (one GEMM per expert) or
   block-schedule a grouped GEMM over the sorted tokens.
3. **Combine (scatter)** — each token went to top-k experts, so the output is a *weighted sum* of k
   expert outputs scattered back to its row — an index-add reduction, not a plain write.

So a real MoE is a **pipeline** of fused stages wired by a sort, not one kernel. This ships two
expert-pipeline drop-ins (plus `moe()` which auto-picks):

- **`moe_per_expert`** — sort by expert, then per expert: cuBLAS gate_up GEMM → fused PolyGLU
  activation (Triton) → cuBLAS down GEMM → weighted scatter. Pure composition, autograd-correct by
  construction. Wins at **low** token counts (loop overhead is small).
- **`moe_grouped`** — ONE block-scheduled grouped-GEMM over all sorted tokens (Triton `tl.dot`) with
  a matched grouped-GEMM backward. Wins at **high** token counts; trades ~1.2× more memory (saves
  intermediates for backward).

**Why naive eager is so slow** — the hand-written MoE (`moe_eager`, the bench baseline) loops
experts, boolean-masks each (`idx == e`), gathers, runs two `F.linear`s + an unfused activation, and
scatters. It is slow for three compounding reasons: (a) the per-expert boolean-mask/index forces a
**GPU→CPU sync every iteration** — the launch queue drains E times per layer; (b) the GLU activation
is unfused elementwise kernels + an intermediate HBM write; (c) **zero GEMM batching** — E tiny GEMMs,
each under-utilizing the device. The fused paths kill all three: one sort instead of E masks, a
fused-activation Triton kernel, and (grouped) a single batched GEMM.

Runnable examples for the MoE kernel (auto-dispatch, forcing a path, and a full `nn.Module` MoE
layer with a training step) are in [`examples/moe_usage.py`](examples/moe_usage.py).

## Benchmarks

`python bench.py` runs all four (or `python bench.py ce xsa conv moe`): forward / backward /
forward+backward via `triton.testing.do_bench`, plus a grad-equivalence check and peak memory.

**`--compile`** (`python bench.py --compile`) wraps the kernel **and** eager forwards in
`torch.compile`. Compilation + Triton autotune happen during warmup, so they are excluded from the
timed step — you get the post-compile steady-state cost (industry-standard). Use this to get honest
numbers on your GPU, and to compare against the *compiled* eager baseline (e.g. compiled standard CE
closes most of the fused-CE time gap, while fused-CE keeps the memory win). `torch.compile` is broken
on some local setups — run `--compile` on the target GPU (T4 / Hopper).

fp16, torch 2.6 / triton 3.7 — speedup = eager ÷ kernel. Re-run on your own hardware: `python bench.py`.

| Kernel (shape) | fwd | bwd | fwd+bwd | peak mem | grad rel |
|---|---|---|---|---|---|
| fused-linear CE (N=4096, H=512, V=81000) | 5.0× | 8.4× | 7.1× | 2.1× less | 8.2e-4 |
| XSA (B=8, Hq=8, S=1024, D=128, Hkv=2) | 2.5× | 2.9× | 2.4× | 1.2× less | 1.0e-3 |
| causal-conv1d router (B=8, S=1024, H=512, E=11, K=4) | 3.4× | 5.2× | 3.9× | 1.2× less | 6.9e-4 |
| MoE per-expert (N=8192, H=512, I=768, E=9, k=2) | 1.6× | 2.1× | 1.9× | 1.05× less | 1.1e-3 |
| MoE grouped (N=8192, H=512, I=768, E=9, k=2) | 2.3× | 3.9× | 3.2× | 0.81× (more) | 1.2e-3 |

Notes:
- **`causal_conv1d_router` and `moe_grouped` use `tl.dot`** for their GEMMs; the others don't.
  `tl.dot` throughput varies a lot by architecture, so re-benchmark these two on your target —
  including **Hopper (H100, sm_90)** — before trusting their numbers. `moe_per_expert` uses cuBLAS
  GEMMs (only the activation is Triton) and ports cleanly.
- **MoE** speedups are vs a naive per-expert eager loop (see "why eager is so slow" above). `grouped`
  is faster but uses ~1.2× more memory (it saves intermediates for the backward); `per-expert` is the
  guaranteed-correct, memory-lean path and the better choice at low token counts.
- **fused-linear CE** speedups are vs naive eager (which materializes the full (N,V) logits). Against
  a `torch.compile`d standard CE the *time* win narrows (compiled std CE is fastest when the logits
  fit), but the **~3.4× memory saving stands** — and it's the only path that fits when standard CE
  OOMs (large N × large vocab). The real comparison is **vs Liger's fused-linear CE** (same niche) —
  see [CE vs Liger](#fused-linear-ce-vs-liger).
- conv-router grad_w has large *absolute* error (~0.25) but ~7e-4 *relative* (TF32 `tl.dot`
  long-reduction order); grad_x is exact.

## Reality check: vs `torch.compile` on T4

Measured **Tesla T4, fp16, `--compile` (both sides compiled)**, fwd+bwd — the production setting:

| kernel | fwd+bwd vs compiled eager | peak mem | verdict |
|---|---|---|---|
| **MoE per-expert** | **2.85×** | 1.08× less | **keep — real speed win** (compile can't fuse routing) |
| fused-linear CE (fused-fwd+bwd) | 0.76× (slower than compiled std CE) | **3.4× less** | **keep — memory/OOM, and beats Liger** (see below) |
| **XSA** | **1.15×** (fwd 1.20×) | 1.0× | **keep — beats inductor** (one fused kernel reads V once vs inductor's two) |
| causal-conv1d router | 0.75× | 1.23× less | fallback only — inductor uses cuDNN conv |
| MoE grouped | **0.10×** | 0.80× | **disabled on Turing** (tl.dot cliff); Ampere+ only |

Takeaway: under `torch.compile`, hand-written Triton beats inductor only where compile **structurally
can't help**: data-dependent MoE dispatch (speed), not materializing the (N,V) logits (CE memory), and
**XSA — where a single fused kernel reads V once and computes ‖v‖² inline, while inductor emits two
kernels and reads V twice** (a ~15–20% traffic saving a graph can't recover, since it won't fuse a
reduction's consumer back into the reduction). Pure *elementwise* ops with no such structure (SwiGLU;
conv, where inductor calls cuDNN) tie or lose — inductor already fuses them at the bandwidth ceiling,
and our `autograd.Function` is an opaque graph-break. The Ampere/eager table above overstates the
elementwise kernels — trust this one for a compiled pipeline.

### Fused-linear CE vs Liger

At the LM-head, compiled standard CE wins on raw speed *when the logits fit*. The fused-linear CE
exists for when they don't — and the established kernel for that niche is **Liger's
`LigerFusedLinearCrossEntropy`**. Measured **T4, fp16, `--compile`, N=16384 (B16×S1024), V=81000,
H=512**, fwd+bwd, compiled standard CE = 197 ms / 3075 MB as the reference:

| kernel | fwd+bwd | peak mem | grad_hidden vs fp32 | grad_weight vs fp32 |
|---|---|---|---|---|
| **ours @192MB** | **260 ms** | **904 MB** (3.4× less) | **1.9e-3** | 7.1e-4 |
| Liger @chunk1024 | 321 ms | 836 MB (3.7× less) | 1.1e-2 | 7.1e-4 |
| Liger @chunk2048 | 283 ms | 1083 MB (2.8× less) | 1.1e-2 | 7.1e-4 |

Ours @192MB is **19% faster than Liger@1024** at +68 MB, and **faster than Liger@2048 at 180 MB
less** — Pareto-ahead either way. Loss is bit-identical to Liger and matches fp32 eager to 1e-6, and
our `grad_hidden` is **~5× tighter to the fp32 reference** than Liger's (we compute the gradient in
the forward chunk loop and overwrite the live logit tile in place — no recompute GEMM, and the
softmax→grad path keeps more fp32 than Liger's). Reproduce: `python bench.py --compile ce_sweep liger_ce_sweep`.
The 192 MB default is the T4 latency knee from `ce_sweep` (flat 260–264 ms across 192–384 MB; the
curve is GEMM-bound, so a smaller budget just buys memory).

## A note on `triton.autotune`

These kernels **do** use `@triton.autotune` — but only keyed on *shape-stable* dimensions
(`K`, `N`, `H`, `I`), never on the per-step-varying token count `M`. That distinction matters for
MoE: autotuning on `M` would re-tune (and stall) every step because each expert's token count
changes per batch, and for the grouped path `BLOCK_M` is **pinned** to the tile schedule (autotuning
it would corrupt the precomputed tiling). So: autotune the dense/fixed dims, keep `BLOCK_M` fixed,
never key on `M`. The first call per new shape pays the tuning cost; with `--compile` that happens
during warmup and is excluded from the timed step.

## Requirements

`torch` (CUDA) + `triton`. That's it. Drop `kernels/` into your project; no install step.

MIT.
