# triton-kernel-fused

Four standalone, drop-in **fused Triton kernels** (forward **and** backward), each a swap-in
replacement for its PyTorch-eager equivalent. No framework, no `pip install` — copy the
`kernels/` folder into your project and import.

```python
from kernels import fused_swiglu, fused_linear_cross_entropy, fused_xsa, causal_conv1d_router
```

Every kernel is wrapped in a `torch.autograd.Function` (trains normally) and is
grad-equivalent to eager within fp16 tolerance (relative error < 1.5e-2, verified by `bench.py`).

## The kernels

| Kernel | Replaces | What it fuses |
|---|---|---|
| `fused_swiglu(gate_up)` | `silu(gate) * up` | SwiGLU activation + gradient, one kernel each. GEMMs stay cuBLAS. |
| `fused_linear_cross_entropy(hidden, weight, labels)` | `F.cross_entropy(hidden @ W.T, labels)` | LM-head GEMM + softmax-CE **without materializing the (N,V) logits** (cut-cross-entropy style, cuBLAS-chunked). |
| `fused_xsa(attn_out, value_states)` | Exclusive Self Attention rejection `y − (y·v̂)v̂` | normalize + dot + reject + GQA broadcast, one kernel each. No repeat_kv copy. |
| `causal_conv1d_router(x, weight)` | `pad → F.conv1d → reshape` (causal) | causal conv1d projection in native (B,S,H) layout, transpose-free fwd + bwd. |

## Usage

**SwiGLU** — operate on a concatenated `gate_up` (M, 2·I), produced by one fused
`gate_up_proj` (Linear → 2·I). Or use the bundled module:
```python
from kernels import fused_swiglu, FusedSwiGLUMLP
out = fused_swiglu(gate_up)                       # (M, I)
mlp = FusedSwiGLUMLP(hidden=512, intermediate_size=1408)   # drop-in nn.Module
# port weights from a separate gate/up/down model:
mlp.load_from_gate_up(m.gate_proj.weight, m.up_proj.weight, m.down_proj.weight)
```

**Fused-linear CE** — pass the pre-logit hidden states and the LM-head weight directly:
```python
loss = fused_linear_cross_entropy(hidden, lm_head.weight, labels)   # hidden (N,H), weight (V,H)
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

## Benchmarks

`python bench.py` runs all four (or `python bench.py swiglu ce xsa conv`): forward / backward /
forward+backward via `triton.testing.do_bench`, plus a grad-equivalence check and peak memory.

fp16, torch 2.6 / triton 3.7 — speedup = eager ÷ kernel. Re-run on your own hardware: `python bench.py`.

| Kernel (shape) | fwd | bwd | fwd+bwd | peak mem | grad rel |
|---|---|---|---|---|---|
| SwiGLU (M=8192, I=768) | 2.1× | 5.2× | 3.4× | 1.2× less | 5.7e-4 |
| fused-linear CE (N=4096, H=512, V=81000) | 5.0× | 8.4× | 7.1× | 2.1× less | 8.2e-4 |
| XSA (B=8, Hq=8, S=1024, D=128, Hkv=2) | 2.5× | 2.9× | 2.4× | 1.2× less | 1.0e-3 |
| causal-conv1d router (B=8, S=1024, H=512, E=11, K=4) | 3.4× | 5.2× | 3.9× | 1.2× less | 6.9e-4 |

Notes:
- **`causal_conv1d_router` uses `tl.dot`** for its GEMMs; the others don't. `tl.dot` throughput
  varies a lot by architecture, so re-benchmark the conv router on your target — including **Hopper
  (H100, sm_90)** — before trusting its numbers.
- **fused-linear CE** speedups are vs naive eager (which materializes the full (N,V) logits). Against
  a `torch.compile`d standard CE the *time* win narrows toward a tie, but the **~2× memory saving
  stands** — and it's the only path that fits when standard CE OOMs (large N × large vocab).
- conv-router grad_w has large *absolute* error (~0.25) but ~7e-4 *relative* (TF32 `tl.dot`
  long-reduction order); grad_x is exact.

## Requirements

`torch` (CUDA) + `triton`. That's it. Drop `kernels/` into your project; no install step.

MIT.
