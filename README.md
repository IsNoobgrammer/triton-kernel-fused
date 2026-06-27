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

## Benchmarks

`python bench.py` runs all four (or `python bench.py swiglu ce xsa conv`): forward / backward /
forward+backward via `triton.testing.do_bench`, plus a grad-equivalence check and peak memory.

**RTX 3050 Laptop (Ampere sm_86), fp16, torch 2.6 / triton 3.7** — speedup = eager ÷ kernel:

| Kernel (shape) | fwd | bwd | fwd+bwd | peak mem | grad rel |
|---|---|---|---|---|---|
| SwiGLU (M=8192, I=768) | 2.1× | 5.2× | 3.4× | 1.2× less | 5.7e-4 |
| fused-linear CE (N=4096, H=512, V=81000) | 5.0× | 8.4× | 7.1× | 2.1× less | 8.2e-4 |
| XSA (B=8, Hq=8, S=1024, D=128, Hkv=2) | 2.5× | 2.9× | 2.4× | 1.2× less | 1.0e-3 |
| causal-conv1d router (B=8, S=1024, H=512, E=11, K=4) | 3.4× | 5.2× | 3.9× | 1.2× less | 6.9e-4 |

## ⚠️ Read before trusting the numbers

- **These are Ampere (sm_86) numbers. Re-run `bench.py` on YOUR GPU.** Triton `tl.dot` GEMMs run
  far slower on **Turing (T4, sm_75)** than on Ampere+. Of these four, **only `causal_conv1d_router`
  uses `tl.dot`** in its hot path — on a T4 its win will shrink and may even regress vs eager;
  benchmark it there before adopting. SwiGLU (elementwise + cuBLAS), CE (cuBLAS-chunked), and XSA
  (elementwise, no GEMM) carry no `tl.dot` and port cleanly.
- **The CE speedups are vs *naive* eager** (which materializes the full (N,V) logits). Against
  `torch.compile`d standard CE the *time* win narrows toward a tie — but the **~2× memory saving
  stands**, and it is the only option that fits when standard CE OOMs (large N × large vocab).
- **conv-router grad_w**: absolute error looks large (~0.25) because grad_w magnitudes are large;
  *relative* error is ~7e-4 (TF32 `tl.dot` long-reduction order). `grad_x` is exact.
- All correctness is fp16 (`atol`-equivalent rel < 1.5e-2). For fp32 use, expect ~1e-5.

## Requirements

`torch` (CUDA) + `triton`. That's it. Drop `kernels/` into your project; no install step.

MIT.
