<div align="center">

<img src="assets/logo.svg" alt="triton-kernel-fused logo" width="120" height="120" />

<h1>triton-kernel-fused</h1>

<p><b>Fused Triton kernels (forward <i>and</i> backward) that beat <code>torch.compile</code> where it structurally can — and don't pretend to where it can't.</b></p>

<p>
Drop-in replacements for eager PyTorch blocks in transformer training on NVIDIA GPUs.<br/>
Tuned and verified on the Turing&nbsp;/&nbsp;Tesla&nbsp;T4 class (<code>sm_75</code>) — correctness is always the gate.
</p>

<p>
<a href="https://isnoobgrammer.github.io/triton-kernel-fused/"><img src="https://img.shields.io/badge/Read_the_Docs-Live-7C3AED?style=for-the-badge&logo=astro&logoColor=white" alt="Documentation" /></a>
&nbsp;
<a href="#benchmarking"><img src="https://img.shields.io/badge/Benchmark-bench.py-1F2937?style=for-the-badge&logo=nvidia&logoColor=76B900" alt="Benchmark" /></a>
&nbsp;
<a href="https://github.com/IsNoobgrammer/triton-kernel-fused"><img src="https://img.shields.io/badge/Source-GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub source" /></a>
</p>

<p>
<img src="https://img.shields.io/badge/Triton-%E2%89%A53.0-2C2C2C?style=flat-square" alt="Triton >= 3.0" />
<img src="https://img.shields.io/badge/PyTorch-%E2%89%A52.4-EE4C2C?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch >= 2.4" />
<img src="https://img.shields.io/badge/CUDA-Turing_sm__75-76B900?style=flat-square&logo=nvidia&logoColor=white" alt="CUDA Turing sm_75" />
<img src="https://img.shields.io/badge/Python-%E2%89%A53.10-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python >= 3.10" />
<img src="https://img.shields.io/badge/License-MIT-A78BFA?style=flat-square" alt="License: MIT" />
</p>

</div>

---

## Why these kernels

These are **not** a general "compile everything" layer. `torch.compile` already saturates memory
bandwidth on plain elementwise ops — you don't beat it there, and we don't try. Each kernel is a tested
replacement for an eager PyTorch block, ships with a custom autograd backward, and targets the places
where the compiler *structurally* can't fuse:

- **data-dependent routing** (MoE dispatch) — the shape isn't known until runtime,
- **terminal-loss fusion** (cross-entropy) — never materialize the giant `(N, V)` logits,
- **read-once reductions** (XSA) — the compiler emits two passes over `V`; we read it once,
- **native-op seams** (`topk` in the router) — ops the compiler must keep as separate library calls,
- **launch-overhead collapse + bigger batched GEMMs** (Muon) — fuse N per-param launches into a few.

Numerical parity (forward output **and** gradients) against the eager reference is never traded for speed.

## Kernels

| Kernel | Replaces | The edge `torch.compile` can't get | T4 result |
|---|---|---|---|
| [`fused_linear_cross_entropy`](https://isnoobgrammer.github.io/triton-kernel-fused/kernels/cross-entropy/) | `lm_head` + `F.cross_entropy` | gradient computed in the forward chunk loop — **never materializes the `(N, V)` logits** | **~3.4× less peak memory** (trains where standard CE OOMs); matches Liger |
| [`fused_xsa`](https://isnoobgrammer.github.io/triton-kernel-fused/kernels/xsa/) | Exclusive Self-Attention correction | one fused kernel reads `V` **once** (the compiler emits two passes) | **~1.15×** fwd+bwd, grad-exact |
| [`FusedMuon`](https://isnoobgrammer.github.io/triton-kernel-fused/kernels/muon/) | Polar-Express Muon optimizer step | `foreach` + `baddbmm` collapse the per-param launch tax; experts batch into one GEMM | **~1.09×** (75M) / **~1.05×** (300M), peak mem ≤ baseline |
| [`fused_router`](https://isnoobgrammer.github.io/triton-kernel-fused/kernels/router/) | conv router (conv → sigmoid → bias → topk → gather) | fuses native `topk` into an in-register epilogue + a merged backward | **~1.11–1.17×** fwd+bwd, exact grads, mem parity |
| [`moe`](https://isnoobgrammer.github.io/triton-kernel-fused/kernels/moe/) | masked per-expert MoE combine | fuses data-dependent dispatch + cuBLAS GEMMs + activation + weighted scatter | **~2.87×** (the data-dependent edge survives compile) |

> **Performance is architecture-specific.** All numbers above are measured on a Tesla T4 (`sm_75`)
> against a `torch.compile`'d eager baseline. Numbers from one GPU class do not transfer to another —
> always re-benchmark on your target hardware. See [Benchmarking](https://isnoobgrammer.github.io/triton-kernel-fused/concepts/benchmarking/).

`fused_router` documents a **conv** router (sigmoid-gate, no-activation) — see its page for which
configurations are covered. `moe` is shaped for the **PolyGLU** expert layout (fused gate+up, per-expert
activation codes) used in BiBo; the per-expert path is general, the activation set is opinionated.

## Requirements

- NVIDIA GPU with CUDA (developed and tuned on Tesla T4, `sm_75`)
- Python ≥ 3.10
- PyTorch ≥ 2.4 with CUDA, Triton ≥ 3.0 (bundled in the CUDA PyTorch wheel)

## Install

The kernels need no installation to be *used* — copy the `kernels/` folder into your project and import
it. To run the benchmarks and examples from a clone:

```bash
uv sync                 # creates .venv with CUDA torch + triton
uv run python bench.py
```

Or, with an existing CUDA PyTorch environment, just run `python bench.py` from the repo root.

## Quick usage

### Fused-linear cross-entropy — trains where standard CE OOMs

```python
import torch
from kernels import fused_linear_cross_entropy

hidden  = torch.randn(16384, 512, device="cuda", dtype=torch.float16, requires_grad=True)
lm_head = torch.randn(81000, 512, device="cuda", dtype=torch.float16, requires_grad=True)
labels  = torch.randint(0, 81000, (16384,), device="cuda")

loss = fused_linear_cross_entropy(hidden, lm_head, labels)   # mean CE, no (N,V) logits materialized
loss.backward()
```

`bwd_logits_budget` (bytes) caps the `(chunk, V)` transient to trade peak memory against launch count;
`ignore_index` matches `F.cross_entropy` (default `-100`).

### XSA correction

```python
import torch
from kernels import fused_xsa

attn_output  = torch.randn(16, 4, 1024, 128, device="cuda", dtype=torch.float16)  # (B, H, S, D)
value_states = torch.randn(16, 2, 1024, 128, device="cuda", dtype=torch.float16)  # (B, Hkv, S, D), GQA
corrected = fused_xsa(attn_output, value_states)                                  # (B, H, S, D)
```

### Fused Muon optimizer

```python
import torch
from kernels import FusedMuon

# 2D/3D weights -> Muon; route 1D params (norms, biases) and embeddings to AdamW upstream.
opt = FusedMuon(model.muon_params(), lr=0.02, momentum=0.95, weight_decay=0.01)

loss = model(batch).loss
loss.backward()
opt.step()
opt.zero_grad(set_to_none=True)
```

`ns_dtype=torch.float16` is the T4 default (engages fp16 tensor cores); use `torch.float32` for a
numerically-safe fallback, `torch.bfloat16` only on Ampere/Hopper. `DistributedMuon` gives a bit-identical
round-robin variant for DDP. See the [Muon page](https://isnoobgrammer.github.io/triton-kernel-fused/kernels/muon/) for `ns_batch_elems`, `scale_mode`,
and the opt-in fast mode.

### Conv MoE router (sigmoid gate)

```python
import torch
from kernels import fused_router, router_bias_update

x      = torch.randn(16, 1024, 512, device="cuda", dtype=torch.float16)  # (B, S, H)
weight = torch.randn(11, 512, 4, device="cuda", dtype=torch.float16)     # (E, H, K) — nn.Conv1d weight
bias   = torch.zeros(11, device="cuda")                                  # selection bias (no grad)

idx, weights = fused_router(x, weight, bias, top_k=2, num_experts=11)    # (B,S,k) long, (B,S,k) fp32

# Auxiliary-loss-free load balancing (heuristic, off the autograd graph):
idx, weights, counts = fused_router(x, weight, bias, top_k=2, num_experts=11, return_counts=True)
router_bias_update(bias, counts, u=0.001)                                # b += u·sign(mean − load)
```

### PolyGLU MoE combine

```python
import torch
from kernels import moe

# top-k indices/weights come from YOUR router; expert weights are stacked per expert:
hidden    = torch.randn(16384, 512, device="cuda", dtype=torch.float16)    # (N, H)
gate_up   = torch.randn(11, 1536, 512, device="cuda", dtype=torch.float16) # (E, 2*I, H) fused gate+up
down      = torch.randn(11, 512, 768, device="cuda", dtype=torch.float16)  # (E, H, I)
act_codes = torch.zeros(11, dtype=torch.int32, device="cuda")              # per expert: 0=SiLU 1=ReLU² 2=Tanh

out = moe(hidden, idx, weights, gate_up, down, act_codes)                  # (N, H) combined output
out.sum().backward()
```

See [`examples/moe_usage.py`](examples/moe_usage.py) for a complete runnable example.

## Benchmarking

`bench.py` times each kernel against a `torch.compile`'d eager baseline (the industry steady state) and
checks gradient parity in the same run:

```bash
python bench.py --compile ce_fit     # fused-linear cross-entropy
python bench.py --compile xsa        # XSA
python bench.py --compile muon       # Fused Muon (add muon_big for the ~300M regime)
python bench.py --compile router     # conv MoE router
python bench.py --compile moe        # PolyGLU MoE
```

Run on the target GPU — kernel performance is architecture-specific, so numbers from one GPU class do
not transfer to another.

## Documentation

The full docs — per-kernel API, the design rationale behind each win, and benchmarking guidance — live at
**[isnoobgrammer.github.io/triton-kernel-fused](https://isnoobgrammer.github.io/triton-kernel-fused/)**.

It is an [Astro Starlight](https://starlight.astro.build) site under [`docs/`](docs/), published to GitHub
Pages automatically on every push to `master`. To preview or edit it locally:

```bash
cd docs && npm install && npm run dev      # http://localhost:4321/triton-kernel-fused
```

## License

MIT
