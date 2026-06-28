# triton-kernel-fused

Drop-in fused Triton kernels (forward **and** backward) for transformer training on NVIDIA GPUs,
tuned and verified on the Turing/T4 class (`sm_75`). Each kernel is a tested replacement for an eager
PyTorch block and ships with a custom autograd backward.

These are not a general "compile everything" layer. `torch.compile` already saturates memory bandwidth
on plain elementwise ops — you don't beat it there. These kernels target the places where the compiler
*structurally* can't: data-dependent routing, terminal-loss fusion, read-once reductions, and
native-op seams (e.g. `topk`) that the compiler must keep as separate library calls.

| Kernel | Replaces | Why it wins over `torch.compile` |
|---|---|---|
| `fused_linear_cross_entropy` | `lm_head` + `F.cross_entropy` | gradient computed in the forward chunk loop — never materializes the `(N, V)` logits (trains where standard CE OOMs) |
| `moe` / `moe_per_expert` | masked per-expert MoE combine | fuses data-dependent dispatch + cuBLAS GEMMs + activation + weighted scatter; the compiler can't fuse routing |
| `fused_router` | conv router (conv → sigmoid → bias → topk → gather) | one fused epilogue replaces native `topk`+gather; merged backward removes the unfused glue |
| `fused_xsa` | Exclusive Self-Attention correction | one fused kernel reads `V` once (the compiler emits two passes) |

All kernels are validated for numerical parity (forward output and gradients) against the eager
reference; correctness is the gate, never traded for speed.

## Requirements

- NVIDIA GPU with CUDA (developed/tuned on Tesla T4, `sm_75`)
- Python ≥ 3.10, PyTorch ≥ 2.4 with CUDA, Triton ≥ 3.0 (bundled in the CUDA PyTorch wheel)

## Install

The kernels need no installation to be used — copy the `kernels/` folder into your project and import
it. To run the benchmarks and examples from a clone:

```bash
uv sync                 # creates .venv with CUDA torch + triton
uv run python bench.py
```

Or, with an existing CUDA PyTorch environment, just run `python bench.py` from the repo root.

## Usage

### Fused-linear cross-entropy

```python
import torch
from kernels import fused_linear_cross_entropy

hidden  = torch.randn(16384, 512, device="cuda", dtype=torch.float16, requires_grad=True)
lm_head = torch.randn(81000, 512, device="cuda", dtype=torch.float16, requires_grad=True)
labels  = torch.randint(0, 81000, (16384,), device="cuda")

loss = fused_linear_cross_entropy(hidden, lm_head, labels)   # mean CE, no (N,V) logits materialized
loss.backward()
```

`bwd_logits_budget` (bytes) caps the chunk size to trade peak memory against launch count;
`ignore_index` matches `F.cross_entropy`.

### Conv MoE router (MiMo-V2.5 / DeepSeek-V3 sigmoid gate)

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

The fused path covers the conv + sigmoid-gate + no-activation configuration. `norm_topk_prob` and
`routed_scaling_factor` are applied in eager so autograd carries their Jacobian.

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

### XSA correction

```python
import torch
from kernels import fused_xsa

attn_output  = torch.randn(16, 4, 1024, 128, device="cuda", dtype=torch.float16)  # (B, H, S, D)
value_states = torch.randn(16, 2, 1024, 128, device="cuda", dtype=torch.float16)  # (B, Hkv, S, D), GQA
corrected = fused_xsa(attn_output, value_states)                                  # (B, H, S, D)
```

## Benchmarking

`bench.py` times each kernel against a `torch.compile`'d eager baseline (the industry steady state) and
checks gradient parity in the same run:

```bash
python bench.py --compile router     # conv MoE router
python bench.py --compile moe        # PolyGLU MoE
python bench.py --compile ce_fit     # fused-linear cross-entropy
python bench.py --compile xsa        # XSA
```

Run on the target GPU — kernel performance is architecture-specific, so numbers from one GPU class do
not transfer to another.

## License

MIT
