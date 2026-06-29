# Contributing

Thanks for considering a contribution. The full guide — with the design philosophy, conventions, and a PR
checklist — lives at **[isnoobgrammer.github.io/triton-kernel-fused/contributing](https://isnoobgrammer.github.io/triton-kernel-fused/contributing/)**.
The essentials:

## Kernels are organized by GPU architecture

Performance does not transfer across GPU classes, so each CUDA compute capability gets its own
self-contained package and its own measured numbers:

```
kernels/
  __init__.py          # namespace + AVAILABLE_ARCHS (no auto-import)
  sm75/                # Turing / Tesla T4 — the reference arch (everything is tuned here today)
    __init__.py        # the public API for this arch
    cross_entropy.py  xsa.py  router.py  moe.py  muon.py
  sm80/                # Ampere — add yours here
  sm90/                # Hopper — ...
```

Import by naming the arch explicitly (`sm_7.5` → `sm75`, `sm_8.0` → `sm80`, ...):

```python
from kernels.sm75 import fused_xsa, moe, FusedMuon       # an arch's public API
from kernels.sm75.moe import moe_grouped                  # advanced / private symbols
```

Find your arch with `torch.cuda.get_device_capability()`.

## The bar

Only fuse where `torch.compile` **structurally can't** — data-dependent routing, terminal-loss fusion,
read-once reductions, native-op seams, or launch-overhead/batched-GEMM collapse. A kernel that reads and
writes the same bytes as the compiled baseline will not be faster; a kernel that merely ties it is not a
contribution.

## Adding a kernel

1. Locate or create `kernels/sm<XX>/` (with an `__init__.py`); add the arch to `AVAILABLE_ARCHS` in
   `kernels/__init__.py`.
2. Implement forward + backward as a `torch.autograd.Function` with a thin public wrapper. Keep files
   self-contained (no cross-kernel imports).
3. Export the public symbols from `kernels/sm<XX>/__init__.py`.
4. **Prove correctness first** — forward output and all gradients vs the eager reference. Correctness is
   the gate and is never traded for speed.
5. Add a `bench_<name>` entry to `bench.py` comparing against the `torch.compile` baseline with a parity
   check, and measure speed on the **target GPU**.

## PR checklist

- [ ] Kernel is in the correct `kernels/sm<XX>/` package and exported from its `__init__.py`.
- [ ] Forward and all gradients pass parity against eager.
- [ ] `bench.py` entry compares against the `torch.compile` baseline and reports parity.
- [ ] Numbers are labelled with GPU class, dtype, and shape, and measured on that GPU.
- [ ] The kernel exploits a real structural seam (not a tie with `torch.compile`).
- [ ] No emoji in code, comments, or docs.

## Local development

```bash
uv sync                                # CUDA torch (cu124) + triton
uv run python bench.py --compile xsa   # benchmark + parity check
cd docs && npm install && npm run dev  # docs site
```
