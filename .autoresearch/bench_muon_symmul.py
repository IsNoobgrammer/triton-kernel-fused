"""OPTIMIZER-LEVEL eval — the regime where FusedMuon's levers actually fire (vs the NS micro).

The NS micro (bench_symmul.py) isolates the symmul kernel on ONE matrix (B=1), where FusedMuon's
foreach + batched-same-shape-state levers are inert, so fused == compiled there. This bench measures
the full optimizer .step() over a BIG-MODEL param set (d=4096, large matrices, gram>=2048 so symmul
fires), where:
  - compiled : per-param NS, torch.compile'd  (runs NS once per parameter — no cross-param batching)
  - fused    : FusedMuon (foreach launch-collapse + ONE batched bmm over same-shape params + baddbmm)
  - amalg    : AmalgamatedMuon (fused's batching + the symmetric symmul FLOP cut on the NS GEMMs)

This is the multiplicative test: fused beats compiled via batching; amalg should beat fused via symmul
ON TOP, so amalg vs compiled = batching x symmul. Memory also flips here: compiled allocates per-param.

Metrics: full step ms (do_bench), peak MB, parity amalg-vs-fused (max|Dp| after 5 steps, fp16 tol).
Run:  <venv>/python .autoresearch/bench_muon_symmul.py [--dtype fp16] [--layers 4] [--d 4096]
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.optim as optim
from triton.testing import do_bench

from kernels.sm75.muon import _PE_COEFFS, newton_schulz
from kernels.sm120.muon import FusedMuon, AmalgamatedMuon

DEV = "cuda"
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}


class CompiledMuon(optim.Optimizer):
    """Per-param Polar-Express Muon with the NS iteration torch.compile'd — the 'compiled' baseline.
    No cross-param batching: NS runs once per parameter (compiled+cached per shape)."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0,
                 ns_dtype=torch.float16):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        self.ns_dtype = ns_dtype
        self._ns = torch.compile(newton_schulz)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            for p in group["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                g = p.grad.to(self.ns_dtype)
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                u = g.add(buf, alpha=momentum) if nesterov else buf
                out = self._ns(u, _PE_COEFFS, self.ns_dtype)
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                if wd:
                    p.mul_(1.0 - lr * wd)
                p.add_(out.to(p.dtype).reshape(p.shape), alpha=-lr * scale)


def make_big_shapes(layers=4, d=4096, ffn=11008):
    """Big-model Muon param inventory: large matrices (gram = min(rows,cols) >= 2048 -> symmul fires).
    Attention forms a big same-shape group (4*layers of d x d) -> the batching lever's sweet spot."""
    sh = []
    for _ in range(layers):
        sh += [(d, d), (d, d), (d, d), (d, d)]          # q, k, v, o
        sh += [(2 * ffn, d), (d, ffn)]                  # dense MLP gate_up, down
    return sh


def make_params(shapes, dtype, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=dtype) for s in shapes]


def prime_grads(params, dtype, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    for p in params:
        p.grad = torch.randn(*p.shape, generator=g, device=DEV, dtype=dtype)


def parity(make_a, make_b, shapes, dtype, steps=5):
    pa, pb = make_params(shapes, dtype, 0), make_params(shapes, dtype, 0)
    oa, ob = make_a(pa), make_b(pb)
    gg = torch.Generator(device=DEV).manual_seed(7)
    worst = 0.0
    for _ in range(steps):
        grads = [torch.randn(*p.shape, generator=gg, device=DEV, dtype=dtype) for p in pa]
        for p, gr in zip(pa, grads): p.grad = gr.clone()
        for p, gr in zip(pb, grads): p.grad = gr.clone()
        oa.step(); ob.step()
        worst = max(worst, max((x.float() - y.float()).abs().max().item() for x, y in zip(pa, pb)))
    return worst


def measure(make_opt, shapes, dtype):
    params = make_params(shapes, dtype, 0)
    opt = make_opt(params)
    prime_grads(params, dtype)
    opt.step()                                          # warm (compile/autotune)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    ms = do_bench(lambda: (prime_grads(params, dtype), opt.step()))
    peak = torch.cuda.max_memory_allocated() / 1e6
    return ms, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=4096)
    ap.add_argument("--ffn", type=int, default=11008)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    dt = _DTYPES[args.dtype]

    shapes = make_big_shapes(args.layers, args.d, args.ffn)
    nparam = sum(torch.tensor(s).prod().item() for s in shapes)
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}")
    print(f"big-model Muon set: {len(shapes)} tensors, {nparam/1e6:.0f}M params, d={args.d}, dtype={args.dtype}\n")

    mk_compiled = lambda ps: CompiledMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=dt)
    mk_fused = lambda ps: FusedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=dt)
    mk_amalg = lambda ps: AmalgamatedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=dt)

    print("PARITY amalg vs fused (max|Dp| after 5 steps):")
    dpar = parity(mk_amalg, mk_fused, shapes, dt)
    print(f"  {dpar:.2e}  {'PASS' if dpar < 2e-2 else 'FAIL'}  (fp16 NS tolerance < 2e-2)\n")

    print("STEP ms (do_bench) + peak MB:")
    tc, mc = measure(mk_compiled, shapes, dt)
    tf, mf = measure(mk_fused, shapes, dt)
    ta, ma = measure(mk_amalg, shapes, dt)
    print(f"  compiled  {tc:8.3f} ms  {mc:8.1f} MB  (1.00x)")
    print(f"  fused     {tf:8.3f} ms  {mf:8.1f} MB  ({tc/tf:.2f}x vs compiled)")
    print(f"  amalg     {ta:8.3f} ms  {ma:8.1f} MB  ({tc/ta:.2f}x vs compiled, {tf/ta:.2f}x vs fused)")
    print(f"\n  HEADLINE: amalg vs compiled = {tc/ta:.2f}x  | amalg mem/compiled = {ma/mc:.2f}x "
          f"({'<=' if ma <= mc else 'OVER'} compiled)")


if __name__ == "__main__":
    main()
