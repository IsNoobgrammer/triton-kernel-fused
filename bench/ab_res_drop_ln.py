"""Paired, interleaved A/B of res_drop_ln fwd+bwd: OLD (a file, e.g. git show HEAD~1:kernels/sm120/res_drop_ln.py)
vs the current kernels/sm120/res_drop_ln.py, at a 1200 s batch of a d=1024 encoder (Nemotron / Parakeet), on a GPU that
may be shared (interleaving cancels most contention). Also checks the two give the same grads (fp32 sum order only).

    python bench/ab_res_drop_ln.py /tmp/old_rdl.py [R C]
"""
import importlib.util
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton

import kernels.sm120.res_drop_ln as new


def load(path):
    spec = importlib.util.spec_from_file_location("old_rdl", path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


old = load(sys.argv[1])
R, C = (int(sys.argv[2]), int(sys.argv[3])) if len(sys.argv) > 3 else (15700, 1024)
res = torch.randn(R, C, device="cuda").requires_grad_()
x = torch.randn(R, C, device="cuda", dtype=torch.bfloat16).requires_grad_()
w, b = (1 + 0.1 * torch.randn(C, device="cuda")).requires_grad_(), (0.1 * torch.randn(C, device="cuda")).requires_grad_()
g_r, g_y = torch.randn(R, C, device="cuda"), torch.randn(R, C, device="cuda")


def fb(mod):
    r, y = mod.res_dropout_layernorm(res, x, w, b, 1e-5, 0.1, 0.5, 123, torch.float32)
    torch.autograd.backward((r, y), (g_r, g_y))


def grads(mod):
    for t in (res, x, w, b):
        t.grad = None
    fb(mod)
    return [t.grad.clone() for t in (res, x, w, b)]


go, gn = grads(old), grads(new)
rel = lambda a, b: ((a.double() - b.double()).norm() / b.double().norm()).item()
print("grads new vs old: " + "  ".join(f"{n} {rel(a, b):.1e}" for n, a, b in zip(("dres", "dx", "dw", "db"), gn, go)))
to, tn = [], []
for _ in range(7):
    to.append(triton.testing.do_bench(lambda: fb(old), warmup=5, rep=60, grad_to_none=[res, x, w, b]))
    tn.append(triton.testing.do_bench(lambda: fb(new), warmup=5, rep=60, grad_to_none=[res, x, w, b]))
med = lambda v: sorted(v)[len(v) // 2]
print(f"AB res_drop_ln R={R} C={C} fwd+bwd: old {med(to):.3f} ms  new {med(tn):.3f} ms  ({med(to) / med(tn):.3f}x)  "
      f"per-rep old {[round(v, 3) for v in to]} new {[round(v, 3) for v in tn]}")
