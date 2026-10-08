"""res_drop_ln (one conformer boundary) vs eager, forward and backward, at the real shape (15000 rows x 512:
a 1200 s batch = 15000 encoder frames), autocast dtypes (residual fp32, x bf16, y fp32), dropout 0.1, factor 0.5.
Sweeps rows-per-program x warps for each kernel.

    python bench/bench_res_drop_ln.py
"""
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import triton

import kernels.sm120.res_drop_ln as rdl

R, C = 15000, 512
res = torch.randn(R, C, device="cuda").requires_grad_()
x = torch.randn(R, C, device="cuda", dtype=torch.bfloat16).requires_grad_()
w, b = torch.ones(C, device="cuda").requires_grad_(), torch.zeros(C, device="cuda").requires_grad_()
g_r, g_y = torch.randn(R, C, device="cuda"), torch.randn(R, C, device="cuda")
drop = torch.nn.Dropout(0.1)


def eager():
    r = res + drop(x) * 0.5
    return r, F.layer_norm(r, (C,), w, b, 1e-5)


def ours():
    return rdl.res_dropout_layernorm(res, x, w, b, 1e-5, 0.1, 0.5, 123, torch.float32)


def time_fb(fn):
    fw = triton.testing.do_bench(lambda: fn(), warmup=10, rep=100)
    def fb():
        r, y = fn()
        torch.autograd.backward((r, y), (g_r, g_y))
    tot = triton.testing.do_bench(fb, warmup=10, rep=100, grad_to_none=[res, x, w, b])
    return fw, tot - fw


fe, be = time_fb(eager)
print(f"eager: fwd {1000 * fe:6.1f} us  bwd {1000 * be:6.1f} us  (x85 per step: {85 * (fe + be):.1f} ms)")
best = {}
for which in ("fwd", "bwd"):
    for br, nw in itertools.product((1, 2, 4, 8, 16), (2, 4, 8)):
        rdl._CFG = {"fwd": None, "bwd": None, which: (br, nw)}
        f, bw = time_fb(ours)
        t = f if which == "fwd" else bw
        best.setdefault(which, []).append((t, br, nw))
    best[which].sort()
    print(f"{which}: best " + "  ".join(f"BR{br}/w{nw} {1000 * t:.1f}us" for t, br, nw in best[which][:4]))
rdl._CFG = {"fwd": best["fwd"][0][1:], "bwd": best["bwd"][0][1:]}
f, bw = time_fb(ours)
print(f"ours (best): fwd {1000 * f:6.1f} us  bwd {1000 * bw:6.1f} us  (x85 per step: {85 * (f + bw):.1f} ms)  "
      f"cfg {rdl._CFG}")
