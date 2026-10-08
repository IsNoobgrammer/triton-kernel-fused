"""relpos_attention vs NeMo's rel_pos attention core under bf16 autocast (what the run1 model runs), fwd and bwd,
at the Lhotse 1200 s bucket shapes (12.5 frames/s, 8 heads x 64), context [70, 13] (training's first choice), with
attention dropout 0.1. bf16 inputs (what the model feeds it under autocast).

    python bench/bench_relpos_attn.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton

from kernels.sm120.relpos_attn import relpos_attention
from parity_check.parity_relpos_attn import nemo_core, nemo_mask

torch.backends.cuda.matmul.allow_tf32 = True
H, D = 8, 64
for name, B, T in (("short 4 s", 240, 50), ("mid 8 s", 150, 100), ("bench_layer batch", 80, 189), ("long 25 s", 48, 312)):
    g = torch.Generator(device="cuda").manual_seed(0)
    xs = [torch.randn(B, T, H, D, device="cuda", generator=g) for _ in range(3)]
    xs += [torch.randn(2 * T - 1, H, D, device="cuda", generator=g), 0.3 * torch.randn(H, D, device="cuda"),
           0.3 * torch.randn(H, D, device="cuda")]
    xs = [(x.to(torch.bfloat16) if i < 4 else x).requires_grad_() for i, x in enumerate(xs)]   # model dtypes
    L = (T * (0.85 + 0.15 * torch.rand(B, device="cuda", generator=g))).long().clamp(1, T)
    mask = nemo_mask(T, L, 70, 13)
    drop = torch.nn.Dropout(0.1)
    def nemo_amp():
        with torch.autocast("cuda", dtype=torch.bfloat16):                  # what the model runs
            return nemo_core(*xs, mask)
    impls = {"nemo": nemo_amp,
             "ours": lambda: relpos_attention(*xs, L, 70, 13, dropout=0.1)}
    W = torch.randn(B, T, H, D, device="cuda")
    res = {}
    for k, fn in impls.items():
        fw = triton.testing.do_bench(fn, warmup=5, rep=40)
        tot = triton.testing.do_bench(lambda: (fn().float() * W).sum().backward(), warmup=5, rep=40, grad_to_none=xs)
        torch.cuda.reset_peak_memory_stats()
        (fn().float() * W).sum().backward()
        res[k] = (fw, tot - fw, torch.cuda.max_memory_allocated() / 2 ** 30)
    n, o = res["nemo"], res["ours"]
    print(f"{name:18s} B={B:3d} T={T:3d}  nemo fwd {n[0]:6.2f} bwd {n[1]:6.2f} ms ({n[2]:5.1f} GB) | ours fwd {o[0]:6.2f} "
          f"bwd {o[1]:6.2f} ms ({o[2]:5.1f} GB) | x{n[0] / o[0]:.1f} fwd x{n[1] / o[1]:.1f} bwd x{(n[0] + n[1]) / (o[0] + o[1]):.1f} total",
          flush=True)
