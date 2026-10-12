"""kl_div_dense / kl_div_topk vs eager (log_softmax x2 + kl + autograd), fwd+bwd, at a 1200 s ASR batch's frame count
(N = 15700 frames) and V = 1025 (Parakeet / Nemotron vocabulary + blank), bf16 student logits, fp32 teacher.

    python bench/bench_kl_div.py [N V]
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton

from kernels.sm120.kl_div import kl_div_dense, kl_div_topk

N, V = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (15700, 1025)
s = torch.randn(N, V, device="cuda").bfloat16().requires_grad_()
t = torch.randn(N, V, device="cuda")
lp, idx = torch.log_softmax(t, -1).topk(8, -1)


def eager():
    lq = torch.log_softmax(s.float(), -1)
    lpt = torch.log_softmax(t, -1)
    (lpt.exp() * (lpt - lq)).sum(-1).sum().backward()


def dense():
    kl_div_dense(s, t).sum().backward()


def topk():
    kl_div_topk(s, idx, lp).sum().backward()


for name, fn in (("eager dense", eager), ("ours dense", dense), ("ours top-8", topk)):
    ms = triton.testing.do_bench(fn, warmup=10, rep=100, grad_to_none=[s])
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    print(f"{name:<12} N={N} V={V}: {ms:.3f} ms fwd+bwd, peak extra {peak:.0f} MiB", flush=True)
