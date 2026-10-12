"""kl_div kernels vs eager vs torch.compile (Inductor) of the same math, fwd+bwd, plus a per-kernel profile of each.
Run with TORCH_LOGS=output_code (stderr) to dump Inductor's generated Triton for reading:

    TORCH_LOGS=output_code python bench/bench_kl_div_compile.py [N V] 2> inductor_kl.log
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
from torch.profiler import ProfilerActivity, profile

from kernels.sm120.kl_div import kl_div_dense, kl_div_topk

N, V = (int(sys.argv[1]), int(sys.argv[2])) if len(sys.argv) > 2 else (15700, 1025)
K, T = 8, 1.0
s = torch.randn(N, V, device="cuda").bfloat16().requires_grad_()
t = torch.randn(N, V, device="cuda")
lp, idx = torch.log_softmax(t / T, -1).topk(K, -1)


def dense_math(s, t):
    lq = torch.log_softmax(s.float() / T, -1)
    lpt = torch.log_softmax(t / T, -1)
    return (lpt.exp() * (lpt - lq)).sum(-1)


def topk_math(s, idx, lp):
    lq = torch.log_softmax(s.float() / T, -1)
    lqk = lq.gather(1, idx)
    p = lp.exp()
    r = (1 - p.sum(1)).clamp(min=0)
    q_rest = (1 - lqk.exp().sum(1)).clamp(min=1e-30)
    return (p * (lp - lqk)).sum(1) + r * (r.clamp(min=1e-30).log() - q_rest.log())


c_dense = torch.compile(dense_math, dynamic=False)
c_topk = torch.compile(topk_math, dynamic=False)
fns = {
    "eager dense": lambda: dense_math(s, t).sum().backward(),
    "inductor dense": lambda: c_dense(s, t).sum().backward(),
    "ours dense": lambda: kl_div_dense(s, t, T).sum().backward(),
    "eager top-8": lambda: topk_math(s, idx, lp).sum().backward(),
    "inductor top-8": lambda: c_topk(s, idx, lp).sum().backward(),
    "ours top-8": lambda: kl_div_topk(s, idx, lp, T).sum().backward(),
}
ref = {}
for name, fn in fns.items():
    s.grad = None
    fn()
    ref[name] = s.grad.float().clone()
rel = lambda a, b: ((a - b).norm() / b.norm()).item()
print(f"grad agreement: inductor-vs-eager dense {rel(ref['inductor dense'], ref['eager dense']):.1e}, ours-vs-eager dense "
      f"{rel(ref['ours dense'], ref['eager dense']):.1e}, ours-vs-eager top-8 {rel(ref['ours top-8'], ref['eager top-8']):.1e}",
      flush=True)
for name, fn in fns.items():
    ms = triton.testing.do_bench(fn, warmup=20, rep=200, grad_to_none=[s])
    torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    fn()
    peak = (torch.cuda.max_memory_allocated() - base) / 2**20
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(5):
            s.grad = None
            fn()
        torch.cuda.synchronize()
    ev = sorted([e for e in prof.key_averages() if e.device_type.name == "CUDA"], key=lambda e: -e.device_time_total)
    ks = "; ".join(f"{e.key[:48]} {e.device_time_total / 5 / 1000:.3f}ms x{e.count // 5}" for e in ev[:6])
    print(f"{name:<15} N={N} V={V}: {ms:.3f} ms fwd+bwd, peak extra {peak:.0f} MiB | kernels/iter: {ks}", flush=True)
