"""Local (RTX 3050 Laptop, sm_86, 4GB) bench: AmalgamatedMuon use_graph False vs True + symmul check.

Run: <BiBo-venv>/python .autoresearch/bench_graph_local.py
Sized to fit 4GB. use_graph=True captures the cuBLAS NS in a CUDA graph (the symmul/torch.compile path
can't be re-captured), so this is symmul-eager vs cuBLAS-in-graph — the launch-bound regime where graphs
can finally help on a slow-CPU laptop (vs the compute-bound wash on T4/Blackwell). Also confirms the
symmetric-matmul win transfers to Ampere (a different arch than the Blackwell it was tuned on).
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from triton.testing import do_bench
from kernels.sm120.muon import FusedMuon
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul
from kernels.sm75.muon import newton_schulz as ns_cublas

DEV = "cuda"


def make_params(shapes, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=torch.float16) for s in shapes]


def prime(params, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    for p in params:
        p.grad = torch.randn(*p.shape, generator=g, device=DEV, dtype=torch.float16)


def measure(make_opt, shapes):
    torch.cuda.empty_cache()
    params = make_params(shapes)
    opt = make_opt(params)
    for _ in range(6):                                          # > graph_warmup (3) so capture happens
        prime(params); opt.step()
    graphed = bool(getattr(opt, "_graph", None) is not None)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ms = do_bench(lambda: (prime(params), opt.step()))
    peak = torch.cuda.max_memory_allocated() / 1e6
    del opt, params
    torch.cuda.empty_cache()
    return ms, peak, graphed


def mixed_shapes():
    sh = []
    for _ in range(3):                                          # small attn+MLP -> symmul gated to cuBLAS
        sh += [(512, 512), (256, 512), (256, 512), (512, 512), (1536, 512), (512, 768)]
    sh += [(2048, 2048), (2048, 2048), (2048, 2048)]            # large -> symmul fires (gram 2048 >= knee)
    return sh


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  sm_{''.join(map(str,torch.cuda.get_device_capability(0)))}"
          f"  | torch {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free\n")

    # 1) symmetric-matmul vs cuBLAS NS, single matrix — does the Blackwell win transfer to Ampere?
    print("=== NS micro (single matrix): symmul vs cuBLAS, sm_86 Ampere ===")
    for d in [1024, 2048, 4096]:
        G = torch.randn(d, d, device=DEV, dtype=torch.float16)
        ns_cublas(G); newton_schulz_symmul(G)                   # warm
        tc = do_bench(lambda: ns_cublas(G, ns_dtype=torch.float16))
        ts = do_bench(lambda: newton_schulz_symmul(G, ns_dtype=torch.float16))
        dmax = (newton_schulz_symmul(G).float() - ns_cublas(G).float()).abs().max().item()
        print(f"  d={d:5d}: cuBLAS {tc:7.3f}ms  symmul {ts:7.3f}ms  -> {tc/ts:.2f}x  | parity max|d|={dmax:.1e}")
        del G; torch.cuda.empty_cache()

    # 2) the ask: amalgamated optimizer, use_graph False vs True
    print("\n=== AmalgamatedMuon optimizer step: use_graph False vs True ===")
    shapes = mixed_shapes()
    nparam = sum(int(torch.tensor(s).prod().item()) for s in shapes)
    print(f"  mixed model: {len(shapes)} tensors, {nparam/1e6:.0f}M params (small gated to cuBLAS + 3x 2048^2 symmul)")
    t_sym, m_sym, _  = measure(lambda ps: FusedMuon(ps, lr=.02, weight_decay=.1, use_symmul=True,  use_graph=False), shapes)
    t_cub, m_cub, _  = measure(lambda ps: FusedMuon(ps, lr=.02, weight_decay=.1, use_symmul=False, use_graph=False), shapes)
    t_cg,  m_cg,  gc = measure(lambda ps: FusedMuon(ps, lr=.02, weight_decay=.1, use_symmul=False, use_graph=True),  shapes)
    t_sg,  m_sg,  gs = measure(lambda ps: FusedMuon(ps, lr=.02, weight_decay=.1, use_symmul=True,  use_graph=True),  shapes)
    print(f"  symmul eager  (symmul=T graph=F):  {t_sym:8.3f} ms   {m_sym:7.0f} MB   <- amalgamated default")
    print(f"  cuBLAS eager  (symmul=F graph=F):  {t_cub:8.3f} ms   {m_cub:7.0f} MB")
    print(f"  cuBLAS graph  (symmul=F graph=T):  {t_cg:8.3f} ms   {m_cg:7.0f} MB   captured={gc}")
    print(f"  symmul graph  (symmul=T graph=T):  {t_sg:8.3f} ms   {m_sg:7.0f} MB   captured={gs}")
    print(f"  -> symmul vs cuBLAS-eager: {t_cub/t_sym:.2f}x speed")
    print(f"  -> symmul-graph vs symmul-eager: {t_sym/t_sg:.2f}x speed, {m_sym/m_sg:.2f}x memory (peak {m_sym:.0f}->{m_sg:.0f} MB)")
    print("DONE")
