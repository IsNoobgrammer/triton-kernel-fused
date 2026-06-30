"""Local sweep (RTX 3050, 4GB): symmul-eager vs symmul-graph as the model scales.

Question: does use_graph=True's memory win hold as the number of symmul matrices grows (more captured
allocations in the graph pool), and does speed stay a wash? Sweep n_blocks of 2048^2 matrices.
Run: <BiBo-venv>/python .autoresearch/bench_graph_sweep.py
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from triton.testing import do_bench
from kernels.sm120.muon import FusedMuon

DEV = "cuda"


def make_params(shapes, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=torch.float16) for s in shapes]


def prime(params, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    for p in params:
        p.grad = torch.randn(*p.shape, generator=g, device=DEV, dtype=torch.float16)


def measure(use_graph, shapes):
    torch.cuda.empty_cache(); torch.cuda.synchronize()
    params = make_params(shapes)
    opt = FusedMuon(params, lr=0.02, weight_decay=0.1, use_symmul=True, use_graph=use_graph)
    try:
        for _ in range(6):
            prime(params); opt.step()
    except RuntimeError as e:
        del opt, params; torch.cuda.empty_cache()
        return None, None, ("OOM" if "out of memory" in str(e).lower() else "ERR")
    graphed = bool(getattr(opt, "_graph", None) is not None)
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    ms = do_bench(lambda: (prime(params), opt.step()))
    peak = torch.cuda.max_memory_allocated() / 1e6
    del opt, params; torch.cuda.empty_cache()
    return ms, peak, graphed


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)} sm_{''.join(map(str,torch.cuda.get_device_capability(0)))}"
          f" | torch {torch.__version__}")
    free, total = torch.cuda.mem_get_info()
    print(f"VRAM: {total/1e9:.1f} GB total, {free/1e9:.1f} GB free\n")
    print("Sweep: n x 2048^2 matrices (all symmul). eager(use_graph=F) vs graph(use_graph=T).\n")
    print(f"{'matrices':>9} | {'eager ms':>9} {'eager MB':>9} | {'graph ms':>9} {'graph MB':>9} | "
          f"{'speed':>6} {'mem':>6} {'captured':>8}")
    print("-" * 78)
    for nmat in [4, 8, 16, 24, 32]:
        shapes = [(2048, 2048)] * nmat
        te, me, _ = measure(False, shapes)
        tg, mg, gc = measure(True, shapes)
        if te is None or tg is None:
            print(f"{nmat:>9} | eager={te or 'OK'} graph={tg or 'OK'}  (stopping — VRAM limit)")
            break
        spd = te / tg
        mem = me / mg if mg else 0
        print(f"{nmat:>9} | {te:9.2f} {me:9.0f} | {tg:9.2f} {mg:9.0f} | {spd:5.2f}x {mem:5.2f}x {str(gc):>8}")
    print("\nDONE")
