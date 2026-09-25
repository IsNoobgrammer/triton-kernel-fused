"""Deterministic MoE reductions vs the atomic ones they replace.
python parity_check/parity_moe_deterministic.py

Each deterministic path must (1) agree with the atomic path to fp32 reordering noise and (2) be
bit-identical across 5 repeats, which the atomic path is not.
"""
import importlib
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels.sm120.moe  # noqa: F401,E402  import sm120 FIRST (circular-import trap)
from kernels.sm120 import moe_fused_glu as FG  # noqa: E402
from kernels.sm120.megakernel.moe.norm_router import rmsnorm_backward  # noqa: E402
M75 = importlib.import_module("kernels.sm75.moe")

torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
E, H, I, N, K = 64, 512, 768, 65536, 6
ok = True


def check(name, det_fn, ref_fn, tol):
    global ok
    outs = [det_fn() for _ in range(5)]
    rep = all(torch.equal(outs[0], o) for o in outs[1:])
    ref = ref_fn()
    rel = float((outs[0].float() - ref.float()).norm() / ref.float().norm().clamp_min(1e-30))
    refs = [ref_fn() for _ in range(4)]
    ref_rep = all(torch.equal(refs[0], r) for r in refs[1:])
    good = rep and rel < tol
    ok &= good
    print(f"  {name:28s} repeatable {rep} | vs atomic rel {rel:.1e} (tol {tol:.0e}) | atomic repeatable {ref_rep}"
          f" -> {'OK' if good else 'FAIL'}")


idx = torch.randint(0, E, (N, K), device=dev)
wt = torch.rand(N, K, device=dev)
st, sw, order, _, _, counts_t = M75._sort_by_expert(idx, wt, E, host=False)
inv = FG.inverse_order(order)
M = N * K
eo = (torch.randn(M, H, device=dev) * 0.5).to(bf)


def ref_combine():
    out = torch.zeros(N, H, device=dev, dtype=torch.float32)
    M75._combine_scatter(eo, sw, st, out)
    return out


check("forward combine", lambda: FG.combine_gather(eo, inv, N, K, w=sw), ref_combine, 1e-6)

tile_gg = FG.build_tile_map(None, counts_t, dev, bm=FG._GG[0], m_rows=M)
dgu = (torch.randn(M, 2 * I, device=dev) * 0.5).to(bf)
W1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(bf)
check("dX gemm + combine", lambda: FG.grouped_gemm_gather(dgu, W1, inv, tile_gg, N, K),
      lambda: FG.grouped_gemm_scatter(dgu, W1, st, tile_gg, N), 1e-6)

row_expert = torch.repeat_interleave(torch.arange(E, device=dev), counts_t, output_size=M)
da = torch.randn(M, device=dev)
check("per-expert theta grad", lambda: M75._ap_grad_from_rows(da, row_expert, E, (E,), dev),
      lambda: torch.zeros(E, device=dev, dtype=torch.float64).index_add_(0, row_expert, da.double()).float(),
      1e-6)   # fp64 reference: the fp32 atomic index_add_ is itself ~1e-6 off over ~6k rows per expert

x = torch.randn(N, H, device=dev).to(bf)
dh = torch.randn(N, H, device=dev).to(bf)
nw = torch.rand(H, device=dev) + 0.5
rstd = torch.rsqrt(x.float().pow(2).mean(-1) + 1e-6)
ref64 = (dh.double() * x.double() * rstd.double()[:, None]).sum(0).float()
check("rmsnorm d_nw", lambda: rmsnorm_backward(x, dh, nw, rstd)[1], lambda: ref64, 1e-5)
print("PARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
