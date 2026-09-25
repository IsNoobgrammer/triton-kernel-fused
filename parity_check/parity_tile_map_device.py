"""Device-built (sync-free, padded) MoE tile maps vs the host-built ones: every consumer kernel must
produce the same output. python parity_check/parity_tile_map_device.py

The device map has ceil(M/bm) + E tiles; the real ones are an exact prefix of the host map and the
padding carries TM = 0 (no rows). Deterministic kernels must match bit-for-bit; the atomic scatter
only to atomic-ordering noise (it is not bitwise reproducible against ITSELF either).
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels.sm120.moe  # noqa: F401,E402  import sm120 FIRST (circular-import trap, see moe.py)
from kernels.sm120 import moe_fused_glu as FG  # noqa: E402
from kernels.sm75.moe import expert_counts  # noqa: E402

torch.manual_seed(0)
dev, bf = "cuda", torch.bfloat16
E, H, I, N_TOK, K = 64, 512, 768, 65536, 6
M = N_TOK * K
# imbalanced routing (early-training shape): Zipf-ish expert popularity, a few empty experts
p = torch.distributions.Dirichlet(torch.full((E,), 0.3)).sample()
p[:3] = 0
e_sorted = torch.multinomial(p, M, replacement=True).to(dev).sort().values
counts_t = expert_counts(e_sorted, E)
counts = counts_t.tolist()
print(f"rows {M}, experts {E}, max/mean {max(counts) / (M / E):.2f}, empty {counts.count(0)}")

ok = True
for bm in sorted({FG._BM, FG._GG[0], FG._BBM, FG._RBM}):
    h = FG.build_tile_map(counts, counts_t, dev, bm=bm)
    torch.cuda.synchronize(); torch.cuda.set_sync_debug_mode("error")
    d = FG.build_tile_map(None, counts_t, dev, bm=bm, m_rows=M)
    torch.cuda.set_sync_debug_mode(0)
    n = h[0].numel()
    pref = all(torch.equal(a, b[:n]) for a, b in zip(h, d))
    pad = bool((d[2][n:] == 0).all()) and bool((d[0][n:] < E).all())
    print(f"  bm {bm:4d}: host {n} tiles, device {d[0].numel()} | prefix equal {pref} | padding empty {pad}")
    ok &= pref and pad

x = (torch.randn(M, H, device=dev) * 0.5).to(bf)
W1 = (torch.randn(E, 2 * I, H, device=dev) * 0.05).to(bf)
W2 = (torch.randn(E, H, I, device=dev) * 0.05).to(bf)
st = torch.randint(0, N_TOK, (M,), device=dev)


def maps(bm):
    return FG.build_tile_map(counts, counts_t, dev, bm=bm), FG.build_tile_map(None, counts_t, dev, bm=bm, m_rows=M)


def same(name, a, b, exact=True):
    global ok
    r = torch.equal(a, b) if exact else float((a.float() - b.float()).abs().max() / b.float().abs().max().clamp_min(1e-30))
    good = r if exact else r < 1e-5
    ok &= bool(good)
    print(f"  {name:34s} {'bit-identical' if exact and r else ('rel %.1e' % r if not exact else 'DIFFERENT')}")


mh, md = maps(FG._BM)
for code, act in ((0, True), (8, False)):
    gh, ih = FG.fused_gate_up_glu(x, W1, mh, code, want_gu=True, act=act)
    gd, idd = FG.fused_gate_up_glu(x, W1, md, code, want_gu=True, act=act)
    same(f"fused_gate_up_glu code {code} gu", gh, gd)
    if act:
        same(f"fused_gate_up_glu code {code} inter", ih, idd)
gh_, gd_ = maps(FG._GG[0])
it = (torch.randn(M, I, device=dev) * 0.5).to(bf)
W2t = W2.transpose(1, 2).contiguous()
same("grouped_gemm (it @ W2)", FG.grouped_gemm(it, W2t, gh_), FG.grouped_gemm(it, W2t, gd_))
ge = (torch.randn(M, H, device=dev) * 0.5).to(bf)
same("grouped_gemm (ge @ W2, dX-inter)", FG.grouped_gemm(ge, W2, gh_), FG.grouped_gemm(ge, W2, gd_))
dgu = (torch.randn(M, 2 * I, device=dev) * 0.5).to(bf)
same("grouped_gemm_scatter (atomic)", FG.grouped_gemm_scatter(dgu, W1, st, gd_, N_TOK),
     FG.grouped_gemm_scatter(dgu, W1, st, gh_, N_TOK), exact=False)
bh, bd = maps(FG._BBM)
gu0, _ = FG.fused_gate_up_glu(x, W1, mh, 0, want_gu=True, act=True)
same("fused_dinter_glu_bwd code 0", FG.fused_dinter_glu_bwd(ge, W2, gu0, bh, 0), FG.fused_dinter_glu_bwd(ge, W2, gu0, bd, 0))
print("PARITY", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
