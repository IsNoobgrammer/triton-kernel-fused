"""ce_factored logits-kernel config per hidden size (train = STORE_E, val = stats only), vs cuBLAS mm."""
import itertools
import torch
import triton

import kernels.sm120.ce_factored as cf
from bench.bench_ce_factored import timed

V, C = 81920, 6553
for K in (512, 2048, 4096, 16384):
    h = torch.randn(C, K, device="cuda").to(torch.bfloat16)
    W = (torch.randn(V, K, device="cuda") / K ** 0.5).to(torch.bfloat16)
    y = torch.randint(0, V, (C,), device="cuda")
    E = torch.empty(C, V, device="cuda", dtype=torch.bfloat16)
    mc, _, _ = timed(lambda: torch.mm(h, W.t(), out=E), it=5)
    res = {True: [], False: []}
    for BM, BN, BK, G, nw, ns in itertools.product((128, 256), (128, 256), (32, 64), (8,), (4, 8), (2, 3, 4)):
        if BM * BN > 256 * 128:
            continue
        cfg = (BM, BN, BK, G, nw, ns)
        cf._lcfg = lambda K, cfg=cfg: cfg
        for st in (True, False):
            try:
                ms, _, _ = timed(lambda: cf._stats(h, W, y, st, E if st else None), it=5)
                res[st].append((round(ms, 3), cfg))
            except Exception:
                pass
    for st in (True, False):
        res[st].sort()
        print(f"K={K:<6d} {'train' if st else 'val  '} top3 {res[st][:3]}   cuBLAS mm alone {mc:.3f}", flush=True)
    del h, W, E
    torch.cuda.empty_cache()
print("LCFG_DONE")
