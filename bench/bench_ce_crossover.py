"""Pick ce_factored.CUBLAS_MIN_K: Triton-epilogue logits vs cuBLAS + in-place exp/stats, per hidden size."""
import torch

import kernels.sm120.ce_factored as cf
from bench.bench_ce_factored import make, step, timed, AMP, prev_ce, ground_truth, rel

V, N = 81920, 32768
for H in (512, 1024, 2048, 3072, 4096, 8192, 16384):
    h, W, y = make(N, H, V)
    row = [f"H={H:<6d}"]
    ms, _, _ = timed(step(prev_ce, h, W, y), it=3, warm=2)
    row.append(f"prev {ms:8.2f}")
    for nm, k in (("triton", 1 << 30), ("cublas", 0)):
        cf.CUBLAS_MIN_K = k
        ms, _, (l, gh, gw) = timed(step(cf.fused_linear_cross_entropy, h, W, y), it=3, warm=2)

        def v():
            with torch.no_grad(), AMP:
                return cf.fused_linear_cross_entropy(h, W, y)
        mv, _, lv = timed(v, it=3, warm=2)
        row.append(f"{nm} train {ms:8.2f} val {mv:7.2f}")
        if H == 4096 and nm == "cublas":
            gt = ground_truth(h, W, y)
            row.append(f"[cublas vs GT: loss {abs(l.item() - gt[0].item()):.1e} gh {rel(gh, gt[1]):.1e} "
                       f"gw {rel(gw, gt[2]):.1e} val-loss {abs(lv.item() - gt[0].item()):.1e}]")
            del gt
    print(" | ".join(row), flush=True)
    del h, W, y
    torch.cuda.empty_cache()
print("XOVER_DONE")
