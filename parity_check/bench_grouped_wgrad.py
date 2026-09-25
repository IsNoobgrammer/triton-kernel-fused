"""MoE weight-grad GEMMs at BiBo board shapes: torch._grouped_mm (sm120 = host loop of cuBLAS GEMMs,
1 sync/call) vs Triton grouped_wgrad (one launch, no sync, deterministic split-K), config sweep.
Reports GPU ms, TFLOPS, host syncs, rel err vs an fp32 per-expert reference, bitwise repeatability.

    python parity_check/bench_grouped_wgrad.py [--sweep]
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kernels.sm120.moe_fused_glu import grouped_wgrad

dev, bf = "cuda", torch.bfloat16
SWEEP = "--sweep" in sys.argv


def gpu_ms(fn, n=20):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def syncs(fn):
    n = [0]
    old = warnings.showwarning
    warnings.showwarning = lambda *a, **k: n.__setitem__(0, n[0] + 1)
    warnings.simplefilter("always")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    fn()
    torch.cuda.set_sync_debug_mode(0)
    warnings.showwarning = old
    return n[0]


def ref(A, B, counts):
    out, r = [], 0
    for c in counts:
        out.append(A[r:r + c].float().t() @ B[r:r + c].float())
        r += c
    return torch.stack(out)


CFGS = [dict()]
if SWEEP:
    CFGS = [dict(CH=ch, BM=bm, BN=bn, BK=bk, num_warps=w, num_stages=st)
            for ch in (4096, 8192, 16384)
            for (bm, bn, bk, w, st) in ((128, 128, 32, 8, 3), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3),
                                        (256, 128, 32, 8, 3), (128, 128, 32, 4, 4), (64, 128, 64, 4, 3))]

ok = True
for label, E, N, K, N1, N2 in (("MoE grad_down    (ge^T @ it)", 64, 65536, 6, 512, 768),
                               ("MoE grad_gate_up (dgu^T @ x)", 64, 65536, 6, 1536, 512),
                               ("L0 grad_down", 8, 65536, 8, 512, 576),
                               ("L0 grad_gate_up", 8, 65536, 8, 1152, 512)):
    torch.manual_seed(0)
    p = torch.distributions.Dirichlet(torch.full((E,), 2.0)).sample()   # mid-training imbalance
    e = torch.multinomial(p, N * K, replacement=True).sort().values
    counts_t = torch.bincount(e, minlength=E).to(dev)
    if E == 64:
        counts_t[5] = 0                                   # an empty expert must come back as zeros
        counts_t[6] += int((torch.bincount(e, minlength=E)[5]))
    counts = counts_t.tolist()
    M = N * K
    A = (torch.randn(M, N1, device=dev) * 0.3).to(bf)
    B = (torch.randn(M, N2, device=dev) * 0.3).to(bf)
    offs = counts_t.cumsum(0).to(torch.int32)
    R = ref(A, B, counts)
    rn = R.norm()
    fl = 2 * M * N1 * N2
    print(f"== {label}: E={E} M={M} ({N1}x{N2}), max/mean rows {max(counts) / (M / E):.2f}")
    f = lambda: torch._grouped_mm(A.t(), B, offs=offs)
    o = f()
    t = gpu_ms(f)
    print(f"   torch._grouped_mm                 {t:7.3f} ms {fl / t / 1e9:6.0f} TF  syncs {syncs(f)}  "
          f"rel {float((o.float() - R).norm() / rn):.2e}")
    best = None
    for cfg in CFGS:
        g = lambda: grouped_wgrad(A, B, offs, cfg)
        try:
            o1 = g()
            if o1 is None:
                continue
            o2 = g()
            t = gpu_ms(g)
        except Exception as ex:
            print(f"   {cfg}: {type(ex).__name__} {str(ex).splitlines()[0][:80]}")
            continue
        rel = float((o1.float() - R).norm() / rn)
        rep = torch.equal(o1, o2)
        zero = E != 64 or bool((o1[5] == 0).all())
        ok &= rep and zero and rel < 3e-3
        tag = ",".join(f"{k}={v}" for k, v in cfg.items()) or "default " + str(
            __import__("kernels.sm120.moe_fused_glu", fromlist=["_WG"])._WG)
        print(f"   grouped_wgrad {tag[:40]:40s} {t:7.3f} ms {fl / t / 1e9:6.0f} TF  syncs {syncs(g)}  "
              f"rel {rel:.2e}  repeat {'bitwise' if rep else 'DIFFERS'}{'' if zero else '  EMPTY EXPERT NOT ZERO'}")
        best = t if best is None else min(best, t)
print("WGRAD PASS" if ok else "WGRAD FAIL")
