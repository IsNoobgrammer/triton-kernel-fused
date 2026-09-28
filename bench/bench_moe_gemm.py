"""MoE grouped-GEMM stages, one at a time: current kernel vs (a) N-fastest tile order (A rows stay in
L2 across the N tiles instead of being re-read from DRAM per N tile) and (b) a config sweep.
Every variant is checked BITWISE against the current kernel (same math, same K order).

    python -m bench.bench_moe_gemm [--skew 0.12]
"""
import argparse
import itertools
import statistics

import torch
import triton
import triton.language as tl

import kernels.sm120.moe_fused_glu as FG
from kernels.sm75.moe import _sort_by_expert

dev = "cuda"


def timed(fn, it=10, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


@triton.jit
def _gg_kernel(A, B, C, TE, TS, TM, NTILES, K: tl.constexpr, N: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NFAST: tl.constexpr):
    if NFAST:
        pid = tl.program_id(0)
        t = pid // (N // BN)
        pid_n = pid % (N // BN)
    else:
        t = tl.program_id(0)
        pid_n = tl.program_id(1)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    Bb = B + e.to(tl.int64) * (K * N)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        b = tl.load(Bb + rk[:, None] * N + rn[None, :])
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


def gg(a, b, counts_t, M, cfg, out=None):
    BM, BN, BK, w, st, nfast = cfg
    TE, TS, TM = FG.build_tile_map(None, counts_t, a.device, bm=BM, m_rows=M)
    K, N = b.shape[1], b.shape[2]
    c = torch.empty(M, N, device=a.device, dtype=a.dtype) if out is None else out
    grid = (TE.numel() * (N // BN),) if nfast else (TE.numel(), N // BN)
    return lambda: _gg_kernel[grid](a, b, c, TE, TS, TM, TE.numel(), K, N, BM, BN, BK, nfast,
                                    num_warps=w, num_stages=st), c


@triton.jit
def _gu_kernel(X, W, GU, TE, TS, TM, XROWS, H: tl.constexpr, I: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, NFAST: tl.constexpr):
    # the forward gate/up GEMM exactly as _gate_up_glu_kernel (GATHER, WRITE_GU, no act)
    if NFAST:
        pid = tl.program_id(0)
        t = pid // (I // BN)
        pid_n = pid % (I // BN)
    else:
        t = tl.program_id(0)
        pid_n = tl.program_id(1)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    Wb = W + e.to(tl.int64) * (2 * I * H)
    xr = tl.load(XROWS + rm, mask=mask_m, other=0).to(tl.int64)
    ag = tl.zeros((BM, BN), tl.float32)
    au = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, H, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + xr[:, None] * H + rk[None, :], mask=mask_m[:, None], other=0.0)
        wg = tl.load(Wb + rn[:, None] * H + rk[None, :])
        wu = tl.load(Wb + (I + rn[:, None]) * H + rk[None, :])
        ag = tl.dot(x, tl.trans(wg), ag)
        au = tl.dot(x, tl.trans(wu), au)
    tl.store(GU + rm[:, None] * (2 * I) + rn[None, :], ag.to(tl.bfloat16), mask=mask_m[:, None])
    tl.store(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), au.to(tl.bfloat16), mask=mask_m[:, None])


def gu_call(x, W, st, counts_t, M, cfg):
    BM, BN, BK, w, s, nfast = cfg
    H, I = x.shape[1], W.shape[1] // 2
    TE, TS, TM = FG.build_tile_map(None, counts_t, x.device, bm=BM, m_rows=M)
    out = torch.empty(M, 2 * I, device=x.device, dtype=x.dtype)
    grid = (TE.numel() * (I // BN),) if nfast else (TE.numel(), I // BN)
    return lambda: _gu_kernel[grid](x, W, out, TE, TS, TM, st, H, I, BM, BN, BK, nfast,
                                    num_warps=w, num_stages=s), out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skew", type=float, default=0.12)
    a = ap.parse_args()
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M = N * k
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(N, E, device=dev, generator=g) + a.skew * torch.randn(E, device=dev, generator=g)
    w, idx = torch.softmax(logits, -1).topk(k, -1)
    st, sw, order, counts, bounds, counts_t = _sort_by_expert(idx, w, E, host=False)
    cnt = counts_t.float()
    print(f"M={M} rows, MaxVio {(cnt.max() / cnt.mean() - 1).item():.2f}, rows/expert {cnt.mean().item():.0f}", flush=True)
    bf = lambda *s: (torch.randn(*s, device=dev, generator=g) * 0.05).to(torch.bfloat16)

    # ---- the three uses of _grouped_gemm_kernel: (label, K, N)
    for lab, K, Nn in (("F3 down   it@Wdn^T", I, H), ("B3 d_int  ge@Wdn", H, I), ("B6 d_x    dgu@Wgu", 2 * I, H)):
        A, B = bf(M, K), bf(E, K, Nn)
        fl = 2 * M * K * Nn
        base_f, base_c = gg(A, B, counts_t, M, (*FG._GG[:3], FG._GG[3], FG._GG[4], False))
        base_f()
        ref = base_c.clone()
        t0 = timed(base_f)
        res = []
        for BM, BN, BK, wp, sg, nf in itertools.product((64, 128, 256), (128, 256), (32, 64), (4, 8), (3, 4), (False, True)):
            if BM * BN > 256 * 128 or Nn % BN or K % BK:
                continue
            try:
                f, c = gg(A, B, counts_t, M, (BM, BN, BK, wp, sg, nf))
                f()
                same = torch.equal(c, ref)
                res.append((timed(f, it=5), (BM, BN, BK, wp, sg, nf), same))
            except Exception:
                pass
        res.sort(key=lambda r: r[0])
        nf_same = [r for r in res if r[1][:5] == (*FG._GG[:3], FG._GG[3], FG._GG[4]) and r[1][5]]
        print(f"\n{lab}: current {FG._GG} {t0:.3f} ms ({fl / t0 / 1e9:.0f} TF)"
              + (f" | same cfg N-fast {nf_same[0][0]:.3f} ms" if nf_same else ""))
        for ms, cfg, same in res[:5]:
            print(f"   {ms:.3f} ms {fl / ms / 1e9:4.0f} TF  {cfg}  bitwise-same {same}", flush=True)
        del A, B

    # ---- F1 gate/up (gather)
    x = bf(N, H)
    W = bf(E, 2 * I, H)
    fl = 2 * M * H * 2 * I
    base_f, base_c = gu_call(x, W, st, counts_t, M, (FG._BM, FG._BN, FG._BK, FG._WARPS, FG._STAGES, False))
    base_f()
    ref = base_c.clone()
    t0 = timed(base_f)
    res = []
    for BM, BN, BK, wp, sg, nf in itertools.product((32, 64, 128), (128, 256), (32, 64), (4, 8), (2, 3, 4), (False, True)):
        if BM * BN > 128 * 256 or I % BN:
            continue
        try:
            f, c = gu_call(x, W, st, counts_t, M, (BM, BN, BK, wp, sg, nf))
            f()
            res.append((timed(f, it=5), (BM, BN, BK, wp, sg, nf), torch.equal(c, ref)))
        except Exception:
            pass
    res.sort(key=lambda r: r[0])
    print(f"\nF1 gate_up x@Wgu^T (gather): current ({FG._BM},{FG._BN},{FG._BK},{FG._WARPS},{FG._STAGES}) "
          f"{t0:.3f} ms ({fl / t0 / 1e9:.0f} TF)")
    for ms, cfg, same in res[:5]:
        print(f"   {ms:.3f} ms {fl / ms / 1e9:4.0f} TF  {cfg}  bitwise-same {same}", flush=True)

    # ---- wgrads (grouped_wgrad takes a cfg)
    offs = counts_t.cumsum(0).to(torch.int32)
    for lab, n1, n2, gather in (("B2 dW_down ge^T@it", H, I, False), ("B5 dW_gu  dgu^T@x", 2 * I, H, True)):
        A = bf(M, n1)
        Bm = bf(N, n2) if gather else bf(M, n2)
        rows = st if gather else None
        fl = 2 * M * n1 * n2
        ref = FG.grouped_wgrad(A, Bm, offs, b_rows=rows).clone()
        t0 = timed(lambda: FG.grouped_wgrad(A, Bm, offs, b_rows=rows))
        res = []
        for CH, BM, BN, BK, wp, sg in itertools.product((8192, 16384, 32768), (64, 128), (64, 128, 256), (32, 64), (4, 8), (3, 4)):
            if BM * BN > 128 * 256 or n1 % BM or n2 % BN:
                continue
            cfg = dict(CH=CH, BM=BM, BN=BN, BK=BK, num_warps=wp, num_stages=sg)
            try:
                o = FG.grouped_wgrad(A, Bm, offs, cfg=cfg, b_rows=rows)
                same = torch.equal(o, ref)
                res.append((timed(lambda: FG.grouped_wgrad(A, Bm, offs, cfg=cfg, b_rows=rows), it=5), cfg, same))
            except Exception:
                pass
        res.sort(key=lambda r: r[0])
        print(f"\n{lab}: current {FG._WG} {t0:.3f} ms ({fl / t0 / 1e9:.0f} TF)")
        for ms, cfg, same in res[:5]:
            print(f"   {ms:.3f} ms {fl / ms / 1e9:4.0f} TF  {cfg}  bitwise-same {same}", flush=True)
        del A, Bm
    print("MOE_GEMM_DONE")


if __name__ == "__main__":
    main()
