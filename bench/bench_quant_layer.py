"""One MoE layer, forward + backward INCLUDING the activation and every other stage: bf16 today vs the
W8A8 recipe (e4m3, e8m0 scale per 32 along each GEMM's K, native MMA, fp32 accumulate).

    python -m bench.bench_quant_layer [--skew 0.35]

1. bf16 layer: moe_per_expert fwd+bwd at the board shapes, every kernel timed (profiler), split into
   forward / backward and GEMM / non-GEMM (activation, sort, gather, combine, ...).
2. The six expert GEMMs standalone, bf16 (our kernels) vs MXFP8 (native dot_scaled):
     F1 x@Wgu^T  F3 it@Wdn^T  (fwd)    B3 ge@Wdn  B6 dgu@Wgu  (dgrad)    B2 ge^T@it  B5 dgu^T@x  (wgrad)
   The wgrads reduce over TOKENS, so their operands are quantized in 32-token blocks (expert-aligned,
   rows padded per expert to a multiple of 128; padding is zeros).
3. The quantize passes each fp8 GEMM needs if done as a separate kernel (the pessimistic bound; fused
   into the producer they approach zero extra -- the producer writes 1 byte/value instead of 2).
4. The composed layer: bf16 total - bf16 GEMMs + fp8 GEMMs (+ standalone quant or fused).
"""
import argparse
import statistics

import torch
import triton
import triton.language as tl

import kernels.sm120.moe_fused_glu as FG
from kernels.sm75.moe import _sort_by_expert
from bench.bench_moe_layer import make, layer_step
from bench.bench_quant_study import quant, _native_kernel

dev = "cuda"


def timed(fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


# ------------------------------------------------------------------ quantize along TOKENS (wgrad operands)
@triton.jit
def _qtok_kernel(X, Q, S, Mp, C: tl.constexpr, BC: tl.constexpr):
    """X (Mp, C) bf16 -> Q (Mp, C) e4m3, S (C, Mp/32) e8m0: one scale per 32 consecutive ROWS of a column."""
    pr = tl.program_id(0)
    pc = tl.program_id(1)
    rows = pr * 32 + tl.arange(0, 32)
    cols = pc * BC + tl.arange(0, BC)
    x = tl.load(X + rows[:, None].to(tl.int64) * C + cols[None, :]).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(x), axis=0), 1e-30)
    e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / 448.0)), -127.0), 127.0)
    q = x * tl.exp2(-e)[None, :]
    tl.store(Q + rows[:, None].to(tl.int64) * C + cols[None, :], q.to(tl.float8e4nv))
    tl.store(S + cols.to(tl.int64) * (Mp // 32) + pr, (e + 127.0).to(tl.uint8))


def qtok(x):
    Mp, C = x.shape
    q = torch.empty(Mp, C, device=x.device, dtype=torch.float8_e4m3fn)
    s = torch.empty(C, Mp // 32, device=x.device, dtype=torch.uint8)
    _qtok_kernel[(Mp // 32, C // 128)](x, q, s, Mp, C, 128, num_warps=4)
    return q, s


@triton.jit
def _mx_wgrad_kernel(A, AS, B, BS, C, PS, PE, Mp, N1: tl.constexpr, N2: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[e] (N1, N2) fp32 = A[rows_e]^T @ B[rows_e], A (Mp, N1) / B (Mp, N2) e4m3 with per-(column,
    32-token) e8m0 scales AS (N1, Mp/32) / BS (N2, Mp/32). rows_e = [PS[e], PE[e]), 128-aligned."""
    pid = tl.program_id(0)
    TN: tl.constexpr = N2 // BN
    TM: tl.constexpr = N1 // BM
    e = pid // (TM * TN)
    r = pid % (TM * TN)
    pm = r // TN
    pn = r % TN
    s0 = tl.load(PS + e)
    s1 = tl.load(PE + e)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    KS = Mp // 32
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(s0, s1, BK):
        rk = k0 + tl.arange(0, BK)
        rs = k0 // 32 + tl.arange(0, BK // 32)
        a = tl.load(A + rk[None, :].to(tl.int64) * N1 + rm[:, None])          # (BM, BK) = A^T tile
        a_s = tl.load(AS + rm[:, None].to(tl.int64) * KS + rs[None, :])
        b = tl.load(B + rk[:, None].to(tl.int64) * N2 + rn[None, :])          # (BK, BN)
        b_s = tl.load(BS + rn[:, None].to(tl.int64) * KS + rs[None, :])
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
    tl.store(C + e.to(tl.int64) * (N1 * N2) + rm[:, None] * N2 + rn[None, :], acc)


def mx_wgrad(aq, as_, bq, bs, ps, pe, E, cfg):
    BM, BN, BK, w, st = cfg
    N1, N2 = aq.shape[1], bq.shape[1]
    c = torch.empty(E, N1, N2, device=aq.device, dtype=torch.float32)
    grid = (E * (N1 // BM) * (N2 // BN),)
    return (lambda: _mx_wgrad_kernel[grid](aq, as_, bq, bs, c, ps, pe, aq.shape[0], N1, N2, BM, BN, BK,
                                           num_warps=w, num_stages=st)), c


def best(make_fn, cfgs):
    out = None
    for cfg in cfgs:
        try:
            f, c = make_fn(cfg)
            f()
            ms = timed(f, it=6)
            if out is None or ms < out[0]:
                out = (ms, cfg, c)
        except Exception:
            pass
    return out


GEMM_KEYS = ("_grouped_gemm", "_gate_up_glu", "_wg_kernel", "_wg_reduce", "_grouped_mm", "gemm", "cutlass", "nvjet")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skew", type=float, default=0.35)
    a = ap.parse_args()
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M = N * k

    # ---- 1. bf16 layer, every kernel, fwd / bwd split
    x, idx, w, gu, dn, codes, theta, maxvio = make(skew=a.skew)
    step = layer_step(x, idx, w, gu, dn, codes, theta)
    tot = timed(step)
    from kernels.sm120.moe import moe_per_expert
    from torch.profiler import profile, ProfilerActivity

    def fwd_only():
        with torch.autocast("cuda", dtype=torch.bfloat16):
            return moe_per_expert(x, idx, w, gu, dn, codes, act_params=theta)
    step(); step(); torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as pf:
        y = fwd_only(); torch.cuda.synchronize()
    n_fwd = sum(1 for e in pf.events() if e.device_type.name == "CUDA")
    del y
    with profile(activities=[ProfilerActivity.CUDA]) as p:
        step(); torch.cuda.synchronize()
    ev = sorted((e for e in p.events() if e.device_type.name == "CUDA"), key=lambda e: e.time_range.start)
    ev = [(e.name, e.time_range.elapsed_us() / 1e3) for e in ev]
    is_g = lambda n: any(kk in n for kk in GEMM_KEYS)
    parts = {("fwd", True): 0.0, ("fwd", False): 0.0, ("bwd", True): 0.0, ("bwd", False): 0.0}
    for i, (n_, ms) in enumerate(ev):
        parts[("fwd" if i < n_fwd else "bwd", is_g(n_))] += ms
    print(f"bf16 layer (N={N} tokens, top{k}, MaxVio {maxvio:.2f}): fwd+bwd wall {tot:.2f} ms, kernel sum "
          f"{sum(v for v in parts.values()):.2f} ms")
    print(f"  fwd: GEMM {parts[('fwd', True)]:.2f}  non-GEMM {parts[('fwd', False)]:.2f} | "
          f"bwd: GEMM {parts[('bwd', True)]:.2f}  non-GEMM {parts[('bwd', False)]:.2f}")
    print("  non-GEMM kernels (activation, sort, gather, combine, ...):")
    agg = {}
    for i, (n_, ms) in enumerate(ev):
        if not is_g(n_):
            key = ("fwd " if i < n_fwd else "bwd ") + n_[:70]
            agg[key] = agg.get(key, 0.0) + ms
    for key, ms in sorted(agg.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {ms:7.3f}  {key}")
    del x, gu, dn

    # ---- 2. the six GEMMs, standalone, bf16 vs mxfp8
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(N, E, device=dev, generator=g) + a.skew * torch.randn(E, device=dev, generator=g)
    _, idx2 = torch.softmax(logits, -1).topk(k, -1)
    _, _, _, _, _, counts_t = _sort_by_expert(idx2, torch.ones_like(idx2, dtype=torch.float32), E, host=False)
    offs = counts_t.cumsum(0).to(torch.int32)
    bf = lambda *s: (torch.randn(*s, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    tm = FG.build_tile_map(None, counts_t, dev, bm=FG._GG[0], m_rows=M)
    res = {}
    cfgs = [(128, 128, 128, 4, 2), (128, 128, 64, 4, 4), (256, 128, 64, 8, 3), (128, 256, 64, 8, 4)]
    for lab, K, Nn in (("F1 x@Wgu^T", H, 2 * I), ("F3 it@Wdn^T", I, H), ("B3 ge@Wdn", H, I), ("B6 dgu@Wgu", 2 * I, H)):
        A_, Bnk = bf(M, K), bf(E, Nn, K)
        t_bf = timed(lambda: FG.grouped_gemm(A_, Bnk.transpose(1, 2), tm))
        aq, as_ = quant(A_, "e4m3", "e8m0", 32)
        bq, bs = quant(Bnk.reshape(E * Nn, K), "e4m3", "e8m0", 32)

        def mk(cfg):
            BM, BN, BK, wp, sg = cfg
            TE, TS, TM = FG.build_tile_map(None, counts_t, dev, bm=BM, m_rows=M)
            c = torch.empty(M, Nn, device=dev, dtype=torch.bfloat16)
            return (lambda: _native_kernel[(TE.numel() * (Nn // BN),)](aq, as_, bq.view(E, Nn, K), bs.view(E, Nn, K // 32),
                                                                     c, TE, TS, TM, K, Nn, "e4m3", BM, BN, BK,
                                                                     num_warps=wp, num_stages=sg)), c
        b8 = best(mk, cfgs)
        # quantize cost: F1's A is the UNSORTED token tensor (N rows, gathered 6x by the GEMM)
        qa = timed(lambda: quant(A_[:N] if lab.startswith("F1") else A_, "e4m3", "e8m0", 32), it=6)
        res[lab] = (t_bf, b8[0], qa)
        del A_, Bnk
    # wgrads: padded, expert-aligned token layout
    pcnt = ((counts_t + 127) // 128) * 128
    pe = torch.cumsum(pcnt, 0).to(torch.int32)
    ps = (pe - pcnt).to(torch.int32)
    Mp = int(pe[-1])
    for lab, N1, N2 in (("B2 ge^T@it", H, I), ("B5 dgu^T@x", 2 * I, H)):
        Aw, Bw = bf(M, N1), bf(M, N2)
        t_bf = timed(lambda: FG.grouped_wgrad(Aw, Bw, offs))
        Ap = torch.zeros(Mp, N1, device=dev, dtype=torch.bfloat16)
        Bp = torch.zeros(Mp, N2, device=dev, dtype=torch.bfloat16)
        src0 = torch.cumsum(counts_t, 0) - counts_t
        for e_ in range(E):
            c_ = int(counts_t[e_]); s_ = int(src0[e_]); d_ = int(ps[e_])
            Ap[d_:d_ + c_] = Aw[s_:s_ + c_]; Bp[d_:d_ + c_] = Bw[s_:s_ + c_]
        aq, as_ = qtok(Ap)
        bq, bs = qtok(Bp)
        b8 = best(lambda cfg: mx_wgrad(aq, as_, bq, bs, ps, pe, E, cfg),
                  [(128, 128, 128, 4, 3), (128, 128, 64, 4, 4), (128, 256, 64, 8, 3), (256, 128, 64, 8, 3)])
        # parity of the fp8 wgrad vs the bf16 kernel on the same inputs (quant error only)
        ref = FG.grouped_wgrad(Aw, Bw, offs).float()
        err = ((b8[2] - ref).norm() / ref.norm()).item()
        qa = timed(lambda: (qtok(Ap), qtok(Bp)), it=6)
        res[lab] = (t_bf, b8[0], qa)
        print(f"  {lab}: fp8 wgrad cfg {b8[1]}, rel err vs bf16 {err:.2e}")
        del Aw, Bw, Ap, Bp
    # weights: quantized once per OPTIMIZER step in both layouts, shared by grad_accum=4 micro-batches
    Wgu, Wdn = bf(E, 2 * I, H), bf(E, H, I)
    qw = timed(lambda: (quant(Wgu.reshape(-1, H), "e4m3", "e8m0", 32),
                        quant(Wgu.transpose(1, 2).reshape(-1, 2 * I).contiguous(), "e4m3", "e8m0", 32),
                        quant(Wdn.reshape(-1, I), "e4m3", "e8m0", 32),
                        quant(Wdn.transpose(1, 2).reshape(-1, H).contiguous(), "e4m3", "e8m0", 32)), it=6)

    print(f"\n{'GEMM':14s} {'bf16 ms':>8s} {'fp8 ms':>8s} {'speedup':>8s} {'quant ms (standalone)':>22s}")
    for lab, (tb, t8, qa) in res.items():
        print(f"{lab:14s} {tb:8.3f} {t8:8.3f} {tb / t8:7.2f}x {qa:22.3f}")
    fw = [l for l in res if l[:2] in ("F1", "F3")]
    bw = [l for l in res if l not in fw]
    gb = lambda ls, j: sum(res[l][j] for l in ls)
    print(f"weights, both layouts, per optimizer step: {qw:.3f} ms -> {qw / 4:.3f} ms per micro-batch")

    # ---- 4. composed layer
    fwd_bf = parts[("fwd", True)] + parts[("fwd", False)]
    bwd_bf = parts[("bwd", True)] + parts[("bwd", False)]
    # scale the standalone GEMM speedups onto the in-layer GEMM kernel time (in-layer kernels differ
    # slightly: F1 gathers, B5 gathers, wgrads split chunks)
    f_fwd = gb(fw, 1) / gb(fw, 0)
    f_bwd = gb(bw, 1) / gb(bw, 0)
    fwd8 = parts[("fwd", True)] * f_fwd + parts[("fwd", False)]
    bwd8 = parts[("bwd", True)] * f_bwd + parts[("bwd", False)]
    qf, qb = gb(fw, 2), gb(bw, 2)
    print(f"\n{'layer (one micro-batch)':34s} {'fwd ms':>8s} {'bwd ms':>8s} {'total':>8s} {'vs bf16':>8s}")
    rows = [("bf16 today", fwd_bf, bwd_bf),
            ("fp8 GEMMs, quant FUSED (ideal)", fwd8, bwd8 + qw / 4),
            ("fp8 GEMMs + standalone quant", fwd8 + qf, bwd8 + qb + qw / 4)]
    for name, f_, b_ in rows:
        print(f"{name:34s} {f_:8.2f} {b_:8.2f} {f_ + b_:8.2f} {(fwd_bf + bwd_bf) / (f_ + b_):7.2f}x")
    print(f"GEMM share of the bf16 layer: {100 * (parts[('fwd', True)] + parts[('bwd', True)]) / (fwd_bf + bwd_bf):.1f}%")
    print("BENCH_QUANT_LAYER_DONE")


if __name__ == "__main__":
    main()
