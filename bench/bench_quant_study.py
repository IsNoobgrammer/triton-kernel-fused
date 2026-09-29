"""8-bit expert-GEMM design study on sm120: accumulator precision, then speed x error per variant.

    python -m bench.bench_quant_study accum     # does the fp8 MMA accumulate in true fp32?
    python -m bench.bench_quant_study speed     # every (element fmt, scale type, block) at board shapes

accum: a row whose exact sum is 4096 + n * 0.25. An fp32 accumulator (24-bit mantissa) keeps every
0.25; a Hopper-style ~14-bit fp8 accumulator loses them once the partial sum reaches 4096 (ulp 0.5).
Tested for plain tl.dot on e4m3 / e5m2, tl.dot_scaled (hardware block scales), and bf16 as control.

speed: grouped expert GEMM (the board MoE layer, 393216 expert rows) for
  native     tl.dot_scaled, element e4m3 or e5m2, e8m0 scale per 32 along K applied INSIDE the MMA.
             Blocks of 64 / 128 are the same kernel with each scale repeated (hardware reads 1 per 32).
  software   DeepSeek-style: tl.dot on fp8 per BLK-wide K chunk into a fresh fp32 partial, then
             acc += partial * s_a[row] * s_b[col]. Scale fp32 / fp16 / e8m0, BLK 32 / 64 / 128.
plus the cost of the quantize pass for each (fmt, scale, block), and each variant's error:
  qerr = GEMM on quantized inputs vs the fp32 GEMM on the original bf16 inputs (what training sees),
  kerr = kernel vs fp64 GEMM on the SAME dequantized inputs (the kernel's own arithmetic error).
"""
import statistics
import sys

import torch
import triton
import triton.language as tl

import kernels.sm120.moe_fused_glu as FG
from kernels.sm75.moe import _sort_by_expert

dev = "cuda"
FMT = {"e4m3": (torch.float8_e4m3fn, 448.0), "e5m2": (torch.float8_e5m2, 57344.0)}


def timed(fn, it=20, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


# ------------------------------------------------------------------ accumulator test
@triton.jit
def _acc_kernel(A, B, C, K: tl.constexpr, BK: tl.constexpr, MODE: tl.constexpr, F: tl.constexpr):
    """64 x 64 tile, K accumulated in BK steps. MODE 0 = tl.dot, 1 = tl.dot_scaled (scales = 2^0)."""
    rm = tl.arange(0, 64)
    rn = tl.arange(0, 64)
    acc = tl.zeros((64, 64), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + rm[:, None] * K + rk[None, :])
        b = tl.load(B + rk[:, None] * 64 + rn[None, :])
        if MODE == 0:
            acc = tl.dot(a, b, acc)
        else:
            one_a = tl.full((64, BK // 32), 127, tl.uint8)
            one_b = tl.full((64, BK // 32), 127, tl.uint8)
            acc = tl.dot_scaled(a, one_a, F, b, one_b, F, acc)
    tl.store(C + rm[:, None] * 64 + rn[None, :], acc)


def accum():
    K = 4096
    n_small = K - 1
    print(f"exact row sum = 4096 + {n_small} * 0.25 = {4096 + 0.25 * n_small}; "
          f"an accumulator that drops sub-ulp addends at 4096 would return 4096.0")
    for name, dt, mode, f in (("bf16 tl.dot (control)", torch.bfloat16, 0, ""),
                              ("e4m3 tl.dot", torch.float8_e4m3fn, 0, ""),
                              ("e5m2 tl.dot", torch.float8_e5m2, 0, ""),
                              ("e4m3 tl.dot_scaled", torch.float8_e4m3fn, 1, "e4m3"),
                              ("e5m2 tl.dot_scaled", torch.float8_e5m2, 1, "e5m2")):
        # the big term FIRST, so every later 0.25 is added to a partial sum already at 4096
        A = torch.full((64, K), 0.5, device=dev)
        B = torch.full((K, 64), 0.5, device=dev)
        A[:, 0], B[0, :] = 256.0, 16.0
        A, B = A.to(dt), B.to(dt)
        C = torch.empty(64, 64, device=dev)
        try:
            _acc_kernel[(1,)](A, B, C, K, 128, mode, f)
            v = C[0, 0].item()
            ok = v == 4096 + 0.25 * n_small
            print(f"  {name:24s} -> {v:.4f}   {'fp32-exact' if ok else 'LOST PRECISION'}")
        except Exception as ex:
            print(f"  {name:24s} -> n/a ({type(ex).__name__}: {str(ex)[:80]})")
    # random-magnitude version: error vs fp64 on identical fp8 inputs, K up to 16384
    g = torch.Generator(device=dev).manual_seed(0)
    for K in (1024, 16384):
        A = (torch.randn(64, K, device=dev, generator=g) * 4).to(torch.float8_e4m3fn)
        B = (torch.randn(K, 64, device=dev, generator=g) * 4).to(torch.float8_e4m3fn)
        ref = A.double() @ B.double()
        for name, mode in (("tl.dot", 0), ("tl.dot_scaled", 1)):
            C = torch.empty(64, 64, device=dev)
            _acc_kernel[(1,)](A, B, C, K, 128, mode, "e4m3")
            err = ((C.double() - ref).norm() / ref.norm()).item()
            print(f"  e4m3 {name:14s} K={K:6d}: rel err vs fp64 {err:.2e}  (fp32 accumulation ~1e-7; 14-bit ~1e-4)")


# ------------------------------------------------------------------ generic quantization
@triton.jit
def _quant_kernel(X, Q, S, M, K: tl.constexpr, BLK: tl.constexpr, NB: tl.constexpr, BM: tl.constexpr,
                  EMAX: tl.constexpr, SC: tl.constexpr, E5: tl.constexpr):
    """Blocks of BLK along the last dim. SC 0 = e8m0 (2^ceil(log2(amax/EMAX)), stored as uint8),
    1 = fp32 amax/EMAX, 2 = fp16 of amax/EMAX nudged up so rounding never shrinks it (no overflow)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_k * NB * BLK + tl.arange(0, NB * BLK)
    mask = rows[:, None] < M
    x = tl.load(X + rows[:, None].to(tl.int64) * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    xb = tl.reshape(x, (BM, NB, BLK))
    amax = tl.maximum(tl.max(tl.abs(xb), axis=2), 1e-30)
    if SC == 0:
        e = tl.minimum(tl.maximum(tl.ceil(tl.log2(amax / EMAX)), -127.0), 127.0)
        s = tl.exp2(e)
        s_store = (e + 127.0).to(tl.uint8)
    elif SC == 1:
        s = amax / EMAX
        s_store = s
    else:
        s16 = (amax / EMAX * (1.0 + 1.0 / 1024.0)).to(tl.float16)
        s16 = tl.maximum(s16, 5.96e-8)                       # fp16's smallest subnormal: never divide by 0
        s = s16.to(tl.float32)
        s_store = s16
    q = xb / s[:, :, None]
    q = tl.reshape(q, (BM, NB * BLK))
    if E5:
        qc = q.to(tl.float8e5)
    else:
        qc = q.to(tl.float8e4nv)
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], qc, mask=mask)
    sc = pid_k * NB + tl.arange(0, NB)
    tl.store(S + rows[:, None].to(tl.int64) * (K // BLK) + sc[None, :], s_store, mask=mask)


SCALES = {"e8m0": (0, torch.uint8), "fp32": (1, torch.float32), "fp16": (2, torch.float16)}


def quant(x, fmt, scale, blk, BM=64):
    R, K = x.shape
    dt, emax = FMT[fmt]
    sc, sdt = SCALES[scale]
    nb = max(1, min(4, 128 // blk, K // blk))
    q = torch.empty(R, K, device=x.device, dtype=dt)
    s = torch.empty(R, K // blk, device=x.device, dtype=sdt)
    _quant_kernel[(triton.cdiv(R, BM), K // (blk * nb))](x, q, s, R, K, blk, nb, BM, emax, sc, fmt == "e5m2",
                                                         num_warps=4)
    return q, s


def dequant(q, s, scale, blk):
    sf = torch.exp2(s.float() - 127.0) if scale == "e8m0" else s.float()
    return q.float() * sf.repeat_interleave(blk, dim=-1)


def to_mx32(s, blk):
    """e8m0 scales per `blk` -> per 32 (what the MMA reads): repeat each scale blk/32 times."""
    return s.repeat_interleave(blk // 32, dim=-1).contiguous()


# ------------------------------------------------------------------ grouped kernels
@triton.jit
def _native_kernel(A, AS, B, BS, C, TE, TS, TM, K: tl.constexpr, N: tl.constexpr, F: tl.constexpr,
                   BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    KS: tl.constexpr = K // 32
    Bb = B + e.to(tl.int64) * (N * K)
    BSb = BS + e.to(tl.int64) * (N * KS)
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        rs = k0 // 32 + tl.arange(0, BK // 32)
        a = tl.load(A + rm[:, None].to(tl.int64) * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        a_s = tl.load(AS + rm[:, None].to(tl.int64) * KS + rs[None, :], mask=mask_m[:, None], other=127)
        b = tl.load(Bb + rn[None, :] * K + rk[:, None])
        b_s = tl.load(BSb + rn[:, None] * KS + rs[None, :])
        acc = tl.dot_scaled(a, a_s, F, b, b_s, F, acc)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _software_kernel(A, AS, B, BS, C, TE, TS, TM, K: tl.constexpr, N: tl.constexpr, BLK: tl.constexpr,
                     SC: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
    """DeepSeek-style: one fp8 MMA chain per BLK-wide K chunk into a fresh fp32 partial, scaled and
    added into the fp32 accumulator (the 'promotion'). Scales per (row, chunk) / (col, chunk)."""
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    mm = tl.load(TM + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    KS: tl.constexpr = K // BLK
    Bb = B + e.to(tl.int64) * (N * K)
    BSb = BS + e.to(tl.int64) * (N * KS)
    acc = tl.zeros((BM, BN), tl.float32)
    for kb in range(0, KS):
        rk = kb * BLK + tl.arange(0, BLK)
        a = tl.load(A + rm[:, None].to(tl.int64) * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        b = tl.load(Bb + rn[None, :] * K + rk[:, None])
        p = tl.dot(a, b)
        sa = tl.load(AS + rm.to(tl.int64) * KS + kb, mask=mask_m, other=0)
        sb = tl.load(BSb + rn * KS + kb)
        if SC == 0:
            sa_f = tl.exp2(sa.to(tl.float32) - 127.0)
            sb_f = tl.exp2(sb.to(tl.float32) - 127.0)
        else:
            sa_f = sa.to(tl.float32)
            sb_f = sb.to(tl.float32)
        acc += p * sa_f[:, None] * sb_f[None, :]
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


def best_of(make, cfgs):
    best = None
    for cfg in cfgs:
        try:
            f, c = make(cfg)
            f()
            ms = timed(f, it=8)
            if best is None or ms < best[0]:
                best = (ms, cfg, f, c)
        except Exception:
            pass
    return best


def speed():
    Ntok, H, E, k, I = 65536, 512, 64, 6, 768
    M = Ntok * k
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(Ntok, E, device=dev, generator=g) + 0.12 * torch.randn(E, device=dev, generator=g)
    _, idx = torch.softmax(logits, -1).topk(k, -1)
    _, _, _, _, _, counts_t = _sort_by_expert(idx, torch.ones_like(idx, dtype=torch.float32), E, host=False)
    bf = lambda *s: (torch.randn(*s, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    summary = {}
    for lab, K, N in (("F1 gate_up", H, 2 * I), ("F3 down", I, H), ("B6 d_x", 2 * I, H)):
        A, Bnk = bf(M, K), bf(E, N, K)
        spike = torch.rand(M, K, device=dev, generator=g) < 0.002       # down_proj-like heavy tail
        A = torch.where(spike, A * 300, A)
        Bkn = Bnk.transpose(1, 2)
        fl = 2 * M * K * N
        ref = torch.empty(M, N, device=dev)
        o = 0
        for e_ in range(E):
            n_ = int(counts_t[e_]); ref[o:o + n_] = A[o:o + n_].float() @ Bkn[e_].float(); o += n_
        tm = FG.build_tile_map(None, counts_t, dev, bm=FG._GG[0], m_rows=M)
        t_bf = timed(lambda: FG.grouped_gemm(A, Bkn, tm))
        print(f"\n{lab} (M={M}, K={K}, N={N})   bf16 ours {t_bf:.3f} ms ({fl / t_bf / 1e9:.0f} TF)")
        print(f"  {'variant':34s} {'gemm ms':>8s} {'x bf16':>7s} {'quant A ms':>10s} {'qerr':>9s} {'kerr':>9s} "
              f"{'A zero%':>8s} {'A sat%':>7s}  best cfg")

        def row(name, ms, qa_ms, c, aq, as_, bq, bs, scale, blk, cfg):
            deq_a, deq_b = dequant(aq, as_, scale, blk), dequant(bq.view(E * N, K), bs.view(E * N, -1), scale, blk)
            deq_b = deq_b.view(E, N, K)
            r64 = torch.empty(M, N, device=dev, dtype=torch.float64)
            o = 0
            for e_ in range(E):
                n_ = int(counts_t[e_]); r64[o:o + n_] = deq_a[o:o + n_].double() @ deq_b[e_].double().t(); o += n_
            qerr = ((c.float() - ref).norm() / ref.norm()).item()
            kerr = ((c.double() - r64).norm() / r64.norm()).item()
            zero = ((deq_a == 0) & (A != 0)).float().mean().item() * 100
            emax = FMT["e5m2" if "e5m2" in name else "e4m3"][1]
            sf = torch.exp2(as_.float() - 127.0) if scale == "e8m0" else as_.float()
            sat = ((A.float().abs() / sf.repeat_interleave(blk, dim=-1)) > emax * 1.0001).float().mean().item() * 100
            print(f"  {name:34s} {ms:8.3f} {t_bf / ms:7.2f} {qa_ms:10.3f} {qerr:9.2e} {kerr:9.2e} {zero:8.4f} {sat:7.4f}  {cfg}",
                  flush=True)
            summary.setdefault(name, []).append((ms, qa_ms, qerr))
            del deq_a, deq_b, r64

        for fmt in ("e4m3", "e5m2"):
            # native: MMA-applied e8m0, blocks 32/64/128 (64/128 = repeated scales)
            for blk in (32, 64, 128):
                aq, as_ = quant(A, fmt, "e8m0", blk)
                bq, bs = quant(Bnk.reshape(E * N, K), fmt, "e8m0", blk)
                qa = timed(lambda: quant(A, fmt, "e8m0", blk), it=8)
                as32, bs32 = to_mx32(as_, blk), to_mx32(bs, blk).view(E, N, K // 32)

                def mk(cfg):
                    BM, BN, BK, w, st = cfg
                    TE, TS, TM = FG.build_tile_map(None, counts_t, dev, bm=BM, m_rows=M)
                    c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                    grid = (TE.numel() * (N // BN),)
                    return (lambda: _native_kernel[grid](aq, as32, bq.view(E, N, K), bs32, c, TE, TS, TM, K, N, fmt,
                                                         BM, BN, BK, num_warps=w, num_stages=st)), c
                b = best_of(mk, [(128, 128, 128, 4, 2), (128, 128, 64, 4, 4), (256, 128, 64, 8, 3), (128, 256, 64, 8, 4)])
                row(f"native {fmt}  e8m0 blk{blk}", b[0], qa, b[3], aq, as_, bq, bs, "e8m0", blk, b[1])
            # software (DeepSeek-style promotion), all scale types
            for scale in ("fp32", "fp16", "e8m0"):
                for blk in (32, 64, 128):
                    if fmt == "e5m2" and scale != "fp32" and blk != 128:
                        continue                                   # keep the e5m2 grid small
                    aq, as_ = quant(A, fmt, scale, blk)
                    bq, bs = quant(Bnk.reshape(E * N, K), fmt, scale, blk)
                    qa = timed(lambda: quant(A, fmt, scale, blk), it=8)
                    sc = SCALES[scale][0]

                    def mk(cfg):
                        BM, BN, w, st = cfg
                        TE, TS, TM = FG.build_tile_map(None, counts_t, dev, bm=BM, m_rows=M)
                        c = torch.empty(M, N, device=dev, dtype=torch.bfloat16)
                        grid = (TE.numel() * (N // BN),)
                        return (lambda: _software_kernel[grid](aq, as_, bq.view(E, N, K), bs.view(E, N, K // blk), c,
                                                               TE, TS, TM, K, N, blk, sc, BM, BN,
                                                               num_warps=w, num_stages=st)), c
                    b = best_of(mk, [(128, 128, 4, 4), (128, 128, 8, 3), (256, 128, 8, 3), (128, 256, 8, 3)])
                    if b is None:
                        print(f"  software {fmt} {scale} blk{blk}: no config compiled")
                        continue
                    row(f"software {fmt} {scale:4s} blk{blk}", b[0], qa, b[3], aq, as_, bq, bs, scale, blk, b[1])
        del A, Bnk, ref
    print("\nSUM over F1 + F3 + B6 (gemm ms | + quantize-A ms | mean qerr):")
    for name, v in summary.items():
        print(f"  {name:34s} {sum(x[0] for x in v):7.3f} | {sum(x[0] + x[1] for x in v):7.3f} | {sum(x[2] for x in v) / len(v):.2e}")
    print("BENCH_QUANT_STUDY_DONE")


if __name__ == "__main__":
    {"accum": accum, "speed": speed}[sys.argv[1] if len(sys.argv) > 1 else "accum"]()
