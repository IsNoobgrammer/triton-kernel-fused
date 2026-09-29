"""W8A8 expert GEMMs on sm120 at the board MoE shapes: is fp8 worth building, and in which format?

    python -m bench.bench_quant_gemm [--skew 0.12] [--sweep]

For each M-grouped expert GEMM of one MoE layer (65536 tokens x top-6 over 64 experts, H=512, I=768):
  bf16 ours     kernels.sm120.moe_fused_glu.grouped_gemm, the training kernel today
  bf16 torch    torch._grouped_mm (cuBLAS grouped)
  fp8 rowwise   torch._scaled_grouped_mm, e4m3, fp32 scale per row of A / per column of B
  mxfp8 triton  a grouped Triton kernel on tl.dot_scaled: e4m3 values, e8m0 (power-of-two) scale
                per 32 elements along K for BOTH operands (OCP MX)
plus dense (ungrouped) cuBLAS bf16 / fp8-rowwise ceilings at the same M, K, N, and the cost of
quantizing A to MXFP8 (a memory pass that a fused epilogue would have to hide).

Error = ||C - C_fp32|| / ||C_fp32||, C_fp32 = the same GEMM in fp32 on the bf16 inputs. Inputs are
Gaussian; the heavy-tailed down_proj input (amax/median 320-1425 on trained models) is the case
per-32 blocks exist for, so --tail adds a column with that shape of outlier.
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
F8 = torch.float8_e4m3fn
F8MAX = 448.0


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


def rel(c, ref):
    return ((c.float() - ref).norm() / ref.norm()).item()


# ------------------------------------------------------------------ MXFP8 quantization
@triton.jit
def _mx_quant_kernel(X, Q, S, M, K: tl.constexpr, BM: tl.constexpr, NB: tl.constexpr):
    """Rows x (NB blocks of 32 along K) per program. e = ceil(log2(amax / 448)) -- the ROUND-UP
    scale, so no element overflows e4m3 (the OCP floor rule can saturate the block max)."""
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    rows = pid_m * BM + tl.arange(0, BM)
    cols = pid_k * NB * 32 + tl.arange(0, NB * 32)
    mask = rows[:, None] < M
    x = tl.load(X + rows[:, None].to(tl.int64) * K + cols[None, :], mask=mask, other=0.0).to(tl.float32)
    xb = tl.reshape(x, (BM, NB, 32))
    amax = tl.max(tl.abs(xb), axis=2)
    e = tl.ceil(tl.log2(tl.maximum(amax, 1e-30) / 448.0))
    e = tl.minimum(tl.maximum(e, -127.0), 127.0)
    q = xb * tl.exp2(-e)[:, :, None]
    tl.store(Q + rows[:, None].to(tl.int64) * K + cols[None, :], tl.reshape(q, (BM, NB * 32)).to(tl.float8e4nv),
             mask=mask)
    sc = pid_k * NB + tl.arange(0, NB)
    tl.store(S + rows[:, None].to(tl.int64) * (K // 32) + sc[None, :], (e + 127.0).to(tl.uint8), mask=mask)


def mx_quant(x, BM=64, NB=4):
    """x (R, K) bf16 -> (q (R, K) e4m3, s (R, K//32) uint8 e8m0), blocks of 32 along the LAST dim."""
    R, K = x.shape
    nb = min(NB, K // 32)
    q = torch.empty(R, K, device=x.device, dtype=F8)
    s = torch.empty(R, K // 32, device=x.device, dtype=torch.uint8)
    _mx_quant_kernel[(triton.cdiv(R, BM), K // (32 * nb))](x, q, s, R, K, BM, nb, num_warps=4)
    return q, s


def mx_dequant(q, s):
    return q.float() * torch.exp2(s.float() - 127.0).repeat_interleave(32, dim=-1)


# ------------------------------------------------------------------ grouped MXFP8 GEMM
@triton.jit
def _mx_gg_kernel(A, AS, B, BS, C, TE, TS, TM, K: tl.constexpr, N: tl.constexpr,
                  BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """C[rows_e] = A[rows_e] @ B[e]^T with A (M, K) e4m3 + (M, K/32) e8m0 and B stored (E, N, K)
    e4m3 + (E, N, K/32) e8m0 -- K-contiguous on both sides, so each scale block is 32 adjacent
    bytes. N-fast tile order like the bf16 kernel."""
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
        acc = tl.dot_scaled(a, a_s, "e4m3", b, b_s, "e4m3", acc)
    tl.store(C + rm[:, None].to(tl.int64) * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


def mx_gg(aq, as_, bq, bs, counts_t, M, N, cfg):
    BM, BN, BK, w, st = cfg
    K = aq.shape[1]
    TE, TS, TM = FG.build_tile_map(None, counts_t, aq.device, bm=BM, m_rows=M)
    c = torch.empty(M, N, device=aq.device, dtype=torch.bfloat16)
    grid = (TE.numel() * (N // BN),)
    return lambda: _mx_gg_kernel[grid](aq, as_, bq, bs, c, TE, TS, TM, K, N, BM, BN, BK,
                                       num_warps=w, num_stages=st), c


MX_CFG = (128, 128, 128, 8, 3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skew", type=float, default=0.12)
    ap.add_argument("--sweep", action="store_true", help="config sweep for the MXFP8 kernel")
    ap.add_argument("--tail", action="store_true", help="also measure error on a heavy-tailed A")
    a = ap.parse_args()
    Ntok, H, E, k, I = 65536, 512, 64, 6, 768
    M = Ntok * k
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(Ntok, E, device=dev, generator=g) + a.skew * torch.randn(E, device=dev, generator=g)
    _, idx = torch.softmax(logits, -1).topk(k, -1)
    w = torch.ones_like(idx, dtype=torch.float32)
    _, _, _, _, _, counts_t = _sort_by_expert(idx, w, E, host=False)
    offs = counts_t.cumsum(0).to(torch.int32)
    cnt = counts_t.float()
    print(f"M={M} rows, rows/expert {cnt.mean().item():.0f}, MaxVio {(cnt.max() / cnt.mean() - 1).item():.2f}", flush=True)
    bf = lambda *s: (torch.randn(*s, device=dev, generator=g) * 0.05).to(torch.bfloat16)

    rows = []
    # (label, K, N): the M-grouped GEMMs of one layer. F1 runs as a gather in training; here the rows
    # are pre-sorted, which is the same GEMM without the gather.
    for lab, K, N in (("F1 gate_up x@Wgu^T", H, 2 * I), ("F3 down    it@Wdn^T", I, H),
                      ("B3 d_int   ge@Wdn", H, I), ("B6 d_x     dgu@Wgu", 2 * I, H)):
        A, Bnk = bf(M, K), bf(E, N, K)                 # B kept (E, N, K): K-contiguous, like nn.Linear
        if a.tail and lab.startswith("F3"):
            spike = torch.rand(M, K, device=dev, generator=g) < 0.002
            A = torch.where(spike, A * 300, A)
        fl = 2 * M * K * N
        Bkn = Bnk.transpose(1, 2)                      # (E, K, N) strided view for the bf16 kernels
        ref = torch.empty(M, N, device=dev)
        o = 0
        for e_ in range(E):
            n_ = int(counts_t[e_]); ref[o:o + n_] = A[o:o + n_].float() @ Bkn[e_].float(); o += n_
        res = {}

        tm = FG.build_tile_map(None, counts_t, dev, bm=FG._GG[0], m_rows=M)
        f = lambda: FG.grouped_gemm(A, Bkn, tm)
        res["bf16 ours"] = (timed(f), rel(f(), ref))
        try:
            f = lambda: torch._grouped_mm(A, Bkn, offs=offs)
            res["bf16 torch grouped"] = (timed(f), rel(f(), ref))
        except Exception as ex:
            res["bf16 torch grouped"] = (float("nan"), f"n/a {type(ex).__name__}")
        # fp8 rowwise: scale per row of A, per output column of B
        sa = (A.float().abs().amax(1) / F8MAX).clamp_min(1e-12)
        sb = (Bnk.float().abs().amax(2) / F8MAX).clamp_min(1e-12)          # (E, N)
        Aq = (A.float() / sa[:, None]).to(F8)
        Bq = (Bnk.float() / sb[:, :, None]).to(F8)                          # (E, N, K)
        try:
            f = lambda: torch._scaled_grouped_mm(Aq, Bq.transpose(1, 2), sa, sb, offs=offs,
                                                 out_dtype=torch.bfloat16)
            res["fp8 rowwise torch grouped"] = (timed(f), rel(f(), ref))
        except Exception as ex:
            res["fp8 rowwise torch grouped"] = (float("nan"), f"n/a {type(ex).__name__}: {str(ex)[:60]}")
        # MXFP8 (Triton dot_scaled)
        aq, as_ = mx_quant(A)
        bq, bs = mx_quant(Bnk.reshape(E * N, K))
        bq, bs = bq.view(E, N, K), bs.view(E, N, K // 32)
        qerr = rel(mx_dequant(aq, as_), A.float())
        try:
            f, c = mx_gg(aq, as_, bq, bs, counts_t, M, N, MX_CFG)
            f()
            res["mxfp8 triton grouped"] = (timed(f), rel(c, ref))
            if a.sweep:
                best = []
                for BM, BN, BK, wp, sg in itertools.product((64, 128, 256), (64, 128, 256), (64, 128), (4, 8), (2, 3, 4)):
                    if BM * BN > 256 * 128 or N % BN or K % BK:
                        continue
                    try:
                        f2, c2 = mx_gg(aq, as_, bq, bs, counts_t, M, N, (BM, BN, BK, wp, sg))
                        f2()
                        best.append((timed(f2, it=5), (BM, BN, BK, wp, sg)))
                    except Exception:
                        pass
                best.sort()
                print(f"   {lab} mxfp8 sweep top-3: " + ", ".join(f"{ms:.3f} ms {cf}" for ms, cf in best[:3]))
        except Exception as ex:
            res["mxfp8 triton grouped"] = (float("nan"), f"n/a {type(ex).__name__}: {str(ex)[:80]}")
        res["  (quantize A -> mxfp8)"] = (timed(lambda: mx_quant(A)), qerr)
        # dense ceilings at the same M, K, N
        Bd = Bkn[0].contiguous()
        res["  dense bf16 cuBLAS"] = (timed(lambda: A @ Bd), float("nan"))
        try:
            Bqd = Bq[0].t()                                                   # (K, N) column-major
            f = lambda: torch._scaled_mm(Aq, Bqd, sa[:, None], sb[0][None, :], out_dtype=torch.bfloat16)
            f()
            res["  dense fp8 rowwise cuBLAS"] = (timed(f), float("nan"))
        except Exception as ex:
            res["  dense fp8 rowwise cuBLAS"] = (float("nan"), f"n/a {type(ex).__name__}")

        t0 = res["bf16 ours"][0]
        print(f"\n{lab}  (M={M}, K={K}, N={N})")
        for name, (ms, err) in res.items():
            tf = "" if ms != ms or name.startswith("  (") else f"{fl / ms / 1e9:5.0f} TF"
            sp = "" if ms != ms else f"{t0 / ms:5.2f}x"
            es = err if isinstance(err, str) else ("" if err != err else f"err {err:.2e}")
            print(f"  {name:28s} {ms:7.3f} ms {tf:9s} {sp:7s} {es}", flush=True)
        rows.append((lab, res))
        del A, Bnk, ref

    print("\nSUM over the 4 M-grouped GEMMs (ms):")
    for name in ("bf16 ours", "fp8 rowwise torch grouped", "mxfp8 triton grouped"):
        tot = sum(r[name][0] for _, r in rows)
        q = sum(r["  (quantize A -> mxfp8)"][0] for _, r in rows) if name.startswith("mx") else 0.0
        print(f"  {name:28s} {tot:7.3f}" + (f"  (+{q:.3f} quantize = {tot + q:.3f})" if q else ""))
    print("BENCH_QUANT_GEMM_DONE")


if __name__ == "__main__":
    main()
