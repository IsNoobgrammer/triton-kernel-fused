"""Remaining MoE GEMM headroom at the board shapes:
  X1  F1 gate_up as ONE plain grouped GEMM over N = 2I (gather A rows) instead of the paired
      gate/up kernel -- radial does not need gate and up in the same program
  X2  B5 dW_gate_up with a CONTIGUOUS pre-gathered x vs the row-gathered B it uses today
  X3  B6 d_x rows buffer bf16 vs fp32 (TKF_MOE_DX_ROWS): time and grad error vs fp32

    python -m bench.bench_moe_misc
"""
import importlib
import itertools

import torch
import triton
import triton.language as tl

import kernels.sm120.moe_fused_glu as FG
from bench.bench_moe_gemm import timed

K75 = importlib.import_module("kernels.sm75.moe")
dev = "cuda"


@triton.jit
def _gg_gather(A, AROWS, B, C, TE, TS, TM, sbe, sbk, sbn, K: tl.constexpr, N: tl.constexpr,
               BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GATHER: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // (N // BN)
    pid_n = pid % (N // BN)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
    rm = r0 + tl.arange(0, BM)
    rn = pid_n * BN + tl.arange(0, BN)
    mask_m = tl.arange(0, BM) < mm
    if GATHER:
        ar = tl.load(AROWS + rm, mask=mask_m, other=0).to(tl.int64)
    else:
        ar = rm.to(tl.int64)
    Bb = B + e.to(tl.int64) * sbe
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(A + ar[:, None] * K + rk[None, :], mask=mask_m[:, None], other=0.0)
        b = tl.load(Bb + rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


def main():
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M = N * k
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(N, E, device=dev, generator=g) + 0.12 * torch.randn(E, device=dev, generator=g)
    wt, idx = torch.softmax(logits, -1).topk(k, -1)
    st, sw, order, _, _, counts_t = K75._sort_by_expert(idx, wt, E, host=False)
    offs = counts_t.cumsum(0).to(torch.int32)
    x = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16)
    Wgu = (torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5).to(torch.bfloat16)

    # ---- X1
    tm = FG.build_tile_map(None, counts_t, dev, m_rows=M)
    ref, _ = FG.fused_gate_up_glu(x, Wgu, tm, 8, want_gu=True, act=False, rows=st)
    t0 = timed(lambda: FG.fused_gate_up_glu(x, Wgu, tm, 8, want_gu=True, act=False, rows=st))
    WT = Wgu.transpose(1, 2)                                   # (E, H, 2I) view
    out = torch.empty(M, 2 * I, device=dev, dtype=torch.bfloat16)
    res = []
    for BM, BN, BK, nw, ns in itertools.product((64, 128, 256), (128, 256), (32, 64), (4, 8), (2, 3, 4)):
        if BM * BN > 256 * 128:
            continue
        te, ts, tmm = FG.build_tile_map(None, counts_t, dev, bm=BM, m_rows=M)
        f = lambda: _gg_gather[(te.numel() * (2 * I // BN),)](x, st, WT, out, te, ts, tmm, WT.stride(0),
                                                               WT.stride(1), WT.stride(2), H, 2 * I, BM, BN, BK, True,
                                                               num_warps=nw, num_stages=ns)
        try:
            f()
            res.append((timed(f, it=5), (BM, BN, BK, nw, ns), torch.equal(out, ref)))
        except Exception:
            pass
    res.sort(key=lambda r: r[0])
    fl = 2 * M * H * 2 * I
    print(f"X1 F1 gate_up: current paired kernel {t0:.3f} ms ({fl / t0 / 1e9:.0f} TF)")
    for ms, c, same in res[:5]:
        print(f"   plain N=2I {ms:.3f} ms {fl / ms / 1e9:.0f} TF {c} bitwise {same}", flush=True)

    # ---- X2
    dgu = (torch.randn(M, 2 * I, device=dev, generator=g) * 1e-2).to(torch.bfloat16)
    t_g = timed(lambda: FG.grouped_wgrad(dgu, x, offs, b_rows=st))
    xs = x.index_select(0, st)
    t_c = timed(lambda: FG.grouped_wgrad(dgu, xs, offs))
    t_gather = timed(lambda: x.index_select(0, st))
    same = torch.equal(FG.grouped_wgrad(dgu, x, offs, b_rows=st), FG.grouped_wgrad(dgu, xs, offs))
    print(f"\nX2 B5 dW_gu: row-gathered B {t_g:.3f} ms | contiguous x_s {t_c:.3f} ms (bitwise {same}) | "
          f"the x_s gather itself {t_gather:.3f} ms", flush=True)
    del xs

    # ---- X3
    inv = FG.inverse_order(order)
    tgg = FG.build_tile_map(None, counts_t, dev, bm=FG._GG[0], m_rows=M)
    r32 = FG.grouped_gemm_gather(dgu, Wgu, inv, tgg, N, k, out_dtype=torch.bfloat16, rows_dtype=torch.float32)
    r16 = FG.grouped_gemm_gather(dgu, Wgu, inv, tgg, N, k, out_dtype=torch.bfloat16, rows_dtype=torch.bfloat16)
    t32 = timed(lambda: FG.grouped_gemm_gather(dgu, Wgu, inv, tgg, N, k, out_dtype=torch.bfloat16, rows_dtype=torch.float32))
    t16 = timed(lambda: FG.grouped_gemm_gather(dgu, Wgu, inv, tgg, N, k, out_dtype=torch.bfloat16, rows_dtype=torch.bfloat16))
    with torch.no_grad():
        rows = torch.empty(M, H, device=dev)
        s = 0
        for ee, c in enumerate(counts_t.tolist()):
            rows[s:s + c] = dgu[s:s + c].float() @ Wgu[ee].float()
            s += c
        ref32 = rows[inv.view(N, k)].sum(1)
    rel = lambda a: ((a.float() - ref32).norm() / ref32.norm()).item()
    print(f"\nX3 B6 d_x + combine: fp32 rows {t32:.3f} ms (err {rel(r32):.3e}) | bf16 rows {t16:.3f} ms (err {rel(r16):.3e})")
    print("MISC_DONE")


if __name__ == "__main__":
    main()
