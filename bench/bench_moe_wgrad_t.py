"""B5 dW_gate_up = dgu^T @ x (gathered x) computed as the TRANSPOSED problem x^T @ dgu with a
transposed store: the narrow side (H=512) becomes N1, where the transposed-pointer A load won big
for dW_down. Bitwise checked against the current grouped_wgrad.

    python -m bench.bench_moe_wgrad_t
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
def _wg_t(X, XROWS, D, C, P, IT_E, IT_S, IT_N, IT_SLOT, ORDER, H, NG, sx, sd,
          NT2: tl.constexpr, NTILE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
          TLOAD: tl.constexpr):
    # acc (BM of H) x (BN of NG=2I) = x[rows]^T @ dgu[rows]; stored transposed into C (E, NG, H)
    pid = tl.program_id(0)
    item = tl.load(ORDER + pid // NTILE)
    e = tl.load(IT_E + item)
    if e < 0:
        return
    tile = pid % NTILE
    r1 = (tile // NT2) * BM + tl.arange(0, BM)          # H columns
    r2 = (tile % NT2) * BN + tl.arange(0, BN)           # 2I columns
    s0 = tl.load(IT_S + item).to(tl.int64)
    n = tl.load(IT_N + item)
    acc = tl.zeros((BM, BN), tl.float32)
    rk = tl.arange(0, BK)
    nfull = (n // BK) * BK
    for k0 in tl.range(0, nfull, BK):
        rows = s0 + k0 + rk
        xr = tl.load(XROWS + rows).to(tl.int64)
        if TLOAD:
            a = tl.load(X + xr[None, :] * sx + r1[:, None])
        else:
            a = tl.trans(tl.load(X + xr[:, None] * sx + r1[None, :]))
        b = tl.load(D + rows[:, None] * sd + r2[None, :])
        acc = tl.dot(a, b, acc)
    for k0 in tl.range(nfull, n, BK):
        mk = (k0 + rk) < n
        rows = s0 + k0 + rk
        xr = tl.load(XROWS + rows, mask=mk, other=0).to(tl.int64)
        if TLOAD:
            a = tl.load(X + xr[None, :] * sx + r1[:, None], mask=mk[None, :], other=0.0)
        else:
            a = tl.trans(tl.load(X + xr[:, None] * sx + r1[None, :], mask=mk[:, None], other=0.0))
        b = tl.load(D + rows[:, None] * sd + r2[None, :], mask=mk[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    slot = tl.load(IT_SLOT + item)
    off = r2[None, :] * H + r1[:, None]                  # transposed store: C[e, 2I col, H col]
    if slot < 0:
        tl.store(C + e.to(tl.int64) * NG * H + off, acc.to(C.dtype.element_ty))
    else:
        tl.store(P + slot.to(tl.int64) * NG * H + off, acc)


def main():
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M, NG = N * k, 2 * I
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(N, E, device=dev, generator=g) + 0.12 * torch.randn(E, device=dev, generator=g)
    wt, idx = torch.softmax(logits, -1).topk(k, -1)
    st, _, _, _, _, counts_t = K75._sort_by_expert(idx, wt, E, host=False)
    offs = counts_t.cumsum(0).to(torch.int32)
    x = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16)
    dgu = (torch.randn(M, NG, device=dev, generator=g) * 1e-2).to(torch.bfloat16)
    ref = FG.grouped_wgrad(dgu, x, offs, b_rows=st).clone()
    t0 = timed(lambda: FG.grouped_wgrad(dgu, x, offs, b_rows=st))
    fl = 2 * M * NG * H
    print(f"B5 current {t0:.3f} ms ({fl / t0 / 1e9:.0f} TF)", flush=True)
    CH = 16384
    end = offs.to(torch.int64)
    cnt = end - torch.cat((end.new_zeros(1), end[:-1]))
    nch = ((cnt + CH - 1) // CH).clamp_min(1)
    cend = torch.cumsum(nch, 0)
    NI = (M + CH - 1) // CH + E
    i = torch.arange(NI, device=dev)
    ie = torch.searchsorted(cend, i, right=True)
    valid = ie < E
    ie = ie.clamp_max(E - 1)
    j = i - (cend - nch)[ie]
    it_s = (end - cnt)[ie] + j * CH
    it_n = torch.where(valid, (cnt[ie] - j * CH).clamp(0, CH), 0).to(torch.int32)
    it_slot = torch.where(valid & (nch[ie] > 1), i, -1).to(torch.int32)
    it_e = torch.where(valid, ie, -1).to(torch.int32)
    order = torch.argsort(it_n.to(torch.int16), descending=True, stable=True)
    out = torch.empty(E, NG, H, device=dev, dtype=torch.bfloat16)
    part = torch.empty(NI, NG, H, device=dev, dtype=torch.float32)
    res = []
    for (BM, BN, BK, nw, ns), tload in itertools.product(
            [(128, 256, 32, 8, 3), (128, 128, 32, 4, 4), (128, 128, 32, 8, 4), (256, 128, 32, 8, 3),
             (128, 256, 64, 8, 2), (256, 128, 64, 8, 2), (64, 256, 32, 8, 3), (128, 128, 64, 8, 3)], (True, False)):
        nt2 = NG // BN
        ntile = (H // BM) * nt2

        def run():
            _wg_t[(NI * ntile,)](x, st, dgu, out, part, it_e, it_s, it_n, it_slot, order, H, NG, x.stride(0),
                                 dgu.stride(0), nt2, ntile, BM, BN, BK, tload, num_warps=nw, num_stages=ns)
            FG._wg_reduce[(E, triton.cdiv(NG * H, 1024))](part, out, (cend - nch), nch, NG * H, BLOCK=1024,
                                                         ACC=False, num_warps=4)
        try:
            run()
            res.append((timed(run, it=5), (BM, BN, BK, nw, ns, "tload" if tload else "trans"), torch.equal(out, ref)))
        except Exception as ex:
            pass
    res.sort(key=lambda r: r[0])
    for ms, c, same in res[:6]:
        print(f"   transposed problem {ms:.3f} ms {fl / ms / 1e9:.0f} TF {c} bitwise {same}", flush=True)
    print("WGT_DONE")


if __name__ == "__main__":
    main()
