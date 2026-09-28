"""MoE weight-gradient kernel variants at the board shapes, bitwise vs the current grouped_wgrad.
  TAIL   unmasked main K loop + one masked tail step (current masks every step)
  TLOAD  load A as a (BM, BK) tile through transposed pointers instead of tl.trans of (BK, BM)
  plus tile shapes / warps / stages.

    python -m bench.bench_moe_wgrad
"""
import itertools

import torch
import triton
import triton.language as tl

import kernels.sm120.moe_fused_glu as FG
from kernels.sm75.moe import _sort_by_expert
from bench.bench_moe_gemm import timed

dev = "cuda"


@triton.jit
def _wg2(A, B, C, P, IT_E, IT_S, IT_N, IT_SLOT, ORDER, BROWS, N1, N2, sa, sb,
         NT2: tl.constexpr, NTILE: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
         GATHER_B: tl.constexpr, TAIL: tl.constexpr, TLOAD: tl.constexpr):
    pid = tl.program_id(0)
    item = tl.load(ORDER + pid // NTILE)
    e = tl.load(IT_E + item)
    if e < 0:
        return
    tile = pid % NTILE
    r1 = (tile // NT2) * BM + tl.arange(0, BM)
    r2 = (tile % NT2) * BN + tl.arange(0, BN)
    s0 = tl.load(IT_S + item).to(tl.int64)
    n = tl.load(IT_N + item)
    acc = tl.zeros((BM, BN), tl.float32)
    rk = tl.arange(0, BK)
    nfull = (n // BK) * BK if TAIL else 0
    for k0 in tl.range(0, nfull, BK):
        rows = s0 + k0 + rk
        if TLOAD:
            a = tl.load(A + rows[None, :] * sa + r1[:, None])
        else:
            a = tl.trans(tl.load(A + rows[:, None] * sa + r1[None, :]))
        if GATHER_B:
            brow = tl.load(BROWS + rows).to(tl.int64)
        else:
            brow = rows
        b = tl.load(B + brow[:, None] * sb + r2[None, :])
        acc = tl.dot(a, b, acc)
    for k0 in tl.range(nfull, n, BK):
        mk = (k0 + rk) < n
        rows = s0 + k0 + rk
        if TLOAD:
            a = tl.load(A + rows[None, :] * sa + r1[:, None], mask=mk[None, :], other=0.0)
        else:
            a = tl.trans(tl.load(A + rows[:, None] * sa + r1[None, :], mask=mk[:, None], other=0.0))
        if GATHER_B:
            brow = tl.load(BROWS + rows, mask=mk, other=0).to(tl.int64)
        else:
            brow = rows
        b = tl.load(B + brow[:, None] * sb + r2[None, :], mask=mk[:, None], other=0.0)
        acc = tl.dot(a, b, acc)
    slot = tl.load(IT_SLOT + item)
    off = r1[:, None] * N2 + r2[None, :]
    if slot < 0:
        tl.store(C + e.to(tl.int64) * N1 * N2 + off, acc.to(C.dtype.element_ty))
    else:
        tl.store(P + slot.to(tl.int64) * N1 * N2 + off, acc)


def wg2(a, b, offs, c, b_rows=None, tail=True, tload=False):
    """grouped_wgrad's host logic verbatim, launching _wg2."""
    CH, BM, BN, BK, nw, ns = c["CH"], c["BM"], c["BN"], c["BK"], c["num_warps"], c["num_stages"]
    M, N1 = a.shape
    N2 = b.shape[1]
    E = offs.numel()
    end = offs.to(torch.int64)
    cnt = end - torch.cat((end.new_zeros(1), end[:-1]))
    nch = ((cnt + CH - 1) // CH).clamp_min(1)
    cend = torch.cumsum(nch, 0)
    NI = (M + CH - 1) // CH + E
    i = torch.arange(NI, device=a.device)
    ie = torch.searchsorted(cend, i, right=True)
    valid = ie < E
    ie = ie.clamp_max(E - 1)
    j = i - (cend - nch)[ie]
    it_s = (end - cnt)[ie] + j * CH
    it_n = torch.where(valid, (cnt[ie] - j * CH).clamp(0, CH), 0)
    it_slot = torch.where(valid & (nch[ie] > 1), i, -1)
    it_e = torch.where(valid, ie, -1)
    order = torch.argsort(it_n.to(torch.int16), descending=True, stable=True)
    out = torch.empty(E, N1, N2, device=a.device, dtype=a.dtype)
    part = torch.empty(NI, N1, N2, device=a.device, dtype=torch.float32)
    nt2 = N2 // BN
    ntile = (N1 // BM) * nt2

    def run():
        _wg2[(NI * ntile,)](a, b, out, part, it_e.to(torch.int32), it_s, it_n.to(torch.int32),
                            it_slot.to(torch.int32), order, b_rows if b_rows is not None else order,
                            N1, N2, a.stride(0), b.stride(0), NT2=nt2, NTILE=ntile, BM=BM, BN=BN, BK=BK,
                            GATHER_B=b_rows is not None, TAIL=tail, TLOAD=tload, num_warps=nw, num_stages=ns)
        FG._wg_reduce[(E, triton.cdiv(N1 * N2, 1024))](part, out, (cend - nch), nch, N1 * N2, BLOCK=1024,
                                                     ACC=False, num_warps=4)
        return out
    return run


def main():
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M = N * k
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(N, E, device=dev, generator=g) + 0.12 * torch.randn(E, device=dev, generator=g)
    w, idx = torch.softmax(logits, -1).topk(k, -1)
    st, _, _, _, _, counts_t = _sort_by_expert(idx, w, E, host=False)
    offs = counts_t.cumsum(0).to(torch.int32)
    bf = lambda *s: (torch.randn(*s, device=dev, generator=g) * 0.05).to(torch.bfloat16)
    for lab, n1, n2, gather in (("B2 dW_down ge^T@it", H, I, False), ("B5 dW_gu dgu^T@x", 2 * I, H, True)):
        A = bf(M, n1)
        Bm = bf(N, n2) if gather else bf(M, n2)
        rows = st if gather else None
        fl = 2 * M * n1 * n2
        ref = FG.grouped_wgrad(A, Bm, offs, b_rows=rows).clone()
        t0 = timed(lambda: FG.grouped_wgrad(A, Bm, offs, b_rows=rows))
        print(f"\n{lab}: current {t0:.3f} ms ({fl / t0 / 1e9:.0f} TF)", flush=True)
        res = []
        for (BM, BN, BK, nw, ns), tail, tload in itertools.product(
                [(128, 128, 32, 4, 4), (128, 128, 32, 8, 4), (128, 128, 64, 8, 3), (128, 256, 32, 8, 3),
                 (256, 128, 32, 8, 3), (128, 256, 64, 8, 2), (256, 128, 64, 8, 2), (64, 128, 32, 4, 4),
                 (128, 64, 32, 4, 4), (256, 256, 32, 8, 2)],
                (False, True), (False, True)):
            if n1 % BM or n2 % BN:
                continue
            c = dict(CH=16384, BM=BM, BN=BN, BK=BK, num_warps=nw, num_stages=ns)
            try:
                f = wg2(A, Bm, offs, c, rows, tail, tload)
                o = f()
                res.append((timed(f, it=5), (BM, BN, BK, nw, ns, "tail" if tail else "mask", "tload" if tload else "trans"),
                            torch.equal(o, ref)))
            except Exception as ex:
                pass
        res.sort(key=lambda r: r[0])
        for ms, cfg, same in res[:8]:
            print(f"   {ms:.3f} ms {fl / ms / 1e9:4.0f} TF  {cfg}  bitwise-same {same}", flush=True)
        del A, Bm
    print("WG_DONE")


if __name__ == "__main__":
    main()
