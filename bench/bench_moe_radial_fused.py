"""Radial forward FUSED into the MoE GEMMs, prototype at the board shapes (M=393216, H=512, I=768).

    current:  F1 gate_up GEMM -> gu | act kernel gu -> it | F3 it @ Wdn^T | (bwd) B2 ge^T @ it
    fused:    F1 also emits per-row, per-N-tile sum(g^2) (PSQ) | F3 applies the act in its A-operand
              load (r from PSQ) | B2 applies it in its B-operand load. `it` is never written.

Numerics differ only through r's summation order (PSQ tiles vs the act kernel's 256-chunks), so the
check is vs an fp32 reference of the whole chain, next to the current pipeline's error.

    python -m bench.bench_moe_radial_fused
"""
import importlib

import torch
import triton
import triton.language as tl

import kernels.sm120.moe_fused_glu as FG
from bench.bench_moe_gemm import timed

K75 = importlib.import_module("kernels.sm75.moe")
dev = "cuda"


@triton.jit
def _f1_psq(X, W, GU, PSQ, TE, TS, TM, XROWS, H: tl.constexpr, I: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    t = pid // (I // BN)
    pid_n = pid % (I // BN)
    mm = tl.load(TM + t)
    if mm == 0:
        return
    e = tl.load(TE + t)
    r0 = tl.load(TS + t)
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
    gb = ag.to(tl.bfloat16)
    tl.store(GU + rm[:, None] * (2 * I) + rn[None, :], gb, mask=mask_m[:, None])
    tl.store(GU + rm[:, None] * (2 * I) + (I + rn[None, :]), au.to(tl.bfloat16), mask=mask_m[:, None])
    gf = gb.to(tl.float32)                                   # r from the ROUNDED gate, as the act kernel
    tl.store(PSQ + rm * (I // BN) + pid_n, tl.sum(gf * gf, axis=1), mask=mask_m)


@triton.jit
def _row_r(PSQ, AL, rows, mrow, I: tl.constexpr, NT: tl.constexpr, EPS: tl.constexpr):
    s = tl.zeros_like(rows.to(tl.float32))
    for j in tl.static_range(NT):                            # fixed tile order: deterministic
        s += tl.load(PSQ + rows * NT + j, mask=mrow, other=0.0)
    r = tl.sqrt(s / I + EPS)
    p = 1.0 / (1.0 + tl.exp(-tl.load(AL + rows, mask=mrow, other=0.0).to(tl.float32)))
    return r, tl.exp(p * tl.log(r))


@triton.jit
def _f3_act(GU, PSQ, AL, B, C, TE, TS, TM, sbe, sbk, sbn, I: tl.constexpr, N: tl.constexpr,
            NT: tl.constexpr, EPS: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
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
    r, rp = _row_r(PSQ, AL, rm, mask_m, I, NT, EPS)
    Bb = B + e.to(tl.int64) * sbe
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, I, BK):
        rk = k0 + tl.arange(0, BK)
        g = tl.load(GU + rm[:, None] * (2 * I) + rk[None, :], mask=mask_m[:, None], other=0.0).to(tl.float32)
        u = tl.load(GU + rm[:, None] * (2 * I) + (I + rk[None, :]), mask=mask_m[:, None], other=0.0).to(tl.float32)
        z = g / r[:, None]
        a = (rp[:, None] * (z * (1.0 / (1.0 + tl.exp(-z)))) * u).to(tl.bfloat16)
        b = tl.load(Bb + rk[:, None] * sbk + rn[None, :] * sbn)
        acc = tl.dot(a, b, acc)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.dtype.element_ty), mask=mask_m[:, None])


@triton.jit
def _b2_act(A, GU, PSQ, AL, C, P, IT_E, IT_S, IT_N, IT_SLOT, ORDER, N1, sa,
            I: tl.constexpr, NT: tl.constexpr, EPS: tl.constexpr, NT2: tl.constexpr, NTILE: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # dW_down[e] = ge[rows_e]^T @ act(gu[rows_e]); A = ge (M, N1=H), B = act rows (M, N2=I) from gu
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
    for k0 in tl.range(0, n, BK):
        mk = (k0 + rk) < n
        rows = s0 + k0 + rk
        a = tl.load(A + rows[None, :] * sa + r1[:, None], mask=mk[None, :], other=0.0)
        r, rp = _row_r(PSQ, AL, rows, mk, I, NT, EPS)
        g = tl.load(GU + rows[:, None] * (2 * I) + r2[None, :], mask=mk[:, None], other=0.0).to(tl.float32)
        u = tl.load(GU + rows[:, None] * (2 * I) + (I + r2[None, :]), mask=mk[:, None], other=0.0).to(tl.float32)
        z = g / r[:, None]
        b = (rp[:, None] * (z * (1.0 / (1.0 + tl.exp(-z)))) * u).to(tl.bfloat16)
        acc = tl.dot(a, b, acc)
    slot = tl.load(IT_SLOT + item)
    off = r1[:, None] * I + r2[None, :]
    if slot < 0:
        tl.store(C + e.to(tl.int64) * N1 * I + off, acc.to(C.dtype.element_ty))
    else:
        tl.store(P + slot.to(tl.int64) * N1 * I + off, acc)


def main():
    N, H, E, k, I = 65536, 512, 64, 6, 768
    M = N * k
    g = torch.Generator(device=dev).manual_seed(0)
    logits = torch.randn(N, E, device=dev, generator=g) + 0.12 * torch.randn(E, device=dev, generator=g)
    wt, idx = torch.softmax(logits, -1).topk(k, -1)
    st, _, _, _, _, counts_t = K75._sort_by_expert(idx, wt, E, host=False)
    offs = counts_t.cumsum(0).to(torch.int32)
    x = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16)
    Wgu = (torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5).to(torch.bfloat16)
    Wdn = (torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5).to(torch.bfloat16)
    ge = (torch.randn(M, H, device=dev, generator=g) * 1e-2).to(torch.bfloat16)
    theta = torch.randn(E, device=dev, generator=g) * 0.3
    row_alpha = torch.repeat_interleave(theta, counts_t, output_size=M).contiguous()
    row_act = torch.full((M,), 8, device=dev, dtype=torch.int32)
    WdnT = Wdn.transpose(1, 2)

    # ---- current pipeline
    tm = FG.build_tile_map(None, counts_t, dev, m_rows=M)
    tgg = FG.build_tile_map(None, counts_t, dev, bm=FG._GG[0], m_rows=M)
    def cur_fwd():
        gu, _ = FG.fused_gate_up_glu(x, Wgu, tm, 8, want_gu=True, act=False, rows=st)
        it = K75._glu_fwd(gu, row_act, code_hint=8, row_alpha=row_alpha)
        eo = FG.grouped_gemm(it, WdnT, tgg)
        return gu, it, eo
    gu_c, it_c, eo_c = cur_fwd()
    dW_c = FG.grouped_wgrad(ge, it_c, offs)
    t_cf = timed(cur_fwd)
    t_cb2 = timed(lambda: FG.grouped_wgrad(ge, it_c, offs))
    t_parts = (timed(lambda: FG.fused_gate_up_glu(x, Wgu, tm, 8, want_gu=True, act=False, rows=st)),
               timed(lambda: K75._glu_fwd(gu_c, row_act, code_hint=8, row_alpha=row_alpha)),
               timed(lambda: FG.grouped_gemm(it_c, WdnT, tgg)))

    # ---- fused pipeline
    BM, BN, BK, nw, ns = FG._BM, FG._BN, FG._BK, FG._WARPS, FG._STAGES
    NT = I // BN
    gu_f = torch.empty(M, 2 * I, device=dev, dtype=torch.bfloat16)
    psq = torch.empty(M, NT, device=dev, dtype=torch.float32)
    eo_f = torch.empty(M, H, device=dev, dtype=torch.bfloat16)
    TE, TS, TM = tm
    TE2, TS2, TM2 = tgg
    GBM, GBN, GBK, gw, gs = FG._GG

    def f1():
        _f1_psq[(TE.numel() * NT,)](x, Wgu, gu_f, psq, TE, TS, TM, st, H, I, BM, BN, BK, num_warps=nw, num_stages=ns)

    def f3(cfg=(GBM, GBN, GBK, gw, gs)):
        bm, bn, bk, w_, s_ = cfg
        te, ts, tmm = (TE2, TS2, TM2) if bm == GBM else FG.build_tile_map(None, counts_t, dev, bm=bm, m_rows=M)
        _f3_act[(te.numel() * (H // bn),)](gu_f, psq, row_alpha, WdnT, eo_f, te, ts, tmm, WdnT.stride(0),
                                           WdnT.stride(1), WdnT.stride(2), I, H, NT, 1e-6, bm, bn, bk,
                                           num_warps=w_, num_stages=s_)

    f1(); f3()
    t_f1, t_f3 = timed(f1), timed(f3)
    best3 = min(((timed(lambda c=c: f3(c), it=5), c) for c in [(128, 256, 64, 8, 3), (128, 256, 32, 8, 4),
                                                            (128, 128, 32, 4, 4), (64, 256, 32, 8, 3),
                                                            (128, 128, 64, 8, 3), (256, 128, 32, 8, 3)]))
    # B2 with the act in its B load
    wgc = dict(FG._WG, **FG._WG_NARROW)
    CH, bm2, bn2, bk2 = wgc["CH"], wgc["BM"], wgc["BN"], wgc["BK"]
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
    it_n = torch.where(valid, (cnt[ie] - j * CH).clamp(0, CH), 0)
    it_slot = torch.where(valid & (nch[ie] > 1), i, -1)
    it_e = torch.where(valid, ie, -1)
    order = torch.argsort(it_n.to(torch.int16), descending=True, stable=True)
    dW_f = torch.empty(E, H, I, device=dev, dtype=torch.bfloat16)
    part = torch.empty(NI, H, I, device=dev, dtype=torch.float32)

    def b2(cfg=(bm2, bn2, bk2, wgc["num_warps"], wgc["num_stages"])):
        bm, bn, bk, w_, s_ = cfg
        nt2 = I // bn
        ntile = (H // bm) * nt2
        _b2_act[(NI * ntile,)](ge, gu_f, psq, row_alpha, dW_f, part, it_e.to(torch.int32), it_s,
                               it_n.to(torch.int32), it_slot.to(torch.int32), order, H, ge.stride(0),
                               I, NT, 1e-6, nt2, ntile, bm, bn, bk, num_warps=w_, num_stages=s_)
        FG._wg_reduce[(E, triton.cdiv(H * I, 1024))](part, dW_f, (cend - nch), nch, H * I, BLOCK=1024,
                                                    ACC=False, num_warps=4)
    b2()
    t_b2 = timed(b2)
    best2 = min(((timed(lambda c=c: b2(c), it=5), c) for c in [(128, 256, 32, 8, 3), (128, 128, 32, 4, 4),
                                                            (128, 128, 32, 8, 4), (64, 256, 32, 8, 3),
                                                            (128, 256, 64, 8, 2)]))

    # ---- numerics vs fp32 reference of the chain
    with torch.no_grad():
        gu32 = gu_c.float()                                    # same bf16 gu in both pipelines
        gg, uu = gu32[:, :I], gu32[:, I:]
        r = torch.sqrt((gg * gg).mean(1, keepdim=True) + 1e-6)
        p = torch.sigmoid(row_alpha)[:, None]
        it32 = r ** p * (gg / r) * torch.sigmoid(gg / r) * uu
        eo32 = torch.empty(M, H, device=dev)
        dW32 = torch.empty(E, H, I, device=dev)
        s = 0
        for ee, c in enumerate(counts_t.tolist()):
            eo32[s:s + c] = it32[s:s + c] @ Wdn[ee].float().t()
            dW32[ee] = ge[s:s + c].float().t() @ it32[s:s + c]
            s += c
    rel = lambda a, b: ((a.float() - b).norm() / b.norm()).item()
    print(f"gu identical between pipelines: {torch.equal(gu_c, gu_f)}")
    print(f"eo  rel err vs fp32: current {rel(eo_c, eo32):.3e}  fused {rel(eo_f, eo32):.3e}")
    print(f"dW  rel err vs fp32: current {rel(dW_c, dW32):.3e}  fused {rel(dW_f, dW32):.3e}")
    print(f"eo fused vs current: rel {rel(eo_f, eo_c.float()):.2e}, bitwise {torch.equal(eo_f, eo_c)}")
    print(f"\ncurrent: F1 {t_parts[0]:.3f} + act {t_parts[1]:.3f} + F3 {t_parts[2]:.3f} = fwd {t_cf:.3f} ms | B2 {t_cb2:.3f}")
    print(f"fused:   F1+psq {t_f1:.3f} + F3-act {t_f3:.3f} (best {best3[0]:.3f} {best3[1]}) | B2-act {t_b2:.3f} "
          f"(best {best2[0]:.3f} {best2[1]})")
    tot_c = t_cf + t_cb2
    tot_f = t_f1 + best3[0] + best2[0]
    print(f"fwd+B2: current {tot_c:.3f} ms -> fused {tot_f:.3f} ms ({tot_c - tot_f:+.3f} saved per layer)")
    print("RFUSE_DONE")


if __name__ == "__main__":
    main()
