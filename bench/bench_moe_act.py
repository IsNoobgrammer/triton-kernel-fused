"""Radial NormSiLU row kernels (code 8) at the board shape, M=393216 rows x I=768: current rowloop
kernels vs variants -- BR rows per program, the code specialised at compile time, warps. Bitwise
checked. Also the card's practical copy bandwidth, to know the ceiling.

    python -m bench.bench_moe_act
"""
import itertools

import torch
import triton
import triton.language as tl

import kernels.sm75.moe as K75
from bench.bench_moe_gemm import timed

dev = "cuda"


@triton.jit
def _fwd_br(GU, AL, OUT, M, I: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr, BI: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = rows < M
    r64 = rows.to(tl.int64)
    aa = tl.load(AL + rows, mask=mr, other=0.0).to(tl.float32)
    acc = tl.zeros([BR, BI], dtype=tl.float32)
    for i0 in tl.static_range(0, I, BI):
        offs = i0 + tl.arange(0, BI)
        g = tl.load(GU + r64[:, None] * (2 * I) + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        acc += g * g
    r = tl.sqrt(tl.sum(acc, axis=1) / I + EPS)
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    rp = tl.exp(p8 * tl.log(r))
    for i0 in tl.static_range(0, I, BI):
        offs = i0 + tl.arange(0, BI)
        g = tl.load(GU + r64[:, None] * (2 * I) + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        u = tl.load(GU + r64[:, None] * (2 * I) + (I + offs)[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        z = g / r[:, None]
        f = z * (1.0 / (1.0 + tl.exp(-z)))
        tl.store(OUT + r64[:, None] * I + offs[None, :], (rp[:, None] * f * u).to(OUT.dtype.element_ty), mask=mr[:, None])


@triton.jit
def _bwd_br(GO, GU, AL, GGU, DA, M, I: tl.constexpr, EPS: tl.constexpr, BR: tl.constexpr, BI: tl.constexpr):
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    mr = rows < M
    r64 = rows.to(tl.int64)
    aa = tl.load(AL + rows, mask=mr, other=0.0).to(tl.float32)
    acc = tl.zeros([BR, BI], dtype=tl.float32)
    for i0 in tl.static_range(0, I, BI):
        offs = i0 + tl.arange(0, BI)
        g = tl.load(GU + r64[:, None] * (2 * I) + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        acc += g * g
    r = tl.sqrt(tl.sum(acc, axis=1) / I + EPS)
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    lr8 = tl.log(r)
    rp = tl.exp(p8 * lr8)
    rpm1 = tl.exp((p8 - 1.0) * lr8)
    sa = tl.zeros([BR, BI], dtype=tl.float32)
    tt = tl.zeros([BR, BI], dtype=tl.float32)
    for i0 in tl.static_range(0, I, BI):
        offs = i0 + tl.arange(0, BI)
        go = tl.load(GO + r64[:, None] * I + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        g = tl.load(GU + r64[:, None] * (2 * I) + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        u = tl.load(GU + r64[:, None] * (2 * I) + (I + offs)[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        gn = g / r[:, None]
        sig = 1.0 / (1.0 + tl.exp(-gn))
        f = gn * sig
        df = sig * (1.0 + gn * (1.0 - sig))
        gu_ = go * u
        sa += gu_ * df * gn
        tt += gu_ * f
    S = tl.sum(sa, axis=1)
    T = tl.sum(tt, axis=1)
    for i0 in tl.static_range(0, I, BI):
        offs = i0 + tl.arange(0, BI)
        go = tl.load(GO + r64[:, None] * I + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        g = tl.load(GU + r64[:, None] * (2 * I) + offs[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        u = tl.load(GU + r64[:, None] * (2 * I) + (I + offs)[None, :], mask=mr[:, None], other=0.0).to(tl.float32)
        gn = g / r[:, None]
        sig = 1.0 / (1.0 + tl.exp(-gn))
        f = gn * sig
        df = sig * (1.0 + gn * (1.0 - sig))
        gu_ = go * u
        gg = rpm1[:, None] * (gu_ * df - (gn / I) * (S - p8 * T)[:, None])
        tl.store(GGU + r64[:, None] * (2 * I) + offs[None, :], gg.to(GGU.dtype.element_ty), mask=mr[:, None])
        tl.store(GGU + r64[:, None] * (2 * I) + (I + offs)[None, :], (go * (rp[:, None] * f)).to(GGU.dtype.element_ty), mask=mr[:, None])
    da = p8 * (1.0 - p8) * rp * lr8 * T
    tl.store(DA + rows, da, mask=mr)


def main():
    M, I = 393216, 768
    g = torch.Generator(device=dev).manual_seed(0)
    gu = torch.randn(M, 2 * I, device=dev, generator=g).to(torch.bfloat16)
    go = (torch.randn(M, I, device=dev, generator=g) * 1e-3).to(torch.bfloat16)
    act = torch.full((M,), 8, device=dev, dtype=torch.int32)
    al = (torch.randn(64, device=dev, generator=g) * 0.3).repeat_interleave(M // 64).contiguous()

    big = torch.empty(int(1.8e9) // 2, device=dev, dtype=torch.bfloat16)
    dst = torch.empty_like(big)
    ms = timed(lambda: dst.copy_(big))
    print(f"copy bandwidth: {2 * big.numel() * 2 / ms / 1e6:.0f} GB/s ({big.numel() * 2 / 1e9:.1f} GB r + w)")
    del big, dst

    ref_f = K75._glu_fwd(gu, act, code_hint=8, row_alpha=al)
    t_f = timed(lambda: K75._glu_fwd(gu, act, code_hint=8, row_alpha=al))
    ref_b, ref_da = K75._glu_bwd(go, gu, act, code_hint=8, row_alpha=al, want_act_grads=True)
    t_b = timed(lambda: K75._glu_bwd(go, gu, act, code_hint=8, row_alpha=al, want_act_grads=True))
    fb, bb = M * I * 2 * 3, M * I * 2 * 5          # bytes: fwd gu r + out w; bwd go r + gu r + ggu w
    print(f"current fwd {t_f:.3f} ms ({fb / t_f / 1e6:.0f} GB/s)   bwd {t_b:.3f} ms ({bb / t_b / 1e6:.0f} GB/s)", flush=True)

    out = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
    ggu = torch.empty_like(gu)
    da = torch.empty(M, device=dev)
    for BR, BI, nw in itertools.product((1, 2, 4, 8, 16), (256,), (2, 4, 8)):
        try:
            f = lambda: _fwd_br[(triton.cdiv(M, BR),)](gu, al, out, M, I, 1e-6, BR, BI, num_warps=nw)
            f()
            sf = torch.equal(out, ref_f)
            tf = timed(f, it=5)
            b = lambda: _bwd_br[(triton.cdiv(M, BR),)](go, gu, al, ggu, da, M, I, 1e-6, BR, BI, num_warps=nw)
            b()
            sb = torch.equal(ggu, ref_b) and torch.equal(da, ref_da)
            tb = timed(b, it=5)
            print(f"  BR={BR:<2d} BI={BI} w={nw}: fwd {tf:.3f} ms ({fb / tf / 1e6:.0f} GB/s) bitwise {sf} | "
                  f"bwd {tb:.3f} ms ({bb / tb / 1e6:.0f} GB/s) bitwise {sb}", flush=True)
        except Exception as ex:
            print(f"  BR={BR} w={nw}: {repr(ex)[:120]}")
    print("ACT_DONE")


if __name__ == "__main__":
    main()
