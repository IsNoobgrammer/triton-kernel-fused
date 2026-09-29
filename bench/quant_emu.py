"""8-bit formats by EMULATION, including e3m4 (no sm120 tensor-core path): error / underflow / saturation.

    python -m bench.quant_emu

1. Validates bench.quant_mnist.emu_round BIT-EXACTLY against torch's e4m3fn and e5m2 casts (so the
   e3m4 numbers come from the same arithmetic the hardware formats use).
2. For e4m3 / e5m2 / e3m4 x scale e8m0 / fp32 / bf16 / fp16 x block 32 / 64 / 128, on three tensors:
     act      Gaussian activations
     tail     + 0.2% spikes x300 and one outlier token x50 (our down_proj input, amax/median ~2e4)
     grad     the tail tensor x 1e-5 (gradient magnitudes: what breaks fp16 scales)
   reports GEMM error (A (4096 x 1024) @ W (1024 x 1024), both quantized along K, vs fp32 on the
   originals), and A's flushed-to-zero / subnormal / saturated rates.
"""
import itertools

import torch

from bench.quant_mnist import FMT, emu_round, fq

dev = "cuda"


def validate():
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1 << 20, device=dev, generator=g) * torch.exp2(torch.randint(-20, 12, (1 << 20,), device=dev,
                                                                                     generator=g).float())
    for name, dt, E, M, bias, mx in (("e4m3", torch.float8_e4m3fn, 4, 3, 7, 448.0),
                                     ("e5m2", torch.float8_e5m2, 5, 2, 15, 57344.0)):
        xs = x.clamp(-mx, mx)
        hw = xs.to(dt).float()
        em = emu_round(xs, E, M, bias, mx)
        n_bad = (hw != em).sum().item()
        print(f"emulator vs torch {name}: {n_bad} of {x.numel()} values differ "
              f"({'BIT-EXACT' if n_bad == 0 else 'MISMATCH'})", flush=True)


def stats(t, fmt, scale, blk):
    _, emax, mn = FMT[fmt]
    y = fq(t, 1, fmt, scale, blk, None)
    xb = t.float().reshape(t.shape[0], -1, blk)
    amax = xb.abs().amax(-1, keepdim=True).clamp_min(1e-30)
    if scale == "e8m0":
        s = torch.exp2(torch.ceil(torch.log2(amax / emax)).clamp(-127, 127))
    elif scale == "fp32":
        s = amax / emax
    elif scale == "bf16":
        s = (amax / emax * (1 + 2 ** -7)).to(torch.bfloat16).float()
    else:
        s = (amax / emax * (1 + 2 ** -10)).to(torch.float16).float().clamp_min(2 ** -24)
    r = (xb / s).abs()
    nz = xb != 0
    flushed = ((y.reshape(xb.shape) == 0) & nz).float().sum().item() / nz.sum().item()
    sub = ((r < mn) & nz).float().sum().item() / nz.sum().item()
    sat = (r > emax * 1.0001).float().mean().item()
    return y, flushed, sub, sat


def main():
    validate()
    g = torch.Generator(device=dev).manual_seed(1)
    A = torch.randn(4096, 1024, device=dev, generator=g)
    W = torch.randn(1024, 1024, device=dev, generator=g) * 0.02
    T = torch.where(torch.rand(A.shape, device=dev, generator=g) < 0.002, A * 300, A)
    T[7] *= 50
    tensors = {"act": A, "tail": T, "grad": T * 1e-5}
    print(f"\n{'fmt':5s} {'scale':5s} {'blk':>4s} | " + " | ".join(
        f"{k:>4s}: {'gemm err':>9s} {'flush%':>7s} {'subn%':>6s} {'sat%':>6s}" for k in tensors))
    for fmt, scale, blk in itertools.product(("e4m3", "e5m2", "e3m4"), ("e8m0", "fp32", "bf16", "fp16"), (32, 64, 128)):
        cells = []
        for k, t in tensors.items():
            ref = t @ W.t()
            y, fl, sb, st = stats(t, fmt, scale, blk)
            wq = fq(W, 1, fmt, scale, blk, None) if k != "grad" else fq(W, 1, fmt, scale, blk, None)
            err = ((y @ wq.t() - ref).norm() / ref.norm()).item()
            cells.append(f"      {err:9.2e} {100 * fl:7.4f} {100 * sb:6.3f} {100 * st:6.3f}")
        print(f"{fmt:5s} {scale:5s} {blk:4d} |" + " |".join(cells), flush=True)
    print("QUANT_EMU_DONE")


if __name__ == "__main__":
    main()
