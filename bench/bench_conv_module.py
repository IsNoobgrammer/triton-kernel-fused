"""conv_module and silu_dropout vs NeMo's eager ops, fwd and fwd+bwd, bf16 autocast, at a real ASR batch
(~2400 s of audio = ~30k encoder frames, C=512, K=9 causal; FFN d_ff 2048, dropout 0.1).

    python bench/bench_conv_module.py [--sweep]          # --sweep: rows per program x warps for conv_module
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import triton

from kernels.sm120 import conv_module as cm
from kernels.sm120.silu_dropout import silu_dropout

B, T, C, K, LEFT = 16, 1900, 512, 9, 8


def eager_conv(g, pad, cw, cb, lw, lb):
    x = F.glu(g, dim=-1).transpose(1, 2)
    x = x.masked_fill(pad.unsqueeze(1), 0.0)
    x = F.conv1d(F.pad(x, (LEFT, 0)), cw, cb, groups=C)
    return F.silu(F.layer_norm(x.transpose(1, 2), (C,), lw, lb, 1e-5)).to(g.dtype)


def bench(fn, args, grad_out):
    def fb():
        for a in args:
            if a.requires_grad:
                a.grad = None
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fn(*args)
        y.backward(grad_out)

    def f():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            fn(*args)
    return triton.testing.do_bench(f) * 1e3, triton.testing.do_bench(fb) * 1e3


def main():
    torch.manual_seed(0)
    g = torch.randn(B, T, 2 * C, device="cuda").bfloat16().requires_grad_()
    lengths = torch.randint(T // 2, T + 1, (B,), device="cuda")
    pad = torch.arange(T, device="cuda")[None, :] >= lengths[:, None]
    ps = [(torch.randn(C, 1, K, device="cuda") / 3).requires_grad_(), torch.zeros(C, device="cuda").requires_grad_(),
          torch.ones(C, device="cuda").requires_grad_(), torch.zeros(C, device="cuda").requires_grad_()]
    dy = torch.randn(B, T, C, device="cuda").bfloat16()
    ours = lambda g_, *p: cm.conv_module(g_, pad, *p, 1e-5, LEFT, out_dtype=torch.bfloat16)
    ef, eb = bench(lambda g_, *p: eager_conv(g_, pad, *p), [g] + ps, dy)
    if "--sweep" in sys.argv:
        for bt in (4, 8, 16, 32):
            for nw in (4, 8, 16):
                cm._CFG.update(fwd=(bt, nw), bwd=(bt, nw))
                f, fb = bench(ours, [g] + ps, dy)
                print(f"  BT {bt:2d} warps {nw:2d}: fwd {f:6.0f} us  bwd {fb - f:6.0f} us", flush=True)
        cm._CFG.update(fwd=None, bwd=None)
    of, ob = bench(ours, [g] + ps, dy)
    print(f"conv_module B={B} T={T} ({B * T} frames): eager fwd {ef:.0f} us fwd+bwd {eb:.0f} us | "
          f"ours fwd {of:.0f} us ({ef / of:.2f}x) fwd+bwd {ob:.0f} us ({eb / ob:.2f}x)")

    x = torch.randn(B * T, 2048, device="cuda").bfloat16().requires_grad_()
    dx = torch.randn(B * T, 2048, device="cuda").bfloat16()
    ef, eb = bench(lambda a: F.dropout(F.silu(a), 0.1, True), [x], dx)
    of, ob = bench(lambda a: silu_dropout(a, 0.1, seed=1), [x], dx)
    print(f"silu_dropout ({B * T}, 2048): eager fwd {ef:.0f} us fwd+bwd {eb:.0f} us | "
          f"ours fwd {of:.0f} us ({ef / of:.2f}x) fwd+bwd {ob:.0f} us ({eb / ob:.2f}x)")


if __name__ == "__main__":
    main()
