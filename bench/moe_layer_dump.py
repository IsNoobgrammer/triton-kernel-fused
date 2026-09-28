"""Dump one MoE layer's output and all grads (radial, board shapes) for bitwise A/B across code versions.

    python -m bench.moe_layer_dump OUT.pt [--skew 0.12]      then compare two dumps with --cmp A B
"""
import sys

import torch

from bench.bench_moe_layer import make, layer_step, timed

if __name__ == "__main__":
    if sys.argv[1] == "--cmp":
        a, b = torch.load(sys.argv[2]), torch.load(sys.argv[3])
        for k in a:
            if k == "ms":
                continue
            d = (a[k].float() - b[k].float()).abs().max().item()
            print(f"  {k:8s} bitwise {torch.equal(a[k], b[k])}  max|diff| {d:.3e}")
        print(f"  layer ms: {a['ms']:.3f} -> {b['ms']:.3f}")
        sys.exit(0)
    skew = float(sys.argv[sys.argv.index("--skew") + 1]) if "--skew" in sys.argv else 0.12
    x, idx, w, gu, dn, codes, theta, mv = make(skew=skew)
    f = layer_step(x, idx, w, gu, dn, codes, theta)
    f()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        from kernels.sm120.moe import moe_per_expert
        with torch.no_grad():
            y = moe_per_expert(x, idx, w, gu, dn, codes, act_params=theta)
    out = dict(y=y, gx=x.grad, ggu=gu.grad, gdn=dn.grad, gth=theta.grad, ms=timed(f, it=15))
    torch.save(out, sys.argv[1])
    print(f"MaxVio {mv:.2f}  layer fwd+bwd {out['ms']:.3f} ms  -> {sys.argv[1]}")
