"""Fused [qk-norm] + attention + [XSA] (kernels/sm120/attn_xsa.py) vs the production path, both
scored against fp64. python parity_check/parity_attn_xsa.py [--bench]

Production path = what BiBo runs today: RMSNorm(q), RMSNorm(k) in bf16 -> SDPA (flash) or flex ->
fused_xsa. The fused kernel passes if every output/grad error vs fp64 is within 1.5x of the
production path's own error, and its backward is bitwise repeatable.
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.sm120.attn_xsa import attn_xsa, attn_xsa_reference  # noqa: E402
from kernels.sm75.xsa import fused_xsa  # noqa: E402

dev, bf = "cuda", torch.bfloat16
H, HKV, D = 4, 2, 128
SC = 1.0 / math.sqrt(D)


def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def production(q, k, v, alpha, wq, wk, window, xsa, qs, ks, backend="flash"):
    if wq is not None:
        q, k = rms(q, wq), rms(k, wk)
    q, k = q * qs, k * ks
    if backend == "flex" or window is not None:
        from torch.nn.attention.flex_attention import flex_attention, create_block_mask
        S = q.shape[2]
        W = window if window is not None else S
        bm = create_block_mask(lambda b, h, i, j: (j <= i) & (i - j < W), None, None, S, S, device=dev)
        global _FLEX
        o = _FLEX(q, k, v, block_mask=bm, scale=SC, enable_gqa=True)
    else:
        G = q.shape[1] // k.shape[1]
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(G, 1), v.repeat_interleave(G, 1),
                                           is_causal=True, scale=SC)
    return fused_xsa(o, v, alpha) if xsa else o


from torch.nn.attention.flex_attention import flex_attention  # noqa: E402
_FLEX = torch.compile(flex_attention, dynamic=False)


def grads(fn, inputs, go):
    ins = [None if t is None else t.detach().clone().requires_grad_() for t in inputs]
    out = fn(*ins)
    out.backward(go)
    return [out.detach()] + [None if t is None else t.grad for t in ins]


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


CASES = [
    ("global, xsa+alpha, qk-norm", dict(window=None, xsa=True, alpha=True, norm=True, qs=1.0, ks=1.0)),
    ("global, xsa+alpha, no qk-norm", dict(window=None, xsa=True, alpha=True, norm=False, qs=1.0, ks=1.0)),
    ("global, qk-norm, q_scale 1.3 k_scale 0.8", dict(window=None, xsa=True, alpha=True, norm=True, qs=1.3, ks=0.8)),
    ("global, xsa off, qk-norm", dict(window=None, xsa=False, alpha=False, norm=True, qs=1.0, ks=1.0)),
    ("window 128, xsa+alpha, qk-norm", dict(window=128, xsa=True, alpha=True, norm=True, qs=1.0, ks=1.0)),
]
NAMES = ["out", "dq", "dk", "dv", "dalpha", "dwq", "dwk"]


def main():
    torch.manual_seed(0)
    B, S = 2, 1024
    ok = True
    for title, c in CASES:
        q = torch.randn(B, H, S, D, device=dev).to(bf)
        k = torch.randn(B, HKV, S, D, device=dev).to(bf)
        v = torch.randn(B, HKV, S, D, device=dev).to(bf)
        alpha = (torch.randn(H, device=dev) * 0.5) if c["alpha"] else None
        wq = (1 + 0.1 * torch.randn(D, device=dev)) if c["norm"] else None
        wk = (1 + 0.1 * torch.randn(D, device=dev)) if c["norm"] else None
        go = torch.randn(B, H, S, D, device=dev).to(bf)
        ins = [q, k, v, alpha, wq, wk]
        kw = dict(scale=SC, window=c["window"], xsa=c["xsa"], q_scale=c["qs"], k_scale=c["ks"])
        f_new = lambda q, k, v, a, wq, wk: attn_xsa(q, k, v, alpha=a, q_norm_w=wq, k_norm_w=wk, **kw)
        f_ref = lambda q, k, v, a, wq, wk: attn_xsa_reference(q, k, v, alpha=a, q_norm_w=wq, k_norm_w=wk,
                                                             dtype=torch.float64, **kw)
        f_prod = lambda q, k, v, a, wq, wk: production(q, k, v, a, wq, wk, c["window"], c["xsa"], c["qs"], c["ks"])
        ref = grads(f_ref, ins, go)
        new = grads(f_new, ins, go)
        prod = grads(f_prod, ins, go)
        reps = [grads(f_new, ins, go) for _ in range(2)]
        repeat = all(torch.equal(a, b) for r in reps for a, b in zip(new, r) if a is not None)
        print(f"== {title}  | bwd repeatable {repeat}")
        ok &= repeat
        for nm, a, p, r in zip(NAMES, new, prod, ref):
            if r is None:
                continue
            en, ep = rel(a, r), rel(p, r)
            good = en <= 1.5 * ep + 1e-6
            ok &= good
            print(f"   {nm:7s} fused {en:.2e}   production {ep:.2e}   {'OK' if good else 'WORSE'}")
    print("PARITY", "PASS" if ok else "FAIL")
    return ok


def bench():
    torch.manual_seed(0)
    B, S = 64, 1024
    q = torch.randn(B, H, S, D, device=dev).to(bf)
    k = torch.randn(B, HKV, S, D, device=dev).to(bf)
    v = torch.randn(B, HKV, S, D, device=dev).to(bf)
    alpha = torch.randn(H, device=dev) * 0.5
    wq, wk = torch.ones(D, device=dev), torch.ones(D, device=dev)
    go = torch.randn(B, H, S, D, device=dev).to(bf)

    def t(fn, n=10):
        ins = [x.detach().clone().requires_grad_() for x in (q, k, v, alpha, wq, wk)]
        for _ in range(3):
            fn(*ins).backward(go)
        e = [torch.cuda.Event(True) for _ in range(3)]
        torch.cuda.synchronize(); e[0].record()
        for _ in range(n):
            fn(*ins)
        e[1].record()
        for _ in range(n):
            fn(*ins).backward(go)
        e[2].record(); torch.cuda.synchronize()
        return e[0].elapsed_time(e[1]) / n, e[1].elapsed_time(e[2]) / n

    print(f"== bench B{B} H{H} Hkv{HKV} S{S} D{D} (qk-norm + attention + xsa, ms)")
    for window in (None, 128):
        rows = [("fused attn_xsa", lambda q, k, v, a, wq, wk: attn_xsa(q, k, v, scale=SC, window=window, alpha=a,
                                                                      q_norm_w=wq, k_norm_w=wk))]
        if window is None:
            rows.append(("production: rmsnorm + sdpa flash + xsa",
                         lambda q, k, v, a, wq, wk: production(q, k, v, a, wq, wk, None, True, 1.0, 1.0, "flash")))
        rows.append(("production: rmsnorm + flex + xsa",
                     lambda q, k, v, a, wq, wk: production(q, k, v, a, wq, wk, window, True, 1.0, 1.0, "flex")))
        print(f"   {'global causal' if window is None else f'window {window}'}")
        for name, fn in rows:
            try:
                tf, tfb = t(fn)
                print(f"     {name:42s} fwd {tf:7.3f}   fwd+bwd {tfb:7.3f}", flush=True)
            except Exception as ex:
                print(f"     {name:42s} FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:100]}", flush=True)


def sweep():
    """Greedy per-kernel tile sweep at the board shape; prints the winners and leaves them in CFG."""
    import kernels.sm120.attn_xsa as AX
    torch.manual_seed(0)
    B, S = 64, 1024
    q = torch.randn(B, H, S, D, device=dev).to(bf)
    k = torch.randn(B, HKV, S, D, device=dev).to(bf)
    v = torch.randn(B, HKV, S, D, device=dev).to(bf)
    alpha = torch.randn(H, device=dev) * 0.5
    wq, wk = torch.ones(D, device=dev), torch.ones(D, device=dev)
    go = torch.randn(B, H, S, D, device=dev).to(bf)

    def run(window, bwd, n=8):
        ins = [x.detach().clone().requires_grad_() for x in (q, k, v, alpha, wq, wk)]
        f = lambda: attn_xsa(*ins[:3], scale=SC, window=window, alpha=ins[3], q_norm_w=ins[4], k_norm_w=ins[5])
        for _ in range(2):
            o = f(); o.backward(go) if bwd else None
        e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
        torch.cuda.synchronize(); e0.record()
        for _ in range(n):
            o = f()
            if bwd:
                o.backward(go)
        e1.record(); torch.cuda.synchronize()
        return e0.elapsed_time(e1) / n

    space = {
        "fwd": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
                ((64, 64, 8, 2), (64, 64, 4, 2), (64, 64, 8, 3), (32, 64, 4, 2), (32, 64, 4, 3), (64, 32, 4, 3), (64, 128, 8, 2))],
        "dkdv": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
                 ((64, 64, 8, 1), (32, 64, 4, 2), (32, 64, 8, 2), (32, 64, 4, 3), (64, 32, 4, 2), (32, 32, 4, 2),
                  (32, 32, 4, 3), (16, 64, 4, 3), (32, 128, 8, 2))],
        "dq": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
               ((64, 64, 8, 1), (32, 64, 4, 2), (32, 64, 8, 2), (64, 32, 8, 2), (64, 32, 4, 2), (32, 32, 4, 2),
                (32, 32, 4, 3), (16, 64, 4, 2), (32, 64, 4, 3))],
    }
    for window in (None, 128):
        print(f"== sweep, {'global causal' if window is None else f'window {window}'}")
        for part in ("fwd", "dkdv", "dq"):
            best = None
            for cfg in space[part]:
                AX.CFG[part] = cfg
                try:
                    t = run(window, bwd=(part != "fwd"))
                except Exception as ex:
                    print(f"   {part:5s} {cfg} FAILED {type(ex).__name__}", flush=True)
                    continue
                print(f"   {part:5s} {cfg}  {t:7.3f} ms", flush=True)
                if best is None or t < best[0]:
                    best = (t, cfg)
            AX.CFG[part] = best[1]
            print(f"   -> {part} best {best[1]} {best[0]:.3f} ms", flush=True)
        print(f"   FINAL {window}: {AX.CFG}  fwd {run(window, False):.3f}  fwd+bwd {run(window, True):.3f}", flush=True)


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    good = main()
    if "--bench" in sys.argv:
        bench()
    sys.exit(0 if good else 1)
