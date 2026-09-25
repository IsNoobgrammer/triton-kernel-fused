"""Fused [qk-norm] + [RoPE] + attention + [XSA] (kernels/sm120/attn_xsa.py) vs the production path,
both scored against fp64.

    python parity_check/parity_attn_xsa.py            parity (all configs, model layout)
    python parity_check/parity_attn_xsa.py --sweep    + per-kernel tile sweep at the board shape
    python parity_check/parity_attn_xsa.py --bench    + board-shape speed vs flash / flex
    python parity_check/parity_attn_xsa.py --scale    + sequence-length scaling: speed, memory, error

Production path = what BiBo runs: RMSNorm(q), RMSNorm(k) -> RoPE (SWA layers) -> SDPA flash / flex ->
fused_xsa. Pass = every output/grad error vs fp64 within 1.5x of production's own, and a bitwise
repeatable backward. Inputs are in the MODEL layout: (B, S, H, D) projections viewed as (B, H, S, D).
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernels.sm120.attn_xsa as AX  # noqa: E402
from kernels.sm120.attn_xsa import attn_xsa, attn_xsa_reference  # noqa: E402
from kernels.sm75.xsa import fused_xsa  # noqa: E402
from torch.nn.attention.flex_attention import flex_attention, create_block_mask  # noqa: E402

dev, bf = "cuda", torch.bfloat16
H, HKV, D = 4, 2, 128
SC = 1.0 / math.sqrt(D)
_FLEX = torch.compile(flex_attention, dynamic=False)
_BM = {}


def block_mask(S, W):
    key = (S, W)
    if key not in _BM:
        _BM[key] = create_block_mask(lambda b, h, i, j: (j <= i) & (i - j < W), None, None, S, S, device=dev)
    return _BM[key]


def rope_tables(S, base=10000.0):
    inv = 1.0 / (base ** (torch.arange(0, D, 2, device=dev).float() / D))
    f = torch.outer(torch.arange(S, device=dev).float(), inv)
    e = torch.cat([f, f], -1)
    return e.cos().to(bf), e.sin().to(bf)


def rms(x, w, eps=1e-6):
    xf = x.float()
    return (xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps) * w.float()).to(x.dtype)


def rot(x):
    return torch.cat((-x[..., D // 2:], x[..., : D // 2]), -1)


def production(q, k, v, alpha, wq, wk, window, xsa, qs, ks, cos, sin, backend):
    if wq is not None:
        q, k = rms(q, wq), rms(k, wk)
    q, k = q * qs, k * ks
    if cos is not None:
        q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    S = q.shape[2]
    if backend == "flash" and window is None:
        G = q.shape[1] // k.shape[1]
        o = F.scaled_dot_product_attention(q, k.repeat_interleave(G, 1), v.repeat_interleave(G, 1),
                                           is_causal=True, scale=SC)
    else:
        o = _FLEX(q, k, v, block_mask=block_mask(S, window if window is not None else S), scale=SC,
                  enable_gqa=True)
    return fused_xsa(o, v, alpha) if xsa else o


def model_layout(B, S, heads):
    """A (B, S, heads*D) projection output viewed as (B, heads, S, D) -- exactly what BiBo passes."""
    return torch.randn(B, S, heads, D, device=dev).to(bf).transpose(1, 2)


def grads(fn, inputs, go):
    # clone() preserves strides, so q/k/v stay in the model's transposed (B, S, H, D) layout
    ins = [None if t is None else t.detach().clone().requires_grad_() for t in inputs]
    out = fn(*ins)
    out.backward(go)
    return [out.detach()] + [None if t is None else t.grad for t in ins]


def rel(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


CASES = [
    ("global, xsa+alpha, qk-norm (BiBo global)", dict(window=None, xsa=True, alpha=True, norm=True, qs=1.0, ks=1.0, rope=False)),
    ("global, xsa+alpha, no qk-norm", dict(window=None, xsa=True, alpha=True, norm=False, qs=1.0, ks=1.0, rope=False)),
    ("global, qk-norm, q_scale 1.3 k_scale 0.8", dict(window=None, xsa=True, alpha=True, norm=True, qs=1.3, ks=0.8, rope=False)),
    ("global, xsa off, qk-norm", dict(window=None, xsa=False, alpha=False, norm=True, qs=1.0, ks=1.0, rope=False)),
    ("window 128, rope, qk-norm, xsa+alpha (BiBo SWA)", dict(window=128, xsa=True, alpha=True, norm=True, qs=1.0, ks=1.0, rope=True)),
    ("window 128, rope, no qk-norm, xsa off", dict(window=128, xsa=False, alpha=False, norm=False, qs=1.0, ks=1.0, rope=True)),
    ("global, rope, qk-norm, xsa+alpha", dict(window=None, xsa=True, alpha=True, norm=True, qs=1.0, ks=1.0, rope=True)),
]
NAMES = ["out", "dq", "dk", "dv", "dalpha", "dwq", "dwk"]


def run_case(title, c, B, S, verbose=True):
    q, k, v = model_layout(B, S, H), model_layout(B, S, HKV), model_layout(B, S, HKV)
    alpha = (torch.randn(H, device=dev) * 0.5) if c["alpha"] else None
    wq = (1 + 0.1 * torch.randn(D, device=dev)) if c["norm"] else None
    wk = (1 + 0.1 * torch.randn(D, device=dev)) if c["norm"] else None
    cos, sin = rope_tables(S) if c["rope"] else (None, None)
    go = torch.randn(B, H, S, D, device=dev).to(bf)
    ins = [q, k, v, alpha, wq, wk]
    kw = dict(scale=SC, window=c["window"], xsa=c["xsa"], q_scale=c["qs"], k_scale=c["ks"], cos=cos, sin=sin)
    f_new = lambda q, k, v, a, wq, wk: attn_xsa(q, k, v, alpha=a, q_norm_w=wq, k_norm_w=wk, **kw)
    f_ref = lambda q, k, v, a, wq, wk: attn_xsa_reference(q, k, v, alpha=a, q_norm_w=wq, k_norm_w=wk,
                                                         dtype=torch.float64, **kw)
    f_prod = lambda q, k, v, a, wq, wk: production(q, k, v, a, wq, wk, c["window"], c["xsa"], c["qs"],
                                                   c["ks"], cos, sin, "flash")
    ref, new, prod = grads(f_ref, ins, go), grads(f_new, ins, go), grads(f_prod, ins, go)
    reps = [grads(f_new, ins, go) for _ in range(2)]
    repeat = all(torch.equal(a, b) for r in reps for a, b in zip(new, r) if a is not None)
    layout = new[1].stride() == ins[0].stride() if new[1] is not None else True
    ok = repeat and layout
    if verbose:
        print(f"== {title} [B{B} S{S}] | bwd repeatable {repeat} | grads keep model layout {layout}")
    for nm, a, p, r in zip(NAMES, new, prod, ref):
        if r is None:
            continue
        en, ep = rel(a, r), rel(p, r)
        good = en <= 1.5 * ep + 1e-6
        ok &= good
        if verbose:
            print(f"   {nm:7s} fused {en:.2e}   production {ep:.2e}   {'OK' if good else 'WORSE'}")
    return ok


def main():
    torch.manual_seed(0)
    ok = True
    for title, c in CASES:
        ok &= run_case(title, c, 2, 1024)
    for S in (256, 4096):                              # other lengths, BiBo global + SWA configs
        ok &= run_case(CASES[0][0], CASES[0][1], 1, S)
        ok &= run_case(CASES[4][0], CASES[4][1], 1, S)
    print("PARITY", "PASS" if ok else "FAIL")
    return ok


def _time(fn, ins, go, n=8):
    for _ in range(2):
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


def _peak(fn, ins, go):
    torch.cuda.synchronize(); torch.cuda.empty_cache()
    base = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    fn(*ins).backward(go)
    torch.cuda.synchronize()
    return (torch.cuda.max_memory_allocated() - base) / 2 ** 20


def _inputs(B, S, norm=True):
    q, k, v = model_layout(B, S, H), model_layout(B, S, HKV), model_layout(B, S, HKV)
    ins = [t.requires_grad_() for t in (q, k, v)]
    alpha = (torch.randn(H, device=dev) * 0.5).requires_grad_()
    wq = torch.ones(D, device=dev, requires_grad=True) if norm else None
    wk = torch.ones(D, device=dev, requires_grad=True) if norm else None
    return ins + [alpha, wq, wk], torch.randn(B, H, S, D, device=dev).to(bf)


def bench():
    torch.manual_seed(0)
    B, S = 64, 1024
    print(f"== board bench B{B} H{H} Hkv{HKV} S{S} D{D}, model layout (ms: fwd | fwd+bwd | peak MiB)")
    for window, rope in ((None, False), (128, True)):
        cos, sin = rope_tables(S) if rope else (None, None)
        ins, go = _inputs(B, S)
        rows = [("fused attn_xsa", lambda q, k, v, a, wq, wk: attn_xsa(
            q, k, v, scale=SC, window=window, alpha=a, q_norm_w=wq, k_norm_w=wk, cos=cos, sin=sin))]
        if window is None:
            rows.append(("rmsnorm + sdpa flash + xsa", lambda q, k, v, a, wq, wk: production(
                q, k, v, a, wq, wk, None, True, 1.0, 1.0, cos, sin, "flash")))
        rows.append(("rmsnorm + rope + flex + xsa" if rope else "rmsnorm + flex + xsa",
                     lambda q, k, v, a, wq, wk: production(q, k, v, a, wq, wk, window, True, 1.0, 1.0, cos, sin, "flex")))
        print(f"   {'global causal (NoPE)' if window is None else f'window {window} + rope'}")
        for name, fn in rows:
            tf, tfb = _time(fn, ins, go)
            print(f"     {name:34s} {tf:7.3f} | {tfb:7.3f} | {_peak(fn, ins, go):8.0f}", flush=True)


def scale():
    """Global causal attention vs sequence length at a FIXED 64k tokens per batch."""
    torch.manual_seed(0)
    print("== seq-len scaling, global causal, 65536 tokens/batch (ms: fwd | fwd+bwd | peak MiB)")
    torch._dynamo.config.recompile_limit = 64
    for S in (512, 1024, 2048, 4096, 8192, 16384):
        B = max(1, 65536 // S)
        torch._dynamo.reset()          # flex compiles per shape; never let it fall back to eager
        ins, go = _inputs(B, S, norm=True)
        rows = [
            ("ours: attention only", lambda q, k, v, a, wq, wk: attn_xsa(q, k, v, scale=SC, xsa=False), False),
            ("flash (enable_gqa)", lambda q, k, v, a, wq, wk: F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=SC, enable_gqa=True), False),
            ("flash, deterministic mode", lambda q, k, v, a, wq, wk: F.scaled_dot_product_attention(
                q, k, v, is_causal=True, scale=SC, enable_gqa=True), True),
            ("flex causal", lambda q, k, v, a, wq, wk: _FLEX(q, k, v, block_mask=block_mask(S, S), scale=SC,
                                                             enable_gqa=True), False),
            ("ours: qk-norm + attn + xsa", lambda q, k, v, a, wq, wk: attn_xsa(
                q, k, v, scale=SC, alpha=a, q_norm_w=wq, k_norm_w=wk), False),
            ("rmsnorm + flash + xsa", lambda q, k, v, a, wq, wk: production(
                q, k, v, a, wq, wk, None, True, 1.0, 1.0, None, None, "flash"), False),
        ]
        print(f"   S={S:5d} B={B:3d}")
        for name, fn, det in rows:
            try:
                torch.use_deterministic_algorithms(det)
                tf, tfb = _time(fn, ins, go, n=5)
                mem = _peak(fn, ins, go)
                print(f"     {name:28s} {tf:8.3f} | {tfb:8.3f} | {mem:8.0f}", flush=True)
            except Exception as ex:
                print(f"     {name:28s} FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:80]}", flush=True)
            finally:
                torch.use_deterministic_algorithms(False)
        if S <= 4096:                                   # accuracy vs fp64 at this length (B=1)
            q1, k1, v1 = (t[:1].detach() for t in ins[:3])
            r = attn_xsa_reference(q1, k1, v1, scale=SC, xsa=False, dtype=torch.float64)
            o = attn_xsa(q1, k1, v1, scale=SC, xsa=False)
            f_ = F.scaled_dot_product_attention(q1, k1, v1, is_causal=True, scale=SC, enable_gqa=True)
            print(f"     fwd rel err vs fp64: ours {rel(o, r):.2e}  flash {rel(f_, r):.2e}", flush=True)


def sweep_long(S=4096):
    """Tile sweep for PURE causal attention at a long sequence (the causal CFG is shared with the
    board's global layers, so the winner must not regress the board shape -- checked after)."""
    torch.manual_seed(0)
    B = 65536 // S
    ins, go = _inputs(B, S, norm=False)
    fn = lambda q, k, v, a, wq, wk: attn_xsa(q, k, v, scale=SC, xsa=False)
    space = {
        "fwd": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
                ((64, 32, 8, 3), (64, 64, 8, 2), (64, 64, 4, 2), (64, 64, 8, 3), (64, 32, 4, 3), (32, 64, 4, 3),
                 (64, 128, 8, 2))],
        "dkdv": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
                 ((64, 32, 8, 1), (32, 32, 4, 2), (32, 64, 8, 2), (64, 64, 8, 1), (32, 64, 4, 2), (64, 32, 8, 2),
                  (32, 128, 8, 1), (64, 128, 8, 1))],
        "dq": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
               ((64, 32, 8, 2), (64, 64, 8, 2), (64, 64, 8, 1), (32, 64, 4, 2), (64, 32, 8, 3), (32, 32, 4, 3),
                (64, 64, 4, 2))],
    }
    print(f"== long sweep, pure causal S={S} B={B}")
    for part in ("fwd", "dkdv", "dq"):
        best = None
        for cfg in space[part]:
            AX.CFG["causal"][part] = cfg
            AX._FIT.clear()
            try:
                tf, tfb = _time(fn, ins, go, n=5)
            except Exception as ex:
                print(f"   {part:5s} {cfg} FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:70]}", flush=True)
                continue
            t = tf if part == "fwd" else tfb
            print(f"   {part:5s} {cfg}  {t:7.3f} ms", flush=True)
            if best is None or t < best[0]:
                best = (t, cfg)
        AX.CFG["causal"][part] = best[1]
        AX._FIT.clear()
        print(f"   -> {part} best {best[1]} {best[0]:.3f} ms", flush=True)
    print(f"   FINAL causal: {AX.CFG['causal']}", flush=True)


def sweep():
    """Greedy per-kernel tile sweep at the board shape (global and SWA configs)."""
    torch.manual_seed(0)
    B, S = 64, 1024
    space = {
        "fwd": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
                ((64, 64, 8, 2), (64, 32, 8, 2), (64, 32, 8, 3), (32, 64, 4, 3), (32, 32, 4, 3), (64, 32, 4, 2))],
        "dkdv": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
                 ((32, 32, 4, 2), (32, 32, 4, 3), (32, 32, 8, 2), (64, 32, 8, 1), (64, 32, 8, 2), (32, 64, 8, 1))],
        "dq": [dict(BM=bm, BN=bn, warps=w, stages=st) for bm, bn, w, st in
               ((64, 32, 8, 2), (32, 32, 4, 2), (32, 32, 4, 3), (64, 32, 4, 3), (32, 64, 4, 2), (64, 64, 8, 1))],
    }
    for window, rope in ((None, False), (128, True)):
        cos, sin = rope_tables(S) if rope else (None, None)
        ins, go = _inputs(B, S)
        fn = lambda q, k, v, a, wq, wk: attn_xsa(q, k, v, scale=SC, window=window, alpha=a, q_norm_w=wq,
                                                 k_norm_w=wk, cos=cos, sin=sin)
        print(f"== sweep, {'global' if window is None else f'window {window} + rope'}")
        mode = "window" if window is not None else "causal"
        base = _BASE_CFG[mode]
        for part in ("fwd", "dkdv", "dq"):
            best = None
            for cfg in space[part]:
                AX.CFG[mode][part] = cfg
                AX._FIT.clear()
                try:
                    tf, tfb = _time(fn, ins, go, n=5)
                except Exception as ex:
                    print(f"   {part:5s} {cfg} FAILED {type(ex).__name__}: {str(ex).splitlines()[0][:70]}", flush=True)
                    continue
                t = tf if part == "fwd" else tfb
                print(f"   {part:5s} {cfg}  {t:7.3f} ms", flush=True)
                if best is None or t < best[0]:
                    best = (t, cfg)
            if best is None:
                print(f"   -> {part}: every config failed, keeping {base[part]}", flush=True)
                AX.CFG[mode][part] = base[part]
                continue
            AX.CFG[mode][part] = best[1]
            print(f"   -> {part} best {best[1]} {best[0]:.3f} ms", flush=True)
        print(f"   FINAL {mode}: {AX.CFG[mode]}", flush=True)


if __name__ == "__main__":
    if "--sweep" in sys.argv:
        sweep()
    if "--sweep_long" in sys.argv:
        sweep_long()
    good = main()
    if "--bench" in sys.argv:
        bench()
    if "--scale" in sys.argv:
        scale()
    sys.exit(0 if good else 1)
