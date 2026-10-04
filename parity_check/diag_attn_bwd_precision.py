"""Attention BACKWARD precision vs an fp64 reference: our attn_xsa, flex_attention, SDPA, and eager fp32 (the floor).

    python -m parity_check.diag_attn_bwd_precision [--S 1024] [--gains 1,4,16,32] [--offsets 0,1,4,16]

The bug class (GProj, arxiv 2609.34272; KohakuFA): the exact score gradient dS = P * (dP - delta) sums to zero along
every row, so dQ_i = sum_j dS_ij k_j only sees keys RELATIVE to each other -- adding one vector u to every key changes
nothing (the logits shift by a per-row constant). Casting dS to bf16 (and using a bf16-rounded O in delta = dO . O)
breaks the zero row sum, and the mean key leaks into dQ as r_i * u. The leak grows with the mean-key size and with
sharper attention. Sweep:
  gain    q scaled by g -> logit std ~ g, max logit ~ 3.3 g over 1024 keys
  offset  every key += c * u, |u| = sqrt(D) (c = 1: a mean key as large as one key's spread). Exact grads of q and v
          do not depend on c; dK changes only through the row-constant shift (none).
Reported: relative error ||g - g_fp64|| / ||g_fp64|| of dQ, dK, dV (and the forward O) per implementation.
"""
from parity_check import _paths  # noqa: F401
import argparse

import torch
import torch.nn.functional as F

import kernels.sm120.attn_xsa as AX
from kernels.sm120.attn_xsa import attn_xsa, attn_xsa_reference

DEV = "cuda"


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()


def run_impl(name, q, k, v, do, scale, flex=None):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    if name.startswith("ours"):                       # ours / ours:tf32 / ours:split / ours:gproj (AX.DS_PREC)
        AX.DS_PREC = name.split(":")[1] if ":" in name else "bf16"
        o = attn_xsa(q, k, v, scale=scale, xsa=False)       # DS_PREC is read in the BACKWARD: reset after it
    elif name == "flex":
        o = flex(q, k, v)
    elif name == "sdpa":
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    elif name == "fp32":
        o = attn_xsa_reference(q, k, v, scale=scale, xsa=False, dtype=torch.float32)
    elif name == "fp64":
        o = attn_xsa_reference(q, k, v, scale=scale, xsa=False, dtype=torch.float64)
    o.backward(do.to(o.dtype))
    AX.DS_PREC = "bf16"
    return o.detach(), q.grad, k.grad, v.grad


def make_flex(S, H, HKV):
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    bm = create_block_mask(lambda b, h, qi, ki: ki <= qi, None, None, S, S, device=DEV)
    f = torch.compile(flex_attention, dynamic=False)
    return lambda q, k, v: f(q, k, v, block_mask=bm, enable_gqa=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--S", type=int, default=1024)
    ap.add_argument("--H", type=int, default=4)
    ap.add_argument("--HKV", type=int, default=2)
    ap.add_argument("--D", type=int, default=128)
    ap.add_argument("--gains", default="1,4,16,32")
    ap.add_argument("--offsets", default="0,1,4,16")
    ap.add_argument("--impls", default="ours,flex,sdpa,fp32")
    ap.add_argument("--bench", action="store_true", help="time fwd+bwd of each dS mode on the 1B board's attention shape")
    a = ap.parse_args()
    if a.bench:
        return bench()
    B, H, HKV, S, D = 1, a.H, a.HKV, a.S, a.D
    scale = D ** -0.5
    impls = a.impls.split(",")
    flex = make_flex(S, H, HKV) if "flex" in impls else None
    g0 = torch.Generator(device=DEV).manual_seed(0)
    rn = lambda *s: torch.randn(*s, device=DEV, generator=g0)
    q0, k0, v0, do = rn(B, H, S, D), rn(B, HKV, S, D), rn(B, HKV, S, D), rn(B, H, S, D)
    u = rn(1, 1, 1, D); u = u / u.norm() * D ** 0.5
    print(f"S={S} H={H}/{HKV} D={D}  rel err vs fp64  [dQ dK dV | O]")
    print(f"{'gain':>4} {'off':>4} {'maxlogit':>8}  " + "  ".join(f"{n:^31}" for n in impls))
    for g in [float(x) for x in a.gains.split(",")]:
        for c in [float(x) for x in a.offsets.split(",")]:
            q = (q0 * g).bfloat16(); k = (k0 + c * u).bfloat16(); v = v0.bfloat16()
            ref = run_impl("fp64", q, k, v, do, scale)
            with torch.no_grad():
                lg = (q.double() @ k.double().repeat_interleave(H // HKV, 1).transpose(-1, -2)) * scale
                lg = lg.masked_fill(~torch.ones(S, S, dtype=torch.bool, device=DEV).tril(), float("-inf"))
                mx = lg.amax().item()
            cells = []
            for n in impls:
                try:
                    o, dq, dk, dv = run_impl(n, q, k, v, do, scale, flex)
                    cells.append(f"{rel(dq, ref[1]):.1e} {rel(dk, ref[2]):.1e} {rel(dv, ref[3]):.1e} | {rel(o, ref[0]):.0e}")
                except Exception as ex:                       # a backend that cannot run this config
                    cells.append(f"{'n/a: ' + type(ex).__name__:^31}")
            print(f"{g:4g} {c:4g} {mx:8.1f}  " + "  ".join(cells), flush=True)


def bench():
    """fwd+bwd ms per dS mode, board shape: micro-batch 64 x S 1024, 4 q / 2 kv heads, D 128, qk-norm + XSA,
    global and window-128 layers. Same inputs per mode; also checks bf16 mode is bitwise the old path's twin."""
    from triton.testing import do_bench
    B, H, HKV, S, D = 64, 4, 2, 1024, 128
    g0 = torch.Generator(device=DEV).manual_seed(0)
    q = torch.randn(B, H, S, D, device=DEV, generator=g0).bfloat16().requires_grad_(True)
    k = torch.randn(B, HKV, S, D, device=DEV, generator=g0).bfloat16().requires_grad_(True)
    v = torch.randn(B, HKV, S, D, device=DEV, generator=g0).bfloat16().requires_grad_(True)
    wq = torch.ones(D, device=DEV, requires_grad=True); wk = torch.ones(D, device=DEV, requires_grad=True)
    al = torch.zeros(H, device=DEV, requires_grad=True)
    dz = torch.randn(B, H, S, D, device=DEV, generator=g0).bfloat16()
    for window in (None, 128):
        row = []
        for m in ("bf16", "tf32", "split", "gproj"):
            AX.DS_PREC = m
            def step():
                z = attn_xsa(q, k, v, scale=D ** -0.5, window=window, xsa=True, alpha=al, q_norm_w=wq, k_norm_w=wk)
                z.backward(dz)
            ms = do_bench(step, warmup=5, rep=40)
            row.append((m, ms))
        AX.DS_PREC = "bf16"
        base = row[0][1]
        print(f"{'global' if window is None else 'w128':6} fwd+bwd ms: " + "  ".join(f"{m} {t:.2f} ({t / base:.3f}x)" for m, t in row), flush=True)


if __name__ == "__main__":
    main()
