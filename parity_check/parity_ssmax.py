"""Parity gate: SSMax query scaling -- numerics, gradient, and the DTYPE it hands to attention.

SSMax multiplies q by C = s*log(n), s a learnable per-head scalar. The scaling itself is trivial;
what is not trivial is that s is an fp32 Parameter and q is bf16 under autocast, so a naive
`q * s * log_n` silently promotes q to fp32 -- SDPA then gets fp32 q against bf16 k/v, and the
full-size fp32 tensor is materialized only to be cast straight back down.

So this gate pins three separate things:
  1. VALUES   -- C = s*log(n) with n the per-query CAUSAL context length, against an fp64 oracle,
                 on both the arange path and the padding-aware context_lens path.
  2. GRADIENT -- ds/dL is unchanged by casting the scale, under a bf16 grad_out (the regime that
                 actually occurs: SDPA emits bf16, so its backward hands back bf16 gradients).
  3. PLUMBING -- q reaches scaled_dot_product_attention as bf16, not fp32. This is the one that
                 kernel-level parity cannot see, and it is the whole point of the change.

Needs the BiBo venv (imports src). Run from the triton-kernel-fused repo:
    python parity_check/parity_ssmax.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BiBo")))
import _paths  # noqa: F401  -- repo root on sys.path
import torch

from src.modeling.attn.ssmax import apply_ssmax_query_scaling

DEV = "cuda"
B, H, S, D = 4, 8, 512, 64


def main():
    ok = True

    # ---- 1. values, vs an fp64 oracle -------------------------------------------------------
    g = torch.Generator(device=DEV).manual_seed(3)
    q32 = torch.randn(B, H, S, D, generator=g, device=DEV, dtype=torch.float32)
    s = torch.nn.Parameter(torch.full((1, H, 1, 1), 0.1443, device=DEV))
    n = torch.arange(1, S + 1, device=DEV, dtype=torch.float64).view(1, 1, S, 1)
    ref = q32.double() * s.detach().double() * torch.log(n)
    got = apply_ssmax_query_scaling(q32, S, s)
    e_val = ((got.double() - ref).norm() / ref.norm()).item()
    print(f"values vs fp64 oracle (fp32 q) : {e_val:.2e}")
    ok &= e_val < 1e-6

    # padding-aware path: n must come from context_lens, NOT the grid position
    cl = torch.randint(1, S + 1, (B, S), device=DEV)
    ref_c = q32.double() * s.detach().double() * torch.log(cl.double().view(B, 1, S, 1).clamp(min=1))
    got_c = apply_ssmax_query_scaling(q32, S, s, cl)
    e_ctx = ((got_c.double() - ref_c).norm() / ref_c.norm()).item()
    print(f"context_lens path              : {e_ctx:.2e}")
    ok &= e_ctx < 1e-6

    # n is PER QUERY POSITION, not one global log(kv_len). If it ever collapses to a constant the
    # feature is a fixed temperature and the whole mechanism is gone -- but the values check above
    # would still pass on a constant-n implementation if the constant happened to match at one row.
    row0 = got[0, 0, 0].norm().item() / q32[0, 0, 0].norm().item()
    rowL = got[0, 0, -1].norm().item() / q32[0, 0, -1].norm().item()
    print(f"C(n=1)={row0:.4f}  C(n={S})={rowL:.4f}  (must differ: n is per-position)")
    ok &= abs(row0) < 1e-6 and abs(rowL - 0.1443 * torch.log(torch.tensor(float(S))).item()) < 1e-4

    # ---- 2. dtype: q must NOT be promoted ---------------------------------------------------
    qb = q32.bfloat16()
    out_b = apply_ssmax_query_scaling(qb, S, s)
    print(f"\nbf16 q -> output dtype {out_b.dtype} (fp32 here means SDPA gets mismatched q/k/v)")
    ok &= out_b.dtype == torch.bfloat16
    out_f = apply_ssmax_query_scaling(q32, S, s)
    ok &= out_f.dtype == torch.float32          # an fp32 run must stay fp32

    # ---- 3. gradient: casting the scale must not change ds/dL under a bf16 grad_out ----------
    # grad_out is bf16 because SDPA emits bf16. Compare against the fp32-scale formulation on the
    # SAME bf16 grad_out -- that is the regime training actually runs in.
    go = torch.randn(B, H, S, D, generator=g, device=DEV).bfloat16()
    logn = torch.log(torch.arange(1, S + 1, device=DEV, dtype=torch.float32).clamp(min=1)).view(1, 1, S, 1)

    def dsdl(cast):
        p = torch.nn.Parameter(torch.full((1, H, 1, 1), 0.1443, device=DEV))
        sc = p * logn
        out = qb * (sc.to(qb.dtype) if cast else sc)
        (out * go.to(out.dtype)).sum().backward()
        return p.grad.clone()

    g_cast, g_fp32 = dsdl(True), dsdl(False)
    e_g = ((g_cast - g_fp32).norm() / g_fp32.norm()).item()
    worst = ((g_cast - g_fp32).abs() / g_fp32.abs().clamp_min(1e-30)).max().item()
    print(f"ds/dL cast-vs-fp32 scale       : rel {e_g:.2e}  worst head {worst:.2e}")
    ok &= e_g < 1e-3

    ok &= plumbing()
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


def plumbing():
    """Does a REAL model hand SDPA a bf16 q? The unit checks above all pass on an implementation
    that is never actually reached, or that is reached with the promotion reintroduced one layer up
    (BiBoAttention could re-multiply, a caller could upcast). Assert the end-to-end dtype."""
    from ablate.common.models import build_arm
    from ablate.common import patches as P
    seen = {}
    m, _ = build_arm("bibo_min", device=DEV, dtype=torch.float32, attn_impl="sdpa",
                     num_experts=8, top_k=2, use_xsa=True, use_ssmax=True)
    P.apply(["liger_norm", "liger_rope", "moe", "xsa"])
    orig = torch.nn.functional.scaled_dot_product_attention

    def spy(q, k, v, *a, **kw):
        seen.setdefault("q", q.dtype)
        seen.setdefault("k", k.dtype)
        return orig(q, k, v, *a, **kw)
    torch.nn.functional.scaled_dot_product_attention = spy
    try:
        ids = torch.randint(0, 81920, (2, 129), device=DEV)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m.model(input_ids=ids[:, :-1], use_cache=False)
    finally:
        torch.nn.functional.scaled_dot_product_attention = orig
    good = seen.get("q") == seen.get("k") == torch.bfloat16
    print(f"\nend-to-end: SDPA receives q={seen.get('q')} k={seen.get('k')}  "
          f"{'OK' if good else '<-- FAIL (q promoted; fp32 q vs bf16 k/v)'}")
    return good


if __name__ == "__main__":
    main()
