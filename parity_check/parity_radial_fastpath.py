"""Gate the radial fast path: outputs AND all gradients must match the per-expert loop.

    python -m parity_check.parity_radial_fastpath

Radial NormSiLU used to be disqualified from the batched grouped-mm path purely by carrying a
per-expert theta (`ap32 is None` gated `uniform`), which cost ~10k tok/s -- not for the two extra
elementwise ops, but because the whole MoE fell back to a 64-iteration Python loop of small GEMMs.
The fix plumbs theta through as a per-ROW vector so the batched GLU kernels can carry it.

This compares the SAME inputs down both paths:

    reference   BIBO_MOE_FORCE_LOOP=1 -> the per-expert loop that shipped and trained every board
    candidate   the batched path with row_alpha

Three gates, all required:
  1. forward outputs match
  2. d_hidden / d_gate_up / d_down match
  3. d_act_params (theta) matches AND IS NON-ZERO

Gate 3 is the one that matters most. The batched backward previously returned None for theta, so a
naive enable would have trained radial with a FROZEN exponent -- numerically silent, and it would
have looked like "radial is no better than normsilu" months later.
"""
import os

import torch

from kernels.sm75.moe import moe_per_expert

DEV = "cuda"
# BiBo's real MoE geometry: hidden 512, moe_intermediate 768, 64 routed experts, top-6.
# Small-E shapes can route every expert down a different branch than the model does.
H, I, E, K, N = 512, 768, 64, 6, 16384


def _inputs(seed=0, dtype=torch.bfloat16):
    torch.manual_seed(seed)
    hidden = torch.randn(N, H, device=DEV, dtype=dtype)
    gu = torch.randn(E, 2 * I, H, device=DEV, dtype=dtype) * 0.02
    dn = torch.randn(E, H, I, device=DEV, dtype=dtype) * 0.02
    idx = torch.randint(0, E, (N, K), device=DEV)
    wt = torch.rand(N, K, device=DEV, dtype=torch.float32)
    wt = wt / wt.sum(-1, keepdim=True)
    codes = torch.full((E,), 8, dtype=torch.int32, device=DEV)      # radial, p = sigmoid(theta)
    theta = (torch.randn(E, device=DEV, dtype=torch.float32) * 0.5)
    return hidden, idx, wt, gu, dn, codes, theta


def _run(force_loop):
    hidden, idx, wt, gu, dn, codes, theta = _inputs()
    prev = os.environ.get("BIBO_MOE_FORCE_LOOP")
    os.environ["BIBO_MOE_FORCE_LOOP"] = "1" if force_loop else "0"
    try:
        h = hidden.detach().requires_grad_(True)
        g = gu.detach().requires_grad_(True)
        d = dn.detach().requires_grad_(True)
        t = theta.detach().requires_grad_(True)
        out = moe_per_expert(h, idx, wt, g, d, codes, act_params=t)
        import kernels.sm75.moe as _m
        path = _m._LAST_PATH
        torch.manual_seed(1234)                       # same upstream grad for both paths
        out.backward(torch.randn_like(out) * 0.01)
        return (out.detach(), h.grad.detach(), g.grad.detach(), d.grad.detach(),
                None if t.grad is None else t.grad.detach()), path
    finally:
        if prev is None:
            os.environ.pop("BIBO_MOE_FORCE_LOOP", None)
        else:
            os.environ["BIBO_MOE_FORCE_LOOP"] = prev


def main():
    ref, ref_path = _run(force_loop=True)
    cand, cand_path = _run(force_loop=False)
    names = ("out", "d_hidden", "d_gate_up", "d_down", "d_theta")
    print(f"N={N} H={H} I={I} E={E} K={K}  radial (act code 8)")
    print(f"  reference path = {ref_path!r}   candidate path = {cand_path!r}\n")
    # If both arms took the same branch the comparison is vacuous and would pass no matter how
    # broken the new path is. Assert the split before reading a single number.
    assert ref_path == "loop", f"reference did NOT take the per-expert loop (got {ref_path!r})"
    assert cand_path in ("gmm", "uniform"), \
        f"candidate did NOT take a batched path (got {cand_path!r}) -- parity below is meaningless"
    print(f"  {'tensor':<12}{'max |diff|':>14}{'rel L2':>12}   verdict")
    ok = True
    for nm, a, b in zip(names, ref, cand):
        if a is None or b is None:
            print(f"  {nm:<12}{'MISSING':>14}{'':>12}   FAIL -- one path produced no gradient")
            ok = False
            continue
        md = (a.double() - b.double()).abs().max().item()
        rel = ((a.double() - b.double()).norm() / a.double().norm().clamp_min(1e-12)).item()
        good = rel < 2e-2
        ok &= good
        print(f"  {nm:<12}{md:>14.3e}{rel:>12.5f}   {'OK' if good else 'FAIL'}")

    # theta must actually MOVE. A frozen exponent is the silent failure this whole check exists for:
    # it would train radial as plain normsilu and look like a negative activation result later.
    for nm, gt in (("reference", ref[4]), ("candidate", cand[4])):
        if gt is None or float(gt.abs().max()) == 0.0:
            print(f"\n  d_theta on the {nm} path is zero/absent -- radial would train with a FROZEN "
                  f"exponent")
            ok = False
    if ok:
        print(f"\n  d_theta non-zero: ref max {ref[4].abs().max():.3e}, "
              f"cand max {cand[4].abs().max():.3e}")
    print("\n" + ("PASS -- batched radial matches the per-expert loop, theta gradient live"
                  if ok else "FAIL -- do NOT ship"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
