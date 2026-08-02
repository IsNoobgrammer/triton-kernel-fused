"""Parity gate: fused_xsa vs BiBo's EAGER apply_xsa, with the learnable per-head alpha.

XSA is now always learnable-strength: Y <- Y - tanh(alpha_h) * (Y.Vn) Vn, alpha a per-head logit.
Two independent implementations of that (src/modeling/attn/xsa.py and kernels/sm75/xsa.py) index
heads by hand, and they index them DIFFERENTLY on the page:

    eager : Yg = Y.view(B, n_kv, g, S, D) and alpha.view(1, n_kv, g, 1, 1)
    kernel: head row = (b*H + kv*GROUP + j)*S + s,  alpha loaded at (kv*GROUP + j)

They agree only because head h == kv*g + j in both. Nothing enforces that, and a permutation is
invisible to any test that gives every head the same alpha -- so this gate gives every head a
DIFFERENT alpha, and runs GQA (H=8, Hkv=2) where a permutation actually changes the answer.

Also pins the two boundary identities that make tanh the right parameterization:
    tanh(0)   = 0  -> XSA is exactly OFF (output == input), so init 0 is a true no-op start
    tanh(inf) = 1  -> recovers the original hard-coded full rejection

Run from the triton-kernel-fused repo with a venv that can import BiBo:
    python parity_check/parity_xsa_alpha.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BiBo")))
import _paths  # noqa: F401  -- repo root on sys.path
import torch
from kernels.sm120.xsa import fused_xsa
from src.modeling.attn.xsa import apply_xsa

DEV = "cuda"
B, H, HKV, S, D = 2, 8, 2, 128, 64


def _inputs(seed, requires_grad=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    y = torch.randn(B, H, S, D, generator=g, device=DEV, dtype=torch.float32)
    v = torch.randn(B, HKV, S, D, generator=g, device=DEV, dtype=torch.float32)
    # one DISTINCT logit per head, spanning negative (amplify) through positive (reject)
    a = torch.linspace(-1.5, 1.5, H, device=DEV, dtype=torch.float32)
    if requires_grad:
        y, v, a = y.requires_grad_(True), v.requires_grad_(True), a.requires_grad_(True)
    return y, v, a


def rel(x, y):
    return ((x - y).norm() / y.norm().clamp_min(1e-12)).item()


def main():
    ok = True

    # ---- 1. fwd + all three grads, GQA, per-head alpha --------------------------------------
    yk, vk, ak = _inputs(7)
    ye, ve, ae = _inputs(7)
    go = torch.randn(B, H, S, D, generator=torch.Generator(device=DEV).manual_seed(8), device=DEV)

    out_k = fused_xsa(yk, vk, ak)
    out_e = apply_xsa(ye, ve, enable_gqa=True, alpha=ae)
    (out_k * go).sum().backward()
    (out_e * go).sum().backward()

    e_f, e_y, e_v, e_a = (rel(out_k, out_e), rel(yk.grad, ye.grad),
                          rel(vk.grad, ve.grad), rel(ak.grad, ae.grad))
    print(f"{'fwd':>10}{'d_Y':>11}{'d_V':>11}{'d_alpha':>11}")
    print(f"{e_f:>10.1e}{e_y:>11.1e}{e_v:>11.1e}{e_a:>11.1e}")
    good = max(e_f, e_y, e_v, e_a) < 2e-5
    ok &= good
    print("  OK" if good else "  <-- FAIL")

    # The d_alpha check above is only meaningful if alpha actually MOVES the output per head.
    # A kernel that ignored A entirely would still match on d_Y/d_V for the a=1 heads.
    with torch.no_grad():
        spread = (fused_xsa(yk.detach(), vk.detach(), ak.detach())
                  - fused_xsa(yk.detach(), vk.detach(), torch.zeros_like(ak))).norm().item()
    print(f"\nalpha is live: |out(alpha) - out(0)| = {spread:.3f} (0 would make d_alpha vacuous)")
    ok &= spread > 1e-3

    # ---- 2. per-head ordering: shuffling alpha must CHANGE the output ------------------------
    # If either side collapsed alpha to a scalar (or permuted heads), a reversed alpha would give
    # the same answer and this passes silently. Both must move, and must move TOGETHER.
    with torch.no_grad():
        a_rev = ak.detach().flip(0)
        k_rev = fused_xsa(yk.detach(), vk.detach(), a_rev)
        e_rev = apply_xsa(ye.detach(), ve.detach(), enable_gqa=True, alpha=a_rev)
        moved = rel(k_rev, out_k.detach())
        agree = rel(k_rev, e_rev)
    print(f"reversed alpha: moves output by {moved:.3f} (must be >0), kernel-vs-eager {agree:.1e}")
    ok &= moved > 1e-3 and agree < 2e-5

    # ---- 3. tanh boundary identities ---------------------------------------------------------
    with torch.no_grad():
        y0, v0, _ = _inputs(21, requires_grad=False)
        off_k = fused_xsa(y0, v0, torch.zeros(H, device=DEV))
        off_e = apply_xsa(y0, v0, enable_gqa=True, alpha=torch.zeros(H, device=DEV))
        full_k = fused_xsa(y0, v0, torch.full((H,), 20.0, device=DEV))   # tanh(20) == 1.0 in fp32
        full_e = apply_xsa(y0, v0, enable_gqa=True, alpha=None)          # the original hard-coded a=1
    e_off_k, e_off_e = rel(off_k, y0), rel(off_e, y0)
    e_full = rel(full_k, full_e)
    print(f"\nalpha=0   -> XSA off  : kernel {e_off_k:.1e}  eager {e_off_e:.1e}")
    print(f"alpha=20  -> full a=1 : kernel-vs-eager-None {e_full:.1e}")
    ok &= max(e_off_k, e_off_e, e_full) < 2e-6

    # ---- 4. alpha=None on both sides (the pre-alpha behaviour still reachable) ----------------
    with torch.no_grad():
        e_none = rel(fused_xsa(y0, v0, None), apply_xsa(y0, v0, enable_gqa=True, alpha=None))
    print(f"alpha=None both sides : {e_none:.1e}")
    ok &= e_none < 2e-6

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
