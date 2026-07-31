"""Parity for CAUTIOUS weight decay in FusedMuon (sm120, the live path).

Rule: decay a coordinate only where the update is already pushing |W| UP. The applied step is
delta_p = (negative alpha) * out, so "delta_p has p's sign" <=> out*p < 0.

Checked WITHOUT reaching inside the optimizer, via an exact algebraic invariant. From the same
initial p0 and the same grad, one step gives
    non-cautious:  p = p0*(1 - lr*wd) + delta
    cautious:      p = p0 - lr*wd*p0*mask + delta
so
    (p_cautious - p_noncautious) / (lr*wd*p0)  ==  1 - mask     elementwise, exactly 0 or 1.
That pins both the magnitude and the placement of the decay. The recovered mask is then checked
against the sign rule using delta, which closes the loop.

Run on the box: python parity_check/parity_cautious_wd.py
"""
import _paths  # noqa: F401  -- repo root on sys.path
import torch

from kernels.sm120.muon import FusedMuon

LR, WD, MOM = 0.01, 0.1, 0.95
R, C = 256, 128


def one_step(cautious, p0, grad, wd=WD):
    p = p0.clone().requires_grad_(True)
    p.grad = grad.clone()
    opt = FusedMuon([{"params": [p]}], lr=LR, momentum=MOM, weight_decay=wd,
                    cautious_decay=cautious)
    opt.step()
    return p.detach().clone()


def main():
    dev = "cuda"
    ok = True
    g = torch.Generator(device=dev).manual_seed(3)
    p0 = torch.randn(R, C, generator=g, device=dev, dtype=torch.float32)
    grad = torch.randn(R, C, generator=g, device=dev, dtype=torch.float32) * 0.05

    p_nc = one_step(False, p0, grad)
    p_c = one_step(True, p0, grad)

    # delta from the non-cautious run (its decay is a known exact factor)
    delta = p_nc - p0 * (1.0 - LR * WD)
    diff = p_c - p_nc
    ratio = diff / (LR * WD * p0)

    # TOL: the two branches compute the same decay in different ORDERS -- non-cautious does
    # p.mul_(1-lam), cautious does p.sub_(p*mask, alpha=lam) -- which differ by 1 fp32 ulp. In these
    # ratio units that floor is eps/lam = 1.19e-7/1e-3 = 1.19e-4, so a 1e-4 tolerance sits exactly ON
    # it and misclassifies ~1.5% of coordinates (measured: 511/32768 off, median 1.09e-4, max gap
    # 1.191e-4 vs eps/lam 1.192e-4 -- an exact match). 1e-3 is 10x above the rounding floor and
    # 1000x below the 0-vs-1 separation, so it discriminates the two classes with margin both ways.
    TOL = 1e-3
    near0 = (ratio.abs() < TOL)
    near1 = ((ratio - 1.0).abs() < TOL)
    binary = (near0 | near1).all().item()
    print(f"diff/(lr*wd*p0) is exactly 0 or 1 everywhere: "
          f"{'YES' if binary else 'NO  <-- FAIL'}   "
          f"(off-values: {(~(near0 | near1)).sum().item()})")
    ok &= binary

    # near0 == decay WAS applied by the cautious run == the mask
    mask = near0
    frac = mask.float().mean().item()
    print(f"fraction of coords decayed: {frac:.3f}  (expect ~0.5 for a random-ish update)")
    ok &= 0.2 < frac < 0.8

    # the mask must be exactly the sign rule: decay where delta pushes |p| up, i.e. delta*p0 > 0
    want = (delta * p0) > 0
    agree = (mask == want).float().mean().item()
    print(f"mask == (delta*p0 > 0): {agree * 100:.2f}% agreement "
          f"{'OK' if agree > 0.999 else '<-- FAIL'}")
    ok &= agree > 0.999

    # wd=0 -> cautious must be a no-op (both flags identical)
    a = one_step(False, p0, grad, wd=0.0)
    b = one_step(True, p0, grad, wd=0.0)
    same = torch.equal(a, b)
    print(f"wd=0: cautious == non-cautious bit-exactly: {'YES' if same else 'NO  <-- FAIL'}")
    ok &= same

    # cautious must actually CHANGE something at wd>0 (guards against a dead flag -- the exact
    # failure mode that shipped per-expert alpha inert)
    moved = not torch.allclose(p_c, p_nc, rtol=1e-9, atol=1e-9)
    print(f"wd>0: cautious differs from non-cautious: {'YES' if moved else 'NO  <-- FAIL (flag dead)'}")
    ok &= moved

    # norms: cautious decays fewer coords, so ||W|| should end LARGER
    print(f"||p|| non-cautious {p_nc.norm():.5f}  cautious {p_c.norm():.5f}")
    ok &= p_c.norm().item() > p_nc.norm().item()

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
