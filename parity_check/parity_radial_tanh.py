"""Parity for act code 10 -- RADIAL NormSiLU with p = TANH(theta) instead of sigmoid.

Code 8 uses p = sigmoid(theta) in (0,1), so the gain r^p is bounded by [1, r] and p->0 is exactly
normsilu. Code 10 uses p = tanh(theta) in (-1,1), which additionally admits gain < 1: r^-1 SHRINKS
high-rms rows, a direction code 8 structurally cannot express. Motivated by the measured dL/dp on a
trained checkpoint being positive in 6/6 layers at low p (ablate/common/theta_grad_probe.py), i.e.
the model asking for a steeper ramp than sigmoid's floor allows.

The two codes must agree EXACTLY wherever sigmoid(theta) == tanh(theta'), and the derivative differs
(p(1-p) vs 1-p^2), so d_theta is the check that actually distinguishes them -- a copy-paste that left
the sigmoid derivative in place would still pass a forward-only test.

Checks fwd, d_gate_up and d_theta vs an autograd eager reference at NEGATIVE and positive theta,
that theta=0 is exactly normsilu (tanh(0)=0 -> r^0=1, unlike code 8 where sigmoid(0)=0.5), that
negative p really shrinks, and an end-to-end pass through moe_per_expert.
    python parity_check/parity_radial_tanh.py
"""
import importlib
import _paths  # noqa: F401  -- repo root on sys.path
import torch
import torch.nn.functional as F

importlib.import_module("kernels.sm120.moe")           # sm120 first (see parity_normed_tiles)
M = importlib.import_module("kernels.sm75.moe")

MROWS, I, EPS, CODE = 512, 768, 1e-6, 10


def eager(gate, up, theta):
    g = gate.float()
    r = torch.sqrt(g.square().mean(-1, keepdim=True) + EPS)
    p = torch.tanh(theta).unsqueeze(-1)
    return r.pow(p) * F.silu(g / r) * up.float()


def main():
    dev = "cuda"
    ok = True
    print(f"{'theta':>7}{'p=tanh':>9}{'fwd':>10}{'d_gateup':>11}{'d_theta':>11}")
    for theta_v in (-2.0, -0.8, 0.0, 0.8, 2.0):        # NEGATIVE theta is the point of code 10
        g = torch.Generator(device=dev).manual_seed(int(theta_v * 10) + 77)
        gu = torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32
                         ).detach().requires_grad_(True)
        go = torch.randn(MROWS, I, generator=g, device=dev, dtype=torch.float32)
        row_act = torch.full((MROWS,), CODE, device=dev, dtype=torch.int32)
        th = torch.full((MROWS,), theta_v, device=dev, dtype=torch.float32).requires_grad_(True)

        out_k = M._glu_fwd(gu, row_act, code_hint=CODE, row_alpha=th.detach())
        ggu_k, dth_k = M._glu_bwd(go, gu, row_act, code_hint=CODE, row_alpha=th.detach(),
                                  want_act_grads=True)
        ref = eager(gu[:, :I], gu[:, I:], th)
        (ref * go).sum().backward()
        rel = lambda x, y: ((x - y).norm() / y.norm().clamp_min(1e-12)).item()
        e1, e2, e3 = rel(out_k, ref), rel(ggu_k, gu.grad), rel(dth_k, th.grad)
        good = max(e1, e2, e3) < 2e-5
        ok &= good
        print(f"{theta_v:>7.1f}{torch.tanh(torch.tensor(theta_v)).item():>9.3f}"
              f"{e1:>10.1e}{e2:>11.1e}{e3:>11.1e}{'' if good else '   <-- FAIL'}")

    # theta = 0 -> p = tanh(0) = 0 -> r^0 = 1 -> EXACTLY normsilu. (Code 8 gives sqrt(r) here;
    # if this arm accidentally ran code 8's sigmoid it would fail by a factor of sqrt(r) ~ 2.)
    print("\ntheta=0 == normsilu exactly (code 8 would give sqrt(r) here):")
    g = torch.Generator(device=dev).manual_seed(5)
    gu = torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32)
    gu[:, :I] *= 4.0   # r ~ 4 like the trained model. At r == 1, r^p is 1 for EVERY p,
                       # so a randn gate makes the shrink check below structurally blind.
    row_act = torch.full((MROWS,), CODE, device=dev, dtype=torch.int32)
    zeros = torch.zeros(MROWS, device=dev, dtype=torch.float32)
    got = M._glu_fwd(gu, row_act, code_hint=CODE, row_alpha=zeros)
    gg = gu[:, :I].float()
    r = torch.sqrt(gg.square().mean(-1, keepdim=True) + EPS)
    want = F.silu(gg / r) * gu[:, I:].float()
    e = ((got - want).norm() / want.norm()).item()
    print(f"  rel {e:.1e} {'OK' if e < 2e-6 else '<-- FAIL'}")
    ok &= e < 2e-6

    # NEGATIVE p must genuinely shrink: gain r^p < 1 where r > 1. This is the whole reason code 10
    # exists, so assert the direction rather than trusting the algebra.
    print("\nnegative p shrinks (the capability code 8 lacks):")
    neg = torch.full((MROWS,), -2.0, device=dev, dtype=torch.float32)
    out_neg = M._glu_fwd(gu, row_act, code_hint=CODE, row_alpha=neg)
    ratio = (out_neg.norm() / want.norm()).item()
    r_mean = r.mean().item()
    print(f"  |out(p=-0.964)| / |out(p=0)| = {ratio:.4f}   (mean r = {r_mean:.2f}, so expect << 1)")
    ok &= ratio < 0.9

    print("\nmissing-alpha guard:")
    try:
        M._glu_fwd(gu, row_act, code_hint=CODE)
        print("  no error raised  <-- FAIL")
        ok = False
    except ValueError as exc:
        print(f"  ValueError OK: {str(exc)[:60]}...")

    ok &= end_to_end(dev)
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


def end_to_end(dev):
    """Through the public entry point, mixing codes 10 / 8 / 2 / 0 in one stack: the dispatcher must
    route act_params per expert and the two radial codes must NOT be confused for each other."""
    H, Ei, E, N, K = 512, 768, 4, 256, 2
    codes_l = [10, 8, 2, 0]
    codes = torch.tensor(codes_l, dtype=torch.int32, device=dev)
    g = torch.Generator(device=dev).manual_seed(11)
    x = torch.randn(N, H, generator=g, device=dev, dtype=torch.float32)
    gup = (torch.randn(E, 2 * Ei, H, generator=g, device=dev, dtype=torch.float32) * 0.02
           ).requires_grad_(True)
    dwn = (torch.randn(E, H, Ei, generator=g, device=dev, dtype=torch.float32) * 0.02
           ).requires_grad_(True)
    idx = torch.randint(0, E, (N, K), generator=g, device=dev)
    wt = torch.rand(N, K, generator=g, device=dev, dtype=torch.float32)
    ap = torch.tensor([[-1.5], [0.5], [1.0], [1.0]], device=dev, dtype=torch.float32
                      ).requires_grad_(True)
    G = torch.randn(N, H, generator=g, device=dev, dtype=torch.float32)

    out = M.moe_per_expert(x, idx, wt, gup, dwn, codes, act_params=ap)
    (out * G).sum().backward()
    dth = ap.grad[:, 0].clone()
    print("\nend-to-end (codes 10/8/2/0 in one stack):")
    print(f"  d_theta = {[round(v, 5) for v in dth.tolist()]}")
    ok = abs(dth[0].item()) > 1e-8 and abs(dth[1].item()) > 1e-8
    print(f"  radial experts (10, 8) both get gradient : {ok}")
    # NOT asserted zero: alpha is the INPUT SCALE for codes 0/2 (z = alpha*gn), so those experts
    # are supposed to have d_theta. Only its MEANING differs from the radial codes.
    print(f"  codes 2/0 use alpha as an input scale -> nonzero grad is correct")

    # central finite difference on expert 0's theta -- catches a wrong dp/dtheta (1-p^2 vs p(1-p))
    with torch.no_grad():
        h = 1e-2
        base = ap.detach().clone()
        losses = []
        for sgn in (+1, -1):
            apx = base.clone(); apx[0, 0] += sgn * h
            losses.append((M.moe_per_expert(x, idx, wt, gup.detach(), dwn.detach(), codes,
                                            act_params=apx) * G).sum().item())
        fd = (losses[0] - losses[1]) / (2 * h)
    rel = abs(fd - dth[0].item()) / max(abs(fd), 1e-12)
    print(f"  d_theta[0] analytic {dth[0].item():+.5f} vs finite-diff {fd:+.5f}  rel {rel:.2e} "
          f"{'OK' if rel < 2e-2 else '<-- FAIL (wrong dp/dtheta?)'}")
    return ok and rel < 2e-2


if __name__ == "__main__":
    main()
