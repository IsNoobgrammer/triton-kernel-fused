"""Parity for act code 8 -- RADIAL NormSiLU: r^p * SiLU(g/r), p = sigmoid(theta), theta per expert.

NormSiLU computes r = rms(gate) and then discards it. Code 8 puts a BOUNDED fraction of the radius
back: p is squashed through a sigmoid so it can never leave (0,1). Bounding is load-bearing -- the
toy round measured full p=1 (raw magnitude passthrough) as harmful and unbounded-learnable p as worse
than a fixed 0.5, while bounded-learnable was the best arm of six.

`row_alpha` carries theta (the exponent LOGIT), NOT an input scale -- code 8 applies no scale to the
normed gate. theta = 0 gives p = 0.5, i.e. sqrt(r)*SiLU(g/r).

Checks fwd, d_gate_up, d_theta against an autograd eager reference, the p=0.5 identity at theta=0,
and -- the check that matters -- an END-TO-END pass through moe_per_expert, because kernel-level
parity cannot see a dispatcher that silently drops the parameter (see parity_expert_alpha.py).
Run on the box: python parity_check/parity_radial.py
"""
import importlib
import _paths  # noqa: F401  -- repo root on sys.path
import torch
import torch.nn.functional as F

importlib.import_module("kernels.sm120.moe")           # sm120 first (see parity_normed_tiles)
M = importlib.import_module("kernels.sm75.moe")

MROWS, I, EPS, CODE = 512, 768, 1e-6, 8


def eager(gate, up, theta):
    g = gate.float()
    r = torch.sqrt(g.square().mean(-1, keepdim=True) + EPS)
    p = torch.sigmoid(theta).unsqueeze(-1)
    return r.pow(p) * F.silu(g / r) * up.float()


def main():
    dev = "cuda"
    ok = True
    print(f"{'theta':>7}{'p':>7}{'fwd':>10}{'d_gateup':>11}{'d_theta':>11}")
    for theta_v in (-2.0, -0.5, 0.0, 0.5, 2.0):
        g = torch.Generator(device=dev).manual_seed(int(theta_v * 10) + 99)
        gu = torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32
                         ).detach().requires_grad_(True)
        go = torch.randn(MROWS, I, generator=g, device=dev, dtype=torch.float32)
        row_act = torch.full((MROWS,), CODE, device=dev, dtype=torch.int32)
        th = torch.full((MROWS,), theta_v, device=dev, dtype=torch.float32).requires_grad_(True)
        ones = torch.ones(MROWS, device=dev, dtype=torch.float32)

        out_k = M._glu_fwd(gu, row_act, code_hint=CODE, row_alpha=th.detach())
        ggu_k, dth_k = M._glu_bwd(go, gu, row_act, code_hint=CODE, row_alpha=th.detach(),
                                     want_act_grads=True)

        ref = eager(gu[:, :I], gu[:, I:], th)
        (ref * go).sum().backward()
        rel = lambda x, y: ((x - y).norm() / y.norm().clamp_min(1e-12)).item()
        e1, e2, e3 = rel(out_k, ref), rel(ggu_k, gu.grad), rel(dth_k, th.grad)
        good = max(e1, e2, e3) < 2e-5
        ok &= good
        p = torch.sigmoid(torch.tensor(theta_v)).item()
        print(f"{theta_v:>7.1f}{p:>7.3f}{e1:>10.1e}{e2:>11.1e}{e3:>11.1e}"
              f"{'' if good else '   <-- FAIL'}")

    # theta=0 must give exactly sqrt(r)*SiLU(g/r)
    print("\ntheta=0 == sqrt(r)*SiLU(g/r):")
    g = torch.Generator(device=dev).manual_seed(5)
    gu = torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32)
    row_act = torch.full((MROWS,), CODE, device=dev, dtype=torch.int32)
    zeros = torch.zeros(MROWS, device=dev, dtype=torch.float32)
    ones = torch.ones(MROWS, device=dev, dtype=torch.float32)
    got = M._glu_fwd(gu, row_act, code_hint=CODE, row_alpha=zeros)
    gg = gu[:, :I].float()
    r = torch.sqrt(gg.square().mean(-1, keepdim=True) + EPS)
    want = r.sqrt() * F.silu(gg / r) * gu[:, I:].float()
    e = ((got - want).norm() / want.norm()).item()
    print(f"  rel {e:.1e} {'OK' if e < 2e-6 else '<-- FAIL'}")
    ok &= e < 2e-6

    # code 8 without alpha must RAISE, not silently default to p=sigmoid(1)=0.731
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
    """Through the public entry point: theta must move the output AND d_theta must match a central
    finite difference of the real loss. Kernel parity above cannot catch a dispatcher that drops
    act_params -- that is exactly how per-expert alpha shipped inert."""
    H, Ei, E, N, K = 512, 768, 4, 256, 2
    codes_l = [8, 2, 8, 0]                       # radial mixed with normsilu and silu
    print("\nend-to-end moe_per_expert (the plumbing test):")
    g = torch.Generator(device=dev).manual_seed(13)
    rn = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float32)
    hid, gup, dwn = rn(N, H) * 0.1, rn(E, 2 * Ei, H) * 0.05, rn(E, H, Ei) * 0.05
    idx = torch.randint(0, E, (N, K), generator=g, device=dev, dtype=torch.int64)
    wt, go = torch.rand(N, K, generator=g, device=dev), rn(N, H)
    codes = torch.tensor(codes_l, dtype=torch.int32, device=dev)
    loss = lambda ap: (M.moe_per_expert(hid, idx, wt, gup, dwn, codes, act_params=ap) * go).sum()

    ap = torch.ones(E, 2, device=dev, dtype=torch.float32)
    ap[:, 0] = torch.tensor([0.0, 1.0, 1.5, 1.0], device=dev)      # theta for the code-8 experts
    base = loss(ap.clone())
    ap2 = ap.clone(); ap2[0, 0] = -2.0                             # move expert 0's exponent only
    live = not torch.allclose(base, loss(ap2), rtol=1e-6)
    print(f"  theta changes the output: {'YES' if live else 'NO   <-- FAIL (code 8 is inert)'}")

    apg = ap.clone().requires_grad_(True)
    loss(apg).backward()
    h, fd_ok = 1e-3, True
    for e in range(E):
        p_, m_ = ap.clone(), ap.clone()
        p_[e, 0] += h; m_[e, 0] -= h
        fd = ((loss(p_) - loss(m_)) / (2 * h)).item()
        an = apg.grad[e, 0].item()
        good = abs(fd - an) <= 2e-3 * max(abs(fd), 1.0)
        # only the code-8 experts should have a nonzero d_theta from the radial term
        fd_ok &= good
        print(f"  expert {e} code {codes_l[e]:<2} d_theta analytic={an:>11.4f} fd={fd:>11.4f}"
              f"{'' if good else '   <-- FAIL'}")
    return live and fd_ok


if __name__ == "__main__":
    main()
