"""Parity for the per-expert INPUT SCALE alpha on the row-fused GLU kernels.

act(alpha * x), x = gate (code 0) or gate/rms(gate) (code 2). alpha sits AFTER the rms for
the normed codes because rms is positively homogeneous -- alpha*g/rms(alpha*g) == g/rms(g), so
scaling before the norm is exactly inert and would train a dead parameter.

There is NO gamma: a per-expert OUTPUT gain is redundant with the router weight (g_e*w*f ==
(g_e*w)*f), so it can express nothing new. Checks fwd, grad wrt gate_up, dalpha, and that alpha==1
is BIT-IDENTICAL to not passing alpha at all.  Run on the box: python parity_check/parity_expert_alpha.py
"""
import importlib
import _paths  # noqa: F401  -- repo root on sys.path
import torch
import torch.nn.functional as F

importlib.import_module("kernels.sm120.moe")           # sm120 first (see parity_normed_tiles)
M = importlib.import_module("kernels.sm75.moe")

MROWS, I, EPS = 512, 768, 1e-6
NAMES = {0: "silu", 2: "normsilu"}


def eager(gate, up, code, alpha):
    g = gate.float()
    x = g if code == 0 else g * torch.rsqrt(g.square().mean(-1, keepdim=True) + EPS)
    z = alpha.unsqueeze(-1) * x          # (rows,) -> broadcast over the intermediate dim
    return F.silu(z) * up.float()        # both live codes are SiLU; only x differs


def main():
    dev = "cuda"
    ok = True
    print(f"{'code':>6}{'act':>11}{'alpha':>7}{'fwd':>10}{'d_gateup':>11}{'d_alpha':>11}")
    for code in (0, 2):
        for alpha in (0.7, 1.0, 1.3):
            g = torch.Generator(device=dev).manual_seed(code * 100 + int(alpha * 10))
            gu = (torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32)
                  ).detach().requires_grad_(True)
            go = torch.randn(MROWS, I, generator=g, device=dev, dtype=torch.float32)
            row_act = torch.full((MROWS,), code, device=dev, dtype=torch.int32)
            a = torch.full((MROWS,), alpha, device=dev, dtype=torch.float32).requires_grad_(True)
            ones = torch.ones(MROWS, device=dev, dtype=torch.float32)

            out_k = M._glu_fwd(gu, row_act, row_alpha=a.detach())
            ggu_k, da_k = M._glu_bwd(go, gu, row_act, row_alpha=a.detach(), want_act_grads=True)

            ref = eager(gu[:, :I], gu[:, I:], code, a)
            (ref * go).sum().backward()
            rel = lambda x, y: ((x - y).norm() / y.norm().clamp_min(1e-12)).item()
            e1, e2, e3 = rel(out_k, ref), rel(ggu_k, gu.grad), rel(da_k, a.grad)
            good = max(e1, e2, e3) < 2e-5
            ok &= good
            print(f"{code:>6}{NAMES[code]:>11}{alpha:>7.1f}{e1:>10.1e}{e2:>11.1e}{e3:>11.1e}"
                  f"{'' if good else '   <-- FAIL'}")

    # alpha == 1 must be BIT-identical to not passing alpha at all (1.0*x is exact)
    print("\nalpha=1 vs no-alpha, bit-exact check:")
    for code in (0, 2):
        g = torch.Generator(device=dev).manual_seed(7)
        gu = torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32)
        ra = torch.full((MROWS,), 1.0, device=dev, dtype=torch.float32)
        row_act = torch.full((MROWS,), code, device=dev, dtype=torch.int32)
        a_out = M._glu_fwd(gu, row_act, row_alpha=ra)
        b_out = M._glu_fwd(gu, row_act)
        same = torch.equal(a_out, b_out)
        ok &= same
        print(f"  code {code} {NAMES[code]:<10} {'IDENTICAL' if same else 'DIFFERS  <-- FAIL'}")

    ok &= end_to_end(dev)
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


def end_to_end(dev):
    """The check the kernel-level tests above CANNOT make: that moe_per_expert actually PLUMBS
    act_params through for non-SiTU codes. It did not -- both the fwd and the bwd gated alpha on
    `codes[e] == 5`, so alpha was inert for silu/normsilu and sat at exactly 1.000 for a
    full training run while still costing the uniform fast path. Kernel parity passed the whole
    time because it called _glu_fwd directly. Verified two ways: alpha must MOVE the output, and
    d_alpha must match a central finite difference of the real loss."""
    H, Ei, E, N, K = 512, 768, 4, 256, 2
    codes_l = [0, 2, 2, 0]   # code 7 removed Jul 31 2026 (deleted activation, not a live path)
    print("\nend-to-end moe_per_expert (the plumbing test):")
    g = torch.Generator(device=dev).manual_seed(11)
    rn = lambda *s: torch.randn(*s, generator=g, device=dev, dtype=torch.float32)
    hid, gup, dwn = rn(N, H) * 0.1, rn(E, 2 * Ei, H) * 0.05, rn(E, H, Ei) * 0.05
    idx = torch.randint(0, E, (N, K), generator=g, device=dev, dtype=torch.int64)
    wt, go = torch.rand(N, K, generator=g, device=dev), rn(N, H)
    codes = torch.tensor(codes_l, dtype=torch.int32, device=dev)
    loss = lambda ap: (M.moe_per_expert(hid, idx, wt, gup, dwn, codes, act_params=ap) * go).sum()

    base = loss(None)
    ap = torch.ones(E, 2, device=dev, dtype=torch.float32)
    ap[:, 0] = torch.tensor([0.7, 1.3, 0.8, 1.2], device=dev)
    apg = ap.clone().requires_grad_(True)
    lv = loss(apg)
    lv.backward()
    live = not torch.allclose(base, lv, rtol=1e-6)
    print(f"  alpha changes the output: {'YES' if live else 'NO   <-- FAIL (alpha is inert)'}")

    h, fd_ok = 1e-3, True
    for e in range(E):
        p, m = ap.clone(), ap.clone()
        p[e, 0] += h; m[e, 0] -= h
        fd = ((loss(p) - loss(m)) / (2 * h)).item()
        an = apg.grad[e, 0].item()
        good = abs(fd - an) <= 2e-3 * max(abs(fd), 1.0)
        fd_ok &= good and an != 0.0
        print(f"  expert {e} code {codes_l[e]:<2} d_alpha analytic={an:>11.4f} fd={fd:>11.4f}"
              f"{'' if good else '   <-- FAIL'}")
    return live and fd_ok


if __name__ == "__main__":
    main()
