"""Parity for the per-expert INPUT SCALE alpha on the row-fused GLU kernels.

act(alpha * x), x = gate (codes 0/1) or gate/rms(gate) (codes 2/6/7). alpha sits AFTER the rms for
the normed codes because rms is positively homogeneous -- alpha*g/rms(alpha*g) == g/rms(g), so
scaling before the norm is exactly inert and would train a dead parameter.

There is NO gamma: a per-expert OUTPUT gain is redundant with the router weight (g_e*w*f ==
(g_e*w)*f), so it can express nothing new. Checks fwd, grad wrt gate_up, dalpha, and that alpha==1
is BIT-IDENTICAL to not passing alpha at all.  Run on the box: python parity_expert_alpha.py
"""
import importlib
import torch
import torch.nn.functional as F

importlib.import_module("kernels.sm120.moe")           # sm120 first (see parity_normed_tiles)
M = importlib.import_module("kernels.sm75.moe")

MROWS, I, EPS = 512, 768, 1e-6
NAMES = {0: "silu", 2: "normsilu", 7: "normsitu"}


def eager(gate, up, code, alpha):
    g = gate.float()
    x = g if code == 0 else g * torch.rsqrt(g.square().mean(-1, keepdim=True) + EPS)
    z = alpha * x
    act = F.silu(z) if code in (0, 2) else torch.tanh(z) * torch.sigmoid(z)
    return act * up.float()


def main():
    dev = "cuda"
    ok = True
    print(f"{'code':>6}{'act':>11}{'alpha':>7}{'fwd':>10}{'d_gateup':>11}{'d_alpha':>11}")
    for code in (0, 2, 7):
        for alpha in (0.7, 1.0, 1.3):
            g = torch.Generator(device=dev).manual_seed(code * 100 + int(alpha * 10))
            gu = (torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32)
                  ).detach().requires_grad_(True)
            go = torch.randn(MROWS, I, generator=g, device=dev, dtype=torch.float32)
            row_act = torch.full((MROWS,), code, device=dev, dtype=torch.int32)
            a = torch.full((MROWS,), alpha, device=dev, dtype=torch.float32).requires_grad_(True)
            ones = torch.ones(MROWS, device=dev, dtype=torch.float32)

            out_k = M._glu_fwd(gu, row_act, row_alpha=a.detach(), row_gamma=ones)
            ggu_k, da_k, _ = M._glu_bwd(go, gu, row_act, row_alpha=a.detach(), row_gamma=ones,
                                        want_situ_grads=True)

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
    for code in (0, 2, 7):
        g = torch.Generator(device=dev).manual_seed(7)
        gu = torch.randn(MROWS, 2 * I, generator=g, device=dev, dtype=torch.float32)
        ra = torch.full((MROWS,), 1.0, device=dev, dtype=torch.float32)
        row_act = torch.full((MROWS,), code, device=dev, dtype=torch.int32)
        a_out = M._glu_fwd(gu, row_act, row_alpha=ra, row_gamma=ra)
        b_out = M._glu_fwd(gu, row_act)
        same = torch.equal(a_out, b_out)
        ok &= same
        print(f"  code {code} {NAMES[code]:<10} {'IDENTICAL' if same else 'DIFFERS  <-- FAIL'}")

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
