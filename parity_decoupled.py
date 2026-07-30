"""Parity for act code 9 -- DECOUPLED SiLU: gamma * SiLU(alpha * g), both per expert, both live.

WHY: in plain SiLU(z) the linear factor and the sigmoid argument are the SAME z, so a single input
scale cannot deliver large magnitude AND an unsaturated gate. Measured on a trained checkpoint: gate
rms ~7, alpha 0.74 => sigma(z) = 0.995, i.e. the gate has collapsed to ReLU. Radial (code 8) escapes
this by pinning the gate argument at rms 1 while r^p carries the magnitude -- and it is the only arm
that has led on train loss. Code 9 asks whether a STATIC per-expert pair does the same job.

Unlike every other GLU code, GAMMA IS LIVE here (codes 0/1/2/6/7/8 pin dgamma to 0). Checks fwd,
d_gate_up, d_alpha, d_gamma against an autograd eager reference, that alpha=gamma=1 reproduces plain
silu EXACTLY, and an end-to-end moe_per_expert finite-difference pass on BOTH parameters.
Run on the box: python parity_decoupled.py
"""
import importlib
import torch
import torch.nn.functional as F

importlib.import_module("kernels.sm120.moe")
M = importlib.import_module("kernels.sm75.moe")

MROWS, I, CODE = 512, 768, 9


def eager(gate, up, alpha, gamma):
    g = gate.float()
    return gamma.unsqueeze(-1) * F.silu(alpha.unsqueeze(-1) * g) * up.float()


def main():
    dev = "cuda"
    ok = True
    print(f"{'alpha':>7}{'gamma':>7}{'fwd':>10}{'d_gateup':>11}{'d_alpha':>11}{'d_gamma':>11}")
    for a_v, g_v in ((1.0, 1.0), (0.15, 4.0), (0.5, 2.0), (2.0, 0.5), (0.05, 8.0)):
        gen = torch.Generator(device=dev).manual_seed(int(a_v * 100) + int(g_v * 10))
        gu = torch.randn(MROWS, 2 * I, generator=gen, device=dev, dtype=torch.float32
                         ).detach().requires_grad_(True)
        go = torch.randn(MROWS, I, generator=gen, device=dev, dtype=torch.float32)
        row_act = torch.full((MROWS,), CODE, device=dev, dtype=torch.int32)
        a = torch.full((MROWS,), a_v, device=dev, dtype=torch.float32).requires_grad_(True)
        gm = torch.full((MROWS,), g_v, device=dev, dtype=torch.float32).requires_grad_(True)

        out_k = M._glu_fwd(gu, row_act, code_hint=CODE, row_alpha=a.detach(), row_gamma=gm.detach())
        ggu_k, da_k, dg_k = M._glu_bwd(go, gu, row_act, code_hint=CODE, row_alpha=a.detach(),
                                       row_gamma=gm.detach(), want_situ_grads=True)

        ref = eager(gu[:, :I], gu[:, I:], a, gm)
        (ref * go).sum().backward()
        rel = lambda x, y: ((x - y).norm() / y.norm().clamp_min(1e-12)).item()
        e = [rel(out_k, ref), rel(ggu_k, gu.grad), rel(da_k, a.grad), rel(dg_k, gm.grad)]
        good = max(e) < 2e-5
        ok &= good
        print(f"{a_v:>7.2f}{g_v:>7.2f}" + "".join(f"{x:>11.1e}" for x in e)
              + ("" if good else "   <-- FAIL"))

    # alpha=gamma=1 must reproduce plain silu (code 0) exactly
    print("\nalpha=gamma=1 == plain silu (code 0):")
    gen = torch.Generator(device=dev).manual_seed(7)
    gu = torch.randn(MROWS, 2 * I, generator=gen, device=dev, dtype=torch.float32)
    ones = torch.ones(MROWS, device=dev, dtype=torch.float32)
    a9 = M._glu_fwd(gu, torch.full((MROWS,), 9, device=dev, dtype=torch.int32),
                    code_hint=9, row_alpha=ones, row_gamma=ones)
    a0 = M._glu_fwd(gu, torch.full((MROWS,), 0, device=dev, dtype=torch.int32), code_hint=0)
    same = torch.equal(a9, a0)
    print(f"  {'IDENTICAL' if same else 'DIFFERS  <-- FAIL'}")
    ok &= same

    print("\nmissing-param guard:")
    try:
        M._glu_fwd(gu, torch.full((MROWS,), 9, device=dev, dtype=torch.int32), code_hint=9,
                   row_alpha=ones)          # gamma absent
        print("  no error  <-- FAIL"); ok = False
    except ValueError as exc:
        print(f"  ValueError OK: {str(exc)[:58]}...")

    ok &= end_to_end(dev)
    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


def end_to_end(dev):
    """Through moe_per_expert: both params must move the output and match finite differences.
    Kernel parity cannot see a dispatcher that drops act_params -- how alpha shipped inert."""
    H, Ei, E, N, K = 512, 768, 4, 256, 2
    codes_l = [9, 2, 9, 0]
    print("\nend-to-end moe_per_expert:")
    gen = torch.Generator(device=dev).manual_seed(21)
    rn = lambda *s: torch.randn(*s, generator=gen, device=dev, dtype=torch.float32)
    hid, gup, dwn = rn(N, H) * 0.1, rn(E, 2 * Ei, H) * 0.05, rn(E, H, Ei) * 0.05
    idx = torch.randint(0, E, (N, K), generator=gen, device=dev, dtype=torch.int64)
    wt, go = torch.rand(N, K, generator=gen, device=dev), rn(N, H)
    codes = torch.tensor(codes_l, dtype=torch.int32, device=dev)
    loss = lambda ap: (M.moe_per_expert(hid, idx, wt, gup, dwn, codes, act_params=ap) * go).sum()

    ap = torch.ones(E, 2, device=dev, dtype=torch.float32)
    ap[:, 0] = torch.tensor([0.2, 1.0, 0.6, 1.0], device=dev)   # alpha
    ap[:, 1] = torch.tensor([3.0, 1.0, 1.8, 1.0], device=dev)   # gamma
    base = loss(ap.clone())
    for col, nm in ((0, "alpha"), (1, "gamma")):
        ap2 = ap.clone(); ap2[0, col] += 0.5
        live = not torch.allclose(base, loss(ap2), rtol=1e-6)
        print(f"  {nm} moves the output: {'YES' if live else 'NO   <-- FAIL (inert)'}")
        if not live:
            return False

    apg = ap.clone().requires_grad_(True)
    loss(apg).backward()
    # h=1e-2, NOT 1e-3: the loss is ~22, whose fp32 resolution is ~2.6e-6, so at h=1e-3 the
    # difference loss(p)-loss(m) (~6e-4) keeps only ~2 significant digits and the FD carries ~0.8%
    # cancellation error. Verified by sweeping h against a fixed analytic value: err went
    # 2.45e-3 (h=1e-3) -> 9.6e-5 (3e-3) -> 1.3e-4 (1e-2) -> 3.2e-5 (3e-2), i.e. error FALLING as h
    # grows = cancellation, not truncation. The loss itself is bit-repeatable (spread 0.0 over 5
    # evals), so this is not the fp32 atomic scatter.
    h, ok = 1e-2, True
    for col, nm in ((0, "alpha"), (1, "gamma")):
        for e in range(E):
            p_, m_ = ap.clone(), ap.clone()
            p_[e, col] += h; m_[e, col] -= h
            fd = ((loss(p_) - loss(m_)) / (2 * h)).item()
            an = apg.grad[e, col].item()
            good = abs(fd - an) <= 2e-3 * max(abs(fd), 1.0)
            ok &= good
            print(f"  e{e} code {codes_l[e]:<2} d_{nm:<5} analytic={an:>10.4f} fd={fd:>10.4f}"
                  f"{'' if good else '   <-- FAIL'}")
    return ok


if __name__ == "__main__":
    main()
