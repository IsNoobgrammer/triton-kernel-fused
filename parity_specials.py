"""Parity gate for the ±Identity special experts (act code 3 = +w*x, code 4 = -w*x).

Code 4 was the ZERO expert (emit nothing) until Jul 26 2026. This gate pins the new semantics on
every path that special-cases it -- moe_eager, moe_per_expert (manual backward), and the sm120
cuBLAS grouped path -- against an INDEPENDENT plain-PyTorch reference, forward and all four grads.

    python parity_specials.py            # needs CUDA (Triton kernels)
"""
import sys
import torch
import torch.nn.functional as F

from kernels.sm75.moe import moe_per_expert, moe_eager

DEV = "cuda"
_NS_EPS = 1e-6


def _act(g, code):
    if code == 0:
        return F.silu(g)
    if code == 1:
        r = F.relu(g.float())
        return (r * r).to(g.dtype)
    if code == 2:
        gf = g.float()
        gf = gf * torch.rsqrt(gf.square().mean(-1, keepdim=True) + _NS_EPS)
        return F.silu(gf).to(g.dtype)
    raise ValueError(code)


def reference(hidden, idx, wt, gate_up_proj, down_proj, codes):
    """Plain PyTorch, no kernel code in the path. GLU block owns weight slot == expert index."""
    N, H = hidden.shape
    I = gate_up_proj.shape[1] // 2
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)
    for e, c in enumerate(codes):
        sel = idx == e                                  # (N, k)
        rows = sel.any(-1)
        if not bool(rows.any()):
            continue
        we = (wt * sel).sum(-1)[rows].unsqueeze(-1)     # (m, 1) -- a token may pick e twice
        x = hidden[rows]
        if c in (3, 4):
            out[rows] += (x * we * (1.0 if c == 3 else -1.0)).float()
            continue
        gu = x @ gate_up_proj[e].t()
        eo = (_act(gu[:, :I], c) * gu[:, I:]) @ down_proj[e].t()
        out[rows] += (eo * we).float()
    return out.to(hidden.dtype)


def run(fn, hidden, idx, wt, gup, dn, codes_t, G):
    h = hidden.clone().requires_grad_(True)
    w = wt.clone().requires_grad_(True)
    a = gup.clone().requires_grad_(True)
    b = dn.clone().requires_grad_(True)
    out = fn(h, idx, w, a, b, codes_t)
    (out.float() * G).sum().backward()
    return out.detach(), h.grad, w.grad, a.grad, b.grad


def rel(x, y):
    return ((x - y).norm() / (y.norm() + 1e-12)).item()


def main():
    torch.manual_seed(0)
    N, H, I, k = 512, 128, 96, 2
    # 6 GLU (silu/radial/normsilu cycled) + 2 +Identity + 2 -Identity -- both signs, multi-expert blocks
    codes = [0, 8, 2, 0, 8, 2, 3, 3, 4, 4]
    E, e_glu = len(codes), 6
    codes_t = torch.tensor(codes, dtype=torch.int32, device=DEV)

    failures = []
    for dtype, tol in ((torch.float32, 2e-5), (torch.bfloat16, 3e-2)):
        hidden = torch.randn(N, H, device=DEV, dtype=dtype) * 0.5
        gup = torch.randn(e_glu, 2 * I, H, device=DEV, dtype=dtype) * 0.05
        dn = torch.randn(e_glu, H, I, device=DEV, dtype=dtype) * 0.05
        idx = torch.randint(0, E, (N, k), device=DEV)
        wt = torch.rand(N, k, device=DEV, dtype=dtype) + 0.1
        G = torch.randn(N, H, device=DEV, dtype=torch.float32)

        ref = run(reference, hidden, idx, wt, gup, dn, codes_t, G)
        paths = [("moe_eager", moe_eager), ("moe_per_expert", moe_per_expert)]
        if torch.cuda.get_device_capability(DEV)[0] >= 12 and dtype is torch.bfloat16:
            from kernels.sm120.moe_grouped import moe_grouped_cublas_polyglu, grouped_supported
            if grouped_supported(hidden, gup, dn):
                paths.append(("sm120 grouped", moe_grouped_cublas_polyglu))

        for name, fn in paths:
            got = run(fn, hidden, idx, wt, gup, dn, codes_t, G)
            rs = [rel(g.float(), r.float()) for g, r in zip(got, ref)]
            ok = max(rs) < tol
            failures += [] if ok else [f"{name}/{dtype}"]
            print(f"  [{'PASS' if ok else 'FAIL'}] {name:<16} {str(dtype):<16} "
                  f"out {rs[0]:.2e} dx {rs[1]:.2e} dw {rs[2]:.2e} dgup {rs[3]:.2e} ddn {rs[4]:.2e}")

    # The property the ± pair exists for: equal weights on +Identity and -Identity cancel exactly,
    # recovering the retired Zero expert's behaviour as a special case of a differentiable pair.
    hidden = torch.randn(N, H, device=DEV) * 0.5
    gup = torch.randn(e_glu, 2 * I, H, device=DEV) * 0.05
    dn = torch.randn(e_glu, H, I, device=DEV) * 0.05
    pair = torch.tensor([[6, 8]], device=DEV).expand(N, 2).contiguous()
    half = torch.full((N, 2), 0.5, device=DEV)
    for name, fn in (("moe_eager", moe_eager), ("moe_per_expert", moe_per_expert)):
        m = fn(hidden, pair, half, gup, dn, codes_t).abs().max().item()
        failures += [] if m < 1e-6 else [f"{name}/cancel"]
        print(f"  [{'PASS' if m < 1e-6 else 'FAIL'}] {name:<16} +Id/-Id cancellation  max|out| {m:.2e}")

    print(f"\n{'PASS' if not failures else 'FAIL ' + ', '.join(failures)} "
          f"-- ±Identity specials (act codes 3/4)")
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()
