"""Localize the grouped-MoE grad break: fused (moe_grouped) vs eager (moe_eager) vs an fp32 torch
truth, split per output/grad tensor and per activation config (PolyGLU 0/1/2 vs all-SiLU).

Run:  ../BiBo/.venv/Scripts/python.exe .autoresearch/grouped_parity.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from kernels.sm75.moe import moe_grouped, moe_eager

DEV = "cuda"


def rel(a, b):
    return (a - b).abs().max().item(), ((a - b).norm() / (b.norm() + 1e-12)).item()


def _act(gate, code, eps=1e-6):
    if code == 0:
        return torch.nn.functional.silu(gate)
    if code == 1:
        r = torch.relu(gate)
        return r * r
    g = gate.float() * torch.rsqrt(gate.float().square().mean(-1, keepdim=True) + eps)
    return torch.nn.functional.silu(g).to(gate.dtype)


def truth(hidden, idx, wt, gup, dwn, codes, *, dt):
    """fp32 (or dt) per-expert reference with autograd — the ground truth."""
    N, H = hidden.shape
    E, twoI, _ = gup.shape
    I = twoI // 2
    k = idx.shape[1]
    h = hidden.to(dt).detach().requires_grad_(True)
    Wg = gup.to(dt).detach().requires_grad_(True)
    Wd = dwn.to(dt).detach().requires_grad_(True)
    w = wt.to(dt).detach().requires_grad_(True)
    out = torch.zeros(N, H, device=DEV, dtype=dt)
    for tk in range(k):
        for e in range(E):
            m = (idx[:, tk] == e)
            if not bool(m.any()):
                continue
            x_e = h[m]
            gu = x_e @ Wg[e].t()
            inter = _act(gu[:, :I], int(codes[e])) * gu[:, I:]
            eo = inter @ Wd[e].t()
            out[m] = out[m] + eo * w[m, tk:tk + 1]
    return out, h, Wg, Wd, w


def run(N, H, I, E, k, codes, label):
    torch.manual_seed(0)
    hid = torch.randn(N, H, device=DEV, dtype=torch.float16)
    gup = (torch.randn(E, 2 * I, H, device=DEV, dtype=torch.float16) * 0.02)
    dwn = (torch.randn(E, H, I, device=DEV, dtype=torch.float16) * 0.02)
    codes_t = torch.tensor(codes, device=DEV, dtype=torch.int32)
    # deterministic routing: random logits -> top-k
    logits = torch.randn(N, E, device=DEV)
    wt_full, idx_full = torch.topk(logits, k, dim=1)
    idx = idx_full.long()
    wt = (torch.sigmoid(wt_full) / 1.0).to(torch.float16)  # unbiased, no norm (isolates the expert math)
    G = torch.randn(N, H, device=DEV, dtype=torch.float32)

    # ---- fused grouped ----
    h2 = hid.clone().requires_grad_(True)
    g2 = gup.clone().requires_grad_(True)
    d2 = dwn.clone().requires_grad_(True)
    w2 = wt.clone().requires_grad_(True)
    o_k = moe_grouped(h2, idx, w2, g2, d2, codes_t)
    (o_k.float() * G).sum().backward()

    # ---- eager ----
    h3 = hid.clone().requires_grad_(True)
    g3 = gup.clone().requires_grad_(True)
    d3 = dwn.clone().requires_grad_(True)
    w3 = wt.clone().requires_grad_(True)
    o_e = moe_eager(h3, idx, w3, g3, d3, codes_t)
    (o_e.float() * G).sum().backward()

    # ---- fp32 truth ----
    o_t, h_t, g_t, d_t, w_t = truth(hid, idx, wt, gup, dwn, codes_t, dt=torch.float32)
    (o_t * G).sum().backward()

    print(f"\n=== {label}  N={N} H={H} I={I} E={E} k={k} codes={codes} ===")
    for name, fk, ek, tk in [
        ("fwd out", o_k.float(), o_e.float(), o_t),
        ("grad_hidden", h2.grad.float(), h3.grad.float(), h_t.grad),
        ("grad_gate_up", g2.grad.float(), g3.grad.float(), g_t.grad),
        ("grad_down", d2.grad.float(), d3.grad.float(), d_t.grad),
        ("grad_wt", w2.grad.float(), w3.grad.float(), w_t.grad),
    ]:
        ka, kr = rel(fk, tk)          # fused vs truth
        ea, er = rel(ek, tk)          # eager vs truth
        fa, fr = rel(fk, ek)          # fused vs eager (the bench metric)
        flag = "  <-- BENCH METRIC" if name == "grad_wt" else ""
        print(f"  {name:14s} fused-vs-truth rel={kr:.2e}  eager-vs-truth rel={er:.2e}  "
              f"fused-vs-eager rel={fr:.2e}{flag}")


if __name__ == "__main__":
    # PolyGLU (mixed 0/1/2) — the bench config shape, scaled down
    run(4096, 256, 256, 6, 2, [0, 1, 2, 0, 1, 2], "PolyGLU mixed")
    # all-SiLU — the common default (moe() dispatches to grouped when all codes <= 2)
    run(4096, 256, 256, 6, 2, [0, 0, 0, 0, 0, 0], "all-SiLU")
    # bench's real shape (reproduce rel~386)
    run(16384, 512, 768, 11, 2, [i % 3 for i in range(9)] + [3, 4], "bench-shape w/ specials")
