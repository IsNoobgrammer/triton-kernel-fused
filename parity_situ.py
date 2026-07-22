"""Parity gate for SiTU (act code 5): moe_per_expert vs moe_eager under autograd.

Checks out / grad_hidden / grad_gate_up_proj / grad_down_proj / grad_topk_weights /
grad_act_params (alpha, gamma) on: all-SiTU, mixed codes, mixed + specials, and an
empty-expert stack. fp32 tight + bf16-autocast loose. Run: python parity_situ.py
"""
import torch

from kernels.sm75.moe import moe_per_expert, moe_eager

DEV = "cuda"
N, H, I, TOPK = 512, 128, 96, 2


def _mk(E_glu, seed=0, dtype=torch.float32):
    g = torch.Generator(device=DEV).manual_seed(seed)
    hid = (torch.randn(N, H, device=DEV, generator=g, dtype=torch.float32) * 0.5).to(dtype)
    gup = (torch.randn(E_glu, 2 * I, H, device=DEV, generator=g, dtype=torch.float32) * 0.1).to(dtype)
    dwn = (torch.randn(E_glu, H, I, device=DEV, generator=g, dtype=torch.float32) * 0.1).to(dtype)
    return hid, gup, dwn, g


def _route(E, g, cap=None):
    lim = cap or E
    scores = torch.rand(N, lim, device=DEV, generator=g)
    wt, idx = scores.topk(TOPK, dim=-1)
    return idx, (wt / wt.sum(-1, keepdim=True))


def _rel(a, b):
    a, b = a.float(), b.float()
    return ((a - b).abs().max() / (b.abs().max() + 1e-8)).item()


def check(name, codes, cap=None, dtype=torch.float32, autocast=False, tol=1e-4, I_=None):
    global I
    I_saved, I = I, (I_ or I)
    E = len(codes)
    E_glu = sum(1 for c in codes if c in (0, 1, 2, 5))
    assert all(c in (0, 1, 2, 5) for c in codes[:E_glu]), "GLU codes must precede specials"
    hid, gup, dwn, g = _mk(E_glu, dtype=dtype)
    idx, wt = _route(E, g, cap)
    act_codes = torch.tensor(codes, dtype=torch.int32, device=DEV)
    ap = torch.stack([torch.rand(E, device=DEV, generator=g) + 0.5,
                      torch.rand(E, device=DEV, generator=g) + 0.5], dim=1)

    def run(fn):
        h, gu, dw, w = (t.detach().clone().requires_grad_(True) for t in (hid, gup, dwn, wt))
        a = ap.detach().clone().requires_grad_(True)
        if autocast:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = fn(h, idx, w, gu, dw, act_codes, a)
        else:
            out = fn(h, idx, w, gu, dw, act_codes, a)
        gsig = torch.randn_like(out.float()) * 0.1
        (out.float() * gsig).sum().backward()
        return out, h.grad, gu.grad, dw.grad, w.grad, a.grad, gsig

    torch.manual_seed(7)   # same grad signal both runs
    o1, gh1, ggu1, gdw1, gw1, ga1, _ = run(moe_per_expert)
    torch.manual_seed(7)
    o2, gh2, ggu2, gdw2, gw2, ga2, _ = run(moe_eager)
    rels = {"out": _rel(o1, o2), "d_hid": _rel(gh1, gh2), "d_gup": _rel(ggu1, ggu2),
            "d_dwn": _rel(gdw1, gdw2), "d_wt": _rel(gw1, gw2)}
    if ga2 is not None and ga2.abs().max() > 0:
        rels["d_ap"] = _rel(ga1, ga2)
    worst = max(rels.values())
    status = "PASS" if worst < tol else "FAIL"
    print(f"[{status}] {name:<28} worst={worst:.2e}  " +
          "  ".join(f"{k}={v:.1e}" for k, v in rels.items()), flush=True)
    I = I_saved
    return worst < tol


def main():
    torch.manual_seed(0)
    ok = True
    # fp32 baseline (tight) — row-fused path (I=96 <= 1024)
    ok &= check("all-situ fp32", [5] * 6)
    ok &= check("all-relu2 fp32", [1] * 6)
    ok &= check("all-normsilu fp32", [2] * 6)
    ok &= check("mixed 0/1/2/5 fp32", [0, 1, 2, 5, 0, 5])
    ok &= check("situ+specials fp32", [0, 5, 2, 5, 3, 4])
    ok &= check("empty-experts fp32", [5, 5, 5, 5, 0, 2], cap=4)   # experts 4,5 get zero rows
    # tiled fallback path (I=1536 > _ROWFUSE_MAX_I) — both paths stay gated
    ok &= check("mixed fp32 I=1536 (tiled)", [0, 1, 2, 5, 0, 5], I_=1536)
    ok &= check("situ+spec fp32 I=1536", [0, 5, 2, 5, 3, 4], I_=1536)
    # bf16 baselines: pure-bf16 tensors AND autocast-on-fp32 (loose, bf16 floor)
    ok &= check("mixed pure-bf16", [0, 1, 2, 5, 0, 5], dtype=torch.bfloat16, tol=3e-2)
    ok &= check("all-normsilu pure-bf16", [2] * 6, dtype=torch.bfloat16, tol=3e-2)
    ok &= check("all-situ bf16-ac", [5] * 6, autocast=True, tol=3e-2)
    ok &= check("mixed+specials bf16-ac", [0, 5, 2, 5, 3, 4], autocast=True, tol=3e-2)
    ok &= check("mixed bf16-ac I=1536", [0, 1, 2, 5, 0, 5], autocast=True, tol=3e-2, I_=1536)
    print("\nPARITY " + ("OK" if ok else "FAILED"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
