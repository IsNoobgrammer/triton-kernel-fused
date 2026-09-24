"""Parity: FusedMuon(scale_mode="muown") vs the reference Muown (github.com/kcc-lion/muown @3bd0c05).

    python parity_check/parity_muown.py [--ref C:/Users/shaur/src/muown]

Both optimizers get identical weights, identical gradients every step, and the SAME orthogonalizer
(ours, fp32), so any gap is the port itself: the (g, v) split, direction momentum, Adam on g,
recomposition and the decoupled-wd resync. The 3D expert stack is fed to the reference as separate
2D params (Muown is per-matrix, per-row, so that is the same math). Also asserts the invariants the
method rests on: ||W_i|| == g_i after every step, and the direction gradient has no radial part.
"""
import argparse
import os
import sys

os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")   # reference decorates its helpers with torch.compile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from kernels.muon import muon_scaling as S
from kernels.sm75.muon import FusedMuon, newton_schulz, _DSV4_COEFFS

SHAPES = [(48, 32), (32, 80), (3, 40, 24)]   # tall, wide, expert stack; none has rows == 3*cols


def run(ref_cls, wd, steps, dev):
    torch.manual_seed(0)
    init = [torch.randn(s, device=dev) * 0.05 for s in SHAPES]
    A = [torch.randn(s, device=dev) for s in SHAPES]

    ours = [torch.nn.Parameter(w.clone()) for w in init]
    ref = []
    for w in init:
        ref += [torch.nn.Parameter(x.clone()) for x in (w.unbind(0) if w.ndim == 3 else [w])]

    opt = FusedMuon(ours, lr=2e-2, weight_decay=wd, scale_mode="muown", ns_dtype=torch.float32)
    ropt = ref_cls(ref, lr=2e-2, weight_decay=wd, betas=S.MUOWN_BETAS, ns_steps=len(_DSV4_COEFFS))
    ropt._zeropower_fn = lambda G, steps: newton_schulz(G, _DSV4_COEFFS, torch.float32)

    worst = 0.0
    for t in range(steps):
        # loss whose gradient changes with W, so momentum and Adam state both matter
        grads = [a * (1 + t % 3) + 3.0 * w.detach() for a, w in zip(A, ours)]
        for p, gr in zip(ours, grads):
            p.grad = gr.clone()
        rg = []
        for gr in grads:
            rg += list(gr.unbind(0)) if gr.ndim == 3 else [gr]
        for p, gr in zip(ref, rg):
            p.grad = gr.clone()
        opt.step()
        ropt.step()

        flat = []
        for p in ours:
            flat += list(p.detach().unbind(0)) if p.ndim == 3 else [p.detach()]
        worst = max(worst, max((a - b.detach()).abs().max().item() for a, b in zip(flat, ref)))

        for p in ours:
            st = opt.state[p].get("muown")
            if st is None:
                continue
            rn = torch.linalg.vector_norm(p.detach().reshape(-1, p.shape[-1]), dim=-1)
            assert torch.allclose(rn, st["g"].reshape(-1), rtol=1e-5, atol=1e-7), "||W_i|| != g_i"
    return worst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="C:/Users/shaur/src/muown")
    ap.add_argument("--steps", type=int, default=25)
    a = ap.parse_args()
    sys.path.insert(0, a.ref)
    from optim.muown import Muown as RefMuown

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # the direction gradient must be tangent to every row (no radial component)
    W, G = torch.randn(4, 16, 8, device=dev), torch.randn(4, 16, 8, device=dev)
    st = S.muown_state(W)
    _, _, gv = S.muown_split(W, G, st["g"], st["vn"])
    radial = (gv * W).sum(-1).abs().max().item()
    assert radial < 1e-4, f"radial component {radial:.2e}"

    ok = True
    for wd in (0.0, 0.1):
        worst = run(RefMuown, wd, a.steps, dev)
        good = worst < 1e-5
        ok &= good
        print(f"wd={wd:<4} {a.steps} steps  max|ours-ref| = {worst:.3e}  {'PASS' if good else 'FAIL'}")
    print(f"radial component of dL/dv: {radial:.2e}")
    print("muown parity PASS" if ok else "muown parity FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
