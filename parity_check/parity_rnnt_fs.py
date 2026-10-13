"""rnnt_joint_loss(..., fs=teacher_nll): full-sum (sequence-level) KD, loss = hard * mean(nll) + kd * mean(|nll - t|).

Gates:
  1. teacher nll far below the student's (every sign +1): grads == the plain loss at hard_weight = hard + kd (rel 1e-6)
  2. mixed signs: grads == the sum over single-utterance plain runs at hard_weight = (hard + kd * sign_b) / B (rel 2e-3;
     batched vs per-utterance GEMMs round differently)
  3. loss value == hard * mean(nll) + kd * mean(|nll - t|)
  4. the plain loss is unchanged by the per-utterance weight plumbing: bitwise vs a second run (repeatable)

    python parity_check/parity_rnnt_fs.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import parity_rnnt_joint as pj
import kernels.sm120.rnnt_joint as rj

AMP = torch.autocast("cuda", dtype=torch.bfloat16)
ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def run(f, g, W, b, y, tl, yl, **kw):
    f, g, W, b = (x.detach().clone().requires_grad_() for x in (f, g, W, b))
    with AMP:
        out = rj.rnnt_joint_loss(f, g, W, b, y, tl, yl, **kw)
    out[0].backward()
    return out, (f.grad, g.grad, W.grad, b.grad)


def rel(a, r):
    return ((a.double() - r.double()).norm() / r.double().norm().clamp(min=1e-30)).item()


def case(name, B, T, U, H, V):
    print(f"{name}: B {B} T {T} U {U} V {V}", flush=True)
    f, g, W, b, y, tl, yl = pj.data(B, T, U, H, V, seed=5)
    hw, kw = 0.3, 0.7
    (l0, nll), g0 = run(f, g, W, b, y, tl, yl)
    (_, _), g0b = run(f, g, W, b, y, tl, yl)
    check("4 plain loss repeatable", all(torch.equal(a, c) for a, c in zip(g0, g0b)), "bitwise")
    (lp, _), gp = run(f, g, W, b, y, tl, yl, hard_weight=hw + kw)
    (lf, _, d), gf = run(f, g, W, b, y, tl, yl, hard_weight=hw, kd_weight=kw, fs=nll.detach() - 1e6)
    e1 = max(rel(a, c) for a, c in zip(gf, gp))
    check("1 all signs +1 == plain at hard + kd", e1 <= 1e-6, f"max rel {e1:.1e}")
    t = nll.detach() + torch.where(torch.arange(B, device=nll.device) % 2 == 0, -5.0, 5.0)
    (lf, nll2, d), gf = run(f, g, W, b, y, tl, yl, hard_weight=hw, kd_weight=kw, fs=t)
    sgn = torch.sign(nll2 - t)
    ref = [torch.zeros_like(x) for x in gf]
    for i in range(B):
        sl = slice(i, i + 1)
        _, gi = run(f[sl], g[sl], W, b, y[sl], tl[sl], yl[sl], hard_weight=float((hw + kw * sgn[i]) / B))
        ref[0][sl] += gi[0]
        ref[1][sl] += gi[1]
        ref[2] += gi[2]
        ref[3] += gi[3]
    e2 = max(rel(a, c) for a, c in zip(gf, ref))
    check("2 mixed signs == per-utterance weighted sum", e2 <= 2e-3,
          f"max rel {e2:.1e} (signs {sgn.tolist()})")
    want = hw * nll2.mean() + kw * (nll2 - t).abs().mean()
    check("3 loss value", abs(lf.item() - want.item()) <= 1e-5 * abs(want.item()), f"{lf.item():.6f} vs {want.item():.6f}")


def main():
    case("small", 4, 30, 10, 64, 129)
    case("wide", 3, 60, 16, 128, 1025)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
