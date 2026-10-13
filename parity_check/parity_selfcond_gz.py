"""selfcond_ctc_pass: a loss on the returned logits (gz) joins the fused backward.

Fused pass with loss = sum(nll) + sum(z * R) vs the same pass with R = 0: the gradient DIFFERENCE must be the extra
term's own gradient, computed independently (dx = R W, dW = R^T x, db = sum R, with x / W the bf16 values the pass
uses). Gates: relative L2 <= 2e-2 on dx / dW / db (dz is bf16), and R = 0 changes nothing (bitwise).

    python parity_check/parity_selfcond_gz.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from kernels.sm120.selfcond_ctc import selfcond_ctc_pass

ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-30)).item()


def run(x0, w0, b0, y, tl_, ul, R):
    x = x0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    b = b0.clone().requires_grad_()
    nll, _, z = selfcond_ctc_pass(x, w, b, None, y, tl_, ul, w0.shape[0] - 1)
    loss = nll.sum() + ((z.float() * R).sum() if R is not None else 0.0)
    loss.backward()
    return x.grad, w.grad, b.grad


def main():
    gen = torch.Generator(device="cuda").manual_seed(0)
    B, T, d, V, U = 4, 50, 64, 129, 8
    x = torch.randn(B, T, d, device="cuda", generator=gen)
    w = torch.randn(V, d, device="cuda", generator=gen) / 8
    b = 0.1 * torch.randn(V, device="cuda", generator=gen)
    y = torch.randint(0, V - 1, (B, U), device="cuda", generator=gen)
    tl_ = torch.tensor([T, T - 7, 30, 20], device="cuda")
    ul = torch.tensor([U, U - 2, 5, 3], device="cuda")
    R = torch.randn(B, T, V, device="cuda", generator=gen)
    g0 = run(x, w, b, y, tl_, ul, None)
    gz0 = run(x, w, b, y, tl_, ul, torch.zeros_like(R))
    check("2 R = 0 changes nothing", all(torch.equal(a, c) for a, c in zip(g0, gz0)), "bitwise")
    gR = run(x, w, b, y, tl_, ul, R)
    Rb = R.bfloat16().float().reshape(B * T, V)                       # dz is bf16 in the pass
    xb = x.bfloat16().float().reshape(B * T, d)
    wb = w.bfloat16().float()
    want = ((Rb @ wb).reshape(B, T, d), Rb.t() @ xb, Rb.sum(0))
    got = tuple(a - c for a, c in zip(gR, g0))
    errs = [rel(gt, wt) for gt, wt in zip(got, want)]
    check("1 grad of sum(z R) joins the backward", max(errs) <= 2e-2,
          f"dx {errs[0]:.1e}  dW {errs[1]:.1e}  db {errs[2]:.1e}")

    # 3. ctc=False (hard weight 0): no CTC at all; loss = sum(z R) + sum(q Rq) through the feedback, vs fp32 eager
    wf0 = torch.randn(2 * d, V, device="cuda", generator=gen) / 8
    Rq = torch.randn(B, T, 2 * d, device="cuda", generator=gen)
    xs, ws, bs, wfs = (t.clone().requires_grad_() for t in (x, w, b, wf0))
    nll, q, z = selfcond_ctc_pass(xs, ws, bs, wfs, y, tl_, ul, V - 1, ctc=False)
    ((z.float() * R).sum() + (q.float() * Rq).sum()).backward()
    xr, wr, br, wfr = (t.bfloat16().float().clone().requires_grad_() for t in (x, w, b, wf0))
    zr = xr.reshape(B * T, d) @ wr.t() + br
    qr = zr.softmax(-1) @ wfr.t()
    ((zr * R.reshape(B * T, V)).sum() + (qr * Rq.reshape(B * T, -1)).sum()).backward()
    e3 = [rel(a, c) for a, c in ((xs.grad, xr.grad), (ws.grad, wr.grad), (bs.grad, br.grad), (wfs.grad, wfr.grad))]
    check("3 ctc=False: only the logit / feedback losses", max(e3) <= 2e-2 and bool((nll == 0).all()),
          f"dx {e3[0]:.1e}  dW {e3[1]:.1e}  db {e3[2]:.1e}  dWfb {e3[3]:.1e}  nll {nll.abs().max().item():.0f}")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
