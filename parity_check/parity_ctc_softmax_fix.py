"""CTC gradients at high confidence (Softmax Collapse, arXiv 2610.31642): ctc_loss.py and selfcond_ctc.py vs torch's
ctc_loss in fp64 on the same inputs.

Each sample gets a hard alignment (mostly blank, its labels at spread frames) and logits = scale * onehot(alignment)
+ noise, with scale per sample from 5 to 40, so the dominant class on every frame has margin ~scale. Reported per
scale: relative error of the frame's dominant-class gradient (|err| / |true|, median over frames) and how often its
sign matches fp64.

    python -m parity_check.parity_ctc_softmax_fix
"""
import torch
import torch.nn.functional as F

from kernels.sm120.ctc_loss import ctc_loss

dev = "cuda"


def make(B, T, V, U, scales, seed=0):
    g = torch.Generator().manual_seed(seed)
    blank = V - 1
    y = torch.randint(0, V - 1, (B, U), generator=g)
    y[:, 1::2] = (y[:, 0::2] + 1) % (V - 1)                     # no immediate repeats: every alignment feasible
    z = torch.randn(B, T, V, generator=g) * 0.3
    for b in range(B):
        al = torch.full((T,), blank)
        pos = torch.linspace(2, T - 3, U).round().long()
        al[pos] = y[b]
        z[b, torch.arange(T), al] += scales[b]
    return z, y, torch.full((B,), T), torch.full((B,), U), blank


def dominant_err(g, gref, z):
    am = z.argmax(-1, keepdim=True)
    a, r = g.gather(-1, am).squeeze(-1).double(), gref.gather(-1, am).squeeze(-1).double()
    return (a - r).abs() / r.abs().clamp_min(1e-300), torch.sign(a) == torch.sign(r), r


def main():
    scales = [5.0, 8.0, 12.0, 17.0, 25.0, 40.0]
    B, T, V, U = len(scales), 120, 256, 12
    z, y, tl_, ul, blank = make(B, T, V, U, scales)
    lp32 = z.log_softmax(-1).to(dev).requires_grad_(True)
    nll = ctc_loss(lp32, y.to(dev), tl_.to(dev), ul.to(dev), blank, zero_infinity=True)
    nll.sum().backward()
    lp64 = z.double().log_softmax(-1).requires_grad_(True)
    nr = F.ctc_loss(lp64.transpose(0, 1), y, tl_, ul, blank=blank, reduction="none", zero_infinity=True)
    nr.sum().backward()
    lp32t = z.log_softmax(-1).to(dev).requires_grad_(True)
    nt = F.ctc_loss(lp32t.transpose(0, 1), y.to(dev), tl_.to(dev), ul.to(dev), blank=blank, reduction="none",
                    zero_infinity=True)
    nt.sum().backward()
    print(f"{'scale':>6} {'ours: dom. grad rel err (median)':>33} {'sign ok':>8} {'torch fp32: rel err':>20} {'sign ok':>8}")
    ok = True
    for b, sc in enumerate(scales):
        e, s, r = dominant_err(lp32.grad[b].cpu(), lp64.grad[b], z[b])
        et, st, _ = dominant_err(lp32t.grad[b].cpu(), lp64.grad[b], z[b])
        live = r.abs() > 0
        print(f"{sc:>6} {e[live].median().item():>33.3e} {s[live].double().mean().item():>8.1%} "
              f"{et[live].median().item():>20.3e} {st[live].double().mean().item():>8.1%}")
        ok &= s[live].all().item() and e[live].median().item() < (1e-1 if sc > 30 else 1e-3)   # true ~e^-40 at 40
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
