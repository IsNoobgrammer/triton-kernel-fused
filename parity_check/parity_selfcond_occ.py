"""selfcond_ctc_pass(..., want_occ=True): the CTC label occupancy occ (B, T, U) vs fp64.

Reference: d(-log P) / d(log p_t(v)) of torch.ctc_loss (fp64, reduction sum) = - the occupancy of symbol v at frame t,
which equals the per-position occupancy when every utterance's tokens are distinct (the test draws them so), on the
same bf16 logits the pass uses.
Gates:
  1. occ == fp64 occupancy (abs <= 2e-3)
  2. per frame: sum_u occ + blank occupancy == 1; per token: sum_t occ >= 1 - 1e-3
  3. want_occ changes nothing else: nll and grads bitwise vs want_occ=False

    python parity_check/parity_selfcond_occ.py
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


def main():
    gen = torch.Generator(device="cuda").manual_seed(0)
    B, T, d, V, U = 4, 60, 64, 129, 10
    x = torch.randn(B, T, d, device="cuda", generator=gen)
    w = torch.randn(V, d, device="cuda", generator=gen) / 4
    b = 0.1 * torch.randn(V, device="cuda", generator=gen)
    b[-1] += 1.5                                                        # blank-leaning, like a real head
    y = torch.stack([torch.randperm(V - 1, generator=torch.Generator().manual_seed(i))[:U] for i in range(B)]).cuda()
    tl_ = torch.tensor([T, T - 9, 40, 30], device="cuda")
    ul = torch.tensor([U, U - 3, 6, 4], device="cuda")
    nll, _, z, occ = selfcond_ctc_pass(x, w, b, None, y, tl_, ul, V - 1, want_occ=True)
    # fp64 reference on the same bf16 logits
    zr = (x.reshape(B * T, d).bfloat16() @ w.bfloat16().t() + b.bfloat16()).double().view(B, T, V)
    lp = zr.log_softmax(-1).detach().requires_grad_()
    loss = torch.nn.functional.ctc_loss(lp.transpose(0, 1), y, tl_, ul, blank=V - 1, reduction="sum",
                                        zero_infinity=True)
    (g,) = torch.autograd.grad(loss, lp)
    occ_sym = -g                                                        # (B, T, V): symbol occupancy per frame
    ref = torch.zeros(B, T, U, dtype=torch.float64, device="cuda")
    for i in range(B):
        for u in range(int(ul[i])):
            ref[i, :int(tl_[i]), u] = occ_sym[i, :int(tl_[i]), y[i, u]]
    e1 = (occ.double() - ref).abs().max().item()
    check("1 occupancy == fp64", e1 <= 2e-3, f"max abs {e1:.1e}")
    tm = torch.arange(T, device="cuda")[None, :] < tl_[:, None]
    tot = occ.double().sum(-1) + occ_sym[..., V - 1]
    e2 = (tot - 1)[tm].abs().max().item()
    per_tok = occ.double().sum(1)
    um = torch.arange(U, device="cuda")[None, :] < ul[:, None]
    check("2 frame sums == 1, token sums >= 1", e2 <= 2e-3 and bool((per_tok[um] >= 1 - 1e-3).all()),
          f"max |frame sum - 1| {e2:.1e}, token occupancy {per_tok[um].min().item():.3f}..{per_tok[um].max().item():.3f}")

    def run(want):
        xs, ws, bs = (t.clone().requires_grad_() for t in (x, w, b))
        out = selfcond_ctc_pass(xs, ws, bs, None, y, tl_, ul, V - 1, want_occ=want)
        out[0].sum().backward()
        return out[0].detach(), xs.grad, ws.grad, bs.grad
    a0, a1 = run(False), run(True)
    check("3 want_occ changes nothing else", all(torch.equal(p, q) for p, q in zip(a0, a1)), "nll + grads bitwise")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
