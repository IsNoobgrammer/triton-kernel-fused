"""selfcond_ctc_pass(..., want_occ=True): the CTC label occupancy occ (B, T, U) vs fp64.

Reference: an explicit fp64 CTC forward-backward (torch.ctc_loss's backward returns exp(log_probs) - occupancy, a
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
    lp = zr.log_softmax(-1).cpu()
    ref = torch.zeros(B, T, U, dtype=torch.float64)
    bocc = torch.zeros(B, T, dtype=torch.float64)                      # blank-state occupancy per frame
    NEG = float("-inf")
    for i in range(B):                                                  # explicit fp64 CTC forward-backward
        Ti, Ui = int(tl_[i]), int(ul[i])
        S = 2 * Ui + 1
        lab = [V - 1 if s % 2 == 0 else int(y[i, s // 2]) for s in range(S)]
        a = torch.full((Ti, S), NEG, dtype=torch.float64)
        be = torch.full((Ti, S), NEG, dtype=torch.float64)
        a[0, 0] = lp[i, 0, lab[0]]
        if S > 1:
            a[0, 1] = lp[i, 0, lab[1]]
        for t in range(1, Ti):
            for s in range(S):
                c = [a[t - 1, s]] + ([a[t - 1, s - 1]] if s >= 1 else []) + \
                    ([a[t - 1, s - 2]] if s >= 2 and s % 2 == 1 and lab[s] != lab[s - 2] else [])
                a[t, s] = torch.logsumexp(torch.stack(c), 0) + lp[i, t, lab[s]]
        be[Ti - 1, S - 1] = lp[i, Ti - 1, lab[S - 1]]
        if S > 1:
            be[Ti - 1, S - 2] = lp[i, Ti - 1, lab[S - 2]]
        for t in range(Ti - 2, -1, -1):
            for s in range(S):
                c = [be[t + 1, s]] + ([be[t + 1, s + 1]] if s + 1 < S else []) + \
                    ([be[t + 1, s + 2]] if s + 2 < S and s % 2 == 1 and lab[s] != lab[s + 2] else [])
                be[t, s] = torch.logsumexp(torch.stack(c), 0) + lp[i, t, lab[s]]
        ll = torch.logsumexp(a[Ti - 1, max(S - 2, 0):S], 0)
        g = torch.exp(a + be - torch.stack([lp[i, :Ti, lab[s]] for s in range(S)], 1) - ll)   # (Ti, S)
        for u in range(Ui):
            ref[i, :Ti, u] = g[:, 2 * u + 1]
        bocc[i, :Ti] = g[:, 0::2].sum(1)
    ref, bocc = ref.cuda(), bocc.cuda()
    e1 = (occ.double() - ref).abs().max().item()
    check("1 occupancy == fp64", e1 <= 2e-3, f"max abs {e1:.1e}")
    tm = torch.arange(T, device="cuda")[None, :] < tl_[:, None]
    tot = occ.double().sum(-1) + bocc
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
