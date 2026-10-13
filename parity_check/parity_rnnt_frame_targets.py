"""rnnt_joint_loss(..., frame_targets=K) (kernels/sm120/rnnt_joint.py _frame_targets_kernel) vs an fp64 reference,
and kl_div_topk's empty-slot handling.

Reference: the same joint as the kernel sees it (bf16 hidden relu(bf16(f + g)), bf16 W / bias, fp64 logits), fp64
alpha / beta by explicit loops, per (t, u) label posterior py = exp(alpha + lpy + beta(t, u+1) - ll), merged per token
per frame, q = py / max(1, sum py), blank = max(0, 1 - sum py).
Gates:
  1. every kernel entry (blank + top tokens) matches the reference probability (abs <= 5e-3: the kernel's joint is bf16
     GEMMs, loss ~8e-4 relative off fp64 in parity_rnnt_joint, which moves posteriors ~3e-3) and the kernel's tokens
     are the reference's top K-1 tokens (ties aside)
  2. sum over frames of the label posteriors = U per utterance (each label is emitted exactly once) within 1e-3
  3. loss unchanged by frame_targets (bitwise)
  4. kl_div_topk with -inf slots == kl_div_topk without them (loss and grad, bitwise)

    python parity_check/parity_rnnt_frame_targets.py
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from kernels.sm120.kl_div import kl_div_topk
from kernels.sm120.rnnt_joint import rnnt_joint_loss

ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def reference(f, g, W, bias, y, tlen, ylen):
    """per utterance: list over t of {token: prob} incl. blank (key = V - 1), and the label posterior total"""
    B, T, H = f.shape
    V = W.shape[0]
    out = []
    for b in range(B):
        tb, U = int(tlen[b]), int(ylen[b])
        h = torch.relu((f[b, :tb, None, :].bfloat16() + g[b, None, :U + 1, :].bfloat16()).bfloat16()).double()
        lg = h @ W.bfloat16().double().T + bias.bfloat16().double()
        lp = torch.log_softmax(lg, -1)
        lpb = lp[..., V - 1]
        lpy = torch.stack([lp[:, u, int(y[b, u])] for u in range(U)], 1) if U else torch.zeros(tb, 0, dtype=torch.float64)
        a = torch.full((tb, U + 1), -math.inf, dtype=torch.float64)
        bt = torch.full((tb, U + 1), -math.inf, dtype=torch.float64)
        for t in range(tb):
            for u in range(U + 1):
                if t == 0 and u == 0:
                    a[t, u] = 0.0
                    continue
                c1 = a[t - 1, u] + lpb[t - 1, u] if t > 0 else -math.inf
                c2 = a[t, u - 1] + lpy[t, u - 1] if u > 0 else -math.inf
                a[t, u] = torch.logaddexp(torch.tensor(c1, dtype=torch.float64), torch.tensor(c2, dtype=torch.float64))
        for t in reversed(range(tb)):
            for u in reversed(range(U + 1)):
                if t == tb - 1 and u == U:
                    bt[t, u] = lpb[t, u]
                    continue
                c1 = bt[t + 1, u] + lpb[t, u] if t < tb - 1 else -math.inf
                c2 = bt[t, u + 1] + lpy[t, u] if u < U else -math.inf
                bt[t, u] = torch.logaddexp(torch.tensor(c1, dtype=torch.float64), torch.tensor(c2, dtype=torch.float64))
        ll = bt[0, 0]
        frames, total = [], 0.0
        for t in range(tb):
            d = {}
            for u in range(U):
                p = math.exp(a[t, u] + lpy[t, u] + bt[t, u + 1] - ll)
                d[int(y[b, u])] = d.get(int(y[b, u]), 0.0) + p
            tot = sum(d.values())
            total += tot
            den = max(tot, 1.0)
            d = {k: v / den for k, v in d.items()}
            d[V - 1] = max(0.0, 1.0 - tot)
            frames.append(d)
        out.append((frames, total, U))
    return out


def main():
    gen = torch.Generator(device="cuda").manual_seed(0)
    B, T, U, H, V, K = 4, 24, 7, 32, 12, 4
    f = torch.randn(B, T, H, device="cuda", generator=gen)
    g = torch.randn(B, U + 1, H, device="cuda", generator=gen)
    W = torch.randn(V, H, device="cuda", generator=gen) / math.sqrt(H)
    bias = 0.1 * torch.randn(V, device="cuda", generator=gen)
    y = torch.randint(0, V - 1, (B, U), device="cuda", generator=gen)
    y[:, 2] = y[:, 1]                                                 # a repeated token: the merge path
    tlen = torch.tensor([T, T - 5, 12, 9], device="cuda")
    ylen = torch.tensor([U, U - 2, 4, 3], device="cuda")
    loss0, _ = rnnt_joint_loss(f, g, W, bias, y, tlen, ylen)
    loss1, _, (idx, lp) = rnnt_joint_loss(f, g, W, bias, y, tlen, ylen, frame_targets=K)
    check("3 loss unchanged", torch.equal(loss0, loss1), f"{loss0.item():.6f} vs {loss1.item():.6f}")
    ref = reference(f.cpu(), g.cpu(), W.cpu(), bias.cpu(), y.cpu(), tlen.cpu(), ylen.cpu())
    worst, topk_ok, tot_err = 0.0, True, 0.0
    for b in range(B):
        frames, total, Ub = ref[b]
        tot_err = max(tot_err, abs(total - Ub))
        for t, d in enumerate(frames):
            kernel = {int(idx[b, t, k]): math.exp(lp[b, t, k].item()) for k in range(K) if lp[b, t, k].item() > -math.inf}
            for tok, pk in kernel.items():
                worst = max(worst, abs(pk - d.get(tok, 0.0)))
            labels = sorted(((v, k) for k, v in d.items() if k != V - 1), reverse=True)[:K - 1]
            want = {k for v, k in labels if v > 1e-6}
            got = {k for k in kernel if k != V - 1}
            if not want <= got and min((v for v, k in labels if k not in got), default=0) > 1e-3:
                topk_ok = False
    check("1 entries vs fp64", worst <= 5e-3 and topk_ok, f"worst |p - ref| {worst:.1e}; top-{K - 1} tokens match {topk_ok}")
    check("2 label posteriors sum to U", tot_err <= 1e-3, f"max |sum - U| {tot_err:.1e}")
    # 4. kl_div_topk: a -inf slot carrying the blank id must change nothing
    N = 64
    s = torch.randn(N, V, device="cuda", generator=gen)
    lpf = torch.log_softmax(torch.randn(N, V, device="cuda", generator=gen), -1)
    lp3, ix3 = lpf.topk(3, -1)
    ix4 = torch.cat([ix3, torch.full((N, 1), V - 1, device="cuda", dtype=ix3.dtype)], 1)
    ix4[:, 0] = V - 1                                                 # blank also a real entry: the clobber case
    lp3[:, 0] = lpf[:, V - 1]
    ix3 = ix4[:, :3].clone()
    lp4 = torch.cat([lp3, torch.full((N, 1), -math.inf, device="cuda")], 1)
    sa = s.clone().requires_grad_()
    la = kl_div_topk(sa, ix3, lp3)
    la.sum().backward()
    sb = s.clone().requires_grad_()
    lb = kl_div_topk(sb, ix4, lp4)
    lb.sum().backward()
    check("4 -inf slots ignored", torch.equal(la, lb) and torch.equal(sa.grad, sb.grad),
          f"loss diff {(la - lb).abs().max().item():.1e}, grad diff {(sa.grad - sb.grad).abs().max().item():.1e}")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
