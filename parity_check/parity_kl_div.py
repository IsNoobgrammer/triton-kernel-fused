"""kl_div_dense / kl_div_topk (kernels/sm120/kl_div.py) vs an fp64 autograd reference.

Gates (rel. L2 error vs fp64, per tensor):
  1. dense loss + ds: fp32 inputs <= 1e-5; bf16 student / teacher logits (exact in fp64) loss <= 1e-5, ds <= 8e-3
     (ds is written in bf16)
  2. top-k loss + ds, same tolerances; the rest-bucket formula against autograd through an explicit fp64 build
  3. top-k with k = V (no rest mass) == dense
  4. bitwise repeatable; masked rows: grad 0 where the incoming grad is 0

    python parity_check/parity_kl_div.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from kernels.sm120.kl_div import kl_div_dense, kl_div_topk

ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-300)).item()


def ref_dense(s, t, T):
    s = s.double().detach().requires_grad_()
    lq = torch.log_softmax(s / T, -1)
    lp = torch.log_softmax(t.double() / T, -1)
    return (lp.exp() * (lp - lq)).sum(-1), s


def ref_topk(s, idx, logp, T):
    s = s.double().detach().requires_grad_()
    lq = torch.log_softmax(s / T, -1)
    lqk = lq.gather(1, idx.long())
    p = logp.double().exp()
    r = (1 - p.sum(1)).clamp(min=0)
    q_rest = (1 - lqk.exp().sum(1)).clamp(min=1e-30)
    kl = (p * (logp.double() - lqk)).sum(1) + torch.where(r > 0, r * (r.clamp(min=1e-30).log() - q_rest.log()), 0 * r)
    return kl, s


def main():
    gen = torch.Generator(device="cuda").manual_seed(0)
    for N, V, T in ((4096, 1025, 1.0), (2000, 2048, 2.0), (513, 5000, 1.0)):
        for dt, tol_g in ((torch.float32, 1e-5), (torch.bfloat16, 8e-3)):
            print(f"\n== N={N} V={V} T={T} {dt}", flush=True)
            s = (3 * torch.randn(N, V, device="cuda", generator=gen)).to(dt)
            t = (3 * torch.randn(N, V, device="cuda", generator=gen)).to(dt)
            w = torch.rand(N, device="cuda", generator=gen)
            w[::7] = 0                                                   # masked rows
            # dense
            kr, sr = ref_dense(s, t, T)
            (kr * w.double()).sum().backward()
            sx = s.detach().clone().requires_grad_()
            ko = kl_div_dense(sx, t, T)
            (ko * w).sum().backward()
            check("1 dense", rel(ko, kr) <= 1e-5 and rel(sx.grad, sr.grad) <= tol_g,
                  f"loss {rel(ko, kr):.1e}  ds {rel(sx.grad, sr.grad):.1e}")
            sx2 = s.detach().clone().requires_grad_()
            ko2 = kl_div_dense(sx2, t, T)
            (ko2 * w).sum().backward()
            check("4 repeatable + mask", torch.equal(ko, ko2) and torch.equal(sx.grad, sx2.grad)
                  and sx.grad[::7].abs().max().item() == 0, "bitwise; masked rows grad 0")
            # top-k from the teacher's full softmax
            K = 8
            lp_full = torch.log_softmax(t.float() / T, -1)
            logp, idx = lp_full.topk(K, -1)
            kr, sr = ref_topk(s, idx, logp, T)
            (kr * w.double()).sum().backward()
            sx = s.detach().clone().requires_grad_()
            ko = kl_div_topk(sx, idx, logp, T)
            (ko * w).sum().backward()
            check("2 top-k (k=8 + rest)", rel(ko, kr) <= 1e-5 and rel(sx.grad, sr.grad) <= tol_g,
                  f"loss {rel(ko, kr):.1e}  ds {rel(sx.grad, sr.grad):.1e}")
            if V <= 2048:                                                # k = V: no rest mass -> the dense KL
                logp, idx = lp_full.topk(V, -1)
                sa = s.detach().clone().requires_grad_()
                ka = kl_div_topk(sa, idx, logp, T)
                (ka * w).sum().backward()
                sb = s.detach().clone().requires_grad_()
                kb = kl_div_dense(sb, t, T)
                (kb * w).sum().backward()
                check("3 top-k (k=V) == dense", rel(ka, kb) <= 1e-4 and rel(sa.grad, sb.grad) <= max(tol_g, 1e-4),
                      f"loss {rel(ka, kb):.1e}  ds {rel(sa.grad, sb.grad):.1e}")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
