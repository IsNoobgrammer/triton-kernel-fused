"""kl_div at the Softmax Cancellation edge (arXiv 2610.31642) vs fp64.

  1. dense, student == teacher: KL 0 and the gradient EXACTLY 0 (the old q - p left a noise floor)
  2. dense, student = teacher + 1e-6 noise: gradient vs fp64 autograd (relative L2)
  3. top-k, student puts all but ~1e-9 of its mass in the teacher's top-k while the teacher keeps rest r ~ 1e-4:
     the old 1 - sum q rounded to 0 (clamped 1e-30) and 1 - r / q_rest blew up; must be finite and match fp64

    python -m parity_check.parity_kl_softmax_fix
"""
import torch

from kernels.sm120.kl_div import kl_div_dense, kl_div_topk

dev = "cuda"
ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}")


def rel(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-300)).item()


def ref_dense(s, t):
    s = s.double().requires_grad_(True)
    p = torch.softmax(t.double(), -1)
    kl = (p * (torch.log_softmax(t.double(), -1) - torch.log_softmax(s, -1))).sum(-1)
    kl.sum().backward()
    return kl.detach(), s.grad


def ref_topk(s, idx, lp):
    s = s.double().requires_grad_(True)
    lq = torch.log_softmax(s, -1)
    lqk = lq.gather(-1, idx)
    p = lp.double().exp()
    r = (1 - p.sum(-1)).clamp_min(0)
    mask = torch.ones_like(lq, dtype=torch.bool).scatter_(-1, idx, False)
    lqr = torch.logsumexp(lq.masked_fill(~mask, -float("inf")), -1)
    kl = (p * (lp.double() - lqk)).sum(-1) + torch.where(r > 0, r * (r.clamp_min(1e-300).log() - lqr), 0)
    kl.sum().backward()
    return kl.detach(), s.grad


def main():
    g = torch.Generator(device=dev).manual_seed(0)
    N, V = 64, 4097
    t = torch.randn(N, V, device=dev, generator=g) * 3
    print("1. dense, student == teacher")
    s = t.clone().requires_grad_(True)
    kl = kl_div_dense(s, t)
    kl.sum().backward()
    check("1", s.grad.abs().max().item() == 0.0 and kl.abs().max().item() < 1e-5,
          f"max |grad| {s.grad.abs().max().item():.3e}  max |KL| {kl.abs().max().item():.3e}")
    print("2. dense, student = teacher + 1e-6")
    s0 = t + 1e-6 * torch.randn(N, V, device=dev, generator=g)
    s = s0.clone().requires_grad_(True)
    kl_div_dense(s, t).sum().backward()
    _, gr = ref_dense(s0, t)
    e = rel(s.grad, gr)
    check("2", e < 1e-2, f"grad rel err vs fp64 {e:.3e}")
    print("3. top-k, confident student, teacher rest ~1e-4")
    K = 8
    idx = torch.stack([torch.randperm(V, device=dev, generator=g)[:K] for _ in range(N)])
    lt = torch.full((N, V), -30.0, device=dev)
    lt.scatter_(-1, idx, torch.randn(N, K, device=dev, generator=g))
    lp_full = torch.log_softmax(lt, -1)
    lp = lp_full.gather(-1, idx) + torch.log1p(-torch.full((N, K), 1e-4, device=dev))   # teacher keeps r = 1e-4
    ss = torch.full((N, V), -25.0, device=dev) + 0.1 * torch.randn(N, V, device=dev, generator=g)
    ss.scatter_(-1, idx, torch.randn(N, K, device=dev, generator=g))                   # student: ~1e-8 outside
    s = ss.clone().requires_grad_(True)
    kl = kl_div_topk(s, idx, lp)
    kl.sum().backward()
    klr, gr = ref_topk(ss, idx, lp)
    fin = torch.isfinite(s.grad).all().item() and torch.isfinite(kl).all().item()
    check("3 finite", fin, f"max |grad| {s.grad.abs().max().item():.3e} (fp64 {gr.abs().max().item():.3e})")
    check("3 loss", rel(kl, klr) < 1e-4, f"loss rel err {rel(kl, klr):.3e}")
    check("3 grad", rel(s.grad, gr) < 1e-2, f"grad rel err {rel(s.grad, gr):.3e}")
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
