"""ce_factored vs an fp64 reference across logit margins 0..45 (Softmax Collapse, arXiv 2610.31642).

Rows are built so the label's logit leads the rest by a chosen margin m. The reference takes the SAME bf16 logits
the kernel sees (L = bf16(h W^T)) and does the softmax in fp64, so the test isolates the softmax / cross-entropy
math from the GEMM rounding. Per margin bucket it reports the relative error of the row's hidden gradient gh and of
the loss, and the sign agreement of the label's own coefficient (gh . W[label] vs the reference).

    python -m parity_check.parity_ce_softmax_fix            # GPU
    TRITON_INTERPRET=1 python -m parity_check.parity_ce_softmax_fix --cpu   # small shapes, Triton interpreter
"""
import argparse
import torch

from kernels.sm120 import ce_factored as cf


def make(n, v, h, dev, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    W = torch.randn(v, h, generator=g) / h ** 0.5
    lab = torch.randint(0, v, (n,), generator=g)
    alpha = torch.linspace(0.0, 60.0, n)                       # label logit, others stay O(1)
    x = torch.randn(n, h, generator=g) * 0.5 / h ** 0.5
    wl = W[lab]
    x = x + alpha[:, None] * wl / (wl * wl).sum(1, keepdim=True)
    return x.to(dev, torch.bfloat16), W.to(dev, torch.bfloat16), lab.to(dev)


def reference(x, W, lab):
    L = (x.float() @ W.float().t()).to(torch.bfloat16).double()     # the bf16 logits every path sees
    n = L.shape[0]
    lse = torch.logsumexp(L, 1)
    others = L.clone()
    others[torch.arange(n), lab] = -float("inf")
    m = L[torch.arange(n), lab] - torch.logsumexp(others, 1)
    nll = torch.nn.functional.softplus(-m)
    q = torch.exp(L - lse[:, None])
    d = q.clone()
    d[torch.arange(n), lab] = -torch.exp(torch.logsumexp(others, 1) - lse)    # exact q_t - 1
    d = d / n
    gh = d @ W.double()
    return nll, gh, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    a = ap.parse_args()
    dev = "cpu" if a.cpu else "cuda"
    if a.cpu and "out_dtype" not in (torch.mm.__doc__ or ""):        # old torch: fp32-accumulate shim
        mm, addmm = torch.mm, torch.addmm
        torch.mm = lambda x, y, out_dtype=None: mm(x.float(), y.float())
        torch.addmm = lambda c, x, y, alpha=1.0, out_dtype=None, out=None: out.copy_(
            addmm(c, x.float(), y.float(), alpha=alpha))
    n, v, h = (256, 384, 64) if a.cpu else (4096, 81920, 512)
    x, W, lab = make(n, v, h, dev)
    xr, Wr = x.clone().requires_grad_(), W.clone().requires_grad_()
    loss = cf.fused_linear_cross_entropy(xr, Wr, lab)
    loss.backward()
    nll_ref, gh_ref, m = reference(x.cpu(), W.cpu(), lab.cpu())
    gh = xr.grad.double().cpu()
    Wd = W.double().cpu()
    lab_c = lab.cpu()
    coef = (gh * Wd[lab_c]).sum(1)
    coef_ref = (gh_ref * Wd[lab_c]).sum(1)
    rel = (gh - gh_ref).norm(dim=1) / gh_ref.norm(dim=1).clamp_min(1e-300)
    print(f"loss {loss.item():.6e} ref {nll_ref.mean().item():.6e}")
    print(f"{'margin':>10} {'rows':>5} {'gh rel err (max)':>17} {'label coef sign ok':>19} {'loss=0 rows':>12}")
    ok = True
    for lo, hi in [(0, 4), (4, 6), (6, 8), (8, 12), (12, 16.6), (16.6, 25), (25, 45)]:
        sel = (m >= lo) & (m < hi)
        if not sel.any():
            continue
        sign = (torch.sign(coef[sel]) == torch.sign(coef_ref[sel])).double().mean().item()
        err = rel[sel].max().item()
        print(f"{lo:>4}-{hi:<5} {int(sel.sum()):>5} {err:>17.3e} {sign:>19.2%} {'-':>12}")
        ok &= sign == 1.0 and err < 5e-2
    print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
