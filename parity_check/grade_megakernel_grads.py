"""Grade BACKWARD parity: fp64 ground truth vs today's stack vs the megakernel.

    python -m parity_check.grade_megakernel_grads

The forward graders and the frozen eval only ever compared forward quantities. That is the wrong
thing to skip when the candidate ships a hand-written backward: a wrong gradient is invisible in the
forward error AND in the timing, and would surface only as a model that trains subtly worse.

Compares d_x, d_norm_weight and d_router_weight against an fp64 reference, for:
    eager (liger norm + eager router)   what the model runs today
    megakernel                          analytic router grads + Triton rmsnorm backward

Separate from bench/eval_mlp_block.py on purpose -- that eval is frozen.
"""
import torch

from kernels.sm120.megakernel.moe.block import _NormRouter

DEV = "cuda"
H, E, K, EPS = 512, 64, 6, 1e-6


def eager_fwd(x, nw, rw, bias, dtype=None):
    """BiBoRMSNorm + BiBoMoERouter, differentiable. dtype=float64 gives the ground truth."""
    dt = dtype or x.dtype
    f = x.to(torch.float32) if dt != torch.float64 else x.to(torch.float64)
    hn = ((f * torch.rsqrt(f.pow(2).mean(-1, keepdim=True) + EPS)) * nw).to(dt)
    logits = hn @ rw
    scores = torch.sigmoid(logits.float() if dt != torch.float64 else logits)
    _, idx = torch.topk(scores + bias, K, dim=-1, sorted=False)
    idx, _ = torch.sort(idx, dim=-1)
    w = scores.gather(-1, idx)
    return hn, idx, w / (w.sum(-1, keepdim=True) + 1e-20)


def grade_fp32_master(T=1024, seed=0):
    """The REAL model layout: bf16 activations, fp32 master weights, narrowed by autocast.

    The bf16-weight grade below cannot see this case, and that gap shipped a backward that passed
    every forward AND backward check here and then died on the first `loss.backward()` of a training
    run -- `dl @ rw.t()` with a bf16 dl and an fp32 rw. A kernel is only graded once it is graded on
    the dtype layout the model actually has.
    """
    torch.manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
    nw = (torch.randn(H, device=DEV, dtype=torch.float32) * 0.1 + 1.0).requires_grad_(True)
    rw = (torch.randn(H, E, device=DEV, dtype=torch.float32) * 0.02).requires_grad_(True)
    bias = torch.randn(E, device=DEV, dtype=torch.float32) * 0.05
    with torch.autocast("cuda", torch.bfloat16):
        hn, idx, w, _gap = _NormRouter.apply(x, nw, rw, bias, K, EPS)
    (hn.float().sum() + w.sum()).backward()
    for name, t, p in (("d_x", x.grad, x), ("d_norm_weight", nw.grad, nw),
                       ("d_router_weight", rw.grad, rw)):
        assert t is not None, f"{name} is None -- no gradient reached the parameter"
        assert t.dtype == p.dtype, f"{name} dtype {t.dtype} != parameter dtype {p.dtype}"
        assert torch.isfinite(t).all(), f"{name} has non-finite entries"
    print(f"  fp32-master layout (bf16 acts, fp32 weights, autocast): backward OK, "
          f"dtypes match, all finite  [T={T}]")


def grade(T=4096, seed=0):
    torch.manual_seed(seed)
    x0 = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    nw0 = torch.randn(H, device=DEV, dtype=torch.float32) * 0.1 + 1.0
    rw0 = torch.randn(H, E, device=DEV, dtype=torch.bfloat16) * 0.02
    bias = torch.randn(E, device=DEV, dtype=torch.float32) * 0.05
    # fixed upstream gradients, so every contender is differentiated at the SAME point
    g_hn = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.01
    g_w = torch.randn(T, K, device=DEV, dtype=torch.float32) * 0.01

    def run(fn, dt):
        x = x0.to(dt).detach().requires_grad_(True)
        nw = nw0.to(torch.float64 if dt == torch.float64 else torch.float32).detach().requires_grad_(True)
        rw = rw0.to(dt).detach().requires_grad_(True)
        hn, idx, w = fn(x, nw, rw, bias.to(nw.dtype))
        # rw gets NO gradient from the hn branch -- only the router path touches it, so .grad is
        # None there rather than zero. Both branches are accumulated because the real block feeds
        # gradients back through hn AND through the routing weights.
        z = lambda t, like: torch.zeros_like(like) if t is None else t.clone()
        (hn * g_hn.to(hn.dtype)).sum().backward(retain_graph=True)
        g1 = (z(x.grad, x), z(nw.grad, nw), z(rw.grad, rw))
        x.grad = nw.grad = rw.grad = None
        (w * g_w.to(w.dtype)).sum().backward()
        g2 = (z(x.grad, x), z(nw.grad, nw), z(rw.grad, rw))
        return tuple(a + b for a, b in zip(g1, g2))

    ref = run(lambda x, nw, rw, b: eager_fwd(x, nw, rw, b, torch.float64), torch.float64)
    eag = run(lambda x, nw, rw, b: eager_fwd(x, nw, rw, b), torch.bfloat16)
    mk = run(lambda x, nw, rw, b: _NormRouter.apply(x, nw, rw, b, K, EPS)[:3], torch.bfloat16)

    print(f"T={T} H={H} E={E} K={K}   ground truth = fp64 eager\n")
    print(f"  {'gradient':<18}{'eager bf16 max':>18}{'megakernel max':>18}   verdict")
    for name, i in (("d_x", 0), ("d_norm_weight", 1), ("d_router_weight", 2)):
        a = (eag[i].double() - ref[i].double()).abs().max().item()
        b = (mk[i].double() - ref[i].double()).abs().max().item()
        scale = ref[i].double().abs().max().item()
        v = "kernel closer" if b < a else ("eager closer" if b > a else "tie")
        print(f"  {name:<18}{a:>18.3e}{b:>18.3e}   {v}   (|ref|max {scale:.2e})")

    # a hand-written backward that is merely CLOSE is not enough -- catch a wrong FORM, e.g. a
    # dropped d(rstd)/dx term, which shows as a large relative error rather than a small absolute one
    for name, i in (("d_x", 0), ("d_norm_weight", 1), ("d_router_weight", 2)):
        rel = ((mk[i].double() - ref[i].double()).norm() / ref[i].double().norm()).item()
        flag = "OK" if rel < 0.05 else "SUSPECT -- wrong form, not rounding"
        print(f"  megakernel {name:<16} relative L2 error {rel:.4f}   {flag}")


if __name__ == "__main__":
    grade()
    print()
    grade_fp32_master()
