"""Grade residual_add against FP64 TRUTH. The kernel must be at least as close as eager, always.

This replaces the bit-identity gate. Bit-identity meant reproducing eager's precision loss, which
encodes autocast's rounding as the specification -- an fp32 run has no bf16 rounding anywhere, so a
kernel tuned to bf16-eager is tuned to an artifact and can be silently wrong at an untested dtype.

Truth is the same formula evaluated in float64 from float64 copies of the inputs. Both the kernel
and eager are scored by relative error against it, forward and backward, and the kernel FAILS if
it is worse than eager on any measured quantity in any layout.

    python -m parity_check.grade_residual_add
"""
import itertools

import torch

from . import _paths  # noqa: F401
from kernels.sm75.residual_add import make_mlp_input

MODES = {"none": lambda t: t,
         "sigmoid": torch.sigmoid,
         "tanh": torch.tanh,
         "2sigmoid": lambda t: 2.0 * torch.sigmoid(t),
         "2tanh": lambda t: 2.0 * torch.tanh(t)}


def _eager(ar, thetas, strms, modes, out_dt):
    """The torch spelling the model actually runs: scalar cast to the stream dtype, product formed
    in the stream dtype, accumulated into whatever the running sum's dtype is."""
    h = ar
    for th, s, m in zip(thetas, strms, modes):
        c = MODES[m](th.float())
        h = h + (c.to(s.dtype) * s).to(torch.promote_types(h.dtype, s.dtype))
    return h.to(out_dt)


def _truth(ar, thetas, strms, modes):
    h = ar.double()
    for th, s, m in zip(thetas, strms, modes):
        h = h + MODES[m](th.double()) * s.double()
    return h


def _relerr(x, truth):
    num = (x.double() - truth).abs().max().item()
    den = truth.abs().max().item()
    return num / max(den, 1e-300)


def _case(ar_dt, s_dts, modes, T=512, H=512, seed=0, device="cuda"):
    torch.manual_seed(seed)
    ar = torch.randn(T, H, device=device, dtype=ar_dt)
    strms = [torch.randn(T, H, device=device, dtype=d) for d in s_dts]
    thetas = [torch.full((1,), v, device=device, dtype=torch.float32)
              for v in (0.6, -0.4, 1.3, 0.2)[:len(s_dts)]]
    out_dt = ar_dt
    for d in s_dts:
        out_dt = torch.promote_types(out_dt, d)

    # ---- forward
    k = make_mlp_input(ar, *itertools.chain(*zip(thetas, strms)), modes=tuple(modes))
    e = _eager(ar, thetas, strms, modes, out_dt)
    t = _truth(ar, thetas, strms, modes)
    fwd = (_relerr(k, t), _relerr(e, t))

    # ---- backward. Same upstream gradient for every path, and a non-uniform one so d_theta is a
    # real reduction rather than a constant times a sum.
    torch.manual_seed(seed + 1)
    w = torch.randn(T, H, device=device, dtype=torch.float32)

    def grads(fn):
        a = ar.clone().requires_grad_(True)
        ss = [s.clone().requires_grad_(True) for s in strms]
        th = [t_.clone().requires_grad_(True) for t_ in thetas]
        out = fn(a, th, ss)
        (out.float() * w).sum().backward()
        return a.grad, [s.grad for s in ss], [t_.grad for t_ in th]

    gk = grads(lambda a, th, ss: make_mlp_input(a, *itertools.chain(*zip(th, ss)),
                                                modes=tuple(modes)))
    ge = grads(lambda a, th, ss: _eager(a, th, ss, modes, out_dt))
    # truth for the backward: closed form. d ar = w ; d s_k = c_k * w ; d th_k = dc_k * sum(w*s_k)
    wd = w.double()
    t_dar = wd
    t_ds, t_dth = [], []
    for th, s, m in zip(thetas, strms, modes):
        td = th.double().reshape(()).clone().requires_grad_(True)
        c = MODES[m](td)
        t_ds.append((c.detach() * wd))
        (c * (wd * s.double()).sum()).backward()
        t_dth.append(td.grad)

    bwd = {
        "d_ar": (_relerr(gk[0], t_dar), _relerr(ge[0], t_dar)),
        "d_stream": (max(_relerr(a, b) for a, b in zip(gk[1], t_ds)),
                     max(_relerr(a, b) for a, b in zip(ge[1], t_ds))),
        "d_theta": (max(_relerr(a, b) for a, b in zip(gk[2], t_dth)),
                    max(_relerr(a, b) for a, b in zip(ge[2], t_dth))),
    }
    return fwd, bwd


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    bf, f32 = torch.bfloat16, torch.float32
    cases = [
        ("1s bf16 / ar fp32   (the carry path)", f32, [bf], ["none"]),
        ("1s fp32 / ar fp32   (fp32 training)",  f32, [f32], ["none"]),
        ("1s bf16 / ar bf16",                    bf,  [bf], ["none"]),
        ("1s fp32 / ar bf16   (mixed)",          bf,  [f32], ["none"]),
        ("2s bf16+fp32 / ar fp32 (carry+emb)",   f32, [bf, f32], ["none", "none"]),
        ("2s bf16+bf16 / ar fp32",               f32, [bf, bf], ["none", "none"]),
        ("2s bounded 2sigmoid+2tanh",            f32, [bf, f32], ["2sigmoid", "2tanh"]),
        ("1s fp32 bounded sigmoid",              f32, [f32], ["sigmoid"]),
    ]
    print(f"{'case':40s} {'quantity':10s} {'kernel':>10s} {'eager':>10s} {'ratio':>8s}")
    worst = 0.0
    fails = []
    for name, ar_dt, s_dts, modes in cases:
        fwd, bwd = _case(ar_dt, s_dts, modes)
        rows = [("forward", fwd)] + list(bwd.items())
        for q, (ke, ee) in rows:
            ratio = ke / ee if ee > 0 else (0.0 if ke == 0 else float("inf"))
            worst = max(worst, ratio)
            flag = ""
            if ke > ee:
                flag = "  <-- WORSE THAN EAGER"
                fails.append((name, q, ke, ee))
            print(f"{name:40s} {q:10s} {ke:10.3e} {ee:10.3e} {ratio:8.2f}{flag}")
    print()
    if fails:
        print(f"FAIL: kernel is worse than eager on {len(fails)} measurement(s)")
        for n, q, ke, ee in fails:
            print(f"   {n} / {q}: kernel {ke:.3e} vs eager {ee:.3e}")
        raise SystemExit(1)
    print(f"PASS: kernel <= eager against fp64 on every quantity "
          f"(worst kernel/eager ratio {worst:.3f})")


if __name__ == "__main__":
    main()
