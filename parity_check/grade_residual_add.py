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

    # ---- backward. The upstream gradient is handed in DIRECTLY, already in the output dtype, so
    # both paths see bit-identical dout and truth can be computed from that same tensor. Going
    # through `(out.float()*w).sum()` instead would let autograd quantize w to the output dtype
    # on its way in, while truth still used the fp32 w -- a common error injected into both sides
    # that swamps the thing being measured. It made the all-bf16 d_theta look 1.34x worse than
    # eager when the real cause was the reference, not the kernel.
    torch.manual_seed(seed + 1)
    w = torch.randn(T, H, device=device, dtype=out_dt)

    def grads(fn):
        a = ar.clone().requires_grad_(True)
        ss = [s.clone().requires_grad_(True) for s in strms]
        th = [t_.clone().requires_grad_(True) for t_ in thetas]
        fn(a, th, ss).backward(gradient=w)
        return a.grad, [s.grad for s in ss], [t_.grad for t_ in th]

    gk = grads(lambda a, th, ss: make_mlp_input(a, *itertools.chain(*zip(th, ss)),
                                                modes=tuple(modes)))
    ge = grads(lambda a, th, ss: _eager(a, th, ss, modes, out_dt))
    # truth for the backward: closed form. d ar = w ; d s_k = c_k * w ; d th_k = dc_k * sum(w*s_k)
    wd = w.double()   # the SAME tensor both paths received
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


_SHORT = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}
_MODE_CYCLE = ["none", "2sigmoid", "tanh", "sigmoid", "2tanh"]


def _all_cases():
    """EXHAUSTIVE over dtype assignments -- attn_read and every stream vary independently.

    A single hand-picked pairing per stream count is how the last contract shipped a kernel that
    was exact on the one layout anyone tested and 1 ULP off on the neighbouring one. There is no
    reason to guess which combination breaks: enumerate them.

    {bf16, fp32, fp16} exhaustively for K=1,2 (3^2 + 3^3 = 36 configs) and {bf16, fp32} for
    K=3,4 (2^4 + 2^5 = 48), so every (attn_read, stream_0..k) assignment the model could ever
    produce is measured. Modes cycle so all five transforms are exercised across the sweep.
    """
    bf, f32, f16 = torch.bfloat16, torch.float32, torch.float16
    for pool, ks in (((bf, f32, f16), (1, 2)), ((bf, f32), (3, 4))):
        for k in ks:
            for combo in itertools.product(pool, repeat=k + 1):
                ar_dt, s_dts = combo[0], list(combo[1:])
                modes = [_MODE_CYCLE[(i + len(s_dts)) % len(_MODE_CYCLE)] for i in range(k)]
                name = f"K{k} ar={_SHORT[ar_dt]} s=" + "+".join(_SHORT[d] for d in s_dts)
                yield name, ar_dt, s_dts, modes


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    cases = list(_all_cases())
    print(f"grading {len(cases)} dtype configurations x 4 quantities "
          f"= {len(cases) * 4} measurements against fp64\n")
    print(f"{'case':34s} {'quantity':10s} {'kernel':>10s} {'eager':>10s} {'ratio':>8s}")
    worst, worst_name = 0.0, ""
    fails, n_meas = [], 0
    for name, ar_dt, s_dts, modes in cases:
        fwd, bwd = _case(ar_dt, s_dts, modes)
        for q, (ke, ee) in [("forward", fwd)] + list(bwd.items()):
            n_meas += 1
            ratio = ke / ee if ee > 0 else (0.0 if ke == 0 else float("inf"))
            if ratio > worst:
                worst, worst_name = ratio, f"{name}/{q}"
            if ke > ee:
                fails.append((name, q, ke, ee))
                print(f"{name:34s} {q:10s} {ke:10.3e} {ee:10.3e} {ratio:8.2f}"
                      f"  <-- WORSE THAN EAGER")
    print(f"\n{n_meas} measurements over {len(cases)} configs")
    if fails:
        print(f"FAIL: kernel worse than eager on {len(fails)}:")
        for n, q, ke, ee in fails:
            print(f"   {n} / {q}: kernel {ke:.3e} vs eager {ee:.3e}")
        raise SystemExit(1)
    print(f"PASS: kernel <= eager on all {n_meas}; worst ratio {worst:.4f} ({worst_name})")


if __name__ == "__main__":
    main()
