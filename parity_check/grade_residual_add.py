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


def _f64(t, what):
    """Ground truth must be FLOAT64, and nothing else. `.float()` is float32 -- using it here
    would quietly grade an fp32 kernel against an fp32 reference and call the agreement accuracy.
    Asserted rather than trusted, because it is invisible when wrong."""
    assert t.dtype == torch.float64, f"{what} is {t.dtype}, must be float64"
    return t


def _truth(ar, thetas, strms, modes):
    h = _f64(ar.double(), "truth attn_read")
    for th, s, m in zip(thetas, strms, modes):
        h = h + _f64(MODES[m](th.double()), "truth theta") * _f64(s.double(), "truth stream")
    return _f64(h, "truth forward")


def _relerr(x, truth):
    """(mean, max) relative error. BOTH, because they answer different questions.

    MEAN is the accuracy: it is stable across seeds and it is what "this kernel is more accurate"
    actually means. MAX over 262k elements at the fp32 rounding floor is a tail statistic decided
    by which individual element happened to round badly -- measured over 5 seeds, the kernel beats
    eager on mean 15/15 times and on max only 10/15, while being 15-20% better on mean every time.
    Gating on max would therefore fail a strictly-better kernel on a coin flip.

    Max is still reported and still gated, but with slack, because a genuine defect (a bad
    transform, a broken tile) shows up as a large max regression, not a 1.02x one.
    """
    _f64(truth, "relerr reference")
    d = (x.double() - truth).abs()
    den = max(truth.abs().max().item(), 1e-300)
    return d.mean().item() / den, d.max().item() / den


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
    wd = _f64(w.double(), "truth dout")   # the SAME tensor both paths received
    t_dar = wd
    t_ds, t_dth = [], []
    for th, s, m in zip(thetas, strms, modes):
        td = th.double().reshape(()).clone().requires_grad_(True)
        c = MODES[m](td)
        t_ds.append(_f64(c.detach() * wd, "truth d_stream"))
        (c * (wd * _f64(s.double(), "truth stream")).sum()).backward()
        t_dth.append(_f64(td.grad, "truth d_theta"))

    bwd = {
        "d_ar": (_relerr(gk[0], t_dar), _relerr(ge[0], t_dar)),
        "d_stream": (max(_relerr(a, b) for a, b in zip(gk[1], t_ds)),
                     max(_relerr(a, b) for a, b in zip(ge[1], t_ds))),
        "d_theta": (max(_relerr(a, b) for a, b in zip(gk[2], t_dth)),
                    max(_relerr(a, b) for a, b in zip(ge[2], t_dth))),
    }
    return fwd, bwd


_SHORT = {torch.bfloat16: "bf16", torch.float32: "fp32", torch.float16: "fp16"}


def _all_cases():
    """EXHAUSTIVE over dtype assignments -- attn_read and every stream vary independently.

    A single hand-picked pairing per stream count is how the last contract shipped a kernel that
    was exact on the one layout anyone tested and 1 ULP off on the neighbouring one. There is no
    reason to guess which combination breaks: enumerate them.

    {bf16, fp32, fp16} exhaustively for K=1,2 (3^2 + 3^3 = 36 configs) and {bf16, fp32} for
    K=3,4 (2^4 + 2^5 = 48), so every (attn_read, stream_0..k) assignment the model could produce
    is measured.

    MODE IS FIXED AT "none". The bounded transforms (sigmoid/tanh/2sigmoid/2tanh) are not on any
    live arm -- every AttnRes run uses carry_scale=unbounded, and 2sigmoid bounding was refuted
    experimentally -- so grading them here only inflates the matrix. They are still exercised, but
    in _bounded_spot_check() below rather than crossed against 84 dtype configs: leaving a code
    path with zero coverage is how the 2*sigmoid(2x)-1 tanh survived in the first place.
    """
    bf, f32, f16 = torch.bfloat16, torch.float32, torch.float16
    for pool, ks in (((bf, f32, f16), (1, 2)), ((bf, f32), (3, 4))):
        for k in ks:
            for combo in itertools.product(pool, repeat=k + 1):
                ar_dt, s_dts = combo[0], list(combo[1:])
                name = f"K{k} ar={_SHORT[ar_dt]} s=" + "+".join(_SHORT[d] for d in s_dts)
                yield name, ar_dt, s_dts, ["none"] * k


def _bounded_spot_check():
    """One case per bounded mode, at the layout most likely to expose a bad transform (fp32
    everywhere, where the transform's own error is not hidden under bf16 quantization)."""
    f32 = torch.float32
    for m in ("sigmoid", "2sigmoid", "tanh", "2tanh"):
        yield f"bounded {m} (fp32/fp32)", f32, [f32], [m]


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    cases = list(_all_cases()) + list(_bounded_spot_check())
    print(f"grading {len(cases)} configurations x 4 quantities "
          f"= {len(cases) * 4} measurements against fp64 "
          f"({len(list(_all_cases()))} dtype configs at mode=none, 4 bounded spot checks)\n")
    MAX_SLACK = 2.0          # a real defect is not a 1.02x max regression
    worst_mean, worst_mean_name = 0.0, ""
    worst_max, worst_max_name = 0.0, ""
    fails, n_meas = [], 0
    for name, ar_dt, s_dts, modes in cases:
        fwd, bwd = _case(ar_dt, s_dts, modes)
        for q, ((k_mu, k_mx), (e_mu, e_mx)) in [("forward", fwd)] + list(bwd.items()):
            n_meas += 1
            r_mu = k_mu / e_mu if e_mu > 0 else (0.0 if k_mu == 0 else float("inf"))
            r_mx = k_mx / e_mx if e_mx > 0 else (0.0 if k_mx == 0 else float("inf"))
            if r_mu > worst_mean:
                worst_mean, worst_mean_name = r_mu, f"{name}/{q}"
            if r_mx > worst_max:
                worst_max, worst_max_name = r_mx, f"{name}/{q}"
            if r_mu > 1.0:
                fails.append((name, q, "MEAN", k_mu, e_mu, r_mu))
                print(f"{name:34s} {q:10s} MEAN {k_mu:10.3e} vs {e_mu:10.3e}  {r_mu:6.2f}x WORSE")
            if r_mx > MAX_SLACK:
                fails.append((name, q, "MAX", k_mx, e_mx, r_mx))
                print(f"{name:34s} {q:10s} MAX  {k_mx:10.3e} vs {e_mx:10.3e}  {r_mx:6.2f}x WORSE")
    print(f"\n{n_meas} measurements over {len(cases)} configs")
    if fails:
        print(f"FAIL on {len(fails)}:")
        for n, q, kind, k, e, r in fails:
            print(f"   {n} / {q} [{kind}]: kernel {k:.3e} vs eager {e:.3e}  ({r:.2f}x)")
        raise SystemExit(1)
    print(f"PASS: kernel mean-error <= eager on all {n_meas} measurements")
    print(f"  worst mean ratio {worst_mean:.4f} ({worst_mean_name})")
    print(f"  worst max  ratio {worst_max:.4f} ({worst_max_name})  [slack {MAX_SLACK}x]")


if __name__ == "__main__":
    main()
