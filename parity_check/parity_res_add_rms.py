"""Grade the "rms" carry mode of make_mlp_input against eager autograd, in fp64 truth.

    python -m parity_check.parity_res_add_rms

    h = attn_read + c * stream / rms(stream)

"rms" replaces the bounded transforms (sigmoid/tanh/2sigmoid/2tanh). Those existed to stop an
unbounded c running away -- the first carry attempt reached 7936 by step 400. Normalising the
STREAM removes the reason for a cage instead of building a better one: c now multiplies a
unit-RMS quantity.

Three things are graded, all against an fp64 eager reference:
  forward, d_attn_read, d_stream, d_theta      kernel must be NO WORSE than fp32 eager

d_stream is the one that matters. y = s/r is a projection, not a rescale: its gradient carries a
`- sn * mean(g*sn)` term, and dropping that is the classic RMSNorm-backward bug -- it produces a
gradient that is close, never NaN, and shows up only as a slow quality drift nothing attributes
back here.
"""
import torch

from kernels.sm120.residual_add import make_mlp_input

DEV = "cuda"
T, H = 4096, 512
RMS_EPS = 1e-6


def eager(attn_read, theta, stream, mode, dt):
    """The eager spelling. At bf16 the result is STORED in bf16, exactly as the model would --
    comparing an fp32 eager output against a bf16 kernel output charges the kernel for a rounding
    step eager also takes, and makes every `out` row read WORSE THAN EAGER for free."""
    ar, sv = attn_read.to(dt), stream.to(dt)
    th = theta.to(dt)                        # eager casts c to the stream dtype -- see exp's else-branch
    if mode == "rms":
        sv = sv * torch.rsqrt(sv.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    c = th.reshape(()) if th.numel() == 1 else th
    out = ar + c * sv
    return out if dt == torch.float64 else out.to(torch.bfloat16)


def _run(fn, dt, per_dim, seed=0):
    torch.manual_seed(seed)
    ar = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    st = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    th = (torch.randn(H if per_dim else 1, device=DEV, dtype=torch.float32) * 0.3 + 1.0)
    g = torch.randn(T, H, device=DEV, dtype=torch.bfloat16) * 0.01

    a = ar.to(dt).detach().requires_grad_(True)
    s = st.to(dt).detach().requires_grad_(True)
    t = th.to(torch.float64 if dt == torch.float64 else torch.float32).detach().requires_grad_(True)
    out = fn(a, t, s)
    out.backward(g.to(out.dtype))
    return out.detach(), a.grad.detach(), s.grad.detach(), t.grad.detach()


def grade(per_dim, mode):
    ref = _run(lambda a, t, s: eager(a, t, s, mode, torch.float64), torch.float64, per_dim)
    # bf16, NOT fp32. The model's eager spelling is `attn_read + c.to(ao.dtype) * ao` on bf16
    # tensors; grading against an fp32 eager compares the kernel to a precision nothing runs and
    # made the kernel look 19x WORSE on scalar d_theta when it is in fact ~38,000x better
    # (eager 22.5 vs fp64 truth 22.4227, kernel 22.4227390).
    eag = _run(lambda a, t, s: eager(a, t, s, mode, torch.bfloat16), torch.bfloat16, per_dim)
    ker = _run(lambda a, t, s: make_mlp_input(a, t, s, modes=(mode,)), torch.bfloat16, per_dim)

    print(f"\n  mode={mode!r}  theta={'per-dim (H,)' if per_dim else 'scalar'}")
    print(f"    {'tensor':<14}{'eager vs fp64':>16}{'kernel vs fp64':>17}   verdict")
    ok = True
    for name, r, e, k in zip(("out", "d_attn_read", "d_stream", "d_theta"), ref, eag, ker):
        de = (e.double() - r.double()).abs().max().item()
        dk = (k.double() - r.double()).abs().max().item()
        # the contract is "at least as close as eager", with a small slack for bf16 tie-breaking
        good = dk <= de * 1.05 + 1e-9
        ok &= good
        print(f"    {name:<14}{de:>16.3e}{dk:>17.3e}   {'OK' if good else 'WORSE THAN EAGER'}")
    return ok


def main():
    ok = True
    for mode in ("none", "rms"):
        for per_dim in (False, True):
            ok &= grade(per_dim, mode)
    # "rms" must actually normalise: a stream scaled by 10x must give the SAME output under rms
    # (up to the eps) and a different one under "none". Without this the mode could be silently
    # inert and every error column above would still read clean.
    torch.manual_seed(1)
    ar = torch.zeros(T, H, device=DEV, dtype=torch.bfloat16)
    st = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    th = torch.ones(H, device=DEV, dtype=torch.float32)
    a = make_mlp_input(ar, th, st, modes=("rms",))
    b = make_mlp_input(ar, th, st * 10.0, modes=("rms",))
    c = make_mlp_input(ar, th, st * 10.0, modes=("none",))
    scale_inv = (a.double() - b.double()).abs().max().item()
    differs = (a.double() - c.double()).abs().max().item()
    print(f"\n  scale-invariance: |rms(s) - rms(10s)| = {scale_inv:.3e}  (must be ~0)")
    print(f"  mode is live    : |rms(10s) - none(10s)| = {differs:.3e}  (must be >> 0)")
    ok &= scale_inv < 5e-2 and differs > 1.0
    print("\n" + ("PASS -- rms mode matches fp64 eager and is genuinely normalising"
                  if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
