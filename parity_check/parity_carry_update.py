"""carry_update (fused h = read + c*f(ao) AND ps + ao) must be BITWISE the unfused production path:
c = 2*sigmoid(theta) on the host, make_mlp_input for h, a separate ps + ao, autograd summing the
two attn_out grads. Every output and every grad, max|fused - unfused| == 0. Then fwd+bwd timing.

    python parity_check/parity_carry_update.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kernels.sm120.residual_add import make_mlp_input, carry_update

dev, bf = "cuda", torch.bfloat16


def unfused(read, theta, ao, ps, mode, csig):
    c = 2.0 * torch.sigmoid(theta) if csig else theta
    h = make_mlp_input(read, c, ao, modes=(mode,))
    return h, (ao if ps is None else ps + ao)


def ref64(read, theta, ao, ps, mode, csig):
    """The formula in fp64 -- ground truth for outputs AND grads (autograd in fp64)."""
    c = 2.0 * torch.sigmoid(theta) if csig else theta
    s_ = ao * torch.rsqrt(ao.pow(2).mean(-1, keepdim=True) + 1e-6) if mode == "rms" else ao
    return read + c * s_, (ao if ps is None else ps + ao)


def run(fn, T, H, vec, boundary, mode, csig, seed=0, dtype=None):
    g = torch.Generator(device=dev).manual_seed(seed)
    read = (torch.randn(T, H, device=dev, generator=g) * 4).to(bf).requires_grad_()
    ao = torch.randn(T, H, device=dev, generator=g).to(bf).requires_grad_()
    ps = None if boundary else (torch.randn(T, H, device=dev, generator=g) * 10).to(bf).requires_grad_()
    theta = (torch.randn(H if vec else 1, device=dev, generator=g) * 0.7).requires_grad_()
    gh = torch.randn(T, H, device=dev, generator=g).to(bf)
    gp = torch.randn(T, H, device=dev, generator=g).to(bf)
    if dtype is not None:                   # fp64 reference: same (bf16-valued) inputs, upcast
        read, ao, gh, gp = (t.detach().to(dtype) for t in (read, ao, gh, gp))
        ps = None if ps is None else ps.detach().to(dtype)
        theta = theta.detach().to(dtype)
        for t in (read, ao, theta) + (() if ps is None else (ps,)):
            t.requires_grad_()
    h, pn = fn(read, theta, ao, ps, mode, csig)
    torch.autograd.backward([h, pn], [gh, gp])
    outs = {"h": h, "ps_new": pn, "d_read": read.grad, "d_ao": ao.grad, "d_theta": theta.grad}
    if ps is not None:
        outs["d_ps"] = ps.grad
    return {k: v.detach().clone() for k, v in outs.items()}


def main():
    ok = True
    for T in (65536, 1000):
        for vec in (True, False):
            for boundary in (False, True):
                for mode in ("none", "rms"):
                    for csig in (True, False):
                        a = run(unfused, T, 512, vec, boundary, mode, csig)
                        b = run(carry_update, T, 512, vec, boundary, mode, csig)
                        bad = {k: (a[k].double() - b[k].double()).abs().max().item() for k in a
                               if not torch.equal(a[k], b[k])}
                        ok &= not bad
                        # and against fp64 truth, every output and every grad
                        r = run(ref64, T, 512, vec, boundary, mode, csig, dtype=torch.float64)
                        errs = {k: (b[k].double() - r[k]).norm().item() / (r[k].norm().item() or 1.0)
                                for k in r}
                        worst = max(errs.values())
                        print(f"T={T:5d} vec={vec:d} boundary={boundary:d} mode={mode:4s} csig={csig:d}: "
                              + ("bitwise OK" if not bad else f"DIFF {bad}")
                              + "  | vs fp64: " + " ".join(f"{k} {v:.1e}" for k, v in errs.items()))
                        ok &= worst < 1e-2              # bf16 outputs/grads: ~4e-3 is the floor
    print("CARRY PARITY PASS" if ok else "CARRY PARITY FAIL")

    T, H = 65536, 512
    read = torch.randn(T, H, device=dev).to(bf).requires_grad_()
    ao = torch.randn(T, H, device=dev).to(bf).requires_grad_()
    ps = torch.randn(T, H, device=dev).to(bf).requires_grad_()
    theta = torch.zeros(H, device=dev).requires_grad_()
    gh, gp = torch.randn(T, H, device=dev).to(bf), torch.randn(T, H, device=dev).to(bf)
    for name, fn in (("unfused (production)", unfused), ("fused carry_update", carry_update)):
        def step():
            read.grad = ao.grad = ps.grad = theta.grad = None
            torch.autograd.backward(list(fn(read, theta, ao, ps, "none", True)), [gh, gp])
        for _ in range(5):
            step()
        torch.cuda.synchronize()
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(50):
            step()
        e1.record()
        torch.cuda.synchronize()
        print(f"   {name:22s} fwd+bwd {e0.elapsed_time(e1) / 50:.3f} ms  (T=65536 H=512 bf16, sigmoid per-dim c)")


if __name__ == "__main__":
    main()
