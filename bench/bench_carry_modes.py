"""All carry parameterisations vs torch.compile, at the board's real shape.

    python -m bench.bench_carry_modes

    h = attn_read + f(theta) * g(stream)

    f in {raw, 2*sigmoid, 2*tanh}      a function of theta -- (H,) values
    g in {identity, /rms}              a function of the stream -- (T,H) values

The point of the sweep: f is per-CHANNEL and cheap, g is a per-ROW reduction and is not. If the
kernel is built right, changing f should cost nothing measurable and only g should move the
number. The previous kernel evaluated f INSIDE the loop in fp64 and made sigmoid look expensive,
which is the thing being corrected.
"""
import torch

from kernels.sm120.residual_add import make_mlp_input

DEV = "cuda"
T, H = 65536, 512
RMS_EPS = 1e-6
F = {"raw": lambda t: t,
     "sigmoid": lambda t: 2.0 * torch.sigmoid(t),
     "tanh": lambda t: 2.0 * torch.tanh(t)}


def eager(attn_read, theta, stream, fname, norm):
    sv = stream
    if norm:
        s32 = sv.float()
        sv = (s32 * torch.rsqrt(s32.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(stream.dtype)
    return attn_read + F[fname](theta).to(sv.dtype) * sv


def _time(fn, mk, backward, iters=30, warmup=8):
    def one():
        a, t, s = mk(backward)
        out = fn(a, t, s)
        if backward:
            out.backward(torch.ones_like(out))
    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(True), torch.cuda.Event(True)
    e0.record()
    for _ in range(iters):
        one()
    e1.record()
    torch.cuda.synchronize()
    return e0.elapsed_time(e1) / iters


def main():
    torch.manual_seed(0)
    ar = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    st = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    th = torch.randn(H, device=DEV, dtype=torch.float32) * 0.3

    def mk(bw):
        return (ar.detach().requires_grad_(bw), th.detach().requires_grad_(bw),
                st.detach().requires_grad_(bw))

    print(f"T={T} H={H} bf16, per-dim c\n")
    print(f"  {'f(theta)':<10}{'stream':<8}{'compile fwd+bwd':>18}{'kernel fwd+bwd':>17}"
          f"{'speedup':>10}")
    for fname in ("raw", "sigmoid", "tanh"):
        for norm in (False, True):
            eag = lambda a, t, s, _f=fname, _n=norm: eager(a, t, s, _f, _n)
            comp = torch.compile(eag, mode="max-autotune", fullgraph=True)
            ker = (lambda a, t, s, _f=fname, _n=norm:
                   make_mlp_input(a, F[_f](t), s, modes=("rms" if _n else "none",)))
            try:
                c = _time(comp, mk, True)
                k = _time(ker, mk, True)
                print(f"  {fname:<10}{'rms' if norm else 'plain':<8}{c:>18.3f}{k:>17.3f}"
                      f"{c / k:>9.2f}x")
            except Exception as e:
                print(f"  {fname:<10}{'rms' if norm else 'plain':<8}FAILED "
                      f"{type(e).__name__}: {str(e)[:40]}")


if __name__ == "__main__":
    main()
