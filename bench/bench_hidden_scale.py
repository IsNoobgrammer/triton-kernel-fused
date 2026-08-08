"""Does the residual-add kernel still beat torch.compile as hidden scales to 7k?

    python -m bench.bench_hidden_scale

The kernel is a PERSISTENT reduction: RBLOCK = next_power_of_2(H), so a whole row lives in
registers and XBLOCK rows are in flight at once. That is free at H=512 (8K floats) and cannot
stay free forever -- at H=7168 one row alone is 8192 floats, and the autotuner has to retreat
to XBLOCK=1 or spill. This finds where that happens, if it happens.

T is held FIXED so H is the only thing moving. Work therefore grows with H; the speedup column
is the answer, the ms column is just how we got there. The chosen autotune config is printed
per row because "which XBLOCK survived" is the whole mechanism.

f is pinned to sigmoid -- the carry-mode sweep already showed f(theta) is free (it is (H,) of
work against (T,H) of traffic). The f-axis is re-checked at both ends only, to confirm that
still holds at 14x the row width and to settle the one tanh+rms outlier.
"""
import torch

from kernels.sm75 import residual_add as ra
from kernels.sm120.residual_add import make_mlp_input

DEV = "cuda"
T = 65536
HIDDENS = (512, 1024, 2048, 3072, 4096, 5120, 6144, 7168)
RMS_EPS = 1e-6
F = {"raw": lambda t: t,
     "sigmoid": lambda t: 2.0 * torch.sigmoid(t),
     "tanh": lambda t: 2.0 * torch.tanh(t)}

# fwd reads ar+st writes out; bwd reads go+st writes d_ar+d_st. 7 passes over (T,H) bf16.
BYTES_PER_ELEM = 7 * 2


def eager(attn_read, theta, stream, fname, norm):
    sv = stream
    if norm:
        s32 = sv.float()
        sv = (s32 * torch.rsqrt(s32.pow(2).mean(-1, keepdim=True) + RMS_EPS)).to(stream.dtype)
    return attn_read + F[fname](theta).to(sv.dtype) * sv


def _time(fn, mk, iters=30, warmup=8):
    def one():
        a, t, s = mk()
        out = fn(a, t, s)
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


def _cfg():
    """Which autotune config won. Empty until the kernel has actually been launched."""
    best = getattr(ra._fwd_kernel, "best_config", None) or {}
    got = getattr(best, "kwargs", {})
    if not got:
        cache = getattr(ra._fwd_kernel, "cache", {})
        vals = list(cache.values())
        if vals:
            best, got = vals[-1], getattr(vals[-1], "kwargs", {})
    return f"x{got.get('XBLOCK', '?')}/w{getattr(best, 'num_warps', '?')}"


def one_row(H, fname, norm, label=None):
    ar = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    st = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    th = torch.randn(H, device=DEV, dtype=torch.float32) * 0.3

    def mk():
        return (ar.detach().requires_grad_(True), th.detach().requires_grad_(True),
                st.detach().requires_grad_(True))

    stream = "rms" if norm else "plain"
    tag = label if label is not None else f"{H:<7}{stream:<8}"
    try:
        comp = torch.compile(lambda a, t, s: eager(a, t, s, fname, norm),
                             mode="max-autotune", fullgraph=True)
        c = _time(comp, mk)
        k = _time(lambda a, t, s: make_mlp_input(a, F[fname](t), s,
                                                 modes=("rms" if norm else "none",)), mk)
        gbs = T * H * BYTES_PER_ELEM / (k * 1e6)
        print(f"  {tag}{c:>14.3f}{k:>14.3f}{c / k:>9.2f}x{gbs:>11.0f}{_cfg():>12}", flush=True)
    except Exception as e:
        print(f"  {tag}FAILED {type(e).__name__}: {str(e)[:44]}", flush=True)
    finally:
        del ar, st, th
        torch.cuda.empty_cache()


def main():
    torch.manual_seed(0)
    p = torch.cuda.get_device_properties(0)
    print(f"{p.name}  {p.multi_processor_count} SMs   T={T} fixed, bf16, per-dim c")

    print(f"\nH sweep, f=sigmoid\n  {'H':<7}{'stream':<8}{'compile ms':>14}{'kernel ms':>14}"
          f"{'speedup':>10}{'kern GB/s':>11}{'cfg':>12}")
    for H in HIDDENS:
        for norm in (False, True):
            one_row(H, "sigmoid", norm)

    print(f"\nf-axis re-check, rms stream\n  {'H':<7}{'f':<8}{'compile ms':>14}{'kernel ms':>14}"
          f"{'speedup':>10}{'kern GB/s':>11}{'cfg':>12}")
    for H in (HIDDENS[0], HIDDENS[-1]):
        for fname in ("raw", "sigmoid", "tanh"):
            one_row(H, fname, True, label=f"{H:<7}{fname:<8}")


if __name__ == "__main__":
    main()
