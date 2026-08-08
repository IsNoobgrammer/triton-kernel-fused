"""Speed: fused make_mlp_input vs eager vs torch.compile, at the board's real shape.

    python -m bench.bench_res_add_rms

    h = attn_read + c * stream / rms(stream)          ("rms" mode, per-dim c)

Times forward and forward+backward separately. The backward is what matters here -- it is where
eager materialises the most intermediates, and it runs once per layer per micro-batch.

torch.compile is the honest bar, not eager: anyone writing this in PyTorch today would compile it,
and a fused kernel that only beats an uncompiled baseline has not earned its maintenance.
"""
import torch

from kernels.sm120.residual_add import make_mlp_input

DEV = "cuda"
T, H = 65536, 512          # batch 64 x seq 1024, hidden 512
RMS_EPS = 1e-6


def eager_rms(attn_read, theta, stream):
    """The model's spelling: c is cast to the stream dtype, exactly as exp's else-branch does."""
    sv = stream * torch.rsqrt(stream.pow(2).mean(-1, keepdim=True) + RMS_EPS)
    return attn_read + theta.to(sv.dtype) * sv


def _bench(fn, tensors, backward, iters=50, warmup=10):
    ar, th, st = tensors
    g = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)

    def one():
        a = ar.detach().requires_grad_(backward)
        s = st.detach().requires_grad_(backward)
        t = th.detach().requires_grad_(backward)
        out = fn(a, t, s)
        if backward:
            out.backward(g.to(out.dtype))

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
    th = (torch.randn(H, device=DEV, dtype=torch.float32) * 0.3 + 1.0)
    tensors = (ar, th, st)

    compiled = torch.compile(eager_rms, mode="max-autotune", fullgraph=True)
    fused = lambda a, t, s: make_mlp_input(a, t, s, modes=("rms",))

    print(f"T={T} H={H} bf16, per-dim c, mode='rms'   (one AttnRes carry write)\n")
    print(f"  {'':<18}{'fwd ms':>10}{'fwd+bwd ms':>13}{'vs compile':>13}")
    res = {}
    for name, fn in (("eager", eager_rms), ("torch.compile", compiled), ("fused kernel", fused)):
        try:
            f = _bench(fn, tensors, backward=False)
            b = _bench(fn, tensors, backward=True)
        except Exception as e:
            print(f"  {name:<18}FAILED {type(e).__name__}: {str(e)[:50]}")
            continue
        res[name] = (f, b)
        ref = res.get("torch.compile")
        rel = f"{ref[1] / b:.2f}x" if ref and name != "torch.compile" else "-"
        print(f"  {name:<18}{f:>10.3f}{b:>13.3f}{rel:>13}")

    if "fused kernel" in res and "torch.compile" in res:
        cf, cb = res["torch.compile"]
        kf, kb = res["fused kernel"]
        print(f"\n  fwd     : {cf / kf:.2f}x vs torch.compile")
        print(f"  fwd+bwd : {cb / kb:.2f}x vs torch.compile")
        # per training step: 10 layers x 4 grad_accum carry writes
        saved = (cb - kb) * 10 * 4
        print(f"  saves {saved:.1f} ms/step at 10 layers x 4 grad_accum "
              f"({100 * saved / 1430:.2f}% of a 1430 ms step)")


if __name__ == "__main__":
    main()
