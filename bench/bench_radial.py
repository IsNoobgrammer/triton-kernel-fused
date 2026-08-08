"""Radial NormSiLU vs torch.compile, fwd and fwd+bwd, at the MoE's real shape.

    python -m bench.bench_radial

Two cases, because they answer different questions:

  A  radial(g) alone          r = rms(g), p = sigmoid(theta), out = r^p * SiLU(g/r)
                              the activation on its own -- is the ACTIVATION the cost?
  B  radial(gate) * up        what the MoE actually computes, and what BatchedGLU replaces.

The kernel only exists in the B shape (it always folds the `* up`), so A runs it with up = 1.
That makes A's kernel column an upper bound on a hypothetical act-only kernel, not a fair
act-only number -- it still reads and multiplies the up half. Called out rather than quietly
compared, since a 2x that came from measuring different work would be worse than no number.

M = 393216 = 65536 tokens x top-6, I = 768: one MoE layer's worth of routed rows.
"""
import torch

from kernels.sm75.moe import BatchedGLU

DEV = "cuda"
M, I = 393216, 768
NS_EPS = 1e-6


def eager_radial(gate, theta):
    """r^p * SiLU(g/r), p = sigmoid(theta). Matches _act_eager code 8."""
    g = gate.float()
    r = torch.sqrt(g.square().mean(-1, keepdim=True) + NS_EPS)
    p = torch.sigmoid(theta.float())
    return (r.pow(p) * torch.nn.functional.silu(g / r)).to(gate.dtype)


def eager_glu(gate_up, theta):
    g, u = gate_up[:, :I], gate_up[:, I:]
    return eager_radial(g, theta) * u


def _time(fn, make, backward, iters=30, warmup=8):
    def one():
        args = make()
        out = fn(*args)
        if backward:
            out.backward(torch.ones_like(out))
    for _ in range(warmup):
        one()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(iters):
        one()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def run_case(tag, note, eager_fn, kernel_fn, make):
    print(f"\n{tag}   {note}")
    print(f"  {'':<18}{'fwd ms':>10}{'fwd+bwd ms':>13}{'vs compile':>13}")
    comp = torch.compile(eager_fn, mode="max-autotune", fullgraph=True)
    res = {}
    for name, fn in (("eager", eager_fn), ("torch.compile", comp), ("fused kernel", kernel_fn)):
        try:
            f = _time(fn, lambda: make(False), backward=False)
            bwd = _time(fn, lambda: make(True), backward=True)
        except Exception as e:
            print(f"  {name:<18}FAILED {type(e).__name__}: {str(e)[:46]}")
            continue
        res[name] = (f, bwd)
        rel = (f"{res['torch.compile'][1] / bwd:.2f}x"
               if "torch.compile" in res and name == "fused kernel" else "-")
        print(f"  {name:<18}{f:>10.3f}{bwd:>13.3f}{rel:>13}")
    if "fused kernel" in res and "torch.compile" in res:
        cf, cb = res["torch.compile"]
        kf, kb = res["fused kernel"]
        print(f"    fwd {cf / kf:.2f}x   fwd+bwd {cb / kb:.2f}x   vs torch.compile")


def main():
    torch.manual_seed(0)
    gate_up = (torch.randn(M, 2 * I, device=DEV, dtype=torch.bfloat16) * 0.5)
    theta = torch.randn(1, device=DEV, dtype=torch.float32) * 0.5
    row_act = torch.full((M,), 8, dtype=torch.int32, device=DEV)     # code 8 = radial, sigmoid p
    row_alpha = theta.expand(M).contiguous()                          # per-ROW theta

    print(f"M={M} (65536 tokens x top-6)  I={I}  bf16")

    # ---- A: activation alone. up is forced to 1 so the kernel computes act * 1.
    gu_ones = gate_up.clone()
    gu_ones[:, I:] = 1.0

    def make_a(bw):
        gu = gu_ones.detach().requires_grad_(bw)
        t = theta.detach().requires_grad_(bw)
        return (gu, t)

    run_case("A  radial(x) only",
             "(kernel still multiplies by up=1 -- an upper bound, not an act-only number)",
             lambda gu, t: eager_radial(gu[:, :I], t),
             lambda gu, t: BatchedGLU.apply(gu, row_act, t.expand(M).contiguous()),
             make_a)

    # ---- B: what the MoE actually runs
    def make_b(bw):
        gu = gate_up.detach().requires_grad_(bw)
        t = theta.detach().requires_grad_(bw)
        return (gu, t)

    run_case("B  radial(gate) * up", "(what BatchedGLU replaces in the MoE)",
             eager_glu,
             lambda gu, t: BatchedGLU.apply(gu, row_act, t.expand(M).contiguous()),
             make_b)


if __name__ == "__main__":
    main()
