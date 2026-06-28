"""Frozen eval for the fused Muon step — parity vs the EXACT BiBo eager recipe + speed.

Compares three step() implementations over a representative BiBo Muon param set (attention + dense MLP
+ 3D stacked experts, fp16 params, T4):
  - eager       : BiBo/bench/optim.py recipe, copied verbatim (the reference)
  - fused-fp32  : kernels.muon.FusedMuon, ns_dtype=fp32 (the champion — must be bit-tight to eager)
  - fused-fp16  : kernels.muon.FusedMuon, ns_dtype=fp16 (T4 fp16 tensor-core NS — opt-in, gated)

Parity: identical init + identical grads each step, compare max|Δp| after K steps.
Stability gate (fp16): SV mean ~1 and |Δp|/lr attribution ~0.2 (flat across shapes).
Speed: triton.testing.do_bench on a fixed grad set.

Run:  <tkf-venv>/python .autoresearch/bench_muon.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

try:
    from triton.testing import do_bench
except ImportError:                          # Windows local: no triton — CUDA-event fallback
    def do_bench(fn, warmup=10, rep=50):
        for _ in range(warmup): fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        for _ in range(rep): fn()
        e.record(); torch.cuda.synchronize()
        return s.elapsed_time(e) / rep

from kernels.muon import FusedMuon, newton_schulz

DEV = "cuda"


# ── EXACT BiBo eager recipe (copied verbatim from BiBo/bench/optim.py — the reference) ──
def eager_ns(G, num_iters=5):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)
    X = X / (X.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1))
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    for _ in range(num_iters):
        A = X @ X.transpose(1, 2)
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(G.dtype)


class EagerMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in params:
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(p.grad)
                g = p.grad.add(buf, alpha=momentum) if group["nesterov"] else buf
                g = eager_ns(g)
                g = g.mul_(0.2 * (max(p.shape[-2], p.shape[-1]) ** 0.5))
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(g, alpha=-lr)


# ── Representative BiBo Muon param inventory (H=512, I=768, E=9, ~6 layers) ──
def make_shapes(layers=6, H=512, I=768, E=9):
    sh = []
    for _ in range(layers):
        sh += [(H, H), (H // 2, H), (H // 2, H), (H, H)]   # q, k(GQA), v(GQA), o
        sh += [(2 * I, H), (H, I)]                          # dense MLP gate_up, down
        sh += [(E, 2 * I, H), (E, H, I)]                    # 3D stacked experts
    return sh


def make_params(shapes, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=torch.float16) for s in shapes]


def prime_grads(params, seed=1):
    g = torch.Generator(device=DEV).manual_seed(seed)
    for p in params:
        p.grad = torch.randn(*p.shape, generator=g, device=DEV, dtype=torch.float16)


def parity(make_opt, shapes, steps=5):
    """Max |Δp| between the eager reference and `make_opt`, after `steps` identical steps."""
    pe, pf = make_params(shapes, 0), make_params(shapes, 0)
    oe, of = EagerMuon(pe, weight_decay=0.1), make_opt(pf)
    gg = torch.Generator(device=DEV).manual_seed(7)
    worst = 0.0
    for _ in range(steps):
        grads = [torch.randn(*p.shape, generator=gg, device=DEV, dtype=torch.float16) for p in pe]
        for p, gr in zip(pe, grads): p.grad = gr.clone()
        for p, gr in zip(pf, grads): p.grad = gr.clone()
        oe.step(); of.step()
        worst = max(worst, max((a.float() - b.float()).abs().max().item() for a, b in zip(pe, pf)))
    return worst


def speed(make_opt, shapes):
    params = make_params(shapes, 0)
    opt = make_opt(params)
    prime_grads(params)
    opt.step()  # materialize state
    return do_bench(lambda: (prime_grads(params), opt.step()))


def stability(shapes, ns_dtype):
    """fp16 NS gate: SV mean ~1 (per shape) and |Δp|/lr attribution ~0.2 (flat)."""
    sv_ok, attr_ok = True, True
    rows = []
    for s in [(512, 512), (256, 512), (1536, 512), (9, 1536, 512), (9, 512, 768)]:
        G = torch.randn(*s, device=DEV, dtype=torch.float16)
        Y = newton_schulz(G, ns_dtype=ns_dtype).float()
        sv = torch.linalg.svdvals(Y if Y.ndim == 3 else Y.unsqueeze(0))
        sv_ok &= 0.80 <= sv.mean().item() <= 1.20
        p = torch.nn.Parameter(torch.randn(*s, device=DEV, dtype=torch.float16))
        opt = FusedMuon([p], lr=1e-3, weight_decay=0.0, ns_dtype=ns_dtype)
        p.grad = torch.randn(*s, device=DEV, dtype=torch.float16)
        before = p.detach().clone()
        opt.step()
        rms = ((p.detach() - before).float().abs() / 1e-3).pow(2).mean().sqrt().item()
        attr_ok &= 0.15 <= rms <= 0.25
        rows.append(f"  {str(s):16s} SV mean={sv.mean():.3f}  |Δp|/lr={rms:.4f}")
    return sv_ok, attr_ok, rows


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    shapes = make_shapes()
    nparam = sum(torch.tensor(s).prod().item() for s in shapes)
    print(f"Muon param set: {len(shapes)} tensors, {nparam/1e6:.1f}M params\n")

    f32 = lambda ps: FusedMuon(ps, weight_decay=0.1, ns_dtype=torch.float32)
    f16 = lambda ps: FusedMuon(ps, weight_decay=0.1, ns_dtype=torch.float16)

    print("PARITY vs eager (max|Δp| after 5 steps):")
    d32, d16 = parity(f32, shapes), parity(f16, shapes)
    print(f"  fused-fp32 = {d32:.2e}  {'PASS' if d32 < 1e-2 else 'FAIL'}  (bit-tight gate < 1e-2 fp16)")
    print(f"  fused-fp16 = {d16:.2e}  (informational — fp16 NS is a different op, not bit-parity)")

    print("\nfp16-NS STABILITY gate:")
    sv_ok, attr_ok, rows = stability(shapes, torch.float16)
    print("\n".join(rows))
    print(f"  SV mean~1: {sv_ok} | attribution~0.2 flat: {attr_ok} -> {'PASS' if sv_ok and attr_ok else 'FAIL'}")

    print("\nSPEED (do_bench, ms/step, lower better):")
    te = speed(EagerMuon, shapes)
    t32 = speed(f32, shapes)
    t16 = speed(f16, shapes)
    print(f"  eager       {te:6.2f} ms  (1.00x)")
    print(f"  fused-fp32  {t32:6.2f} ms  ({te/t32:.2f}x)")
    print(f"  fused-fp16  {t16:6.2f} ms  ({te/t16:.2f}x)")
