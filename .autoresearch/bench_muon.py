"""Frozen eval for the fused Polar-Express Muon step — parity vs the EXACT baseline + speed.

Baseline = nprime06/parameter-golf Polar-Express Muon (single-GPU path, distributed stripped), copied
verbatim: 5 per-iteration NS coeff tuples, bf16 NS, Jordan aspect-ratio scale max(1,rows/cols)**0.5.

Compares over a representative BiBo Muon param set (attention + dense MLP + 3D experts, fp16 params, T4):
  - baseline    : the reference recipe above (eager; the original @torch.compile'd it)
  - fused-bf16  : kernels.muon.FusedMuon, ns_dtype=bf16 (the champion — must be tight to baseline)
  - fused-fp16  : kernels.muon.FusedMuon, ns_dtype=fp16 (T4 fp16 tensor cores; bf16 has none on sm_75)

Parity: identical init + grads each step, max|Δp| after K steps. Stability: NS output SV mean ~1, NaN-free.
Run:  ../BiBo/.venv/Scripts/python.exe .autoresearch/bench_muon.py
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

from kernels.muon import FusedMuon, newton_schulz, _PE_COEFFS

DEV = "cuda"


# ── EXACT baseline (nprime06/parameter-golf Polar-Express Muon, single-GPU; distributed stripped) ──
def baseline_ns(G, coeffs=_PE_COEFFS, eps=1e-7):
    was_2d = G.ndim == 2
    if was_2d:
        G = G.unsqueeze(0)
    X = G.bfloat16()
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
    for a, b, c in coeffs:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    if was_2d:
        X = X.squeeze(0)
    return X


class BaselineMuon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                g = p.grad.bfloat16()
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = baseline_ns(update)
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(update.to(dtype=p.dtype), alpha=-lr * scale)


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
    pe, pf = make_params(shapes, 0), make_params(shapes, 0)
    oe, of = BaselineMuon(pe, weight_decay=0.1), make_opt(pf)
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
    opt.step()
    return do_bench(lambda: (prime_grads(params), opt.step()))


def stability(ns_dtype):
    """NS output quality: SV mean ~1 per shape, NaN-free."""
    ok, rows = True, []
    for s in [(512, 512), (256, 512), (1536, 512), (9, 1536, 512), (9, 512, 768)]:
        G = torch.randn(*s, device=DEV, dtype=torch.float16)
        Y = newton_schulz(G, ns_dtype=ns_dtype).float()
        nan = bool(Y.isnan().any())
        sv = torch.linalg.svdvals(Y if Y.ndim == 3 else Y.unsqueeze(0))
        ok &= (not nan) and (0.80 <= sv.mean().item() <= 1.20)
        rows.append(f"  {str(s):16s} SV mean={sv.mean():.3f} min={sv.min():.3f} max={sv.max():.3f} nan={nan}")
    return ok, rows


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    shapes = make_shapes()
    nparam = sum(torch.tensor(s).prod().item() for s in shapes)
    print(f"Muon param set: {len(shapes)} tensors, {nparam/1e6:.1f}M params (Polar-Express, Jordan scale)\n")

    bf16 = lambda ps: FusedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.bfloat16)
    fp16 = lambda ps: FusedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16)

    print("PARITY vs baseline (max|Δp| after 5 steps):")
    dbf, dfp = parity(bf16, shapes), parity(fp16, shapes)
    print(f"  fused-bf16 = {dbf:.2e}  {'PASS' if dbf < 2e-2 else 'FAIL'}  (bf16 tolerance gate < 2e-2)")
    print(f"  fused-fp16 = {dfp:.2e}  (informational — fp16 NS is a different op, not bit-parity)")

    print("\nfp16-NS STABILITY gate (SV mean ~1, NaN-free):")
    ok, rows = stability(torch.float16)
    print("\n".join(rows))
    print(f"  -> {'PASS' if ok else 'FAIL'}")

    print("\nSPEED (do_bench, ms/step, lower better):")
    tb = speed(BaselineMuon, shapes)
    t16 = speed(fp16, shapes)
    tbf = speed(bf16, shapes)
    print(f"  baseline    {tb:6.2f} ms  (1.00x)")
    print(f"  fused-bf16  {tbf:6.2f} ms  ({tb/tbf:.2f}x)")
    print(f"  fused-fp16  {t16:6.2f} ms  ({tb/t16:.2f}x)")
