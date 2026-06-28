"""FROZEN local eval for the CE latency loop (RTX 3050, proxy shape — the optimization set).

Benches, as raw Triton (no torch.compile): the lifted compiled-CE BASELINE (ce_compiled) vs our
chunked fused-linear CE at several budgets. Reports fwd+bwd ms, peak MB, grad_rel vs eager.
Objective = minimize our latency s.t. peak <= our-low-mem peak AND grad PASS. DO NOT edit to flatter.

    python .autoresearch/ce_eval_local.py            # proxy N=4096 V=32000 H=512 (fits 4GB)
    python .autoresearch/ce_eval_local.py 8192 81000  # custom N V (needs more VRAM / T4)
"""
import sys, gc
sys.path.insert(0, ".")
import torch, torch.nn.functional as F
from triton.testing import do_bench
from kernels.cross_entropy import fused_linear_cross_entropy
from ce_compiled import compiled_cross_entropy

DEV, DT = "cuda", torch.float16
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4096
V = int(sys.argv[2]) if len(sys.argv) > 2 else 32000
H = int(sys.argv[3]) if len(sys.argv) > 3 else 512
MB = 1024 * 1024


def _leaf(*shape):
    # proper leaf: scale in-place so the tensor itself is the leaf (NOT a non-leaf `*0.1` graph node,
    # which would get freed after the first backward and break the warmup loop).
    return torch.randn(*shape, device=DEV, dtype=DT).mul_(0.1).requires_grad_(True)


def _peak(step):
    for _ in range(3):
        step()
    torch.cuda.synchronize(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


def _fbms(make_loss, leaves):
    for _ in range(4):
        for l in leaves: l.grad = None
        make_loss().backward()
    torch.cuda.synchronize()
    def step():
        for l in leaves: l.grad = None
        make_loss().backward()
    return do_bench(step)


def _grad(fn):
    h = _leaf(N, H); w = _leaf(V, H)
    lab = torch.randint(0, V, (N,), device=DEV)
    fn(h, w, lab).backward()
    gh, gw = h.grad.clone(), w.grad.clone()
    h2 = h.detach().clone().requires_grad_(True); w2 = w.detach().clone().requires_grad_(True)
    F.cross_entropy((h2 @ w2.t()).float(), lab).backward()
    rh = ((gh - h2.grad).abs().max() / (h2.grad.abs().max() + 1e-9)).item()
    rw = ((gw - w2.grad).abs().max() / (w2.grad.abs().max() + 1e-9)).item()
    return max(rh, rw)


def run(name, fn):
    h = _leaf(N, H); w = _leaf(V, H)
    lab = torch.randint(0, V, (N,), device=DEV)
    leaves = [h, w]
    ms = _fbms(lambda: fn(h, w, lab), leaves)
    pk = _peak(lambda: (fn(h, w, lab).backward()))
    rel = _grad(fn)
    print(f"  {name:22s} fwd+bwd {ms:8.3f} ms | peak {pk:7.0f} MB | grad_rel {rel:.2e} "
          f"({'PASS' if rel < 1.5e-2 else 'FAIL'})")
    del h, w, lab, leaves; gc.collect(); torch.cuda.empty_cache()
    return ms, pk, rel


print(f"GPU {torch.cuda.get_device_name(0)} | CE proxy N={N} V={V} H={H} | dtype={DT}")
print(f"  (N,V) fp16 logits would be {N*V*2/MB:.0f} MB  — baseline materializes this, ours does not")
base_ms, base_pk, _ = run("compiled (baseline)", lambda h, w, l: compiled_cross_entropy(h, w, l))
# fused-fwd+bwd (the only path now): grad computed in forward, no recompute. budget = memory dial.
for tag, bud in [("ours_192MB", 192 * MB), ("ours_128MB", 128 * MB), ("ours_64MB", 64 * MB)]:
    ms, pk, rel = run(tag, lambda h, w, l, _b=bud: fused_linear_cross_entropy(h, w, l, bwd_logits_budget=_b))
    print(f"      -> latency {ms/base_ms:.2f}x baseline | peak {base_pk/max(pk,1):.2f}x less than baseline")
