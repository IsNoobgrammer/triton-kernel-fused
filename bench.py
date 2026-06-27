"""3-phase benchmark (forward / backward / forward+backward) of every fused kernel vs
its PyTorch-eager equivalent, plus a grad-equivalence check.

    python bench.py            # all kernels, default shapes, fp16
    python bench.py swiglu     # one kernel

Timing: triton.testing.do_bench (median ms). Speedup = eager / kernel (>1 = kernel faster).
Grad check: max|Δ| of every input grad vs eager, same upstream cotangent.

⚠️ Numbers are GPU-specific. Triton tl.dot GEMMs are far slower on Turing (T4, sm_75) than
on Ampere+; re-run on YOUR target GPU before trusting any speedup. The printed header states
the GPU these numbers came from.
"""
import sys
import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench

from kernels import fused_swiglu, fused_linear_cross_entropy, fused_xsa, causal_conv1d_router
from kernels.moe import moe_per_expert, moe_grouped, moe_eager

DTYPE = torch.float16
DEV = "cuda"


def _stats(kernel_step, eager_step, leaves):
    """Return (kernel_ms, eager_ms) for a full fwd+bwd step, zeroing leaf grads each iter."""
    k = do_bench(kernel_step, grad_to_none=leaves)
    e = do_bench(eager_step, grad_to_none=leaves)
    return k, e


def _fwd_ms(fn):
    with torch.no_grad():
        return do_bench(fn)


def _bwd_ms(make_loss, leaves):
    loss = make_loss()
    return do_bench(lambda: loss.backward(retain_graph=True), grad_to_none=leaves)


def _gdiff(pairs):
    """pairs: list of (kernel_grad, eager_grad). Return (abs_max, rel_max)."""
    a = max((kg - eg).abs().max().item() for kg, eg in pairs)
    r = max(((kg - eg).abs().max() / (eg.abs().max() + 1e-9)).item() for kg, eg in pairs)
    return a, r


def _report(name, kf, ef, kb, eb, kfb, efb, gabs, grel, peak_k, peak_e):
    print(f"\n=== {name} ===")
    print(f"  forward      kernel {kf:7.3f} ms | eager {ef:7.3f} ms | {ef/kf:5.2f}x")
    print(f"  backward     kernel {kb:7.3f} ms | eager {eb:7.3f} ms | {eb/kb:5.2f}x")
    print(f"  fwd+bwd      kernel {kfb:7.3f} ms | eager {efb:7.3f} ms | {efb/kfb:5.2f}x")
    print(f"  peak mem     kernel {peak_k:6.0f} MB | eager {peak_e:6.0f} MB | {peak_e/max(peak_k,1):5.2f}x less")
    print(f"  grad vs eager: abs {gabs:.2e} | rel {grel:.2e}  ({'PASS' if grel < 1.5e-2 else 'CHECK'})")


def _peak(step):
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


# ───────────────────────── SwiGLU ─────────────────────────
def bench_swiglu(M=8192, I=768):
    gu = torch.randn(M, 2 * I, device=DEV, dtype=DTYPE)
    G = torch.randn(M, I, device=DEV, dtype=DTYPE)

    def eager(t):
        gate, up = t[:, :I], t[:, I:]
        return F.silu(gate) * up

    # grad check (shared cotangent)
    a = gu.clone().requires_grad_(True); b = gu.clone().requires_grad_(True)
    fused_swiglu(a).backward(G); eager(b).backward(G)
    gabs, grel = _gdiff([(a.grad, b.grad)])

    kf = _fwd_ms(lambda: fused_swiglu(gu))
    ef = _fwd_ms(lambda: eager(gu))
    x = gu.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: (fused_swiglu(x) * G).sum(), [x])
    eb = _bwd_ms(lambda: (eager(x) * G).sum(), [x])
    kstep = lambda: (fused_swiglu(x) * G).sum().backward()
    estep = lambda: (eager(x) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [x])
    _report("SwiGLU activation (M=%d I=%d)" % (M, I), kf, ef, kb, eb, kfb, efb, gabs, grel,
            _peak(kstep), _peak(estep))


# ─────────────────── fused-linear CE ───────────────────
def bench_ce(N=4096, H=512, V=81000):
    hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    w = torch.randn(V, H, device=DEV, dtype=DTYPE) * 0.1
    lab = torch.randint(0, V, (N,), device=DEV)

    def eager(h, ww):
        # fair fp16 baseline: fp16 GEMM (as in a real autocast model), CE math in fp32
        return F.cross_entropy((h @ ww.t()).float(), lab)

    a = hid.clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
    b = hid.clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
    fused_linear_cross_entropy(a, wa, lab).backward(); eager(b, wb).backward()
    gabs, grel = _gdiff([(a.grad, b.grad), (wa.grad, wb.grad)])

    kf = _fwd_ms(lambda: fused_linear_cross_entropy(hid, w, lab))
    ef = _fwd_ms(lambda: eager(hid, w))
    h2 = hid.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: fused_linear_cross_entropy(h2, w2, lab), [h2, w2])
    eb = _bwd_ms(lambda: eager(h2, w2), [h2, w2])
    kstep = lambda: fused_linear_cross_entropy(h2, w2, lab).backward()
    estep = lambda: eager(h2, w2).backward()
    kfb, efb = _stats(kstep, estep, [h2, w2])
    _report("fused-linear CE (N=%d H=%d V=%d)" % (N, H, V), kf, ef, kb, eb, kfb, efb, gabs, grel,
            _peak(kstep), _peak(estep))


# ───────────────────────── XSA ─────────────────────────
def bench_xsa(B=8, Hq=8, S=1024, D=128, Hkv=2):
    Y = torch.randn(B, Hq, S, D, device=DEV, dtype=DTYPE)
    V = torch.randn(B, Hkv, S, D, device=DEV, dtype=DTYPE)
    G = torch.randn(B, Hq, S, D, device=DEV, dtype=DTYPE)
    g = Hq // Hkv

    def eager(y, v):
        Yg = y.view(B, Hkv, g, S, D)
        Vn = F.normalize(v, dim=-1).unsqueeze(2)
        return (Yg - (Yg * Vn).sum(-1, keepdim=True) * Vn).reshape(B, Hq, S, D)

    ya = Y.clone().requires_grad_(True); va = V.clone().requires_grad_(True)
    yb = Y.clone().requires_grad_(True); vb = V.clone().requires_grad_(True)
    fused_xsa(ya, va).backward(G); eager(yb, vb).backward(G)
    gabs, grel = _gdiff([(ya.grad, yb.grad), (va.grad, vb.grad)])

    kf = _fwd_ms(lambda: fused_xsa(Y, V))
    ef = _fwd_ms(lambda: eager(Y, V))
    y2 = Y.clone().requires_grad_(True); v2 = V.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: (fused_xsa(y2, v2) * G).sum(), [y2, v2])
    eb = _bwd_ms(lambda: (eager(y2, v2) * G).sum(), [y2, v2])
    kstep = lambda: (fused_xsa(y2, v2) * G).sum().backward()
    estep = lambda: (eager(y2, v2) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [y2, v2])
    _report("XSA (B=%d Hq=%d S=%d D=%d Hkv=%d)" % (B, Hq, S, D, Hkv), kf, ef, kb, eb, kfb, efb, gabs, grel,
            _peak(kstep), _peak(estep))


# ─────────────── causal-conv1d router ───────────────
def bench_conv_router(B=8, S=1024, H=512, E=11, K=4):
    x = torch.randn(B, S, H, device=DEV, dtype=DTYPE)
    w = torch.randn(E, H, K, device=DEV, dtype=DTYPE) * 0.02
    G = torch.randn(B * S, E, device=DEV, dtype=DTYPE)

    def eager(xx, ww):
        xp = F.pad(xx.transpose(1, 2), (K - 1, 0))   # (B,H,S+K-1)
        return F.conv1d(xp, ww).transpose(1, 2).reshape(B * S, E)   # (B*S,E)

    xa = x.clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
    xb = x.clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
    causal_conv1d_router(xa, wa).backward(G); eager(xb, wb).backward(G)
    gabs, grel = _gdiff([(xa.grad, xb.grad), (wa.grad, wb.grad)])

    kf = _fwd_ms(lambda: causal_conv1d_router(x, w))
    ef = _fwd_ms(lambda: eager(x, w))
    x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: (causal_conv1d_router(x2, w2) * G).sum(), [x2, w2])
    eb = _bwd_ms(lambda: (eager(x2, w2) * G).sum(), [x2, w2])
    kstep = lambda: (causal_conv1d_router(x2, w2) * G).sum().backward()
    estep = lambda: (eager(x2, w2) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [x2, w2])
    _report("causal-conv1d router (B=%d S=%d H=%d E=%d K=%d)" % (B, S, H, E, K),
            kf, ef, kb, eb, kfb, efb, gabs, grel, _peak(kstep), _peak(estep))


# ───────────────────────── MoE (PolyGLU) ─────────────────────────
def bench_moe(N=8192, H=512, I=768, E=9, top_k=2):
    # act_codes: PolyGLU groups of 3 (SiLU/ReLU²/Tanh)
    act_codes = torch.tensor([i % 3 for i in range(E)], device=DEV, dtype=torch.int32)
    hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    gup = (torch.randn(E, 2 * I, H, device=DEV, dtype=DTYPE) * 0.02)
    dwn = (torch.randn(E, H, I, device=DEV, dtype=DTYPE) * 0.02)
    logits = torch.randn(N, E, device=DEV, dtype=DTYPE)
    wt_full, idx = torch.topk(torch.softmax(logits.float(), -1), top_k, dim=-1)
    wt_full = wt_full.to(DTYPE)
    print(f"\n(MoE: N={N} rows*top_k={N*top_k} {'>=' if N*top_k>=4096 else '<'} GROUPED_MIN_TOKENS={4096}; H={H} I={I} E={E} k={top_k})")

    def run(variant):
        # grad check vs eager (shared cotangent G)
        G = torch.randn(N, H, device=DEV, dtype=DTYPE)
        def mk():
            return (hid.clone().requires_grad_(True), gup.clone().requires_grad_(True),
                    dwn.clone().requires_grad_(True), wt_full.clone().requires_grad_(True))
        hk, gk, dk, wk = mk(); (variant(hk, idx, wk, gk, dk, act_codes) * G).sum().backward()
        he, ge, de, we = mk(); (moe_eager(he, idx, we, ge, de, act_codes) * G).sum().backward()
        gabs, grel = _gdiff([(hk.grad, he.grad), (gk.grad, ge.grad), (dk.grad, de.grad), (wk.grad, we.grad)])
        # timing
        kf = _fwd_ms(lambda: variant(hid, idx, wt_full, gup, dwn, act_codes))
        ef = _fwd_ms(lambda: moe_eager(hid, idx, wt_full, gup, dwn, act_codes))
        h2 = hid.clone().requires_grad_(True); g2 = gup.clone().requires_grad_(True)
        d2 = dwn.clone().requires_grad_(True); w2 = wt_full.clone().requires_grad_(True)
        leaves = [h2, g2, d2, w2]
        kb = _bwd_ms(lambda: (variant(h2, idx, w2, g2, d2, act_codes) * G).sum(), leaves)
        eb = _bwd_ms(lambda: (moe_eager(h2, idx, w2, g2, d2, act_codes) * G).sum(), leaves)
        kstep = lambda: (variant(h2, idx, w2, g2, d2, act_codes) * G).sum().backward()
        estep = lambda: (moe_eager(h2, idx, w2, g2, d2, act_codes) * G).sum().backward()
        kfb, efb = _stats(kstep, estep, leaves)
        return kf, ef, kb, eb, kfb, efb, gabs, grel, _peak(kstep), _peak(estep)

    for vname, vfn in [("per-expert", moe_per_expert), ("grouped", moe_grouped)]:
        try:
            _report(f"MoE {vname} vs eager", *run(vfn))
        except Exception as ex:
            print(f"\n=== MoE {vname} vs eager ===\n  FAILED: {type(ex).__name__}: {ex}")


BENCHES = {"swiglu": bench_swiglu, "ce": bench_ce, "xsa": bench_xsa, "conv": bench_conv_router,
           "moe": bench_moe}

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype={DTYPE} | torch {torch.__version__} | triton {triton.__version__}")
    which = sys.argv[1:] or list(BENCHES)
    for name in which:
        try:
            BENCHES[name]()
        except torch.cuda.OutOfMemoryError:
            print(f"\n=== {name} ===\n  OOM at this shape (eager baseline likely materialized full logits). Try smaller.")
            torch.cuda.empty_cache()
