"""3-phase benchmark (forward / backward / forward+backward) of every fused kernel vs
its PyTorch-eager equivalent, plus a grad-equivalence check.

    python bench.py                 # all kernels, fp16. Default shapes = BiBo's REAL training step:
                                    #   16384 tokens (B16*S1024), hidden 512, vocab 81000,
                                    #   dense-MLP I=1024, MoE I=768/E=9/k=2, attn Hq4/Hkv2/D128.
                                    # At 16384 tok the (N,V) logit matrix is ~2.65 GB fp16 — the
                                    # regime where never-materializing CE actually matters.
    python bench.py swiglu moe      # selected kernels
    python bench.py --compile       # measure vs torch.compile'd eager (industry steady-state)
    python bench.py --compile --dump-triton swiglu   # ALSO print inductor's generated Triton to
                                                     # stderr — read it, then hand-iterate to beat it

Timing: triton.testing.do_bench (median ms) after explicit warmup. Speedup = eager / kernel
(>1 = kernel faster). Grad check: max|Δ| of every input grad vs eager, same upstream cotangent.

--compile (industry-standard steady-state): wraps ONLY the EAGER baseline in torch.compile —
the kernel runs as its native Triton (eager). This is the correct, fair comparison: "does the
hand-written kernel beat what torch.compile gives PyTorch for free?" We do NOT compile the kernel
itself — wrapping a custom autograd.Function in torch.compile is both unrepresentative and crashes
some (e.g. Liger SiLUMul: "leaf Variable ... in-place operation"). Compilation + Triton autotune
run during warmup, excluded from the timed step. torch.compile is broken on some local setups; run
--compile on the target GPU (T4 / Hopper).

⚠️ Numbers are GPU-specific. Triton tl.dot GEMMs are far slower on Turing (T4, sm_75) than
on Ampere+; re-run on YOUR target GPU. The printed header states the GPU.
"""
import os
import sys
import json
# --dump-triton: make inductor PRINT the Triton it generates for each compiled fn (to stderr).
# This is the forum tactic (marksaroufim): compile is a great *starting point* for a kernel — read
# its generated Triton and hand-iterate from there. Must be set BEFORE `import torch`. Implies
# --compile (only compiled fns emit code). On Kaggle: `python bench.py --compile --dump-triton swiglu`
# then read the `# kernel ...` blocks in stderr — that's inductor's kernel for you to beat.
if "--dump-triton" in sys.argv:
    os.environ["TORCH_LOGS"] = "output_code"
import torch
import torch.nn.functional as F
import triton
from triton.testing import do_bench

from kernels import fused_swiglu, fused_linear_cross_entropy, fused_xsa, causal_conv1d_router
from kernels.moe import moe_per_expert, moe_grouped, moe_grouped_cublas, moe_eager

DTYPE = torch.float16
DEV = "cuda"
COMPILE = False   # set by --compile in __main__
JSON_OUT = False  # set by --json in __main__ (emits one @@RESULT line per kernel for the loop harness)


def _c(fn):
    """torch.compile(fn) when --compile is set, else fn unchanged. Compilation + autotune run
    during the warmup passes in the timing helpers, so they are excluded from the timed step."""
    return torch.compile(fn) if COMPILE else fn


def _warm(fn, n=4):
    """Run fn a few times + sync so compilation/autotune is done before timing (excluded from it)."""
    for _ in range(n):
        fn()
    torch.cuda.synchronize()


def _stats(kernel_step, eager_step, leaves):
    """Return (kernel_ms, eager_ms) for a full fwd+bwd step, zeroing leaf grads each iter."""
    _warm(kernel_step); _warm(eager_step)
    return (do_bench(kernel_step, grad_to_none=leaves),
            do_bench(eager_step, grad_to_none=leaves))


def _step_ms(step, leaves):
    """Time one full fwd+bwd step (fresh graph each call) after warmup. Backward is derived as
    (fwd+bwd − fwd) by the caller — reliable under torch.compile, unlike retain_graph re-backward."""
    _warm(step)
    return do_bench(step, grad_to_none=leaves)


def _fwd_ms(fn):
    with torch.no_grad():
        _warm(fn)
        return do_bench(fn)


def _bwd_ms(make_loss, leaves):
    loss = make_loss()
    step = lambda: loss.backward(retain_graph=True)
    _warm(step)
    return do_bench(step, grad_to_none=leaves)


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
    if JSON_OUT:
        print("@@RESULT " + json.dumps({
            "name": name, "fwd_x": round(ef / kf, 3), "bwd_x": round(eb / kb, 3),
            "fwdbwd_x": round(efb / kfb, 3), "mem_x_less": round(peak_e / max(peak_k, 1), 3),
            "kernel_ms": round(kfb, 3), "eager_ms": round(efb, 3),
            "grad_rel": float(f"{grel:.2e}"), "pass": grel < 1.5e-2}))


def _peak(step):
    _warm(step, 2)                                      # compile/autotune cached before measuring
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    step(); torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1e6


# ───────────────────────── SwiGLU ─────────────────────────
def bench_swiglu(M=16384, I=1024):   # BiBo dense-MLP: B16*S1024 tokens, intermediate_size=1024
    gu = torch.randn(M, 2 * I, device=DEV, dtype=DTYPE)
    G = torch.randn(M, I, device=DEV, dtype=DTYPE)

    def eager(t):
        gate, up = t[:, :I], t[:, I:]
        return F.silu(gate) * up

    # grad check (shared cotangent)
    a = gu.clone().requires_grad_(True); b = gu.clone().requires_grad_(True)
    fused_swiglu(a).backward(G); eager(b).backward(G)
    gabs, grel = _gdiff([(a.grad, b.grad)])

    K = fused_swiglu; E = _c(eager)   # kernel runs native Triton (eager); only the baseline is compiled
    kf = _fwd_ms(lambda: K(gu))
    ef = _fwd_ms(lambda: E(gu))
    x = gu.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: (K(x) * G).sum(), [x])
    eb = _bwd_ms(lambda: (E(x) * G).sum(), [x])
    kstep = lambda: (K(x) * G).sum().backward()
    estep = lambda: (E(x) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [x])
    _report("SwiGLU activation (M=%d I=%d)" % (M, I), kf, ef, kb, eb, kfb, efb, gabs, grel,
            _peak(kstep), _peak(estep))


# ─────────────────── fused-linear CE (chunk-budget candidate sweep) ───────────────────
def bench_ce(N=16384, H=512, V=81000):   # BiBo training: B16*S1024 tokens, hidden 512, vocab 81000
    hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    w = torch.randn(V, H, device=DEV, dtype=DTYPE) * 0.1
    lab = torch.randint(0, V, (N,), device=DEV)

    def eager_full(h, ww, y):
        # naive baseline: fp16 GEMM (autocast) → fp32 logits → CE. Materializes the (rows,V) matrix;
        # at training N this is exactly what OOMs a T4 — that's the point of a never-materialize CE.
        return F.cross_entropy((h @ ww.t()).float(), y)

    torch.cuda.empty_cache()
    # --- grad check on a SMALL slice so the fp32 (Nc,V) reference fits (the full-N ref OOMs by design) ---
    Nc = min(N, 2048)
    a = hid[:Nc].clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
    fused_linear_cross_entropy(a, wa, lab[:Nc]).backward()
    bb = hid[:Nc].clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
    eager_full(bb, wb, lab[:Nc]).backward()
    gabs, grel = _gdiff([(a.grad, bb.grad), (wa.grad, wb.grad)])
    del a, wa, bb, wb; torch.cuda.empty_cache()

    # --- compiled-eager baseline at FULL N — best-effort: it materializes the (N,V) logits and may OOM ---
    print(f"\n(CE: N={N} H={H} V={V}  — eager = compiled F.cross_entropy; grad-check on {Nc}-row slice)")
    ef = efb = peak_e = float("nan"); eager_ok = False
    try:
        Eg = _c(lambda h, ww: eager_full(h, ww, lab))
        ef = _fwd_ms(lambda: Eg(hid, w))
        he = hid.clone().requires_grad_(True); we = w.clone().requires_grad_(True)
        estep = lambda: Eg(he, we).backward()
        efb = _step_ms(estep, [he, we]); peak_e = _peak(estep)
        eager_ok = True
        del he, we
    except torch.cuda.OutOfMemoryError:
        print(f"  compiled-eager CE OOM at N={N} (materializes the (N,V) logits) — our kernel is the "
              f"ENABLING path here, not just faster. Standalone kernel numbers below.")
    torch.cuda.empty_cache()

    # --- our chunked CE at full N (never materializes (N,V); should fit where eager can't) ---
    MB = 1024 * 1024
    for vname, budget in [("ce_384MB", 384 * MB), ("ce_1GB", 1024 * MB), ("ce_128MB", 128 * MB)]:
        try:
            K = (lambda h, ww, _b=budget: fused_linear_cross_entropy(h, ww, lab, bwd_logits_budget=_b))
            kf = _fwd_ms(lambda: K(hid, w))
            h2 = hid.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
            kstep = lambda: K(h2, w2).backward()
            kfb = _step_ms(kstep, [h2, w2]); peak_k = _peak(kstep)
            if eager_ok:
                _report(f"CE {vname} (N={N} V={V})", kf, ef, max(kfb - kf, 0.0), max(efb - ef, 0.0),
                        kfb, efb, gabs, grel, peak_k, peak_e)
            else:
                print(f"\n=== CE {vname} (N={N} V={V}) — eager OOM, kernel standalone ===")
                print(f"  forward {kf:7.3f} ms | fwd+bwd {kfb:7.3f} ms | peak {peak_k:6.0f} MB "
                      f"| grad rel {grel:.2e} ({'PASS' if grel < 1.5e-2 else 'CHECK'})  [ENABLES training where eager OOMs]")
            del h2, w2; torch.cuda.empty_cache()
        except Exception as ex:
            print(f"\n=== CE {vname} ===\n  FAILED: {type(ex).__name__}: {str(ex).splitlines()[0]}")
            torch.cuda.empty_cache()


# ───────────────────────── XSA ─────────────────────────
def bench_xsa(B=16, Hq=4, S=1024, D=128, Hkv=2):   # BiBo training: batch 16, 4 q-heads / 2 kv-heads, head_dim 128
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

    K = fused_xsa; E = _c(eager)
    kf = _fwd_ms(lambda: K(Y, V))
    ef = _fwd_ms(lambda: E(Y, V))
    y2 = Y.clone().requires_grad_(True); v2 = V.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: (K(y2, v2) * G).sum(), [y2, v2])
    eb = _bwd_ms(lambda: (E(y2, v2) * G).sum(), [y2, v2])
    kstep = lambda: (K(y2, v2) * G).sum().backward()
    estep = lambda: (E(y2, v2) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [y2, v2])
    _report("XSA (B=%d Hq=%d S=%d D=%d Hkv=%d)" % (B, Hq, S, D, Hkv), kf, ef, kb, eb, kfb, efb, gabs, grel,
            _peak(kstep), _peak(estep))


# ─────────────── causal-conv1d router ───────────────
def bench_conv_router(B=16, S=1024, H=512, E=11, K=4):   # BiBo training: batch 16, 11 routed experts
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

    kfn = causal_conv1d_router; efn = _c(eager)   # not 'E' — E is the expert count here
    kf = _fwd_ms(lambda: kfn(x, w))
    ef = _fwd_ms(lambda: efn(x, w))
    x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
    kb = _bwd_ms(lambda: (kfn(x2, w2) * G).sum(), [x2, w2])
    eb = _bwd_ms(lambda: (efn(x2, w2) * G).sum(), [x2, w2])
    kstep = lambda: (kfn(x2, w2) * G).sum().backward()
    estep = lambda: (efn(x2, w2) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [x2, w2])
    _report("causal-conv1d router (B=%d S=%d H=%d E=%d K=%d)" % (B, S, H, E, K),
            kf, ef, kb, eb, kfb, efb, gabs, grel, _peak(kstep), _peak(estep))


# ───────── outsourced reference kernels (Liger) — do THEY beat compiled eager? ─────────
def bench_liger_swiglu(M=16384, I=1024):
    try:
        from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    except Exception as ex:
        print(f"\n=== Liger SwiGLU ===\n  SKIPPED — liger_kernel not installed ({ex}). `pip install liger-kernel`.")
        return
    gate = torch.randn(M, I, device=DEV, dtype=DTYPE)
    up = torch.randn(M, I, device=DEV, dtype=DTYPE)
    G = torch.randn(M, I, device=DEV, dtype=DTYPE)
    liger = lambda g, u: LigerSiLUMulFunction.apply(g, u)
    eager = lambda g, u: F.silu(g) * u
    ga = gate.clone().requires_grad_(True); ua = up.clone().requires_grad_(True)
    gb = gate.clone().requires_grad_(True); ub = up.clone().requires_grad_(True)
    liger(ga, ua).backward(G); eager(gb, ub).backward(G)
    gabs, grel = _gdiff([(ga.grad, gb.grad), (ua.grad, ub.grad)])
    K = liger; Eg = _c(eager)
    kf = _fwd_ms(lambda: K(gate, up)); ef = _fwd_ms(lambda: Eg(gate, up))
    g2 = gate.clone().requires_grad_(True); u2 = up.clone().requires_grad_(True)
    kstep = lambda: (K(g2, u2) * G).sum().backward()
    estep = lambda: (Eg(g2, u2) * G).sum().backward()
    kfb, efb = _stats(kstep, estep, [g2, u2])
    _report("Liger SwiGLU (M=%d I=%d)" % (M, I), kf, ef, max(kfb - kf, 0.0), max(efb - ef, 0.0),
            kfb, efb, gabs, grel, _peak(kstep), _peak(estep))


def bench_liger_ce(N=16384, H=512, V=81000, force_chunk=None):
    """force_chunk=C: scope-patch Liger's FLCE so its chunk_size = C rows (=> ceil(N/C) chunks).
    Maps the memory<->speed trade: Liger's default tiny chunk (next_power_of_2(cdiv(BT,cdiv(V,H)))
    = 32 here -> ~128 chunks) is slow; bigger chunks approach compiled-eager speed but spend more
    memory. The patch only rebinds the `triton` NAME in Liger's FLCE module (its inner CE kernel
    is a different module, untouched) and is restored in finally. force_chunk=None -> Liger default."""
    try:
        import liger_kernel.ops.fused_linear_cross_entropy as _flce
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction as LFLCE
    except Exception as ex:
        print(f"\n=== Liger CE ===\n  SKIPPED — liger_kernel not installed ({ex}). `pip install liger-kernel`.")
        return

    class _ShimTriton:                       # delegates to real triton; pins chunk_size = force_chunk
        def __getattr__(self, n): return getattr(triton, n)
        def next_power_of_2(self, _x): return force_chunk
    _orig = _flce.triton
    if force_chunk is not None:
        _flce.triton = _ShimTriton()
    try:
        hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
        w = torch.randn(V, H, device=DEV, dtype=DTYPE) * 0.1
        lab = torch.randint(0, V, (N,), device=DEV)
        def liger(h, ww):
            out = LFLCE.apply(h, ww, lab)
            return out[0] if isinstance(out, (tuple, list)) else out   # Liger returns (loss, z_loss)
        eager = lambda h, ww: F.cross_entropy((h @ ww.t()).float(), lab)
        a = hid.clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
        b = hid.clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
        liger(a, wa).backward(); eager(b, wb).backward()
        gabs, grel = _gdiff([(a.grad, b.grad), (wa.grad, wb.grad)])
        K = liger; Eg = _c(eager)
        kf = _fwd_ms(lambda: K(hid, w)); ef = _fwd_ms(lambda: Eg(hid, w))
        h2 = hid.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
        kstep = lambda: K(h2, w2).backward()
        estep = lambda: Eg(h2, w2).backward()
        kfb, efb = _stats(kstep, estep, [h2, w2])
        if force_chunk is None:
            tag = "Liger CE (default chunk)"
        else:
            tag = "Liger CE chunk=%d (%d chunks)" % (force_chunk, -(-N // force_chunk))
        _report("%s (N=%d V=%d)" % (tag, N, V), kf, ef, max(kfb - kf, 0.0),
                max(efb - ef, 0.0), kfb, efb, gabs, grel, _peak(kstep), _peak(estep))
    finally:
        _flce.triton = _orig                 # always restore


def bench_liger_ce_sweep(N=16384, H=512, V=81000):
    """Sweep Liger CE chunk size to map the memory<->speed curve: small chunk = slow + low mem,
    big chunk = ~compile speed + higher mem."""
    print(f"\n--- Liger CE chunk-size sweep (N={N} V={V}); default heuristic chunk = 32 ---")
    for c in (16384, 8192, 4096, 2048, 1024, 512, 256, 128, 64, 32):
        if c <= N:
            bench_liger_ce(N, H, V, force_chunk=c)


# ───────────────────────── MoE (PolyGLU) ─────────────────────────
def bench_moe(N=16384, H=512, I=768, E=9, top_k=2):   # BiBo training: 16384 tokens, moe_intermediate 768, 9 GLU experts
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
        # timing (compiled forwards under --compile)
        K = variant; Eg = _c(moe_eager)
        kf = _fwd_ms(lambda: K(hid, idx, wt_full, gup, dwn, act_codes))
        ef = _fwd_ms(lambda: Eg(hid, idx, wt_full, gup, dwn, act_codes))
        h2 = hid.clone().requires_grad_(True); g2 = gup.clone().requires_grad_(True)
        d2 = dwn.clone().requires_grad_(True); w2 = wt_full.clone().requires_grad_(True)
        leaves = [h2, g2, d2, w2]
        kstep = lambda: (K(h2, idx, w2, g2, d2, act_codes) * G).sum().backward()
        estep = lambda: (Eg(h2, idx, w2, g2, d2, act_codes) * G).sum().backward()
        kfb, efb = _stats(kstep, estep, leaves)
        return kf, ef, max(kfb - kf, 0.0), max(efb - ef, 0.0), kfb, efb, gabs, grel, _peak(kstep), _peak(estep)

    variants = [("per-expert", moe_per_expert), ("grouped", moe_grouped)]
    if torch.cuda.get_device_capability()[0] >= 8:
        variants.append(("grouped_cublas", moe_grouped_cublas))   # bf16/sm_80+ only — see moe.py
    else:
        print("\n=== MoE grouped_cublas ===\n  SKIPPED — torch._grouped_mm is bf16/sm_80+; this GPU is sm_<80 (Turing).")
    for vname, vfn in variants:
        try:
            _report(f"MoE {vname} vs eager", *run(vfn))
        except Exception as ex:
            print(f"\n=== MoE {vname} vs eager ===\n  FAILED: {type(ex).__name__}: {ex}")


# NOTE: Cut Cross Entropy (Apple cut_cross_entropy) was benched and REMOVED — on T4 it's
# fwd+bwd 0.08x (catastrophic; CCE targets Ampere/Hopper, its T4-fitting autotune config is
# terrible) despite 2.7x less memory. Same family as our CE/Liger CE: a memory-for-speed trade
# that loses hard on time vs compiled eager. Not worth the slow autotune to re-confirm.


# ───────── outsourced kernels (bassrehab/triton-kernels) — forward-only kernels ─────────
def _ensure_bassrehab():
    """Make triton_kernels (bassrehab) importable: clone on first use, blank its __init__s to dodge
    the eager broken-import bug. Raises on failure (callers catch)."""
    import os, sys, tempfile, subprocess
    try:
        import triton_kernels.swiglu  # noqa: F401
        return
    except Exception:
        pass
    BR = os.path.join(tempfile.gettempdir(), "bassrehab_triton_kernels")
    if not os.path.isdir(BR):
        subprocess.run(["git", "clone", "--depth", "1", "-q",
                        "https://github.com/bassrehab/triton-kernels", BR], check=True)
    for p in ("triton_kernels/__init__.py", "triton_kernels/moe/__init__.py"):
        fp = os.path.join(BR, p)
        if os.path.exists(fp):
            open(fp, "w").close()
    if BR not in sys.path:
        sys.path.insert(0, BR)
    import triton_kernels.swiglu  # noqa: F401


def bench_bassrehab_swiglu(M=16384, I=1024):
    """bassrehab swiglu_fused(gate, up). Auto-detects backward: full fwd+bwd if differentiable,
    else forward-only (output match + fwd speed)."""
    try:
        _ensure_bassrehab()
        from triton_kernels.swiglu import swiglu_fused
    except Exception as ex:
        print(f"\n=== bassrehab SwiGLU ===\n  SKIPPED — couldn't import ({type(ex).__name__}: {ex}).")
        return
    gate = torch.randn(M, I, device=DEV, dtype=DTYPE)
    up = torch.randn(M, I, device=DEV, dtype=DTYPE)
    G = torch.randn(M, I, device=DEV, dtype=DTYPE)
    eager = lambda g, u: F.silu(g) * u
    ga = gate.clone().requires_grad_(True); ua = up.clone().requires_grad_(True)
    try:
        probe = swiglu_fused(ga, ua)
        has_bwd = bool(probe.requires_grad)
    except Exception as ex:
        print(f"\n=== bassrehab SwiGLU ===\n  FAILED forward: {type(ex).__name__}: {ex}")
        return
    if has_bwd:
        gb = gate.clone().requires_grad_(True); ub = up.clone().requires_grad_(True)
        probe.backward(G); eager(gb, ub).backward(G)
        gabs, grel = _gdiff([(ga.grad, gb.grad), (ua.grad, ub.grad)])
        K = (lambda g, u: swiglu_fused(g, u)); Eg = _c(eager)
        kf = _fwd_ms(lambda: K(gate, up)); ef = _fwd_ms(lambda: Eg(gate, up))
        g2 = gate.clone().requires_grad_(True); u2 = up.clone().requires_grad_(True)
        kstep = lambda: (K(g2, u2) * G).sum().backward()
        estep = lambda: (Eg(g2, u2) * G).sum().backward()
        kfb, efb = _stats(kstep, estep, [g2, u2])
        _report("bassrehab SwiGLU (M=%d I=%d)" % (M, I), kf, ef, max(kfb - kf, 0.0),
                max(efb - ef, 0.0), kfb, efb, gabs, grel, _peak(kstep), _peak(estep))
    else:
        with torch.no_grad():
            ko = swiglu_fused(gate, up); eo = eager(gate, up)
        match = (ko.float() - eo.float()).abs().max().item() / (eo.float().abs().max().item() + 1e-9)
        K = (lambda: swiglu_fused(gate, up)); Eg = _c(lambda: eager(gate, up))
        kf = _fwd_ms(K); ef = _fwd_ms(Eg)
        print(f"\n=== bassrehab SwiGLU (FORWARD-ONLY, M=%d I=%d) ===" % (M, I))
        print(f"  forward      kernel {kf:7.3f} ms | eager {ef:7.3f} ms | {ef/kf:5.2f}x")
        print(f"  output rel-max vs eager: {match:.2e}  ({'MATCH' if match < 5e-2 else 'MISMATCH'})  [no backward]")
        if JSON_OUT:
            print("@@RESULT " + json.dumps({"name": "bassrehab_swiglu_fwd_only",
                                            "fwd_x": round(ef / kf, 3), "out_rel": float(f"{match:.2e}"), "backward": False}))


def bench_bassrehab_moe(N=16384, H=512, FFN=768, E=9, top_k=2):
    """bassrehab fused_moe_forward (forward-only, standard SwiGLU, self-routing) vs compiled-eager
    standard-SwiGLU MoE forward. Forward speed + output match only (no backward in their kernel)."""
    try:
        _ensure_bassrehab()
        from triton_kernels.moe.fused_moe import fused_moe_forward
    except Exception as ex:
        print(f"\n=== bassrehab fused MoE (fwd-only) ===\n  SKIPPED — couldn't import ({type(ex).__name__}: {ex}).")
        return

    x = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    rw = torch.randn(E, H, device=DEV, dtype=DTYPE) * 0.02
    wg = torch.randn(E, FFN, H, device=DEV, dtype=DTYPE) * 0.02
    wu = torch.randn(E, FFN, H, device=DEV, dtype=DTYPE) * 0.02
    wd = torch.randn(E, H, FFN, device=DEV, dtype=DTYPE) * 0.02

    def eager(xx):
        probs = torch.softmax(xx @ rw.t(), dim=-1).float()
        tw, idx = torch.topk(probs, top_k, dim=-1)
        tw = (tw / tw.sum(-1, keepdim=True)).to(xx.dtype)
        out = torch.zeros_like(xx)
        for e in range(E):
            hit = (idx == e)
            rows = hit.any(-1)
            if not bool(rows.any()):
                continue
            we = (tw * hit).sum(-1)[rows]
            xe = xx[rows]
            h = F.silu(xe @ wg[e].t()) * (xe @ wu[e].t())
            out[rows] += (h @ wd[e].t()) * we.unsqueeze(-1)
        return out

    try:
        kfwd = lambda: fused_moe_forward(x, rw, wg, wu, wd, E, top_k, gating="softmax")[0]
        with torch.no_grad():
            ko = kfwd(); eo = eager(x)
        match = (ko.float() - eo.float()).abs().max().item() / (eo.float().abs().max().item() + 1e-9)
        K = kfwd
        Eg = _c(eager)
        kf = _fwd_ms(K)
        ef = _fwd_ms(lambda: Eg(x))
        peak_k = _peak(lambda: K()); peak_e = _peak(lambda: Eg(x))
        print(f"\n=== bassrehab fused MoE (FORWARD-ONLY, N={N} H={H} FFN={FFN} E={E} k={top_k}) ===")
        print(f"  forward      kernel {kf:7.3f} ms | eager {ef:7.3f} ms | {ef/kf:5.2f}x")
        print(f"  peak mem     kernel {peak_k:6.0f} MB | eager {peak_e:6.0f} MB | {peak_e/max(peak_k,1):5.2f}x less")
        print(f"  output rel-max vs eager: {match:.2e}  ({'MATCH' if match < 5e-2 else 'MISMATCH'})  [no backward]")
        if JSON_OUT:
            print("@@RESULT " + json.dumps({"name": "bassrehab_moe_fwd_only", "fwd_x": round(ef / kf, 3),
                                            "mem_x_less": round(peak_e / max(peak_k, 1), 3),
                                            "out_rel": float(f"{match:.2e}"), "backward": False}))
    except Exception as ex:
        print(f"\n=== bassrehab fused MoE (fwd-only) ===\n  FAILED: {type(ex).__name__}: {ex}")


BENCHES = {"swiglu": bench_swiglu, "ce": bench_ce, "xsa": bench_xsa, "conv": bench_conv_router,
           "moe": bench_moe, "liger_swiglu": bench_liger_swiglu, "liger_ce": bench_liger_ce,
           "bassrehab": bench_bassrehab_moe, "bassrehab_swiglu": bench_bassrehab_swiglu,
           "liger_ce_sweep": bench_liger_ce_sweep}

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    COMPILE = "--compile" in sys.argv
    JSON_OUT = "--json" in sys.argv
    if "--dump-triton" in sys.argv:
        COMPILE = True  # dumping is meaningless without compiled fns
    args = [a for a in sys.argv[1:] if a not in ("--compile", "--json", "--dump-triton")]
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype={DTYPE} | torch {torch.__version__} | "
          f"triton {triton.__version__} | compile={'ON' if COMPILE else 'off'}")
    # Default run = head-to-head vs compiled eager: OUR MoE/CE/SwiGLU + the OUTSOURCED reference
    # kernels (Liger SwiGLU + Liger fused-linear CE). Answers "do ANY hand-written kernels — ours OR
    # the famous external ones — beat torch.compile on this GPU?" (xsa/conv named-only.)
    which = args or ["moe", "ce", "swiglu", "liger_swiglu", "liger_ce", "liger_ce_sweep",
                     "bassrehab_swiglu", "bassrehab"]
    for name in which:
        try:
            BENCHES[name]()
        except torch.cuda.OutOfMemoryError:
            print(f"\n=== {name} ===\n  OOM at this shape (eager baseline likely materialized full logits). Try smaller.")
            torch.cuda.empty_cache()
        except Exception as ex:
            # one crashing contender must never abort the sweep
            print(f"\n=== {name} ===\n  CRASHED: {type(ex).__name__}: {str(ex).splitlines()[0] if str(ex) else ex}")
            torch.cuda.empty_cache()
