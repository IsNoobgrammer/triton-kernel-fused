"""3-phase benchmark (forward / backward / forward+backward) of every fused kernel vs
its PyTorch-eager equivalent, plus a grad-equivalence check.

    python bench.py                 # all kernels, fp16. Default shapes = BiBo's REAL training step:
                                    #   16384 tokens (B16*S1024), hidden 512, vocab 81000,
                                    #   dense-MLP I=1024, MoE I=768/E=9/k=2, attn Hq4/Hkv2/D128.
                                    # At 16384 tok the (N,V) logit matrix is ~2.65 GB fp16 — the
                                    # regime where never-materializing CE actually matters.
    python bench.py swiglu moe      # selected kernels
    python bench.py --compile       # measure vs torch.compile'd eager (industry steady-state)
    python bench.py --compile --profile moe   # ALSO print a torch.profiler kernel breakdown for MoE:
                                              # CUDA launches/iter + per-op self-CUDA time (the
                                              # fusion-opportunity map — where time goes, what's fusable)
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

from kernels import fused_linear_cross_entropy, fused_xsa
from kernels.moe import moe_per_expert, moe_grouped, moe_grouped_cublas, moe_eager

DTYPE = torch.float16
DEV = "cuda"
COMPILE = False   # set by --compile in __main__
JSON_OUT = False  # set by --json in __main__ (emits one @@RESULT line per kernel for the loop harness)
PROFILE = False   # set by --profile: emit a torch.profiler kernel breakdown (launch count + per-op CUDA time)
NO_SPECIAL = False  # set by --no-special: bench MoE with 0 special experts (E_glu all-GLU) for the A/B vs the stack


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


def _profile(label, step, iters=20, leaves=None):
    """torch.profiler kernel breakdown for an already-built `step`. Prints CUDA kernel launches/iter
    (the fusion-opportunity signal: fewer launches = more fused) + per-op self-CUDA time so you can
    SEE where time goes and what's fusable, not infer it from inductor's output. leaves: zero their
    grads each iter for a fwd+bwd step."""
    from torch.profiler import profile, ProfilerActivity

    def _dev_us(e):   # self device (CUDA) time, robust across torch versions
        return getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0.0)

    _warm(step, 3)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            if leaves:
                for lf in leaves:
                    lf.grad = None
            step()
        torch.cuda.synchronize()
    ka = prof.key_averages()
    n_launch = sum(e.count for e in ka if _dev_us(e) > 0) / iters   # device-side ops = kernel launches
    print(f"\n  --- profile: {label}  (~{n_launch:.0f} CUDA kernel launches/iter; fewer = more fused) ---")
    for key in ("self_device_time_total", "self_cuda_time_total", "cuda_time_total"):
        try:
            print(ka.table(sort_by=key, row_limit=15)); break
        except Exception:
            continue


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

    # --- our fused-fwd+bwd CE at full N (grad in forward, no recompute) at two budgets ---
    MB = 1024 * 1024
    for vname, budget in [("ce_192MB", 192 * MB), ("ce_128MB", 128 * MB)]:
        try:
            a = hid[:Nc].clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
            fused_linear_cross_entropy(a, wa, lab[:Nc], bwd_logits_budget=budget).backward()
            bb = hid[:Nc].clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
            eager_full(bb, wb, lab[:Nc]).backward()
            gabs_m, grel_m = _gdiff([(a.grad, bb.grad), (wa.grad, wb.grad)])
            del a, wa, bb, wb; torch.cuda.empty_cache()
            K = (lambda h, ww, _b=budget: fused_linear_cross_entropy(h, ww, lab, bwd_logits_budget=_b))
            kf = _fwd_ms(lambda: K(hid, w))
            h2 = hid.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
            kstep = lambda: K(h2, w2).backward()
            kfb = _step_ms(kstep, [h2, w2]); peak_k = _peak(kstep)
            if eager_ok:
                _report(f"CE {vname} (N={N} V={V})", kf, ef, max(kfb - kf, 0.0), max(efb - ef, 0.0),
                        kfb, efb, gabs_m, grel_m, peak_k, peak_e)
            else:
                print(f"\n=== CE {vname} (N={N} V={V}) — eager OOM, kernel standalone ===")
                print(f"  forward {kf:7.3f} ms | fwd+bwd {kfb:7.3f} ms | peak {peak_k:6.0f} MB "
                      f"| grad rel {grel_m:.2e} ({'PASS' if grel_m < 1.5e-2 else 'CHECK'})  [ENABLES training where eager OOMs]")
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


def bench_router_full(B=16, S=1024, H=512, E=11, K=4, top_k=2):   # BiBo conv router: 11 experts, top-2
    """WHOLE conv router (conv+sigmoid+bias-select+topk+gather+norm) as ONE fused op vs the same
    pipeline in torch.compile'd eager. The fused win is the transpose-free conv + folded-away glue
    (sigmoid/bias/gather) that compile can't pull out of cuDNN. Verifies grad_x/grad_w equivalence,
    idx agreement, and that the in-kernel count == bincount. topk/norm stay eager on both sides."""
    from kernels.router import fused_router, _count_experts
    x = torch.randn(B, S, H, device=DEV, dtype=DTYPE)
    w = torch.randn(E, H, K, device=DEV, dtype=DTYPE) * 0.02
    bias = torch.zeros(E, device=DEV, dtype=torch.float32)
    G = torch.randn(B, S, top_k, device=DEV, dtype=torch.float32)        # upstream grad on weights

    def eager(xx, ww):
        xp = F.pad(xx.transpose(1, 2), (K - 1, 0))
        logits = F.conv1d(xp, ww).transpose(1, 2).reshape(B, S, E).float()
        scores = torch.sigmoid(logits)
        sel = scores + bias
        _, idx = torch.topk(sel, top_k, dim=-1)
        wt = scores.gather(-1, idx)
        wt = wt / (wt.sum(-1, keepdim=True) + 1e-20)
        return idx, wt

    efn = _c(lambda xx, ww: eager(xx, ww)[1])      # compiled-eager baseline (cuDNN conv + fused glue)
    # SHIPPED: the 'cudnn' fused router (cuDNN conv padding=K-1 + fused top-k epilogue + merged manual
    # backward). T4 win 1.11-1.17x fwd+bwd, exact grads, mem parity. Other backends (tldot/cublas/
    # readonce/tlconv/ref) were refuted on T4 and removed — see .autoresearch/reflections.md.
    if True:
        # ── grad equivalence (Rule 1) + idx/count agreement — in fp32 (idx exact, isolates math) ──
        xf = x.float(); wf = w.float(); Gf = G.float()
        xa = xf.clone().requires_grad_(True); wa = wf.clone().requires_grad_(True)
        xb = xf.clone().requires_grad_(True); wb = wf.clone().requires_grad_(True)
        ik, wk, ck = fused_router(xa, wa, bias, top_k, E, return_counts=True)
        ie, we = eager(xb, wb)
        (wk * Gf).sum().backward(); (we * Gf).sum().backward()
        gabs, grel = _gdiff([(xa.grad, xb.grad), (wa.grad, wb.grad)])
        idx_agree = (ik.sort(-1).values == ie.sort(-1).values).float().mean().item()
        cnt_ok = bool((ck == torch.bincount(ie.reshape(-1), minlength=E).int()).all().item())
        nan = False
        for _ in range(2):
            xt = x.clone().requires_grad_(True); wt2 = w.clone().requires_grad_(True)
            _, wo = fused_router(xt, wt2, bias, top_k, E)
            (wo * G).sum().backward()
            nan |= bool(wo.isnan().any() or xt.grad.isnan().any() or wt2.grad.isnan().any())
        print(f"  [cudnn] [fp32] idx-agree {idx_agree:.4f} | count==bincount {cnt_ok} | NaN-free {not nan}")

        # ── 3-phase timing (Rule 3) vs compiled eager ──
        kfn = lambda xx, ww: fused_router(xx, ww, bias, top_k, E)[1]
        kf = _fwd_ms(lambda: kfn(x, w)); ef = _fwd_ms(lambda: efn(x, w))
        x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
        kb = _bwd_ms(lambda: (kfn(x2, w2) * G).sum(), [x2, w2])
        eb = _bwd_ms(lambda: (efn(x2, w2) * G).sum(), [x2, w2])
        kstep = lambda: (kfn(x2, w2) * G).sum().backward()
        estep = lambda: (efn(x2, w2) * G).sum().backward()
        kfb, efb = _stats(kstep, estep, [x2, w2])
        _report("FULL conv router [cudnn] (B=%d S=%d H=%d E=%d K=%d k=%d)" % (B, S, H, E, K, top_k),
                kf, ef, kb, eb, kfb, efb, gabs, grel, _peak(kstep), _peak(estep))
        if PROFILE:
            _profile("fused router [cudnn] fwd+bwd", kstep, leaves=[x2, w2])
    if PROFILE:
        x2 = x.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
        _profile("compiled-eager router fwd+bwd", lambda: (efn(x2, w2) * G).sum().backward(), leaves=[x2, w2])


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
        # grad check on a small slice — the full-N fp32 (N,V) reference OOMs a T4 by design
        Nc = min(N, 2048); labc = lab[:Nc]
        a = hid[:Nc].clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
        b = hid[:Nc].clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
        liger(a, wa).backward(); F.cross_entropy((b @ wb.t()).float(), labc).backward()
        gabs, grel = _gdiff([(a.grad, b.grad), (wa.grad, wb.grad)])
        del a, wa, b, wb; torch.cuda.empty_cache()
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
    """Liger CE swept on the SAME memory-budget grid as bench_ce_sweep (128..512MB step 64), so it
    overlays directly on ours + compiled: each budget -> Liger chunk_rows = budget//(V*2) (the same
    sizing our chunked CE uses), driven via the next_power_of_2 shim. Compact one-line-per-budget:
    fwd+bwd ms, peak MB, x-vs-compiled, x-less-mem, grad PASS/CHECK. Answers 'is Liger better than
    ours at equal memory?' on one table."""
    try:
        import liger_kernel.ops.fused_linear_cross_entropy as _flce
        from liger_kernel.ops.fused_linear_cross_entropy import LigerFusedLinearCrossEntropyFunction as LFLCE
    except Exception as ex:
        print(f"\n=== Liger CE sweep ===\n  SKIPPED — liger_kernel not installed ({ex}). `pip install liger-kernel`.")
        return
    hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    w = torch.randn(V, H, device=DEV, dtype=DTYPE) * 0.1
    lab = torch.randint(0, V, (N,), device=DEV)
    MB = 1024 * 1024
    print(f"\n(Liger CE chunk sweep: N={N} V={V} H={H}, 128..512MB step 64 — same grid as ce_sweep)")
    # compiled-eager baseline (same as ce_sweep) — best-effort
    efb = peak_e = float("nan"); eager_ok = False
    try:
        Eg = _c(lambda h, ww: F.cross_entropy((h @ ww.t()).float(), lab))
        he = hid.clone().requires_grad_(True); we = w.clone().requires_grad_(True)
        estep = lambda: Eg(he, we).backward()
        efb = _step_ms(estep, [he, we]); peak_e = _peak(estep); eager_ok = True
        print(f"  compiled baseline: fwd+bwd {efb:8.2f} ms | peak {peak_e:6.0f} MB")
        del he, we
    except torch.cuda.OutOfMemoryError:
        print("  compiled eager OOM — reporting Liger standalone ms/peak only")
    torch.cuda.empty_cache()

    class _ShimTriton:                       # delegates to real triton; pins chunk_size = `rows`
        def __init__(self, rows): self.rows = rows
        def __getattr__(self, n): return getattr(triton, n)
        def next_power_of_2(self, _x): return self.rows
    _orig = _flce.triton
    Nc = min(N, 2048)
    labc = lab[:Nc]

    def liger(h, ww, labels):
        out = LFLCE.apply(h, ww, labels)
        return out[0] if isinstance(out, (tuple, list)) else out   # Liger returns (loss, z_loss)

    # Liger's inner CE kernel uses chunk_size as a Triton BLOCK_SIZE constexpr -> it MUST be a power
    # of 2 (a non-pow2 chunk is a CompilationError). So we can't hit arbitrary MB budgets like our
    # kernel can; we sweep the pow2 chunk sizes whose (chunk,V) fp16 transient fits under the 512MB
    # ceiling ({256,512,1024,2048}; 4096 = 663MB > ceiling) and report each one's MEASURED peak, so
    # it still overlays on ce_sweep by the memory axis. This IS Liger's real memory<->speed frontier.
    for rows in (256, 512, 1024, 2048):
        approx_mb = rows * V * 2 // MB
        _flce.triton = _ShimTriton(rows)
        try:
            # grad check vs eager fp32 CE on a small slice (full-N fp32 ref OOMs by design)
            a = hid[:Nc].clone().requires_grad_(True); wa = w.clone().requires_grad_(True)
            liger(a, wa, labc).backward()
            b = hid[:Nc].clone().requires_grad_(True); wb = w.clone().requires_grad_(True)
            F.cross_entropy((b @ wb.t()).float(), labc).backward()
            _, grel = _gdiff([(a.grad, b.grad), (wa.grad, wb.grad)])
            del a, wa, b, wb; torch.cuda.empty_cache()

            h2 = hid.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
            kstep = lambda: liger(h2, w2, lab).backward()
            kfb = _step_ms(kstep, [h2, w2]); peak_k = _peak(kstep)
            lat = f"{efb / kfb:.2f}x" if eager_ok else "  n/a"
            memx = f"{peak_e / max(peak_k, 1):.2f}x less" if eager_ok else ""
            pf = "PASS" if grel < 1.5e-2 else "CHECK"
            print(f"  chunk {rows:5d} (~{approx_mb:3d}MB): fwd+bwd {kfb:8.2f} ms | peak {peak_k:6.0f} MB | "
                  f"{lat} compiled | {memx} | grad {pf}")
            del h2, w2; torch.cuda.empty_cache()
        except Exception as ex:
            print(f"  chunk {rows:5d}: FAILED {type(ex).__name__}: {str(ex).splitlines()[0]}")
            torch.cuda.empty_cache()
        finally:
            _flce.triton = _orig                 # always restore


# ───────────────────────── MoE (PolyGLU) ─────────────────────────
def bench_moe(N=16384, H=512, I=768, E_glu=9, n_special=2, top_k=2):   # BiBo STACK: 9 PolyGLU + Identity + Zero = 11 routed
    # BiBo's real routed layout: 9 GLU experts (PolyGLU groups of 3: SiLU/ReLU²/Tanh) + 2 param-free
    # specials (Identity=weighted passthrough, Zero=noop). act_codes: 0/1/2=GLU (weight slot), 3=Identity,
    # 4=Zero. The specials are nearly free AND absorb ~n_special/E of routings -> the GLU GEMMs get
    # smaller M, so the stack is slightly CHEAPER than E_glu-all-GLU at the same token budget.
    if NO_SPECIAL:
        n_special = 0
    E = E_glu + n_special
    act_codes = torch.tensor([i % 3 for i in range(E_glu)] + [3, 4][:n_special], device=DEV, dtype=torch.int32)
    hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    gup = (torch.randn(E_glu, 2 * I, H, device=DEV, dtype=DTYPE) * 0.02)   # weights only for the GLU experts
    dwn = (torch.randn(E_glu, H, I, device=DEV, dtype=DTYPE) * 0.02)
    logits = torch.randn(N, E, device=DEV, dtype=DTYPE)                   # routing over all 11
    wt_full, idx = torch.topk(torch.softmax(logits.float(), -1), top_k, dim=-1)
    wt_full = wt_full.to(DTYPE)
    print(f"\n(MoE STACK: N={N} rows*top_k={N*top_k}; H={H} I={I} E={E} routed = {E_glu} GLU + Identity + Zero; k={top_k})")
    print("  BASELINE = compiled `moe_eager` (per-expert mask + loop + weighted scatter) = the "
          "Qwen3MoE / HF compute pattern, under torch.compile. 'x' columns are vs THAT (the real bar).")

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

    # Only the per-expert path runs by default. Baseline (the "eager" column) = compiled `moe_eager`
    # = the Qwen3MoE / HF MoE compute pattern (mask + per-expert loop + index_add) under torch.compile.
    # DISABLED (T4, useless): `grouped` (tl.dot — 0.07x, the Turing tl.dot cliff) and `bassrehab`
    # (fwd-only, 0.09x AND wrong output rel 0.76). `grouped_cublas` is bf16/sm_80+ only. Re-enable
    # grouped/bassrehab by hand if benching on Ampere+.
    variants = [("per-expert", moe_per_expert)]
    for vname, vfn in variants:
        try:
            _report(f"MoE {vname} vs Qwen3MoE-eager (compiled)", *run(vfn))
        except Exception as ex:
            print(f"\n=== MoE {vname} ===\n  FAILED: {type(ex).__name__}: {ex}")

    if PROFILE:
        # Where does the time actually go on THIS GPU, and how many launches can we fuse away?
        # Profile kernel fwd, compiled-eager fwd (launch-count contrast), and kernel fwd+bwd.
        K = moe_per_expert; Eg = _c(moe_eager)
        G = torch.randn(N, H, device=DEV, dtype=DTYPE)
        with torch.no_grad():
            _profile("per-expert FORWARD", lambda: K(hid, idx, wt_full, gup, dwn, act_codes))
            _profile("compiled-eager FORWARD", lambda: Eg(hid, idx, wt_full, gup, dwn, act_codes))
        h2 = hid.clone().requires_grad_(True); g2 = gup.clone().requires_grad_(True)
        d2 = dwn.clone().requires_grad_(True); w2 = wt_full.clone().requires_grad_(True)
        _profile("per-expert FWD+BWD",
                 lambda: (K(h2, idx, w2, g2, d2, act_codes) * G).sum().backward(),
                 leaves=[h2, g2, d2, w2])


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


def bench_ce_fit():
    """CONFIG A — compiled CE FITS. BiBo's current training step (N=16384, V=81000): the (N,V) fp16
    logits = ~2.65 GB, compiled-eager peak ~3 GB on a 16 GB T4. Expectation: compile WINS (~2.3x
    faster); our chunked CE is redundant here — its only edge is ~3x less peak at the 128MB budget.
    Lesson: at today's scale, just let the compiler do CE."""
    bench_ce(N=16384, H=512, V=81000)


def bench_ce_oom():
    """CONFIG B — compiled CE does NOT fit. Long-context roadmap (N=131072 = B16*S8192, V=81000):
    the (N,V) fp16 logits alone = ~21 GB > 16 GB T4 -> compiled eager OOMs. Our chunked CE never
    materializes (N,V) (peak ~1-2 GB at the 128MB budget) -> it's the ONLY path that runs. This is
    the regime where fused-linear CE is the ENABLING kernel, not the faster one."""
    bench_ce(N=131072, H=512, V=81000)


def bench_ce_sweep(N=16384, H=512, V=81000):
    """Sweep the default FUSED-fwd+bwd CE chunk budget 128..512MB (step 64) at ce_fit. Fused = grad
    computed in forward, no recompute (vs Liger). Reports fwd+bwd ms, peak, x-vs-compiled."""
    hid = torch.randn(N, H, device=DEV, dtype=DTYPE) * 0.1
    w = torch.randn(V, H, device=DEV, dtype=DTYPE) * 0.1
    lab = torch.randint(0, V, (N,), device=DEV)
    MB = 1024 * 1024
    print(f"\n(CE chunk sweep: N={N} V={V} H={H}, fused-fwd+bwd mode, 128..512MB step 64)")
    efb = peak_e = float("nan"); eager_ok = False
    try:
        Eg = _c(lambda h, ww: F.cross_entropy((h @ ww.t()).float(), lab))
        he = hid.clone().requires_grad_(True); we = w.clone().requires_grad_(True)
        estep = lambda: Eg(he, we).backward()
        efb = _step_ms(estep, [he, we]); peak_e = _peak(estep); eager_ok = True
        print(f"  compiled baseline: fwd+bwd {efb:8.2f} ms | peak {peak_e:6.0f} MB")
        del he, we
    except torch.cuda.OutOfMemoryError:
        print("  compiled eager OOM — reporting kernel standalone ms/peak only")
    torch.cuda.empty_cache()
    for mb in range(128, 512 + 1, 64):
        try:
            K = (lambda h, ww, _b=mb * MB: fused_linear_cross_entropy(h, ww, lab, bwd_logits_budget=_b))
            h2 = hid.clone().requires_grad_(True); w2 = w.clone().requires_grad_(True)
            kstep = lambda: K(h2, w2).backward()
            kfb = _step_ms(kstep, [h2, w2]); peak_k = _peak(kstep)
            lat = f"{efb / kfb:.2f}x" if eager_ok else "  n/a"
            memx = f"{peak_e / max(peak_k, 1):.2f}x less" if eager_ok else ""
            print(f"  {mb:3d}MB: fwd+bwd {kfb:8.2f} ms | peak {peak_k:6.0f} MB | {lat} compiled | {memx}")
            del h2, w2; torch.cuda.empty_cache()
        except Exception as ex:
            print(f"  {mb:3d}MB: FAILED {type(ex).__name__}: {str(ex).splitlines()[0]}")
            torch.cuda.empty_cache()


# ───────────────────────── Muon optimizer (Polar-Express) ─────────────────────────
# Quintic Moonlight coeffs (BiBo's recipe) — used ONLY to parity-check the FUSION against BiBo's Muon.
_QUINTIC = ((3.4445, -4.7750, 2.0315),) * 5


def _import_bibo_muon():
    """Best-effort import of BiBo's trusted Muon (../BiBo/bench/optim.py) via importlib — the parity
    anchor for the fusion. Returns the class or None (T4 box may not have BiBo checked out alongside)."""
    import importlib.util
    p = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "BiBo", "bench", "optim.py"))
    if not os.path.exists(p):
        return None
    try:
        spec = importlib.util.spec_from_file_location("bibo_optim", p)
        m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
        return m.Muon
    except Exception as ex:
        print(f"  (BiBo Muon import failed: {type(ex).__name__}: {str(ex).splitlines()[0]}) — using PE reference only")
        return None


def _make_baseline_ns(ns_dtype):
    """Unfused, separate-ops Polar-Express NS (the reference) at a chosen dtype. Norm is fp32 (an fp16
    sum-of-squares overflows); the iteration GEMMs run in ns_dtype — so baseline-fp16 vs fused-fp16 is a
    fair same-precision fusion comparison. Plain fn so --compile can wrap it (compile WORKS for fp16/fp32
    on T4; it SKIPS bf16 — which is exactly why bf16 isn't a T4 baseline)."""
    from kernels.muon import _PE_COEFFS

    def baseline_ns(G, coeffs=_PE_COEFFS, eps=1e-7):
        was_2d = G.ndim == 2
        if was_2d:
            G = G.unsqueeze(0)
        X = G.float()
        X = X / (X.norm(dim=(-2, -1), keepdim=True) + eps)
        transposed = X.size(-2) > X.size(-1)
        if transposed:
            X = X.mT
        X = X.to(ns_dtype)
        for a, b, c in coeffs:
            A = X @ X.mT
            B = b * A + c * (A @ A)
            X = a * X + B @ X
        X = X.float()
        if transposed:
            X = X.mT
        if was_2d:
            X = X.squeeze(0)
        return X.to(G.dtype)

    return baseline_ns


class _BaselineMuon(torch.optim.Optimizer):
    """Reference Muon (PE coeffs, Jordan scale) — unfused per-param baseline. `compute_dtype` is the
    grad+momentum dtype (matches FusedMuon's ns_dtype so baseline-mixed vs fused-mixed differs ONLY in
    foreach+baddbmm); `ns_fn` runs the NS at the chosen precision. Master params keep their own dtype."""

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0,
                 ns_fn=None, compute_dtype=torch.float32):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay))
        self.ns_fn = ns_fn
        self.compute_dtype = compute_dtype

    @torch.no_grad()
    def step(self):
        for grp in self.param_groups:
            lr, mom, wd = grp["lr"], grp["momentum"], grp["weight_decay"]
            for p in grp["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                g = p.grad.to(self.compute_dtype)
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]; buf.mul_(mom).add_(g)
                u = g.add(buf, alpha=mom) if grp["nesterov"] else buf
                u = self.ns_fn(u)
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(u.to(p.dtype), alpha=-lr * scale)


def _muon_shapes(layers=6, H=512, I=768, E=9):
    sh = []
    for _ in range(layers):
        sh += [(H, H), (H // 2, H), (H // 2, H), (H, H)]      # q, k(GQA), v(GQA), o
        sh += [(2 * I, H), (H, I)]                             # dense MLP gate_up, down
        sh += [(E, 2 * I, H), (E, H, I)]                       # 3D stacked experts
    return sh


# Muon master weights are fp32 (the realistic mixed-precision setup): params/grads fp32, NS in fp16.
_MASTER = torch.float32


def _muon_params(shapes, seed=0, dtype=_MASTER):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return [torch.randn(*s, generator=g, device=DEV, dtype=dtype) for s in shapes]


def _opt_ms(make_opt, shapes):
    """Time one optimizer step alone. Grads are set ONCE (the step only READS p.grad, never consumes
    it) so randn generation stays OUT of the timed region — otherwise `normal_` pollutes the numbers."""
    params = _muon_params(shapes, 0)
    opt = make_opt(params)
    gg = torch.Generator(device=DEV).manual_seed(1)
    for p in params:
        p.grad = torch.randn(*p.shape, generator=gg, device=DEV, dtype=_MASTER)
    opt.step()                                                 # materialize state, warm autotune/compile
    _warm(opt.step)
    return do_bench(opt.step)


def _opt_peak(make_opt, shapes):
    """Peak CUDA memory of one optimizer step (params + grads + momentum state + NS transients)."""
    params = _muon_params(shapes, 0)
    opt = make_opt(params)
    gg = torch.Generator(device=DEV).manual_seed(1)

    def prime():
        for p in params:
            p.grad = torch.randn(*p.shape, generator=gg, device=DEV, dtype=_MASTER)
    prime(); opt.step()                                        # materialize momentum + warm
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    prime(); opt.step(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    del params, opt; torch.cuda.empty_cache()
    return peak


def _muon_parity(make_opt, make_ref, shapes, steps=5):
    """max|Δp| between reference and candidate after `steps` identical steps from identical init+grads."""
    pr, pc = _muon_params(shapes, 0), _muon_params(shapes, 0)
    oref, ocand = make_ref(pr), make_opt(pc)
    gg = torch.Generator(device=DEV).manual_seed(7)
    worst = 0.0
    for _ in range(steps):
        grads = [torch.randn(*p.shape, generator=gg, device=DEV, dtype=_MASTER) for p in pr]
        for p, gr in zip(pr, grads): p.grad = gr.clone()
        for p, gr in zip(pc, grads): p.grad = gr.clone()
        oref.step(); ocand.step()
        worst = max(worst, max((a.float() - b.float()).abs().max().item() for a, b in zip(pr, pc)))
    return worst


def bench_muon(layers=6):
    """Polar-Express Muon: fused (batched-state same-shape NS) vs the baseline, with a hard parity gate
    FIRST. Single-GPU always; distributed A (replicated) vs B (round-robin) under torchrun. `muon` =
    6 layers (~75M); `muon_big` = 24 layers (~300M) where the batched GEMMs dominate and the per-param
    CPU launch overhead amortizes (the regime that matters for real training).
    """
    from kernels.muon import FusedMuon, DistributedMuon, newton_schulz
    shapes = _muon_shapes(layers=layers)
    nparam = sum(int(torch.tensor(s).prod()) for s in shapes)
    print(f"\n=== Muon (Polar-Express) — {len(shapes)} tensors, {nparam/1e6:.1f}M params ===")
    print("  master weights fp32, NS in fp16 (mixed) — the realistic T4 setup. bf16 omitted: no bf16")
    print("  tensor cores on sm_75 AND torch.compile skips bf16 there (0.97x loser). Parity ref = FULL fp32 Muon.")
    WD, LR = 0.1, 0.02
    ns32, ns16 = _make_baseline_ns(torch.float32), _make_baseline_ns(torch.float16)
    # PARITY REFERENCE = full fp32 Muon (everything fp32 — the mathematical truth).
    ref = lambda ps: _BaselineMuon(ps, lr=LR, weight_decay=WD, ns_fn=ns32, compute_dtype=torch.float32)

    # ── PHASE 1: PARITY GATE (correctness before speed; all vs the full-fp32 Muon) ──
    BiBoMuon = _import_bibo_muon()
    if BiBoMuon is not None:
        d = _muon_parity(
            lambda ps: FusedMuon(ps, lr=3e-4, weight_decay=WD, coeffs=_QUINTIC,
                                 ns_dtype=torch.float32, scale_mode="moonlight"),
            lambda ps: BiBoMuon(ps, lr=3e-4, momentum=0.95, weight_decay=WD), shapes)
        print(f"  parity  fused(quintic,fp32) vs BiBo Muon   = {d:.2e}  {'PASS' if d < 1e-4 else 'FAIL'}  (trusted anchor)")
    else:
        print("  (BiBo Muon unavailable — checkout ../BiBo next to repo for the trusted anchor)")
    # FUSION correctness: fused-fp32 vs the full-fp32 reference (isolates foreach+baddbmm; fp32 params -> ~1e-6)
    df32 = _muon_parity(lambda ps: FusedMuon(ps, lr=LR, weight_decay=WD, ns_dtype=torch.float32), ref, shapes)
    print(f"  parity  fused-fp32 vs full-fp32 Muon       = {df32:.2e}  {'PASS' if df32 < 1e-4 else 'FAIL'}  (isolates the fusion)")
    # PRECISION fidelity: mixed (fp16 NS) vs the full-fp32 truth — informational (different op, not bit-parity)
    dmix = _muon_parity(lambda ps: FusedMuon(ps, lr=LR, weight_decay=WD, ns_dtype=torch.float16), ref, shapes)
    print(f"  parity  fused-mixed(fp16NS) vs full-fp32    = {dmix:.2e}  (fp16-NS precision diff, informational)")
    # fp16-NS stability (T4 path): SV mean ~1, NaN-free
    ok = True
    for s in [(512, 512), (256, 512), (1536, 512), (9, 1536, 512), (9, 512, 768)]:
        Y = newton_schulz(torch.randn(*s, device=DEV, dtype=torch.float32), ns_dtype=torch.float16).float()
        sv = torch.linalg.svdvals(Y if Y.ndim == 3 else Y.unsqueeze(0))
        ok &= (not bool(Y.isnan().any())) and (0.80 <= sv.mean().item() <= 1.20)
    print(f"  fp16-NS stability (SV~1, NaN-free)         = {'PASS' if ok else 'FAIL'}")

    # ── PHASE 2: SINGLE-GPU — ms / speedup / peak mem / parity. Speedup baseline = MIXED (fp16 NS),
    # our kernel = same MIXED, so the speedup is purely the fusion. |Δp| is vs the full-fp32 truth. ──
    # NS is called with ~6 distinct shapes (+ internal transposes); the default recompile_limit (8) is
    # hit -> the compiled baseline falls back to EAGER for the overflow shapes (an unfairly slow bar).
    # Bump it so EVERY shape gets its own compiled kernel — the strongest, fairest baseline.
    import torch._dynamo as _dyn                                # aliased: don't rebind the local `torch`
    for _attr in ("recompile_limit", "cache_size_limit"):
        if hasattr(_dyn.config, _attr):
            setattr(_dyn.config, _attr, 64)
    variants = [
        ("full-fp32",      lambda ps: _BaselineMuon(ps, lr=LR, weight_decay=WD, ns_fn=_c(ns32), compute_dtype=torch.float32)),
        ("baseline-mixed", lambda ps: _BaselineMuon(ps, lr=LR, weight_decay=WD, ns_fn=_c(ns16), compute_dtype=torch.float16)),
        ("fused-mixed",    lambda ps: FusedMuon(ps, lr=LR, weight_decay=WD, ns_dtype=torch.float16)),
    ]
    res = {}
    for name, mk in variants:
        d = 0.0 if name == "full-fp32" else _muon_parity(mk, ref, shapes)
        res[name] = (_opt_ms(mk, shapes), _opt_peak(mk, shapes), d)
    den = res["baseline-mixed"][0]                              # speedup is vs the mixed baseline
    print(f"\n  single-GPU step ({'compiled ' if COMPILE else ''}baseline-mixed = 1.00x):")
    print(f"    {'variant':16s} {'ms':>9s} {'vs mixed':>9s} {'peak MB':>9s} {'|Δp| vs fp32':>14s}")
    for name, (t, mem, d) in res.items():
        tag = "  <- T4 path" if name == "fused-mixed" else ("  (parity ref)" if name == "full-fp32" else "")
        print(f"    {name:16s} {t:9.2f} {den/t:8.2f}x {mem:9.0f} {d:14.2e}{tag}")
    print(f"    => fusion-only win (mixed vs mixed) = {den/res['fused-mixed'][0]:.2f}x")

    # --profile: launch-count + per-op CUDA time for fused vs compiled-baseline step (the fusion signal —
    # foreach + baddbmm should issue FEWER launches than the unfused per-param baseline).
    if PROFILE:
        def _mk_step(mk):
            ps = _muon_params(shapes, 0); opt = mk(ps)
            gg = torch.Generator(device=DEV).manual_seed(1)
            def prime():
                for p in ps:
                    p.grad = torch.randn(*p.shape, generator=gg, device=DEV, dtype=_MASTER)
            prime(); opt.step()
            return lambda: (prime(), opt.step())
        _profile("fused-mixed step",
                 _mk_step(lambda ps: FusedMuon(ps, lr=LR, weight_decay=WD, ns_dtype=torch.float16)))
        _profile(f"{'compiled ' if COMPILE else ''}baseline-mixed step",
                 _mk_step(lambda ps: _BaselineMuon(ps, lr=LR, weight_decay=WD, ns_fn=_c(ns16), compute_dtype=torch.float16)))

    # ── PHASE 3: DISTRIBUTED A (replicated) vs B (round-robin) — only under torchrun ──
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        rank, ws = dist.get_rank(), dist.get_world_size()
        # exactness: B must give bit-identical weights to A (same grads -> same weights, work just relocated)
        pa, pb = _muon_params(shapes, 0), _muon_params(shapes, 0)
        oa = FusedMuon(pa, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16)
        ob = DistributedMuon(pb, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16)
        gg = torch.Generator(device=DEV).manual_seed(11)
        worst = 0.0
        for _ in range(3):
            grads = [torch.randn(*p.shape, generator=gg, device=DEV, dtype=_MASTER) for p in pa]
            for p, gr in zip(pa, grads): p.grad = gr.clone()    # DDP would all-reduce; same grads here
            for p, gr in zip(pb, grads): p.grad = gr.clone()
            oa.step(); ob.step()
            worst = max(worst, max((x.float() - y.float()).abs().max().item() for x, y in zip(pa, pb)))
        ta = _opt_ms(lambda ps: FusedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16), shapes)
        tbd = _opt_ms(lambda ps: DistributedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16), shapes)
        if rank == 0:
            print(f"\n  distributed (world_size={ws}):")
            print(f"    exactness  B vs A weights = {worst:.2e}  {'PASS' if worst < 1e-3 else 'FAIL'}  (must be exact)")
            print(f"    A replicated   {ta:7.3f} ms/rank  (1.00x)")
            print(f"    B round-robin  {tbd:7.3f} ms/rank  ({ta/tbd:.2f}x)   [each rank does ~1/{ws} of the NS]")
    else:
        print("\n  (distributed A/B skipped — run `torchrun --nproc_per_node=2 bench.py --compile muon` on 2x T4)")


BENCHES = {"ce": bench_ce, "ce_fit": bench_ce_fit, "ce_oom": bench_ce_oom,
           "muon": bench_muon, "muon_big": lambda: bench_muon(layers=24),
           "ce_sweep": bench_ce_sweep,
           "xsa": bench_xsa, "router": bench_router_full,
           "moe": bench_moe, "liger_swiglu": bench_liger_swiglu, "liger_ce": bench_liger_ce,
           "bassrehab": bench_bassrehab_moe, "bassrehab_swiglu": bench_bassrehab_swiglu,
           "liger_ce_sweep": bench_liger_ce_sweep}

if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    # Under `torchrun --nproc_per_node=N`: init NCCL + pin this rank's GPU (for the muon distributed A/B).
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        DEV = f"cuda:{os.environ['LOCAL_RANK']}"
        if not dist.is_initialized():
            dist.init_process_group("nccl")
    COMPILE = "--compile" in sys.argv
    JSON_OUT = "--json" in sys.argv
    PROFILE = "--profile" in sys.argv
    NO_SPECIAL = "--no-special" in sys.argv
    if "--dump-triton" in sys.argv:
        COMPILE = True  # dumping is meaningless without compiled fns
    args = [a for a in sys.argv[1:] if a not in ("--compile", "--json", "--dump-triton", "--profile", "--no-special")]
    print(f"GPU: {torch.cuda.get_device_name(0)} | dtype={DTYPE} | torch {torch.__version__} | "
          f"triton {triton.__version__} | compile={'ON' if COMPILE else 'off'}")
    # Default run = head-to-head vs compiled eager: OUR MoE/CE/SwiGLU + the OUTSOURCED reference
    # kernels (Liger SwiGLU + Liger fused-linear CE). Answers "do ANY hand-written kernels — ours OR
    # the famous external ones — beat torch.compile on this GPU?" (xsa/conv named-only.)
    which = args or ["moe", "ce", "liger_swiglu", "liger_ce", "liger_ce_sweep",
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
