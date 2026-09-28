"""Factored-softmax CE (kernels/sm120/ce_factored.py) vs the previous fused kernel, torch.compile and
Liger. fwd+bwd, fp32 master W under bf16 autocast (the training setup). Dummy shapes only.

    python -m bench.bench_ce_factored parity|val|mtp|scale [...]
"""
import statistics
import sys

import torch
import torch.nn.functional as F

from kernels.sm120.cross_entropy import fused_linear_cross_entropy as prev_ce
from kernels.sm120.ce_factored import fused_linear_cross_entropy as fact_ce, fused_linear_cross_entropy_heads

dev = "cuda"
AMP = torch.autocast("cuda", dtype=torch.bfloat16)


def timed(fn, it=10, warm=3):
    """median ms of fn() (fwd+bwd inside), peak extra GB, last result"""
    torch.cuda.synchronize()
    ts = []
    for k in range(warm + it):
        if k == warm:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        out = fn()
        e.record()
        torch.cuda.synchronize()
        if k >= warm:
            ts.append(s.elapsed_time(e))
    return statistics.median(ts), (torch.cuda.max_memory_allocated() - base) / 2 ** 30, out


def make(N, H, V, wscale=0.05, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    h = torch.randn(N, H, device=dev, generator=g).to(torch.bfloat16).requires_grad_(True)
    W = (torch.randn(V, H, device=dev, generator=g) * wscale / (H / 512) ** 0.5).requires_grad_(True)
    y = torch.randint(0, V, (N,), device=dev, generator=g)
    y[::101] = -100
    return h, W, y


def step(ce, h, W, y, **kw):
    def f():
        h.grad = W.grad = None
        with AMP:
            loss = ce(h, W, y, **kw)
        loss.backward()
        return loss.detach(), h.grad, W.grad
    return f


def liger_fn(chunk_mem_const):
    from liger_kernel.transformers.functional import liger_fused_linear_cross_entropy as L

    def ce(h, W, y):
        # Liger does not autocast: hand it the same bf16 operands ours see; fp32 grad accumulation
        return L(h, W.to(torch.bfloat16), y, ignore_index=-100, accum_dtype=torch.float32,
                 chunk_mem_const=chunk_mem_const)
    return ce


_compiled = None


def compiled_ce(h, W, y):
    global _compiled
    if _compiled is None:
        _compiled = torch.compile(lambda h, W, y: F.cross_entropy(F.linear(h, W).float(), y, ignore_index=-100))
    return _compiled(h, W, y)


def ground_truth(h, W, y):
    with torch.no_grad():
        hf, Wf = h.float(), W.to(torch.bfloat16).float()
        lg = hf @ Wf.t()
        ok = y != -100
        lse = torch.logsumexp(lg, 1)
        loss = (lse - lg.gather(1, torch.where(ok, y, 0)[:, None]).squeeze(1))[ok].mean()
        lg.sub_(lse[:, None]).exp_()
        lg[ok, y[ok]] -= 1
        lg[~ok] = 0
        lg /= ok.sum()
        gh, gw = lg @ Wf, lg.t() @ hf
        del lg
    return loss, gh, gw


def rel(a, r):
    return ((a.float() - r).norm() / r.norm()).item()


def parity():
    N, H, V = 32768, 512, 81920
    for wscale, tag in ((0.05, "normal logits (max ~6)"), (0.3, "large logits (max ~40)"),
                        (0.6, "out-of-window logits (max ~80) -> fixup path")):
        h, W, y = make(N, H, V, wscale)
        gt = ground_truth(h, W, y)
        # grads are taken wrt the bf16 operands everyone sees, so GT uses bf16(W) too
        print(f"\n== {tag}   N={N} H={H} V={V}   (errors are vs fp32 ground truth)", flush=True)
        arms = [("prev fused 1GB (current)", step(prev_ce, h, W, y)),
                ("prev fused 4GB", step(prev_ce, h, W, y, bwd_logits_budget=4 << 30)),
                ("factored 1GB", step(fact_ce, h, W, y)),
                ("factored 4GB", step(fact_ce, h, W, y, bwd_logits_budget=4 << 30))]
        if wscale == 0.05:
            arms += [("torch.compile", step(compiled_ce, h, W, y)),
                     ("liger default", step(liger_fn(1), h, W, y)),
                     ("liger unlimited (1 chunk)", step(liger_fn(1 << 30), h, W, y))]
        res = {}
        for nm, f in arms:
            try:
                ms, gb, (l, gh, gw) = timed(f)
                l2, gh2, gw2 = f()
                det = torch.equal(gh, gh2) and torch.equal(gw, gw2)
                res[nm] = ms
                print(f"  {nm:28s} {ms:7.2f} ms  peak +{gb:5.1f} GB | loss err {abs(l.item() - gt[0].item()):.1e} "
                      f"gh {rel(gh, gt[1]):.2e} gw {rel(gw, gt[2]):.2e} | rerun-bitwise {det}", flush=True)
            except Exception as ex:
                print(f"  {nm:28s} FAILED {repr(ex)[:200]}", flush=True)
            torch.cuda.empty_cache()
        if "liger unlimited (1 chunk)" in res:
            b = res["liger unlimited (1 chunk)"]
            print("  speed vs liger unlimited: " + ", ".join(f"{k} {b / v:.2f}x" for k, v in res.items()))
        del h, W, y, gt
        torch.cuda.empty_cache()


def fixup():
    """the out-of-window pass must be EXERCISED, not just present: force it onto every row"""
    import kernels.sm120.ce_factored as cf
    N, H, V = 32768, 512, 81920
    for wscale in (0.05, 0.6):
        h, W, y = make(N, H, V, wscale)
        gt = ground_truth(h, W, y)
        with torch.no_grad(), AMP:
            mx = (h @ W.t()).float().amax(1)
        n_out = int(((mx < cf._LO) | (mx > cf._HI)).sum())
        base = step(fact_ce, h, W, y)()
        base = [t.clone() for t in base]
        lo, hi = cf._LO, cf._HI
        cf._LO, cf._HI = float("inf"), float("-inf")          # every row out of window
        try:
            ms, _, (l, gh, gw) = timed(step(fact_ce, h, W, y), it=5)
        finally:
            cf._LO, cf._HI = lo, hi
        print(f"  wscale {wscale}: {n_out}/{N} rows naturally out of window | ALL rows forced: {ms:.2f} ms, "
              f"loss err {abs(l.item() - gt[0].item()):.1e} gh {rel(gh, gt[1]):.2e} gw {rel(gw, gt[2]):.2e} | "
              f"vs normal path: gh {rel(gh, base[1].float()):.1e} gw {rel(gw, base[2]):.1e}", flush=True)
        del h, W, y, gt
        torch.cuda.empty_cache()


def val():
    N, H, V = 32768, 512, 81920
    h, W, y = make(N, H, V)
    with torch.no_grad():
        for nm, ce in (("prev fused", prev_ce), ("factored val path", fact_ce)):
            def f():
                with AMP:
                    return ce(h, W, y)
            ms, gb, l = timed(f)
            print(f"  val {nm:20s} {ms:7.2f} ms  peak +{gb:5.2f} GB  loss {l.item():.6f}", flush=True)


def mtp(max_heads=8):
    N, H, V = 32768, 512, 81920
    h, W, y = make(N, H, V)
    hs = [h] + [torch.randn_like(h).requires_grad_(True) for _ in range(max_heads - 1)]
    ys = [y] + [torch.randint(0, V, (N,), device=dev) for _ in range(max_heads - 1)]
    print(f"== MTP heads, {N} tokens/head, H={H} V={V}, weights 1 + 0.3 per extra head", flush=True)
    for k in range(1, max_heads + 1):
        ws = [1.0] + [0.3] * (k - 1)

        def old():
            W.grad = None
            for t in hs[:k]:
                t.grad = None
            with AMP:
                loss = sum(w * prev_ce(t, W, yy) for w, t, yy in zip(ws, hs[:k], ys[:k]))
            loss.backward()

        def new(budget=None):
            def f():
                W.grad = None
                for t in hs[:k]:
                    t.grad = None
                with AMP:
                    loss, _ = fused_linear_cross_entropy_heads(hs[:k], W, ys[:k], ws, bwd_logits_budget=budget)
                loss.backward()
            return f
        try:
            to, go, _ = timed(old, it=5)
            tn, gn, _ = timed(new(), it=5)
            t4, g4, _ = timed(new(4 << 30), it=5)
            print(f"  heads {k}: prev {to:7.2f} ms ({to / k:5.2f}/head) | factored 1GB {tn:7.2f} ({tn / k:5.2f}/head) "
                  f"| factored 4GB {t4:7.2f} ({t4 / k:5.2f}/head)  -> {to / t4:.2f}x", flush=True)
        except torch.OutOfMemoryError:
            print(f"  heads {k}: OOM", flush=True)
            break
        torch.cuda.empty_cache()


def scale():
    base = dict(N=32768, H=512, V=81920)
    axes = [("V", [32768, 81920, 131072, 262144]),
            ("H", [512, 1024, 2048, 4096, 8192, 16384]),
            ("N", [8192, 32768, 131072, 524288])]
    for big in (False, True):
        b = dict(base) if not big else dict(N=32768, H=4096, V=262144)
        print(f"\n==== around {b}", flush=True)
        for ax, vals in axes:
            for v in vals:
                s = dict(b, **{ax: v})
                if big and ax == "N" and v > 131072:
                    continue
                try:
                    h, W, y = make(s["N"], s["H"], s["V"])
                    out = []
                    for nm, ce in (("prev", prev_ce), ("fact", fact_ce)):
                        for bud in (1 << 30, 4 << 30):
                            ms, gb, _ = timed(step(ce, h, W, y, bwd_logits_budget=bud), it=3, warm=2)
                            out.append((nm, bud >> 30, ms, gb))
                    fl = 6 * s["N"] * s["H"] * s["V"] / 1e12
                    best_p = min(o[2] for o in out if o[0] == "prev")
                    best_f = min(o[2] for o in out if o[0] == "fact")
                    print(f"  {ax}={v:<7d} " + " | ".join(f"{n}{g}G {m:8.2f}ms" for n, g, m, _ in out)
                          + f" | speedup {best_p / best_f:.2f}x, fact {fl / best_f * 1e3:.0f} TF", flush=True)
                    del h, W, y
                except torch.OutOfMemoryError:
                    print(f"  {ax}={v}: OOM", flush=True)
                torch.cuda.empty_cache()


if __name__ == "__main__":
    for part in sys.argv[1:] or ["parity", "val", "mtp", "scale"]:
        print(f"\n######## {part}", flush=True)
        globals()[part]()
    print("CEF_DONE")
