"""All-active MoE (top_k == E, the L0 ensemble 0:8:8:576): the DENSE path must be at least as close
to an fp32 reference (per-expert loop, autocast off) as the sparse bf16 path it replaces, on the
output and every gradient; bitwise repeatable; zero host syncs. Then fwd+bwd timing.

    python parity_check/parity_moe_dense.py
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import importlib
M = importlib.import_module("kernels.sm75.moe")   # the package re-exports a `moe` function over the module name

dev = "cuda"
N, H, E, I = 65536, 512, 8, 576
codes = torch.full((E,), 8, device=dev, dtype=torch.int64)   # one buffer, as in the model: its host
                                                             # copy is cached per tensor


def inputs(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    hidden = torch.randn(N, H, device=dev, generator=g)
    scores = torch.randn(N, E, device=dev, generator=g)
    wt, idx = scores.softmax(-1).topk(E, dim=-1)
    gu = torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5
    dn = torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5
    ap = torch.randn(E, device=dev, generator=g) * 0.3
    go = torch.randn(N, H, device=dev, generator=g)
    return hidden, idx, wt, gu, dn, ap, go


def run(mode, seed=0):
    hidden, idx, wt, gu, dn, ap, go = inputs(seed)
    leaves = [t.requires_grad_() for t in (hidden, wt, gu, dn, ap)]
    M.DENSE_ALL_ACTIVE = mode == "dense"
    os.environ["BIBO_MOE_FORCE_LOOP"] = "1" if mode == "ref" else "0"
    if mode == "ref":
        out = M.moe_per_expert(hidden, idx, wt, gu, dn, codes, ap)
    else:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = M.moe_per_expert(hidden, idx, wt, gu, dn, codes, ap)
    path = M._LAST_PATH
    out.float().backward(go.to(out.dtype))
    res = {"out": out.detach().float()}
    for n, t in zip(("d_hidden", "d_wt", "d_gate_up", "d_down", "d_theta"), leaves):
        res[n] = t.grad.detach().float()
    return res, path


def syncs(mode):
    n = [0]
    import traceback

    def _w(msg, *a, **k):
        if "called a synchronizing" in str(msg):
            n[0] += 1
            fr = [x for x in traceback.extract_stack()[:-1] if "kernels" in x.filename or "parity" in x.filename]
            print(f"      sync ({mode}): " + " <- ".join(f"{x.filename.split('/')[-1]}:{x.lineno} {x.line}" for x in fr[-2:][::-1])[:220])
    warnings.showwarning = _w
    warnings.simplefilter("always")
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    run(mode)
    torch.cuda.set_sync_debug_mode(0)
    return n[0]


ref, p0 = run("ref")
sp, p1 = run("sparse")
de, p2 = run("dense")
de2, _ = run("dense")
print(f"paths: ref={p0} sparse={p1} dense={p2}")
ok = p2 == "dense" and p1 != "dense"
for k in ref:
    rn = ref[k].norm().item() or 1.0
    es = (sp[k] - ref[k]).norm().item() / rn
    ed = (de[k] - ref[k]).norm().item() / rn
    rep = torch.equal(de[k], de2[k])
    good = rep and ed <= max(es * 1.05, 1e-6)
    ok &= good
    print(f"   {k:10s} rel err vs fp32: sparse {es:.2e}  dense {ed:.2e}  repeat {'bitwise' if rep else 'DIFFERS'}"
          f"  {'OK' if good else 'WORSE'}")
ns, nd = syncs("sparse"), syncs("dense")
print(f"   host syncs per fwd+bwd: sparse {ns}  dense {nd}")
ok &= nd == 0
for mode in ("sparse", "dense"):
    for _ in range(3):
        run(mode)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(10):
        run(mode)
    b.record()
    torch.cuda.synchronize()
    print(f"   {mode:6s} fwd+bwd (incl. input setup) {a.elapsed_time(b) / 10:.2f} ms")
print("DENSE PARITY PASS" if ok else "DENSE PARITY FAIL")
