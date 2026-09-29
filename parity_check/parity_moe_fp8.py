"""MXFP8 MoE (kernels.sm120.moe_fp8) vs eager fp32, next to the bf16 path vs eager fp32.

    python -m parity_check.parity_moe_fp8 [--N 16384] [--time]

The bar is the bf16 path's own distance from fp32: fp8 is expected to be ~10x further (e4m3 has 3
mantissa bits vs bf16's 8), and every output / grad is reported as that ratio too. Radial experts
with theta ~ U(-2, 2) and gate_up scaled so the gate rms r ~ 3 -- random unit weights give r ~ 1,
which makes r^p ~ 1 for ANY p and hides a wrong or ignored theta (radial-parity lesson).
Also: bitwise determinism of two fp8 runs, per-operand flush/saturation, and fwd+bwd timing.
"""
import argparse
import importlib
import statistics

import torch

K75 = importlib.import_module("kernels.sm75.moe")
F8 = importlib.import_module("kernels.sm120.moe_fp8")
dev = "cuda"


def make(N, H=512, E=64, k=6, I=768, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    x = torch.randn(N, H, device=dev, generator=g)
    logits = torch.randn(N, E, device=dev, generator=g) + 0.35 * torch.randn(E, device=dev, generator=g)
    w, idx = torch.softmax(logits, -1).topk(k, -1)
    w = (w / w.sum(-1, keepdim=True)).float()
    gu = torch.randn(E, 2 * I, H, device=dev, generator=g) * H ** -0.5
    gu[:, :I] *= 3.0                                         # gate rms r ~ 3, so r^p depends on theta
    dn = torch.randn(E, H, I, device=dev, generator=g) * I ** -0.5
    theta = (torch.rand(E, device=dev, generator=g) * 4 - 2)
    gy = torch.randn(N, H, device=dev, generator=g)
    codes = torch.full((E,), 8, device=dev, dtype=torch.int32)
    P = lambda t: torch.nn.Parameter(t.clone())
    return x, idx, w, P(gu), P(dn), P(theta), codes, gy


def run(path, x0, idx, w0, gu, dn, th, codes, gy):
    for p in (gu, dn, th):
        p.grad = None
    x = x0.clone().requires_grad_(True)
    w = w0.clone().requires_grad_(True)
    if path == "fp32":
        y = K75.moe_eager(x, idx, w, gu, dn, codes, act_params=th[:, None])
    else:
        fn = K75.moe_per_expert if path == "bf16" else F8.moe_fp8
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = fn(x.bfloat16(), idx, w, gu, dn, codes, act_params=th)
    (y.float() * gy).sum().backward()
    return {"y": y.detach().float(), "d_x": x.grad.float(), "d_w (router wts)": w.grad.float(),
            "d_gate_up": gu.grad.float().clone(), "d_down": dn.grad.float().clone(),
            "d_theta": th.grad.float().clone()}


def timed(fn, it=10, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(it):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e))
    return statistics.median(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=16384)
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--E", type=int, default=64)
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--I", type=int, default=768)
    a = ap.parse_args()
    args = make(a.N, E=a.E, k=a.k, I=a.I)
    ref = run("fp32", *args)
    bf = run("bf16", *args)
    F8.STATS = {}
    f8 = run("fp8", *args)
    stats, F8.STATS = F8.STATS, None
    f8b = run("fp8", *args)
    rel = lambda u, v: ((u - v).norm() / v.norm()).item()
    print(f"N={a.N} tokens, E={a.E} top-{a.k}, I={a.I}, radial theta U(-2,2), gate rms ~3")
    print(f"{'tensor':18s} {'bf16 vs fp32':>13s} {'fp8 vs fp32':>12s} {'fp8 / bf16':>11s}")
    ok = True
    for kk in ref:
        eb, e8 = rel(bf[kk], ref[kk]), rel(f8[kk], ref[kk])
        # gate: fp8 error within 20x of the bf16 path's own error (e4m3 has 3 mantissa bits vs bf16's 7:
        # ~10-16x is the expected band; an absolute bar fails L0's dtheta, where bf16 itself is 2x worse)
        ok &= e8 < 20 * max(eb, 1e-6)
        print(f"{kk:18s} {eb:13.2e} {e8:12.2e} {e8 / max(eb, 1e-12):10.1f}x")
    det = all(torch.equal(f8[kk], f8b[kk]) for kk in f8)
    print(f"fp8 two runs bitwise identical: {det}")
    # leak: every call caches tile maps; 40 calls (80 inserts) must stay under the 64-entry cap with flat
    # allocated memory (the pre-fill in _prep once bypassed eviction: +6.8 MB/step in training)
    small = make(4096, E=a.E, k=a.k, I=a.I)
    mem = []
    for _ in range(40):
        run("fp8", *small)
        torch.cuda.synchronize(); mem.append(torch.cuda.memory_allocated())
    MX = importlib.import_module("kernels.sm120.mxfp8")
    leak_ok = len(MX._TM_CACHE) <= 64 and mem[-1] - mem[9] < 2 ** 20
    print(f"leak: tile-map cache {len(MX._TM_CACHE)} entries (cap 64), allocated drift calls 10->40 "
          f"{(mem[-1] - mem[9]) / 2 ** 20:+.2f} MiB -> {'ok' if leak_ok else 'LEAK'}")
    ok &= leak_ok
    print("fp8 operands: flushed-to-0 % / saturated % (mean over calls)")
    for tag, v in stats.items():
        print(f"   {tag:18s} {statistics.mean(x[0] for x in v):8.4f} / {statistics.mean(x[1] for x in v):.4f}")
    if a.time:
        big = make(65536)

        def step(path):
            def f():
                run(path, *big)
            return f
        tb, t8 = timed(step("bf16")), timed(step("fp8"))
        print(f"fwd+bwd at N=65536 (incl. autograd glue): bf16 {tb:.2f} ms | fp8 {t8:.2f} ms | {tb / t8:.2f}x")
    print("MOE_FP8_PARITY", "PASS" if ok and det else "FAIL")


if __name__ == "__main__":
    main()
