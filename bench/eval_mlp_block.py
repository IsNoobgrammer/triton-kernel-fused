"""FROZEN EVAL for the MoE MLP-block megakernel. Do not edit to make a number look better.

    python -m bench.eval_mlp_block                 # baseline only
    python -m bench.eval_mlp_block --candidate     # baseline + megakernel

Scores a whole-block implementation. The objective is the LAP, not the sector: a candidate that
speeds up norm+router while slowing the expert GEMMs is a loss if the block total does not improve.

Per shape it reports:
    fwd_ms            forward only
    fwdbwd_ms         forward + backward through real autograd  (PRIMARY)
    peak_gb           peak allocated during fwd+bwd
    out_err           max |block output - fp64 reference|
    flips             tokens routed to a different expert than fp64 would choose

BASELINE is the stack the model actually runs: liger RMSNorm + EAGER router + kernels.sm120.moe.moe.
The patch list is {liger_norm, liger_rope, moe, xsa} -- there is no router patch. Three separate
times this session a benchmark accidentally measured a hand-written reference instead of the real
path; the baseline here is assembled from the same calls ablate/common/patches.py makes.

Splits: optimization shapes are what training uses; held-out shapes are never tuned against and are
where promotion is judged.
"""
import argparse
import json
import time

import torch

OPT_SHAPES = [(32, 1024), (64, 1024)]
HELD_OUT = [(32, 2048), (16, 4096)]
H, E, K, I = 512, 64, 6, 768
EPS = 1e-6
DEV = "cuda"


def make_weights(seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return dict(
        nw=torch.randn(H, device=DEV, dtype=torch.float32, generator=g) * 0.1 + 1.0,
        rw=torch.randn(H, E, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02,
        bias=torch.randn(E, device=DEV, dtype=torch.float32, generator=g) * 0.05,
        # LAYOUTS ARE (E, 2I, H) and (E, H, I) -- taken from the probe in patches.py, not guessed.
        # Transposing them does not raise: it returns an invalid gradient and then an illegal
        # memory access several shapes later.
        gu=torch.randn(E, 2 * I, H, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02,
        dn=torch.randn(E, H, I, device=DEV, dtype=torch.bfloat16, generator=g) * 0.02,
    )


def baseline_block(x, w, codes):
    """liger RMSNorm -> eager router -> moe() kernel. What the model runs today."""
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    # moe_per_expert, NOT moe(): BIBO_MOE_DISPATCH defaults to per_expert and the
    # grouped path was measured a wash on real steps.
    from kernels.sm120.moe import moe_per_expert
    hn = LigerRMSNormFunction.apply(x, w["nw"], EPS, 0.0, "llama", False)
    scores = torch.sigmoid((hn @ w["rw"]).float())
    _, idx = torch.topk(scores + w["bias"], K, dim=-1, sorted=False)
    tw = scores.gather(-1, idx)
    # the router returns long indices and fp32 weights; match it exactly
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)
    return moe_per_expert(hn, idx.long(), tw.float(), w["gu"], w["dn"], codes)


def reference_fp64(x, w):
    """Ground truth: the same math end to end in double. Slow by design; small shapes only."""
    xf = x.double()
    rstd = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + EPS)
    hn = (xf * rstd).double() * w["nw"].double()
    scores = torch.sigmoid(hn @ w["rw"].double())
    _, idx = torch.topk(scores + w["bias"].double(), K, dim=-1, sorted=False)
    idx, _ = torch.sort(idx, dim=-1)
    tw = scores.gather(-1, idx)
    tw = tw / (tw.sum(-1, keepdim=True) + 1e-20)
    gu64, dn64 = w["gu"].double(), w["dn"].double()
    out = torch.zeros_like(hn)
    for k in range(K):
        e = idx[:, k]
        for ex in e.unique():
            m = e == ex
            g = hn[m] @ gu64[ex].T      # (E,2I,H) layout
            gate, up = g[:, :I], g[:, I:]
            act = torch.nn.functional.silu(gate) * up
            out[m] += (act @ dn64[ex].T) * tw[m, k : k + 1]   # (E,H,I) layout
    return out, idx


def time_block(fn, x, iters=10, bwd=True):
    """Returns (ms, peak_gb). Peak is measured over the timed region only."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    for _ in range(3):
        o = fn(x)
        if bwd:
            o.sum().backward()
            x.grad = None
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = fn(x)
        if bwd:
            o.sum().backward()
            x.grad = None
    torch.cuda.synchronize()
    ms = 1000 * (time.perf_counter() - t0) / iters
    return ms, torch.cuda.max_memory_allocated() / 1e9


def score(block_fn, name, shapes, seed=0, acc_tokens=2048):
    """Score one block implementation across shapes. Returns a list of per-shape dicts."""
    w = make_weights(seed)
    codes = torch.zeros(E, dtype=torch.int32, device=DEV)
    rows = []
    for B, S in shapes:
        T = B * S
        # SEEDED per shape. Unseeded, out_err moved 5.7e-02 -> 5.4e-02 between two runs of the
        # SAME code, making the accuracy objective pure noise. Timing is unaffected either way.
        gx = torch.Generator(device=DEV).manual_seed(1234 + B * 100003 + S)
        x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16, generator=gx).requires_grad_(True)
        fn = lambda t: block_fn(t, w, codes)
        try:
            f_ms, _ = time_block(fn, x, bwd=False)
            fb_ms, peak = time_block(fn, x, bwd=True)
        except Exception as e:
            rows.append(dict(B=B, S=S, error=f"{type(e).__name__}: {str(e)[:90]}"))
            print(f"  {name:<14} B={B:<3} S={S:<5} FAILED {type(e).__name__}: {str(e)[:70]}",
                  flush=True)
            continue
        # accuracy on a SUBSET: the fp64 reference is O(E) python loops and cannot run at 65k rows
        xs = x[:acc_tokens].detach()
        with torch.no_grad():
            got = block_fn(xs, w, codes)
        ref, idx64 = reference_fp64(xs, w)
        err = (got.double() - ref).abs().max().item()
        rows.append(dict(B=B, S=S, fwd_ms=round(f_ms, 4), fwdbwd_ms=round(fb_ms, 4),
                         peak_gb=round(peak, 3), out_err=err))
        print(f"  {name:<14} B={B:<3} S={S:<5} fwd {f_ms:7.3f} ms  fwd+bwd {fb_ms:8.3f} ms  "
              f"peak {peak:6.2f} GB  err {err:.3e}", flush=True)
        del x
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", action="store_true", help="also score the megakernel")
    ap.add_argument("--held_out", action="store_true", help="score the held-out shapes instead")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    shapes = HELD_OUT if a.held_out else OPT_SHAPES
    split = "held_out" if a.held_out else "opt"
    print(f"FROZEN EVAL  split={split}  H={H} E={E} K={K} I={I}  shapes={shapes}")

    res = {"split": split, "shapes": shapes}
    print("\nBASELINE = liger RMSNorm + eager router + moe() kernel")
    res["baseline"] = score(baseline_block, "baseline", shapes)

    if a.candidate:
        try:
            from kernels.sm120.megakernel.moe import megakernel_block
            print("\nCANDIDATE = megakernel")
            res["candidate"] = score(megakernel_block, "megakernel", shapes)
        except ImportError as e:
            print(f"\n[candidate not available yet: {e}]")

    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
        print(f"\nwrote {a.out}")
    return res


if __name__ == "__main__":
    main()
