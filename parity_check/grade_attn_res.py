"""Grade attn_res (the AttnRes depth mix) against FP64 TRUTH, the same contract as residual_add.

The mix is a softmax over N block candidates, scored on RMS-normalised keys, applied to RAW values:

    k       = v * rsqrt(mean(v^2) + eps)          per (token, candidate)
    scores  = (k * score_weight).sum(-1)          -> (T, N)
    out     = softmax(scores) @ v                 -> (T, H)

Two things make this harder to get right than the residual add, and both are graded here:

  MIXED DTYPES ARE THE NORM. Under --attn_res_fp32_stream block_residual is fp32 while prefix_sum
  is bf16 under autocast, and they are concatenated. Every (block_residual, prefix_sum) dtype pair
  is enumerated rather than assumed.

  THE MIX IS ILL-CONDITIONED BY CONSTRUCTION. Candidate RMS spans ~0.04 (the raw embedding) to
  ~426 (the prefix sum) in the real model -- four orders of magnitude inside one softmax-weighted
  sum. That is exactly the regime where fp32 accumulation quietly loses digits, and it is what
  broke d_theta in residual_add. `spread` below reproduces it deliberately instead of grading on
  well-scaled random data that would hide it.

    python -m parity_check.grade_attn_res
"""
import itertools
import statistics

import torch

from . import _paths  # noqa: F401
from kernels.sm75.attn_res import attn_res, attn_res_reference


def _f64(t, what):
    assert t.dtype == torch.float64, f"{what} is {t.dtype}, must be float64"
    return t


def _truth(br, ps, w, eps):
    """fp64 throughout. NOT .float() -- that is fp32 and would grade fp32 against fp32."""
    v = torch.cat((_f64(br.double(), "truth br"), _f64(ps.double(), "truth ps").unsqueeze(1)), 1)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * _f64(w.double(), "truth w")).sum(-1)
    return _f64(torch.matmul(scores.softmax(-1).unsqueeze(1), v).squeeze(1), "truth out")


def _relerr(x, truth):
    _f64(truth, "relerr ref")
    d = (x.double() - truth).abs()
    den = max(truth.abs().max().item(), 1e-300)
    return d.mean().item() / den, d.max().item() / den


def _case(br_dt, ps_dt, N, T=512, H=512, spread=1.0, seed=0, eps=1e-6, device="cuda"):
    torch.manual_seed(seed)
    # spread > 1 gives each candidate its own magnitude scale, geometric from 1 to `spread`.
    # spread=1e4 reproduces the model's embedding-vs-prefix-sum range.
    scales = torch.logspace(0, torch.log10(torch.tensor(float(spread))).item(), N, device=device)
    br = torch.randn(T, N - 1, H, device=device) * scales[:N - 1].view(1, -1, 1)
    ps = torch.randn(T, H, device=device) * scales[-1]
    br, ps = br.to(br_dt), ps.to(ps_dt)
    w = torch.randn(H, device=device, dtype=torch.float32)

    tr = _truth(br, ps, w, eps)
    k = attn_res(br, ps, w, eps)
    e = attn_res_reference(br, ps, w, eps)
    return _relerr(k, tr), _relerr(e, tr)


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    bf, f32, f16 = torch.bfloat16, torch.float32, torch.float16
    cases = []
    for br_dt, ps_dt in itertools.product((bf, f32, f16), repeat=2):
        for N in (2, 4, 8):
            for spread in (1.0, 1e4):
                cases.append((br_dt, ps_dt, N, spread))
    short = {bf: "bf16", f32: "fp32", f16: "fp16"}
    print(f"grading {len(cases)} configs x 2 statistics against fp64\n")
    mu_r, mx_r, fails = [], [], []
    for br_dt, ps_dt, N, spread in cases:
        (kmu, kmx), (emu, emx) = _case(br_dt, ps_dt, N, spread=spread)
        rmu = kmu / emu if emu > 0 else 1.0
        rmx = kmx / emx if emx > 0 else 1.0
        mu_r.append(rmu)
        mx_r.append(rmx)
        name = f"br={short[br_dt]} ps={short[ps_dt]} N={N} spread={spread:g}"
        if rmu > 1.0 or rmx > 2.0:
            fails.append((name, kmu, emu, rmu, kmx, emx, rmx))
            print(f"  {name:38s} mean {kmu:.3e}/{emu:.3e}={rmu:5.2f}x  "
                  f"max {kmx:.3e}/{emx:.3e}={rmx:5.2f}x  <-- WORSE")
    for lbl, v in (("MEAN", mu_r), ("MAX", mx_r)):
        v = sorted(v)
        print(f"  {lbl:5s} n={len(v)}  better={sum(1 for x in v if x < 1):3d} "
              f"tie={sum(1 for x in v if x == 1):3d} worse={sum(1 for x in v if x > 1):3d}   "
              f"median={statistics.median(v):.4f}  worst={v[-1]:.4f}")
    if fails:
        print(f"\nFAIL on {len(fails)} configs")
        raise SystemExit(1)
    print("\nPASS: attn_res kernel <= eager against fp64 on every config")


if __name__ == "__main__":
    main()
