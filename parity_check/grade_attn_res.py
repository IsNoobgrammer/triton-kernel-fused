"""Grade attn_res (the AttnRes depth mix) against FP64 TRUTH, the same contract as residual_add.

The mix is a softmax over N block candidates, scored on RMS-normalised keys, applied to RAW values:

    k       = v * rsqrt(mean(v^2) + eps)          per (token, candidate)
    scores  = (k * score_weight).sum(-1)          -> (T, N)
    out     = softmax(scores) @ v                 -> (T, H)

BF16 IN, FP32 SCORES, BF16 OUT -- and that is the only configuration graded. block_residual and
prefix_sum are both bf16 (--attn_res_fp32_stream false); the kernel widens to fp32 for the RMS,
the dot product, the softmax and the mixture; the store rounds back to bf16. Eager does the same,
so the two agree on 99.98% of elements with identical error against fp64.

The dtype sweep this file used to run ({bf16,fp32,fp16} x both operands) policed a MIXED stack
where an fp32 stream met a bf16 one inside one kernel. Going uniformly bf16 dissolved that
problem rather than solving it, so the sweep now grades configurations nothing runs. What remains
is the axis that is still real:

  THE MIX IS ILL-CONDITIONED BY CONSTRUCTION. Candidate RMS spans ~0.04 (the raw embedding) to
  ~426 (the prefix sum) in the real model -- four orders of magnitude inside one softmax-weighted
  sum. That is exactly the regime where fp32 accumulation quietly loses digits, and it is what
  broke d_theta in residual_add. `spread` below reproduces it deliberately instead of grading on
  well-scaled random data that would hide it. N (2, 4, 8) covers block sizes 1 and 3 at 10 layers.

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


def _truth(br, ps, w, eps, score_mode=0):
    """fp64 throughout. NOT .float() -- that is fp32 and would grade fp32 against fp32."""
    v = torch.cat((_f64(br.double(), "truth br"), _f64(ps.double(), "truth ps").unsqueeze(1)), 1)
    k = v * torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + eps)
    scores = (k * _f64(w.double(), "truth w")).sum(-1)
    if score_mode == 0:
        probs = scores.softmax(-1)
    else:
        sp = torch.sigmoid(scores)
        probs = sp / sp.sum(-1, keepdim=True).clamp_min(1e-30)
    return _f64(torch.matmul(_f64(probs, "truth probs").unsqueeze(1), v).squeeze(1), "truth out")


def _relerr(x, truth):
    _f64(truth, "relerr ref")
    d = (x.double() - truth).abs()
    den = max(truth.abs().max().item(), 1e-300)
    return d.mean().item() / den, d.max().item() / den


def _case(br_dt, ps_dt, N, T=512, H=512, spread=1.0, seed=0, eps=1e-6, device="cuda",
          score_mode=0):
    torch.manual_seed(seed)
    # spread > 1 gives each candidate its own magnitude scale, geometric from 1 to `spread`.
    # spread=1e4 reproduces the model's embedding-vs-prefix-sum range.
    scales = torch.logspace(0, torch.log10(torch.tensor(float(spread))).item(), N, device=device)
    br = torch.randn(T, N - 1, H, device=device) * scales[:N - 1].view(1, -1, 1)
    ps = torch.randn(T, H, device=device) * scales[-1]
    br, ps = br.to(br_dt), ps.to(ps_dt)
    w = torch.randn(H, device=device, dtype=torch.float32)

    tr = _truth(br, ps, w, eps, score_mode)
    k = attn_res(br, ps, w, eps, score_mode)
    e = attn_res_reference(br, ps, w, eps, score_mode)
    return _relerr(k, tr), _relerr(e, tr)


def _grade_backward(N=4, T=256, H=256, spread=1e4, seed=3, eps=1e-6, device="cuda"):
    """Kernel gradients vs the eager reference, per score mode.

    The forward grid above cannot catch a wrong backward -- a bad prefactor trains badly and
    silently. signorm's is NOT the same as softmax's: dp_i/dx_k is p_k(delta_ik - p_i) for
    softmax but (s_k(1-s_k)/S)(delta_ik - p_i) for signorm, and that factor was derived by hand.
    fp64 inputs so the reference is a real target and neither side is measuring bf16 rounding.
    """
    print("\nbackward, kernel vs eager reference (fp64 inputs):")
    worst = 0.0
    for mode in (0, 1):
        torch.manual_seed(seed)
        scales = torch.logspace(0, torch.log10(torch.tensor(float(spread))).item(), N,
                                device=device)
        br0 = (torch.randn(T, N - 1, H, device=device, dtype=torch.float64)
               * scales[:N - 1].view(1, -1, 1))
        ps0 = torch.randn(T, H, device=device, dtype=torch.float64) * scales[-1]
        w0 = torch.randn(H, device=device, dtype=torch.float32)
        gout = torch.randn(T, H, device=device, dtype=torch.float64)

        def grads(fn):
            br = br0.clone().requires_grad_(True)
            ps = ps0.clone().requires_grad_(True)
            w = w0.clone().requires_grad_(True)
            fn(br, ps, w, eps, mode).backward(gradient=gout)
            return br.grad, ps.grad, w.grad

        gk = grads(attn_res)
        ge = grads(attn_res_reference)
        for nm, a, b in zip(("d_br", "d_ps", "d_w"), gk, ge):
            den = max(b.abs().max().item(), 1e-300)
            r = (a - b).abs().max().item() / den
            worst = max(worst, r)
            print(f"  {'softmax' if mode == 0 else 'signorm'} {nm:5s} rel {r:.3e}")
    # Kernel and reference differ only in accumulation order here, so this is a rounding-level
    # check, not a tolerance to tune. A wrong prefactor lands at O(1), not O(1e-6).
    assert worst < 1e-5, f"backward disagrees with the reference at {worst:.3e} -- check the mode"
    print(f"  backward OK, worst {worst:.3e}")


def main():
    assert torch.cuda.is_available(), "needs a GPU"
    bf, f32, f16 = torch.bfloat16, torch.float32, torch.float16
    # BF16 EVERYWHERE. block_residual and prefix_sum are both bf16 in the shipped config
    # (--attn_res_fp32_stream false), the kernel scores and softmaxes in fp32 internally, and the
    # output goes back to bf16. The old 9-way dtype product policed a mixed stack that no longer
    # exists. N and the magnitude spread stay -- those are real axes, dtype no longer is.
    # Both score modes: 0 = softmax (every arm before Aug 5), 1 = sigmoid/sum. Mode 1 keeps the
    # convex-combination property -- the weights still sum to 1 -- so the ill-conditioning that
    # `spread` reproduces applies to it identically and it gets the same grid, not a token case.
    cases = []
    for mode in (0, 1):
        for N in (2, 4, 8):
            for spread in (1.0, 1e4):
                cases.append((bf, bf, N, spread, mode))
    short = {bf: "bf16", f32: "fp32", f16: "fp16"}
    # At bf16 in/out the kernel and eager are the SAME computation to within float-comparison
    # noise -- measured ratios 0.999999879 to 1.000000053, i.e. differences in the 8th decimal.
    # A strict `ratio > 1.0` test fails on the last bit of a float division and reports ties as
    # regressions. MEAN_SLACK sits well below any real defect (the cancelling-tanh bug read
    # 1.29x) and well above this noise.
    MEAN_SLACK, MAX_SLACK = 1.001, 2.0
    print(f"grading {len(cases)} configs x 2 statistics against fp64 "
          f"(mean slack {MEAN_SLACK}x, max slack {MAX_SLACK}x)\n")
    mu_r, mx_r, fails = [], [], []
    for br_dt, ps_dt, N, spread, mode in cases:
        (kmu, kmx), (emu, emx) = _case(br_dt, ps_dt, N, spread=spread, score_mode=mode)
        rmu = kmu / emu if emu > 0 else 1.0
        rmx = kmx / emx if emx > 0 else 1.0
        mu_r.append(rmu)
        mx_r.append(rmx)
        name = (f"{'softmax' if mode == 0 else 'signorm'} br={short[br_dt]} "
                f"ps={short[ps_dt]} N={N} spread={spread:g}")
        if rmu > MEAN_SLACK or rmx > MAX_SLACK:
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
    print("\nforward PASS: attn_res kernel <= eager against fp64 on every config")
    _grade_backward()
    print("PASS")


if __name__ == "__main__":
    main()
