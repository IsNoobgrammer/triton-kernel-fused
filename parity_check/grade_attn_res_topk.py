"""Grade the SPARSE-DEPTH (top-k) AttnRes path: kernel vs eager, forward and all three grads.

Four things are checked, because the kernel being right is NOT the same as the feature being
reachable -- an earlier round shipped a kernel that graded clean while the flag driving it was
dead, and the arm silently trained the baseline.

  1. SELECTION SEMANTICS -- exactly k survivors, prefix_sum always among them, and every survivor
     outscores every dropped candidate.
  2. TIE SENSITIVITY     -- hard top-k is DISCONTINUOUS in the scores, so two candidates within
     fp32 epsilon can be ordered differently by an fp32 kernel and an fp64 reference. That is
     inherent to the mechanism, not a defect, but it must be MEASURED rather than absorbed into a
     max-error number. Tokens whose selection flips are counted and excluded from (3).
  3. FORWARD ACCURACY    -- kernel vs eager against real fp64 truth, in the production dtype
     layout (block_residual fp32, stream bf16), at every N block_size=1 produces.
  4. BACKWARD            -- kernel vs eager at fp64 inputs with a SHARED upstream gradient. A
     wrong prefactor lands at O(1), not O(1e-6), so this is a rounding check and not a tolerance
     to tune. The selected face changes which lanes are live but not the per-lane derivative.
  5. NOT INERT           -- topk must change the output wherever k < N and be bit-identical to
     dense wherever k >= N. A clean grade with no behavioural delta is the failure this catches.

  python -m parity_check.grade_attn_res_topk
"""
import sys

import torch

from . import _paths  # noqa: F401
from kernels.sm75.attn_res import attn_res, attn_res_reference, _topk_mask_reference

DEV = "cuda"
# The candidate pool block_size=1 actually produces across its 20 mix sites plus the output mix.
POOL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
KS = (4, 6)
EPS = 1e-6


def _scores(br, ps, w, dtype):
    """Depth scores in `dtype`. Mirrors the kernel: RMS pulled out of the contraction."""
    v = torch.cat((br.to(dtype), ps.to(dtype).unsqueeze(1)), dim=1)
    inv = torch.rsqrt(v.pow(2).mean(-1, keepdim=True) + EPS)
    return (v * inv * w.to(dtype)).sum(-1), v


def _truth(br, ps, w, topk, score_mode=1):
    """fp64 throughout. NOT .float() -- attn_res_reference casts internally, so calling it with
    double inputs grades fp32 against fp32 and every ratio comes back exactly 1.00x."""
    s, v = _scores(br, ps, w, torch.float64)
    s = _topk_mask_reference(s, topk)
    if score_mode == 0:
        p = s.softmax(-1)
    else:
        sp = torch.sigmoid(s)
        p = sp / sp.sum(-1, keepdim=True).clamp_min(1e-30)
    assert p.dtype == torch.float64
    return torch.matmul(p.unsqueeze(1), v).squeeze(1)


def _mk(N, seed, br_dt=torch.float32, ps_dt=torch.bfloat16):
    torch.manual_seed(seed)
    # 1e4 magnitude spread reproduces the real range: raw embedding rms ~0.04 vs prefix sum ~426.
    scales = torch.logspace(0, 4, N, device=DEV)
    br = torch.randn(4096, N - 1, 512, device=DEV) * scales[:N - 1].view(1, -1, 1)
    ps = torch.randn(4096, 512, device=DEV) * scales[-1]
    return br.to(br_dt), ps.to(ps_dt), torch.randn(512, device=DEV)


def _flips(br, ps, w, topk):
    """Fraction of tokens where fp32 and fp64 scoring pick a DIFFERENT survivor set."""
    if not topk or topk >= br.shape[1] + 1:
        return 0.0, None
    m32 = torch.isfinite(_topk_mask_reference(_scores(br, ps, w, torch.float32)[0], topk))
    m64 = torch.isfinite(_topk_mask_reference(_scores(br, ps, w, torch.float64)[0], topk))
    same = (m32 == m64).all(-1)
    return (~same).float().mean().item(), same


def _relerr(x, truth, keep):
    d = (x.double() - truth).abs()[keep]
    den = max(truth[keep].abs().max().item(), 1e-300)
    return d.mean().item() / den, d.max().item() / den


def check_selection():
    bad = 0
    for N in POOL:
        for k in (2,) + KS + (8,):
            s = torch.randn(2048, N, device=DEV)
            live = torch.isfinite(_topk_mask_reference(s, k))
            if k >= N:
                if not live.all():
                    print(f"  FAIL N={N} k={k}: k>=N must keep every candidate"); bad += 1
                continue
            if not live[:, -1].all():
                print(f"  FAIL N={N} k={k}: prefix_sum row was dropped"); bad += 1
            cnt = live.sum(-1)
            if not (cnt == k).all():
                print(f"  FAIL N={N} k={k}: kept {cnt.min().item()}..{cnt.max().item()}"); bad += 1
            key = s.clone(); key[:, -1] = float("inf")
            lo = key.masked_fill(~live, float("inf")).min(-1).values
            hi = key.masked_fill(live, float("-inf")).max(-1).values
            if not (lo >= hi).all():
                print(f"  FAIL N={N} k={k}: a dropped score beat a kept one"); bad += 1
    return bad


def check_forward():
    """kernel_err / eager_err against fp64 truth. Ratio <= 1 means the kernel is at least as
    accurate as eager, which is this kernel's standing contract."""
    bad = 0
    print(f"{'N':>3} {'k':>3} {'flip%':>7} | {'kern mean/max':>21} {'ratio mean/max':>15}")
    print("-" * 56)
    for N in POOL:
        for k in (0,) + KS:
            if k and k >= N:
                continue
            br, ps, w = _mk(N, seed=N * 100 + k)
            flip, same = _flips(br, ps, w, k)
            keep = torch.ones(br.shape[0], dtype=torch.bool, device=DEV) if same is None else same
            tr = _truth(br, ps, w, k)
            km, kx = _relerr(attn_res(br, ps, w, EPS, 1, k), tr, keep)
            em, ex = _relerr(attn_res_reference(br, ps, w, EPS, 1, k), tr, keep)
            rm = km / em if em else float("nan")
            rx = kx / ex if ex else float("nan")
            worse = rm > 1.5 or rx > 1.5
            bad += worse
            print(f"{N:>3} {k or '-':>3} {flip*100:>6.2f}% | {km:.2e}/{kx:.2e} "
                  f"{rm:>6.2f}x/{rx:>5.2f}x" + ("   <-- WORSE THAN EAGER" if worse else ""))
    return bad


def check_backward():
    """fp64 inputs, SHARED upstream gradient. Both sides then differ only in accumulation order,
    so a correct prefactor lands at rounding level; a wrong one lands at O(1)."""
    worst, bad = 0.0, 0
    for N in (5, 8, 11):
        for k in (0,) + KS:
            if k and k >= N:
                continue
            torch.manual_seed(N + k)
            scales = torch.logspace(0, 4, N, device=DEV)
            br0 = torch.randn(256, N - 1, 256, device=DEV, dtype=torch.float64) * scales[:N-1].view(1, -1, 1)
            ps0 = torch.randn(256, 256, device=DEV, dtype=torch.float64) * scales[-1]
            w0 = torch.randn(256, device=DEV)
            gout = torch.randn(256, 256, device=DEV, dtype=torch.float64)   # SAME for both sides

            def grads(fn):
                a = br0.clone().requires_grad_(True)
                b = ps0.clone().requires_grad_(True)
                c = w0.clone().requires_grad_(True)
                fn(a, b, c, EPS, 1, k).backward(gradient=gout)
                return a.grad, b.grad, c.grad

            row = []
            for nm, x, y in zip(("d_br", "d_ps", "d_w"), grads(attn_res), grads(attn_res_reference)):
                r = (x - y).abs().max().item() / max(y.abs().max().item(), 1e-300)
                worst = max(worst, r)
                row.append(f"{nm} {r:.2e}")
            flag = "  <-- O(1), WRONG PREFACTOR" if max(
                float(c.split()[1]) for c in row) > 1e-5 else ""
            bad += bool(flag)
            print(f"  N={N:>2} k={k or '-':>2} | " + "  ".join(row) + flag)
    print(f"  worst {worst:.3e} (contract: < 1e-5)")
    return bad


def check_not_inert():
    bad = 0
    for N in POOL:
        br, ps, w = _mk(N, seed=N)
        dense = attn_res(br, ps, w, EPS, 1, 0)
        for k in KS:
            d = (attn_res(br, ps, w, EPS, 1, k).float() - dense.float()).abs().max().item()
            if k >= N and d != 0.0:
                print(f"  FAIL N={N} k={k}: k>=N must be bit-identical to dense, got {d:.3e}")
                bad += 1
            elif k < N and d == 0.0:
                print(f"  FAIL N={N} k={k}: top-k is INERT, output identical to dense")
                bad += 1
    return bad


def main():
    print("=== 1. selection semantics ===")
    b1 = check_selection(); print(f"  {'OK' if not b1 else f'{b1} FAILURES'}\n")
    print("=== 2+3. forward vs fp64 truth, signorm, br fp32 / ps bf16 ===")
    b3 = check_forward(); print(f"  {'OK' if not b3 else f'{b3} WORSE THAN EAGER'}\n")
    print("=== 4. backward, kernel vs eager, fp64 inputs + shared gout ===")
    b4 = check_backward(); print(f"  {'OK' if not b4 else f'{b4} FAILURES'}\n")
    print("=== 5. top-k is not inert ===")
    b5 = check_not_inert(); print(f"  {'OK' if not b5 else f'{b5} FAILURES'}\n")
    total = b1 + b3 + b4 + b5
    print("PASS" if not total else f"FAIL ({total})")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
