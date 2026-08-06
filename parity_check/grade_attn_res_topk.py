"""Grade the SPARSE-DEPTH (top-k) AttnRes path: kernel vs eager, forward and all three grads.

Three separate things are checked, because the kernel being right is NOT the same as the feature
being reachable -- an earlier round shipped a kernel that graded clean while the flag driving it
was dead, and the arm silently trained the baseline.

  1. SELECTION SEMANTICS  -- the reference top-k mask does what it claims: exactly k survivors,
     prefix_sum always among them, survivors are the k highest scores.
  2. NUMERICS             -- kernel vs eager on fwd + dbr/dps/dw, against fp64 truth, at every
     N and k that block_size=1 actually produces.
  3. IT IS NOT INERT      -- topk=k must CHANGE the output wherever k < N, and must be bit-equal
     to dense wherever k >= N. A "passing" grade with no behavioural delta is the failure mode
     this file exists to catch.

  python -m parity_check.grade_attn_res_topk
"""
import itertools
import sys

import torch

from kernels.sm75.attn_res import attn_res, attn_res_reference, _topk_mask_reference

DEV = "cuda"
# The pool block_size=1 actually produces across the 20 mix sites, plus the output mix.
POOL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
T, H = 4096, 512


def _mk(N, dtype_br, dtype_ps, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    # 1e4 magnitude spread reproduces the real range: raw embedding rms ~0.04 vs prefix sum ~426.
    br = torch.randn(T, N - 1, H, generator=g, device=DEV, dtype=torch.float32)
    br[:, 0] *= 1e-2
    if N > 2:
        br[:, -1] *= 1e2
    ps = torch.randn(T, H, generator=g, device=DEV, dtype=torch.float32) * 1e2
    w = torch.randn(H, generator=g, device=DEV, dtype=torch.float32)
    return br.to(dtype_br), ps.to(dtype_ps), w


def check_selection():
    """The mask rule itself, on scores whose ordering we control."""
    bad = 0
    for N, k in itertools.product(POOL, (2, 4, 6, 8)):
        s = torch.randn(1024, N, device=DEV)
        m = _topk_mask_reference(s, k)
        live = torch.isfinite(m)
        if k >= N:
            if not live.all():
                print(f"  FAIL N={N} k={k}: k>=N must keep everything"); bad += 1
            continue
        if not live[:, -1].all():
            print(f"  FAIL N={N} k={k}: prefix_sum row dropped"); bad += 1
        cnt = live.sum(-1)
        if not (cnt == k).all():
            print(f"  FAIL N={N} k={k}: kept {cnt.min().item()}..{cnt.max().item()}, want {k}")
            bad += 1
        # survivors (excluding the forced last row) must dominate every dropped candidate
        key = s.clone(); key[:, -1] = float("inf")
        lo = key.masked_fill(~live, float("inf")).min(-1).values
        hi = key.masked_fill(live, float("-inf")).max(-1).values
        if not (lo >= hi).all():
            print(f"  FAIL N={N} k={k}: a dropped score beat a kept one"); bad += 1
    return bad


def _grads(fn, br, ps, w, **kw):
    br = br.detach().clone().requires_grad_(True)
    ps = ps.detach().clone().requires_grad_(True)
    w = w.detach().clone().requires_grad_(True)
    out = fn(br, ps, w, 1e-6, **kw)
    torch.manual_seed(0)
    out.backward(torch.randn_like(out))
    return out.detach(), br.grad, ps.grad, w.grad


def _relerr(a, b):
    return ((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30)).item()


def check_numerics():
    """Kernel vs eager, both against an fp64 reference, on the production dtype layout."""
    bad = 0
    hdr = f"{'N':>3} {'k':>3} | {'fwd':>18} {'dbr':>18} {'dps':>18} {'dw':>18}"
    print(hdr); print("-" * len(hdr))
    for N in POOL:
        for k in (0, 4, 6):
            if k and k >= N:
                continue
            # production layout: block_residual fp32 (seeded from the fp32 embedding), stream bf16
            br, ps, w = _mk(N, torch.float32, torch.bfloat16, seed=N * 100 + k)
            kw = dict(score_mode=1, topk=k)
            ker = _grads(attn_res, br, ps, w, **kw)
            eag = _grads(attn_res_reference, br, ps, w, **kw)
            ref = _grads(attn_res_reference, br.double(), ps.double(), w.double(), **kw)
            cells, worse = [], 0
            for kx, ex, rx in zip(ker, eag, ref):
                ke, ee = _relerr(kx, rx), _relerr(ex, rx)
                ratio = ke / ee if ee > 0 else (0.0 if ke == 0 else float("inf"))
                cells.append(f"{ke:.2e}/{ratio:5.2f}x")
                if ratio > 1.5:
                    worse += 1
            bad += worse
            flag = "  <-- WORSE THAN EAGER" if worse else ""
            print(f"{N:>3} {k if k else '-':>3} | " + " ".join(f"{c:>18}" for c in cells) + flag)
    return bad


def check_not_inert():
    """topk must bite when k<N and must be a no-op when k>=N. Guards the dead-flag failure."""
    bad = 0
    for N in POOL:
        br, ps, w = _mk(N, torch.float32, torch.bfloat16, seed=N)
        dense = attn_res(br, ps, w, 1e-6, score_mode=1, topk=0)
        for k in (4, 6):
            sparse = attn_res(br, ps, w, 1e-6, score_mode=1, topk=k)
            delta = (sparse.float() - dense.float()).abs().max().item()
            if k >= N:
                if delta != 0.0:
                    print(f"  FAIL N={N} k={k}: k>=N must be bit-identical to dense, got {delta:.3e}")
                    bad += 1
            elif delta == 0.0:
                print(f"  FAIL N={N} k={k}: topk is INERT -- output identical to dense")
                bad += 1
    return bad


def main():
    torch.manual_seed(0)
    print("=== 1. selection semantics ===")
    b1 = check_selection()
    print(f"  {'OK' if not b1 else str(b1) + ' FAILURES'}\n")
    print("=== 2. numerics: kernel_err/eager_ratio vs fp64 truth (signorm, br fp32 / ps bf16) ===")
    b2 = check_numerics()
    print(f"  {'OK -- never worse than eager' if not b2 else str(b2) + ' CELLS WORSE THAN EAGER'}\n")
    print("=== 3. top-k is not inert ===")
    b3 = check_not_inert()
    print(f"  {'OK' if not b3 else str(b3) + ' FAILURES'}\n")
    total = b1 + b2 + b3
    print("PASS" if not total else f"FAIL ({total})")
    return 1 if total else 0


if __name__ == "__main__":
    sys.exit(main())
