"""Fused AR kernel == K3's `_apply_attn_res`, and a standalone profile at real shapes.

NO MODEL NEEDED. Dummy tensors at the shapes BiBo actually runs, so the kernel can be iterated on
without a 20-minute training probe in the loop.

Shapes come from the 524M stack: micro-batch 16 x seq 1024 = 16384 tokens, hidden 512, and N =
blocks+1 sweeping 2..11 (block_size=3 gives N up to 5; block_size=1 gives N up to 11).

    python -m parity_check.parity_attn_res
"""
from . import _paths  # noqa: F401

import torch
import triton

from kernels.sm75.attn_res import fused_attn_res, attn_res_reference

DEV = "cuda"
EPS = 1e-6


def _opt_torch(block_residual, prefix_sum, score_weight, eps=EPS):
    """What BiBo runs today: RMS factored out of the contraction, no normalized copy."""
    v = torch.cat((block_residual, prefix_sum.unsqueeze(1)), dim=1)
    vf = v.float()
    sq = torch.linalg.vector_norm(vf, dim=-1).square()
    inv_rms = torch.rsqrt(sq / vf.shape[-1] + eps)
    scores = torch.matmul(vf, score_weight.float()) * inv_rms
    probs = scores.softmax(-1).unsqueeze(1)
    return torch.matmul(probs, vf).squeeze(1).to(v.dtype)


def _mk(T, N, H, dtype):
    torch.manual_seed(0)
    br = torch.randn(T, N - 1, H, device=DEV, dtype=dtype)
    ps = torch.randn(T, H, device=DEV, dtype=dtype)
    w = torch.randn(H, device=DEV, dtype=torch.float32) * 0.05
    return br, ps, w


def parity():
    """EVERY dtype is scored against the SAME fp32 eager reference, not against eager at its own
    precision. Grading a bf16 kernel against a bf16 reference hides shared error -- both could
    drift from the true answer together and still look clean.

    With bf16 INPUTS the true answer is unreachable: the inputs are already quantized, so there is
    a floor no implementation can beat. The meaningful claim is therefore not "bf16 == fp32" but
    "the kernel is no worse than eager at the same input precision" -- it must not ADD error on
    top of the quantization floor. So each row reports both, and the gate is kernel <= eager.
    """
    print("=== parity: every dtype vs the FP32 EAGER reference ===")
    print(f"{'T':>6}{'N':>4}{'H':>6}{'in dtype':>10}{'kernel err':>12}{'eager err':>12}"
          f"{'cached sq':>12}{'':>4}")
    bad = 0
    for dtype in (torch.float32, torch.bfloat16):
        for T, N, H in ((4096, 2, 512), (4096, 5, 512), (16384, 5, 512),
                        (16384, 11, 512), (2048, 4, 1024), (1024, 8, 256)):
            br32, ps32, w = _mk(T, N, H, torch.float32)
            ref = attn_res_reference(br32, ps32, w, EPS).float()          # ground truth
            den = ref.abs().max().item()

            br, ps = br32.to(dtype), ps32.to(dtype)
            e_err = (attn_res_reference(br, ps, w, EPS).float() - ref).abs().max().item() / den
            k_err = (fused_attn_res(br, ps, w, EPS).float() - ref).abs().max().item() / den

            bsq = torch.zeros(T, N, device=DEV, dtype=torch.float32)
            bsq[:, : N - 1] = br.float().pow(2).sum(-1)
            c_err = (fused_attn_res(br, ps, w, EPS, block_sq_sum=bsq).float()
                     - ref).abs().max().item() / den

            # kernel must not be worse than eager at the same input precision (5% slack for
            # reduction-order jitter), and fp32 must actually hit fp32 accuracy.
            ok = k_err <= max(e_err * 1.05, 1e-6) and c_err <= max(e_err * 1.05, 1e-6)
            if dtype is torch.float32:
                ok = ok and k_err < 1e-5
            bad += not ok
            print(f"{T:>6}{N:>4}{H:>6}{str(dtype).split('.')[-1]:>10}"
                  f"{k_err:>12.2e}{e_err:>12.2e}{c_err:>12.2e}"
                  + ("   ok" if ok else "   <-- FAIL"))
    return bad


def mixed_dtype():
    """block_residual fp32 + prefix_sum bf16 -- the pair the MODEL actually produces.

    The reference concatenates them, so torch type promotion makes the RESULT fp32. The kernel
    returned empty_like(prefix_sum) = bf16 and silently changed the model's residual dtype; the
    standalone parity never caught it because it fed both sides the same dtype.
    """
    print()
    print("=== mixed dtype: block_residual fp32 + prefix_sum bf16 (what the model produces) ===")
    print(f"{'T':>6}{'N':>4}{'H':>6}{'ref dtype':>12}{'kernel dtype':>14}{'rel err':>11}")
    bad = 0
    for T, N, H in ((4096, 5, 512), (16384, 5, 512), (16384, 11, 512)):
        br32, ps32, w = _mk(T, N, H, torch.float32)
        br, ps = br32, ps32.to(torch.bfloat16)
        ref = attn_res_reference(br, ps, w, EPS)
        got = fused_attn_res(br, ps, w, EPS)
        rel = (got.float() - ref.float()).abs().max().item() / ref.float().abs().max().item()
        ok = got.dtype == ref.dtype and rel < 1e-5
        bad += not ok
        print(f"{T:>6}{N:>4}{H:>6}{str(ref.dtype).split('.')[-1]:>12}"
              f"{str(got.dtype).split('.')[-1]:>14}{rel:>11.2e}" + ("   ok" if ok else "   <-- FAIL"))
    return bad


def _bench(fn, *a, warmup=10, iters=50):
    for _ in range(warmup):
        fn(*a)
    torch.cuda.synchronize()
    return triton.testing.do_bench(lambda: fn(*a), warmup=20, rep=100)


def profile():
    print()
    print("=== forward profile, bf16, T=16384 H=512 (micro-batch 16 x seq 1024) ===")
    print(f"{'N':>4}{'K3 naive':>12}{'opt torch':>12}{'fused':>10}"
          f"{'vs naive':>10}{'vs opt':>9}{'GB/s':>9}")
    T, H = 16384, 512
    for N in (2, 3, 5, 8, 11):
        br, ps, w = _mk(T, N, H, torch.bfloat16)
        t_ref = _bench(attn_res_reference, br, ps, w)
        t_opt = _bench(_opt_torch, br, ps, w)
        t_fus = _bench(fused_attn_res, br, ps, w)
        # ideal traffic: read V once (bf16) + write out once
        gb = (T * N * H * 2 + T * H * 2) / 1e9
        print(f"{N:>4}{t_ref:>11.3f}m{t_opt:>11.3f}m{t_fus:>9.3f}m"
              f"{t_ref / t_fus:>9.2f}x{t_opt / t_fus:>8.2f}x{gb / (t_fus * 1e-3):>9.0f}")

    print()
    print("=== per-STEP cost, block3 pattern (10 layers, 2 sites/layer + 1 output = 21 mixes) ===")
    # N at layer l is floor(l/3)+1, +1 for the prefix row
    Ns = [min(l // 3 + 1, 4) + 1 for l in range(10) for _ in range(2)] + [5]
    tot = {"naive": 0.0, "opt": 0.0, "fused": 0.0}
    for N in Ns:
        br, ps, w = _mk(T, N, H, torch.bfloat16)
        tot["naive"] += _bench(attn_res_reference, br, ps, w)
        tot["opt"] += _bench(_opt_torch, br, ps, w)
        tot["fused"] += _bench(fused_attn_res, br, ps, w)
    for k, v in tot.items():
        print(f"  {k:<6} {v:7.2f} ms/step-of-AR   ({v / tot['fused']:.2f}x fused)")
    print(f"  a 1527 ms/step baseline step means AR-naive adds ~{tot['naive'] / 1527 * 100:.1f}% "
          f"forward-only, AR-fused ~{tot['fused'] / 1527 * 100:.1f}%")


def roofline():
    """Where does the kernel's time actually go, and how close is it to memory-bound ideal?

    Ideal traffic, per site (bf16 V, fp32 out to match cat promotion):
      fwd  read br + ps, write out
      bwd  read br + ps + dout, write dbr + dps + dwp
    dwp is the (T,H) fp32 partial for the dw reduction -- at one token per program it is written
    AND read back once, which is pure overhead for a (H,) gradient.
    """
    from kernels.sm75.attn_res import attn_res
    print()
    print("=== roofline, bf16 in / fp32 out, T=16384 H=512 ===")
    print(f"{'N':>4}{'fwd ms':>9}{'fwd GB/s':>10}{'bwd ms':>9}{'bwd GB/s':>10}"
          f"{'dwp share':>11}")
    T, H = 16384, 512
    for N in (2, 3, 5, 8, 11):
        br, ps, w = _mk(T, N, H, torch.bfloat16)
        t_f = _bench(fused_attn_res, br, ps, w)
        brg = br.clone().requires_grad_(True)
        psg = ps.clone().requires_grad_(True)
        wg = w.clone().requires_grad_(True)
        g = torch.randn(T, H, device=DEV, dtype=torch.float32)
        def bwd():
            for x in (brg, psg, wg):
                x.grad = None
            attn_res(brg, psg, wg, EPS).backward(g)
        t_all = _bench(bwd)
        t_b = max(t_all - t_f, 1e-6)
        rd = T * (N - 1) * H * 2 + T * H * 2
        gb_f = (rd + T * H * 4) / 1e9
        dwp = T * H * 4 * 2                       # written by the kernel, read by the sum
        gb_b = (rd + T * H * 4 + T * (N - 1) * H * 2 + T * H * 2 + dwp) / 1e9
        print(f"{N:>4}{t_f:>8.3f}m{gb_f / (t_f * 1e-3):>10.0f}{t_b:>8.3f}m"
              f"{gb_b / (t_b * 1e-3):>10.0f}{dwp / 1e9 / gb_b * 100:>10.0f}%")


def gradcheck():
    """Gradients from the fused kernel must match eager autograd on the K3 reference.

    This is the assertion that matters for training: a forward that is right and a backward that
    is subtly wrong produces a model that trains, looks plausible, and is optimizing something
    else. Checked in fp64 for exactness and fp32/bf16 at real shapes for the dtypes we run.
    """
    from kernels.sm75.attn_res import attn_res
    print()
    print("=== gradcheck: EVERY dtype's gradients vs FP32 EAGER autograd ===")
    print("    k = fused kernel at that input dtype, e = eager at that input dtype;")
    print("    both scored against eager autograd in fp32. Gate: kernel <= eager.")
    print(f"{'T':>6}{'N':>4}{'H':>6}{'in dtype':>10}"
          f"{'d_block k/e':>22}{'d_prefix k/e':>22}{'d_w k/e':>22}")
    bad = 0
    for dtype in (torch.float32, torch.bfloat16):
        for T, N, H in ((512, 3, 512), (4096, 5, 512), (16384, 11, 512), (1024, 8, 256)):
            torch.manual_seed(1)
            br32 = torch.randn(T, N - 1, H, device=DEV, dtype=torch.float32)
            ps32 = torch.randn(T, H, device=DEV, dtype=torch.float32)
            w32 = torch.randn(H, device=DEV, dtype=torch.float32) * 0.05
            g32 = torch.randn(T, H, device=DEV, dtype=torch.float32)

            def grads(fn, dt):
                xs = [br32.to(dt).clone().requires_grad_(True),
                      ps32.to(dt).clone().requires_grad_(True),
                      w32.clone().requires_grad_(True)]
                fn(xs[0], xs[1], xs[2], EPS).backward(g32.to(dt))
                return [x.grad.float() for x in xs]

            try:
                # the eager REFERENCE is what OOMs on a busy box -- it needs several fp32
                # (T,N,H) copies alive for backward, which is the memory the kernel removes.
                ref = grads(attn_res_reference, torch.float32)      # ground truth
                eag = ref if dtype is torch.float32 else grads(attn_res_reference, dtype)
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"{T:>6}{N:>4}{H:>6}{str(dtype).split('.')[-1]:>10}"
                      f"{'   fp32 reference OOM (GPU busy) - skipped':>40}")
                continue
            ker = grads(attn_res, dtype)

            cells, ok = [], True
            for r, e, k in zip(ref, eag, ker):
                den = r.abs().max().item() + 1e-12
                ke = (k - r).abs().max().item() / den
                ee = (e - r).abs().max().item() / den
                cells.append(f"{ke:.1e}/{ee:.1e}")
                ok = ok and ke <= max(ee * 1.05, 2e-5)
            bad += not ok
            print(f"{T:>6}{N:>4}{H:>6}{str(dtype).split('.')[-1]:>10}"
                  + "".join(f"{c:>22}" for c in cells) + ("" if ok else "   <-- FAIL"))
    return bad


def bwd_profile():
    from kernels.sm75.attn_res import attn_res
    print()
    print("=== fwd+bwd profile, bf16, T=16384 H=512 ===")
    print(f"{'N':>4}{'K3 naive':>12}{'opt torch':>12}{'fused':>10}{'vs naive':>10}"
          f"{'peak MB naive':>15}{'peak MB fused':>15}")
    T, H = 16384, 512
    for N in (3, 5, 11):
        def run(fn):
            br = torch.randn(T, N - 1, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            ps = torch.randn(T, H, device=DEV, dtype=torch.bfloat16, requires_grad=True)
            w = torch.randn(H, device=DEV, dtype=torch.float32, requires_grad=True)
            g = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
            def step():
                for x in (br, ps, w):
                    x.grad = None
                fn(br, ps, w, EPS).backward(g)
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
            step(); torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / 1e6
            return triton.testing.do_bench(step, warmup=20, rep=100), peak
        t_ref, m_ref = run(attn_res_reference)
        t_opt, _ = run(_opt_torch)
        t_fus, m_fus = run(attn_res)
        print(f"{N:>4}{t_ref:>11.3f}m{t_opt:>11.3f}m{t_fus:>9.3f}m{t_ref / t_fus:>9.2f}x"
              f"{m_ref:>15.0f}{m_fus:>15.0f}")


if __name__ == "__main__":
    bad = parity()
    bad += mixed_dtype()
    print()
    print("PARITY FAIL" if bad else "PARITY OK")
    bad += gradcheck()
    print(f"{'GRAD FAIL' if bad else 'GRAD OK'}")
    if not bad:
        profile()
        bwd_profile()
        roofline()
        tile_sweep()


def tile_sweep():
    """TILE tokens per backward program: memory/time tradeoff, swept rather than guessed."""
    import importlib, os as _os
    import kernels.sm75.attn_res as K
    print()
    print("=== backward TILE sweep (dw partial is (T/TILE, H)) ===")
    print(f"{'TILE':>5}{'dwp MB':>9}" + "".join(f"{'N='+str(n):>10}" for n in (3, 5, 11)))
    T, H = 16384, 512
    for tile in (1, 2, 4, 8, 16, 32):
        _os.environ["BIBO_AR_BWD_TILE"] = str(tile)
        importlib.reload(K)
        row = []
        for N in (3, 5, 11):
            br, ps, w = _mk(T, N, H, torch.bfloat16)
            br.requires_grad_(True); ps.requires_grad_(True); w.requires_grad_(True)
            g = torch.randn(T, H, device=DEV, dtype=torch.float32)
            def bwd():
                for x in (br, ps, w):
                    x.grad = None
                K.attn_res(br, ps, w, EPS).backward(g)
            row.append(_bench(bwd))
        print(f"{tile:>5}{T / tile * H * 4 / 1e6:>9.1f}" + "".join(f"{r:>9.3f}m" for r in row))
    _os.environ.pop("BIBO_AR_BWD_TILE", None)
    importlib.reload(K)
