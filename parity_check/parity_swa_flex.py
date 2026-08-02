"""Parity gate: FlexAttention SWA vs BiBo's eager banded core -- values, grads, and the sink identity.

The flex path is not a transcription of the eager path, it is an algebraic reformulation, so it
needs a real gate rather than a shape check:

  1. BAND      -- flex must attend to exactly the same (q, kv) pairs. A block mask works on 128x128
                  blocks, so a window that is not a block multiple (or a q_len that is not) is
                  where an off-by-one hides. Tested at W = 64 / 128 / 200 / 512, including a
                  non-multiple window and a non-multiple sequence length.
  2. SINK      -- eager appends a value-less column of logit beta; flex instead scales the output
                  by sigmoid(lse - beta). Those are equal only if lse is natural-log over SCALED
                  scores. A missing `scale=` or a log2 lse passes every shape check and silently
                  changes the sink strength, so this is compared numerically against eager.
  3. GRADS     -- d_q, d_k, d_v and d_sink. The sink gradient is the new path (it now flows through
                  a sigmoid rather than a softmax column) and is the most likely thing to be wrong.
  4. GQA       -- H != H_kv with enable_gqa, since flex broadcasts instead of repeat_kv.

Needs the BiBo venv and CUDA. Run from the triton-kernel-fused repo:
    python parity_check/parity_swa_flex.py
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BiBo")))
import _paths  # noqa: F401
import torch

from src.modeling.attn import swa as SWA
from src.modeling.attn.utils import causal_band_mask, eager_attention_forward

DEV = "cuda"


def _inputs(B, H, HKV, S, D, seed, grad=True):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.randn(B, H, S, D, generator=g, device=DEV, dtype=torch.float32)
    k = torch.randn(B, HKV, S, D, generator=g, device=DEV, dtype=torch.float32)
    v = torch.randn(B, HKV, S, D, generator=g, device=DEV, dtype=torch.float32)
    s = torch.randn(H, generator=g, device=DEV, dtype=torch.float32) * 0.5
    if grad:
        q, k, v, s = (t.detach().requires_grad_(True) for t in (q, k, v, s))
    return q, k, v, s


def _eager(q, k, v, sinks, W, scaling, groups):
    m = causal_band_mask(q.shape[-2], k.shape[-2], W, q.dtype, q.device)
    return eager_attention_forward(q, k, v, m, scaling, groups, sinks=sinks)[0]


def rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    if not SWA._HAS_FLEX:
        print("flex_attention unavailable in this torch -- gate cannot run")
        raise SystemExit(1)
    ok = True
    D, TOL = 64, 2e-4      # tf32-era tolerance; flex and eager reduce in a different order

    print(f"{'case':<28}{'fwd':>10}{'d_q':>10}{'d_k':>10}{'d_v':>10}{'d_sink':>10}")
    # non-multiple window (200) and non-multiple length (300) are the block-boundary cases
    for (B, H, HKV, S, W, sink) in [(2, 4, 2, 512, 128, True),
                                    (2, 4, 2, 512, 128, False),
                                    (2, 4, 2, 512, 64, True),
                                    (2, 4, 2, 512, 200, True),
                                    (2, 4, 2, 512, 512, True),
                                    (2, 4, 4, 300, 128, True),
                                    (1, 8, 2, 256, 128, True)]:
        scaling, groups = D ** -0.5, H // HKV
        go = torch.randn(B, H, S, D, generator=torch.Generator(device=DEV).manual_seed(5),
                         device=DEV)
        qf, kf, vf, sf = _inputs(B, H, HKV, S, D, 11)
        qe, ke, ve, se = _inputs(B, H, HKV, S, D, 11)
        of = SWA.swa_attention(qf, kf, vf, sf if sink else None, sliding_window=W,
                               num_key_value_groups=groups, scaling=scaling)[0]
        oe = _eager(qe, ke, ve, se if sink else None, W, scaling, groups)
        (of * go).sum().backward()
        (oe * go).sum().backward()
        es = rel(sf.grad, se.grad) if sink else 0.0
        errs = [rel(of, oe), rel(qf.grad, qe.grad), rel(kf.grad, ke.grad), rel(vf.grad, ve.grad), es]
        good = max(errs) < TOL
        ok &= good
        tag = f"S{S} W{W} H{H}/{HKV} {'sink' if sink else 'nosink'}"
        print(f"{tag:<28}" + "".join(f"{e:>10.1e}" for e in errs) + ("" if good else "  <-- FAIL"))

    # ---- the band is REALLY enforced (a mask that silently widens still matches nothing above) --
    # Feed one-hot values so the output reads back exactly which kv positions were attended.
    print("\nband exactness (output must be zero outside the window):")
    B, H, S, W = 1, 2, 384, 128
    q = torch.randn(B, H, S, D, device=DEV)
    k = torch.randn(B, H, S, D, device=DEV)
    v = torch.eye(S, device=DEV)[None, None].expand(B, H, S, S).contiguous()[..., :D * 0 + S]
    # v is (B,H,S,S): row j is the one-hot for kv position j, so out[.., q, j] = attn weight (q, j)
    out = SWA.swa_attention(q, k, v, None, sliding_window=W, num_key_value_groups=1,
                            scaling=D ** -0.5)[0]
    i = torch.arange(S, device=DEV)
    allowed = (i[:, None] >= i[None, :]) & ((i[:, None] - i[None, :]) < W)
    leak = out[0, 0].masked_select(~allowed).abs().max().item()
    mass = out[0, 0].masked_select(allowed).sum().item() / S
    print(f"  max weight outside band {leak:.2e} (must be ~0), mean in-band mass {mass:.4f} (~1)")
    ok &= leak < 1e-6 and abs(mass - 1.0) < 1e-3

    # ---- the sink must actually DO something, and more sink = less output -----------------------
    print("\nsink is live and monotone (bigger beta -> more mass parked -> smaller output):")
    q, k, v, _ = _inputs(2, 4, 2, 512, D, 3, grad=False)
    norms = []
    for b in (-4.0, 0.0, 4.0):
        s = torch.full((4,), b, device=DEV)
        o = SWA.swa_attention(q, k, v, s, sliding_window=128, num_key_value_groups=2,
                              scaling=D ** -0.5)[0]
        norms.append(o.norm().item())
    print(f"  |out| at beta=-4/0/+4: {norms[0]:.3f} {norms[1]:.3f} {norms[2]:.3f}")
    ok &= norms[0] > norms[1] > norms[2]

    print("\nALL PASS" if ok else "\nFAILURES ABOVE")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
