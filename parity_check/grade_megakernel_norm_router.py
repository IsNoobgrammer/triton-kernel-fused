"""Grade the fused RMSNorm+router megakernel against an FP64 ground truth AND today's stack.

    python -m parity_check.grade_megakernel_norm_router

Three contenders, one fp64 reference:

    eager bf16          pure PyTorch, BiBoRMSNorm + BiBoMoERouter semantics
    liger + eager       TODAY'S STACK. The model patches {liger_norm, liger_rope, moe, xsa} --
                        there is NO router patch, so only the norm is accelerated and the router
                        runs plain PyTorch. This is the baseline the megakernel has to beat.
    megakernel          the fused replacement

"Closer to fp64" is reported, not asserted on. This project has already paid for the lesson that a
more-accurate kernel can cost real bpb, so the numbers go side by side and the call is a judgement.

The router-weight error is DECOMPOSED. Comparing dense [T,E] expert->weight maps, a token routed to
a different expert than fp64 differs by that entire weight (~0.17), which swamps any arithmetic
error and makes every contender read as ~1e-1 regardless of precision. Restricting to tokens whose
selection AGREES shows what the weight math is actually worth; the flip count is reported separately
because a flip is categorical, not a rounding error.
"""
import torch

from kernels.sm120.megakernel.moe import norm_router_forward, norm_router_reference

DEV = "cuda"


def _mapping(idx, w, E):
    """[T,K] indices + weights -> dense [T,E], so comparison is order-independent (eager's
    topk(sorted=False) leaves index order undefined)."""
    out = torch.zeros(idx.shape[0], E, device=idx.device, dtype=torch.float64)
    out.scatter_(1, idx.long(), w.to(torch.float64))
    return out


def _router_from_hn(hn, rw, bias, K):
    """The eager router, given an already-normed hidden. Mirrors ffn/router.py exactly."""
    scores = torch.sigmoid((hn @ rw).float())
    _, idx = torch.topk(scores + bias, K, dim=-1, sorted=False)
    idx, _ = torch.sort(idx, dim=-1)
    w = scores.gather(-1, idx)
    return idx.to(torch.int32), w / (w.sum(-1, keepdim=True) + 1e-20)


def _maxerr(a, b):
    return (a.double() - b.double()).abs().max().item()


def grade(T=8192, H=512, E=64, K=6, seed=0, eps=1e-6):
    torch.manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    nw = torch.randn(H, device=DEV, dtype=torch.float32) * 0.1 + 1.0
    rw = torch.randn(H, E, device=DEV, dtype=torch.bfloat16) * 0.02
    bias = torch.randn(E, device=DEV, dtype=torch.float32) * 0.05   # non-zero: exercises sel-vs-weight

    hn64, idx64, w64, rstd64 = norm_router_reference(
        x.double(), nw.double(), rw.double(), bias.double(), K, eps, dtype=torch.float64)
    map64 = _mapping(idx64, w64, E)
    sel64 = map64 > 0

    hnE, idxE, wE, rstdE = norm_router_reference(x, nw, rw, bias, K, eps)
    mapE = _mapping(idxE, wE, E)

    # today's stack: liger norm, eager router. EXACT call from ablate/common/patches.py.
    try:
        from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        hnL = LigerRMSNormFunction.apply(x, nw, eps, 0.0, "llama", False)
        idxL, wL = _router_from_hn(hnL, rw, bias, K)
        mapL, liger = _mapping(idxL, wL, E), True
    except Exception as e:
        print(f"  [liger unavailable: {type(e).__name__}: {str(e)[:70]}]")
        hnL, mapL, liger = hnE, mapE, False

    hnK, idxK, wK, rstdK, cntK = norm_router_forward(x, nw, rw, bias, K, eps, write_hn=True)
    mapK = _mapping(idxK, wK, E)

    lab = "liger+eager (TODAY)" if liger else "liger N/A"
    cands = [("eager bf16", hnE, mapE), (lab, hnL, mapL), ("megakernel", hnK, mapK)]

    print(f"\nT={T} H={H} E={E} K={K}   ground truth = fp64 eager")
    print(f"  {'contender':<22}{'h_norm':>12}{'weights(all)':>15}{'weights(agree)':>16}"
          f"{'selection':>11}{'flips':>9}")
    for name, hn, m in cands:
        ok = ((m > 0) == sel64).all(1)
        wagree = ((m[ok] - map64[ok]).abs().max().item() if ok.any() else float("nan"))
        print(f"  {name:<22}{_maxerr(hn, hn64):>12.3e}{_maxerr(m, map64):>15.3e}"
              f"{wagree:>16.3e}{ok.float().mean().item():>10.4%}{(~ok).sum().item():>9d}")

    s = mapK.sum(1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3), \
        f"kernel weights do not sum to 1: min {s.min():.6f} max {s.max():.6f}"
    ref_cnt = torch.bincount(idxK.reshape(-1).long(), minlength=E).to(torch.int32)
    assert torch.equal(cntK, ref_cnt), "counts != bincount(indices)"
    assert cntK.sum().item() == T * K, f"counts sum {cntK.sum().item()} != T*K {T*K}"
    print(f"  weights sum to 1, counts exact (sum {cntK.sum().item()}, "
          f"min {cntK.min().item()} max {cntK.max().item()}): OK")


if __name__ == "__main__":
    for T in (4096, 65536):
        grade(T=T)
