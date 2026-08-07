"""Grade the fused RMSNorm+router megakernel phase against an FP64 eager ground truth.

    python -m parity_check.grade_megakernel_norm_router

Reports THREE contenders against the same fp64 reference, because "our kernel is accurate" is not
the question -- the question is whether it moved relative to what the model was trained with:

    eager bf16   what the model runs today
    megakernel   the fused replacement
    (fp64 eager) ground truth, computed end to end in double

A kernel that is closer to fp64 than eager is NOT automatically better here: this project has
already paid for that lesson once, where a more-accurate kernel cost real bpb. So the numbers are
reported side by side and the verdict is a judgement, not an assert on the fp64 error alone.

Top-k indices are compared as an expert->weight MAPPING rather than positionally: eager calls
topk(sorted=False), so its index order is undefined and there is nothing positional to match.
"""
import torch

from kernels.sm120.megakernel.moe import norm_router_forward, norm_router_reference

DEV = "cuda"


def _mapping(idx, w, E):
    """[T,K] indices + weights -> dense [T,E] so comparison is order-independent."""
    out = torch.zeros(idx.shape[0], E, device=idx.device, dtype=torch.float64)
    out.scatter_(1, idx.long(), w.to(torch.float64))
    return out


def grade(T=8192, H=512, E=64, K=6, seed=0, eps=1e-6):
    torch.manual_seed(seed)
    x = torch.randn(T, H, device=DEV, dtype=torch.bfloat16)
    nw = torch.randn(H, device=DEV, dtype=torch.float32) * 0.1 + 1.0
    rw = (torch.randn(H, E, device=DEV, dtype=torch.bfloat16) * 0.02)
    # a NON-ZERO bias, or the whole selection-vs-weight distinction goes untested
    bias = (torch.randn(E, device=DEV, dtype=torch.float32) * 0.05)

    # ---- ground truth in fp64, end to end
    hn64, idx64, w64, rstd64 = norm_router_reference(
        x.double(), nw.double(), rw.double(), bias.double(), K, eps, dtype=torch.float64)
    map64 = _mapping(idx64, w64, E)

    # ---- contender A: eager in the dtype the model actually runs
    hnE, idxE, wE, rstdE = norm_router_reference(x, nw, rw, bias, K, eps)
    mapE = _mapping(idxE, wE, E)

    # ---- contender B: the megakernel
    hnK, idxK, wK, rstdK, cntK = norm_router_forward(x, nw, rw, bias, K, eps, write_hn=True)
    mapK = _mapping(idxK, wK, E)

    def err(a, b):
        d = (a.double() - b.double()).abs()
        return d.max().item(), d.mean().item()

    print(f"T={T} H={H} E={E} K={K}  ground truth = fp64 eager\n")
    print(f"{'quantity':<22}{'eager bf16 max':>18}{'megakernel max':>18}   verdict")
    rows = [("h_norm", hnE, hnK, hn64), ("router weights", mapE, mapK, map64),
            ("rstd", rstdE, rstdK, rstd64)]
    for name, a, b, ref in rows:
        ea, _ = err(a, ref)
        eb, _ = err(b, ref)
        verdict = "kernel closer" if eb < ea else ("eager closer" if eb > ea else "tie")
        print(f"{name:<22}{ea:>18.3e}{eb:>18.3e}   {verdict}")

    # selection agreement: which experts were picked, ignoring order
    selE = (mapE > 0).sum(1)
    sel_match_kernel = ((mapK > 0) == (map64 > 0)).all(1).float().mean().item()
    sel_match_eager = ((mapE > 0) == (map64 > 0)).all(1).float().mean().item()
    print(f"\n{'expert SELECTION vs fp64':<22}{sel_match_eager:>17.4%}{sel_match_kernel:>18.4%}"
          f"   (fraction of tokens picking the identical expert set)")
    assert (selE == K).all(), "reference did not select exactly K experts per token"

    # a selection flip is categorical, not a rounding error: the token goes to a different expert
    # entirely, so it is worth separating from the weight error above.
    flips = (~((mapK > 0) == (map64 > 0)).all(1)).sum().item()
    print(f"tokens whose selection differs from fp64: {flips} / {T}")

    # weights must still be a valid distribution regardless
    s = mapK.sum(1)
    assert torch.allclose(s, torch.ones_like(s), atol=1e-3), \
        f"kernel weights do not sum to 1: min {s.min():.6f} max {s.max():.6f}"
    print("kernel weights sum to 1 per token: OK")

    # per-expert counts must match a bincount of the indices exactly, and total T*K. This is the
    # tile map's input: a wrong count silently drops or duplicates tokens in the grouped GEMM.
    ref_cnt = torch.bincount(idxK.reshape(-1).long(), minlength=E).to(torch.int32)
    assert torch.equal(cntK, ref_cnt), (
        f"counts mismatch: max delta {(cntK.long()-ref_cnt.long()).abs().max().item()}")
    assert cntK.sum().item() == T * K, f"counts sum {cntK.sum().item()} != T*K {T*K}"
    print(f"per-expert counts exact (sum {cntK.sum().item()} = T*K, "
          f"min {cntK.min().item()} max {cntK.max().item()}): OK")
    return dict(eager=err(mapE, map64)[0], kernel=err(mapK, map64)[0], flips=flips)


if __name__ == "__main__":
    for T in (4096, 65536):
        grade(T=T)
        print("-" * 78)
