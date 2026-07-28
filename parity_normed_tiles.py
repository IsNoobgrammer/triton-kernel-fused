"""Regression check: the RMS-normed act codes (2/6/7) must take the Triton grouped GEMMs.

The fused GEMM+GLU epilogue genuinely cannot serve codes 2/6/7 (it holds a BN-wide slice of a row,
not the whole row, so it can't see the per-row RMS). But the THREE generic grouped GEMMs -- it@W2,
grad_inter, and dX-with-fused-scatter -- are activation-agnostic, and they used to be gated on
`fused_supported()` along with the epilogue. That silently sent every normed run back to
torch._grouped_mm plus a separate index_add_.

This asserts (a) the tile map is built for every code, (b) `grouped_gemm` actually fires on the
BiBo shape rather than returning None, and (c) numerics are unchanged vs moe_eager. Run on the box:
    python parity_normed_tiles.py
"""
import importlib

import torch

# kernels/sm75/__init__.py re-exports a `moe` FUNCTION, which shadows the submodule of the same
# name -- `from kernels.sm75 import moe` hands back the function. Grab the module itself.
M = importlib.import_module("kernels.sm75.moe")

H, I, E, TOP_K, N = 512, 768, 8, 2, 4096
CODES = {0: "silu", 1: "relu2", 2: "normsilu", 6: "normrelu2", 7: "normsitu"}


def build(code, dtype, dev):
    g = torch.Generator(device=dev).manual_seed(0)
    hid = (torch.randn(N, H, generator=g, device=dev, dtype=torch.float32) * 0.5)
    w1 = torch.randn(E, 2 * I, H, generator=g, device=dev, dtype=torch.float32) * (H ** -0.5)
    w2 = torch.randn(E, H, I, generator=g, device=dev, dtype=torch.float32) * (I ** -0.5)
    logits = torch.randn(N, E, generator=g, device=dev, dtype=torch.float32)
    wt, idx = logits.softmax(-1).topk(TOP_K, dim=-1)
    codes = torch.full((E,), code, device=dev, dtype=torch.int32)
    ts = [t.to(dtype).detach().requires_grad_(True) for t in (hid, w1, w2)]
    return ts, idx.contiguous(), wt.to(dtype).detach(), codes


def run(fn, code, dtype, dev):
    (hid, w1, w2), idx, wt, codes = build(code, dtype, dev)
    out = fn(hid, idx, wt, w1, w2, codes)
    out.square().mean().backward()
    return out.float(), [t.grad.float() for t in (hid, w1, w2)]


def main():
    dev = "cuda"
    assert torch.cuda.is_available(), "needs a GPU"
    if not hasattr(torch, "_grouped_mm") or torch.cuda.get_device_capability(dev)[0] < 8:
        raise SystemExit("grouped path unavailable on this device -- nothing to check")

    # (b) the tile map must be built, and grouped_gemm must not bail to None, for EVERY code.
    fired = {"map": 0, "gemm": 0, "scatter": 0}
    fg = M._FUSED_GLU
    assert fg is not None, "sm120 fused-GLU module did not import"
    orig = (fg.build_tile_map, fg.grouped_gemm, fg.grouped_gemm_scatter)

    def wrap(key, f, none_ok=False):
        def inner(*a, **k):
            r = f(*a, **k)
            assert none_ok or r is not None, f"{key} returned None -- shape does not fit the tuned tiles"
            fired[key] += r is not None
            return r
        return inner

    fg.build_tile_map = wrap("map", orig[0])
    fg.grouped_gemm = wrap("gemm", orig[1])
    fg.grouped_gemm_scatter = wrap("scatter", orig[2])
    try:
        ok = True
        for code, name in CODES.items():
            for k in fired:
                fired[k] = 0
            ref_o, ref_g = run(M.moe_eager, code, torch.float32, dev)
            got_o, got_g = run(M.moe_per_expert, code, torch.bfloat16, dev)
            assert fired["map"], f"code {code} ({name}): no tile map built -- back on the cuBLAS path"
            # fwd it@W2 always; bwd grad_inter only for the normed codes (0/1 fuse it into the epilogue)
            want_gemm = 1 if code in (0, 1) else 2
            assert fired["gemm"] >= want_gemm, \
                f"code {code}: grouped_gemm fired {fired['gemm']}x, want >={want_gemm}"
            assert fired["scatter"] == 1, f"code {code}: dX scatter fired {fired['scatter']}x, want 1"
            rel = [((a - b).norm() / b.norm().clamp_min(1e-12)).item()
                   for a, b in zip([got_o] + got_g, [ref_o] + ref_g)]
            good = max(rel) < 5e-2                      # bf16 vs fp32 reference
            ok &= good
            print(f"code {code:>2} {name:<10} {'PASS' if good else 'FAIL'}  "
                  f"rel out/dx/dw1/dw2 = " + " ".join(f"{r:.1e}" for r in rel) +
                  f"   [gemm x{fired['gemm']}, scatter x{fired['scatter']}]")
        print("ALL PASS" if ok else "FAILURES ABOVE")
        raise SystemExit(0 if ok else 1)
    finally:
        fg.build_tile_map, fg.grouped_gemm, fg.grouped_gemm_scatter = orig


if __name__ == "__main__":
    main()
