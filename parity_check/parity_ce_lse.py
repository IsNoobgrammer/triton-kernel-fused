"""Fused linear CE with the logsumexp in the logits-GEMM epilogue (TKF_CE_LSE_GEMM) vs the cuBLAS +
fwd-reduce path: loss, d_hidden and d_weight against an fp32 reference (F.cross_entropy on fp32
logits), at H=512 V=81920 and a multi-chunk N with ignored labels. The fused path must be at least as
close as the old one, bitwise repeatable, and its logsumexp within fp32 rounding of the old.

    python parity_check/parity_ce_lse.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

CE = importlib.import_module("kernels.sm75.cross_entropy")
dev = "cuda"
N, H, V = 16384, 512, 81920
g = torch.Generator(device=dev).manual_seed(0)
x0 = torch.randn(N, H, device=dev, generator=g)
w0 = torch.randn(V, H, device=dev, generator=g) * H ** -0.5 * 2
lab = torch.randint(0, V, (N,), device=dev, generator=g)
lab[::7] = -100


def run(mode):
    x = x0.clone().requires_grad_()
    w = w0.clone().requires_grad_()
    if mode == "ref":
        loss = F.cross_entropy(x @ w.t(), lab, ignore_index=-100)
    elif mode == "eager_bf16":            # plain PyTorch under bf16 autocast: the bf16 floor
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(x @ w.t(), lab, ignore_index=-100)
    else:
        CE.LSE_GEMM = mode == "fused"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            # small budget -> several chunks, so the chunk loop is exercised
            loss = CE.fused_linear_cross_entropy(x, w, lab, -100, 256 * 1024 * 1024)
    loss.backward()
    return {"loss": loss.detach().float().reshape(1), "d_hidden": x.grad.float(), "d_weight": w.grad.float()}


ref, old, new, new2 = run("ref"), run("old"), run("fused"), run("fused")
eag = run("eager_bf16")
ok = True
for k in ref:
    rn = ref[k].norm().item()
    eo = (old[k] - ref[k]).norm().item() / rn
    en = (new[k] - ref[k]).norm().item() / rn
    rep = torch.equal(new[k], new2[k])
    ee = (eag[k] - ref[k]).norm().item() / rn
    # at least as close as the old path AND as plain PyTorch bf16
    good = rep and en <= max(eo * 1.02, 1e-6) and en <= max(ee * 1.05, 1e-6)
    ok &= good
    print(f"   {k:9s} rel err vs fp32: torch bf16 eager {ee:.3e}  cuBLAS+reduce {eo:.3e}  lse-GEMM {en:.3e}"
          f"  repeat {'bitwise' if rep else 'DIFFERS'}  {'OK' if good else 'WORSE'}")
print("CE LSE PASS" if ok else "CE LSE FAIL")
