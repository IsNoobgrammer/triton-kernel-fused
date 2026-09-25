"""Expert dW accumulated straight into the fp32 param .grad (TKF_MOE_ACC_GRAD) vs the autograd return:
two micro-batches, board MoE layer (64 experts, top-6, I=768) and the L0 dense layer (8/8, I=576).
Accumulated .grad must be at least as close to an fp32 reference (per-expert loop, autocast off) as
the autograd path, bitwise repeatable, and the non-expert grads must be unchanged.

    python parity_check/parity_moe_accgrad.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

M = importlib.import_module("kernels.sm75.moe")
dev = "cuda"
H = 512


def run(case, mode, N=65536):
    E, K, I = case
    torch.manual_seed(0)
    gu = torch.nn.Parameter(torch.randn(E, 2 * I, H, device=dev) * H ** -0.5)
    dn = torch.nn.Parameter(torch.randn(E, H, I, device=dev) * I ** -0.5)
    ap = torch.nn.Parameter(torch.randn(E, device=dev) * 0.3)
    codes = CODES[E]
    M.ACC_GRAD = mode == "acc"
    os.environ["BIBO_MOE_FORCE_LOOP"] = "1" if mode == "ref" else "0"
    hs = []
    for micro in range(2):
        g = torch.Generator(device=dev).manual_seed(100 + micro)
        hidden = torch.randn(N, H, device=dev, generator=g).requires_grad_()
        wt, idx = torch.randn(N, E, device=dev, generator=g).softmax(-1).topk(K, dim=-1)
        wt = wt.detach().requires_grad_()
        go = torch.randn(N, H, device=dev, generator=g)
        if mode == "ref":
            out = M.moe_per_expert(hidden, idx, wt, gu, dn, codes, ap)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = M.moe_per_expert(hidden, idx, wt, gu, dn, codes, ap)
        out.float().backward(go.to(out.dtype))
        hs += [hidden.grad.float(), wt.grad.float()]
    return {"d_gate_up": gu.grad.float(), "d_down": dn.grad.float(), "d_theta": ap.grad.float(),
            "d_hidden(1)": hs[0], "d_wt(2)": hs[3]}, M._LAST_PATH


CODES = {E: torch.full((E,), 8, device=dev, dtype=torch.int64) for E in (8, 64)}
ok = True
for case in ((64, 6, 768), (8, 8, 576)):
    ref, _ = run(case, "ref")
    old, p1 = run(case, "autograd")
    new, p2 = run(case, "acc")
    new2, _ = run(case, "acc")
    print(f"== E={case[0]} top-{case[1]} I={case[2]}  path={p2}")
    for k in ref:
        rn = ref[k].norm().item() or 1.0
        eo = (old[k] - ref[k]).norm().item() / rn
        en = (new[k] - ref[k]).norm().item() / rn
        rep = torch.equal(new[k], new2[k])
        good = rep and en <= eo * 1.02
        ok &= good
        print(f"   {k:12s} rel err vs fp32: autograd {eo:.3e}  acc-into-.grad {en:.3e}  "
              f"repeat {'bitwise' if rep else 'DIFFERS'}  {'OK' if good else 'WORSE'}")
# token gather folded into the GEMM loads (TKF_MOE_GATHER) must be BITWISE the materialized x_s
case = (64, 6, 768)
M.GATHER_X = False
nog, _ = run(case, "acc")
M.GATHER_X = True
gat, _ = run(case, "acc")
same = all(torch.equal(nog[k], gat[k]) for k in nog)
ok &= same
print(f"== gather-on-load vs materialized x_s (E=64): {'bitwise identical' if same else 'DIFFERS'}")
print("ACCGRAD PASS" if ok else "ACCGRAD FAIL")
