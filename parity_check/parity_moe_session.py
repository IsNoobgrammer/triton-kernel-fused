"""MoE fwd AND bwd parity for every Sep 25-26 change, against an fp32 per-expert-loop reference
(autocast off), for the board's sparse layer (64 experts, top-6, I=768) and the L0 all-active layer
(8/8, I=576), over two accumulated micro-batches:

    old  = every new path OFF: torch._grouped_mm dW, autograd grad return, materialized x_s, sparse L0
    new  = every new path ON:  grouped_wgrad, dW into .grad, gather-on-load, dense L0
           (+ always-on: histogram counts, partitioned chunk order, bf16 combine stores)

Compared: the forward output of EACH micro-batch, d_hidden and d_wt of each micro-batch, and the
accumulated d_gate_up, d_down, d_theta. new must be at least as close to fp32 as old (<= 2% slack),
bitwise repeatable, and gather on/off must be bitwise identical in BOTH directions.

    python parity_check/parity_moe_session.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

M = importlib.import_module("kernels.sm75.moe")
dev = "cuda"
H = 512
CODES = {E: torch.full((E,), 8, device=dev, dtype=torch.int64) for E in (8, 64)}


def config(mode):
    new = mode == "new"
    M.WGRAD = "triton" if new else "torch"
    M.ACC_GRAD = new
    M.GATHER_X = new
    M.DENSE_ALL_ACTIVE = new
    os.environ["BIBO_MOE_FORCE_LOOP"] = "1" if mode == "ref" else "0"


def run(case, mode, gather=None, N=65536):
    E, K, I = case
    config(mode)
    if gather is not None:
        M.GATHER_X = gather
    torch.manual_seed(0)
    gu = torch.nn.Parameter(torch.randn(E, 2 * I, H, device=dev) * H ** -0.5)
    dn = torch.nn.Parameter(torch.randn(E, H, I, device=dev) * I ** -0.5)
    ap = torch.nn.Parameter(torch.randn(E, device=dev) * 0.3)
    r = {}
    for micro in range(2):
        g = torch.Generator(device=dev).manual_seed(100 + micro)
        hidden = torch.randn(N, H, device=dev, generator=g).requires_grad_()
        wt, idx = torch.randn(N, E, device=dev, generator=g).softmax(-1).topk(K, dim=-1)
        wt = wt.detach().requires_grad_()
        go = torch.randn(N, H, device=dev, generator=g)
        if mode == "ref":
            out = M.moe_per_expert(hidden, idx, wt, gu, dn, CODES[E], ap)
        else:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = M.moe_per_expert(hidden, idx, wt, gu, dn, CODES[E], ap)
        out.float().backward(go.to(out.dtype))
        r[f"out[{micro}]"] = out.detach().float()
        r[f"d_hidden[{micro}]"] = hidden.grad.float()
        r[f"d_wt[{micro}]"] = wt.grad.float()
    r.update({"d_gate_up(acc)": gu.grad.float(), "d_down(acc)": dn.grad.float(),
              "d_theta(acc)": ap.grad.float()})
    return r, M._LAST_PATH


ok = True
for case in ((64, 6, 768), (8, 8, 576)):
    ref, _ = run(case, "ref")
    old, p_old = run(case, "old")
    new, p_new = run(case, "new")
    new2, _ = run(case, "new")
    print(f"== E={case[0]} top-{case[1]} I={case[2]}   old path={p_old}  new path={p_new}")
    for k in ref:
        rn = ref[k].norm().item() or 1.0
        eo = (old[k] - ref[k]).norm().item() / rn
        en = (new[k] - ref[k]).norm().item() / rn
        rep = torch.equal(new[k], new2[k])
        good = rep and en <= eo * 1.02
        ok &= good
        print(f"   {k:15s} rel err vs fp32: old {eo:.3e}  new {en:.3e}  repeat {'bitwise' if rep else 'DIFFERS'}"
              f"  {'OK' if good else 'WORSE'}")

a, _ = run((64, 6, 768), "new", gather=False)
b, _ = run((64, 6, 768), "new", gather=True)
bad = [k for k in a if not torch.equal(a[k], b[k])]
ok &= not bad
print(f"== gather-on-load vs materialized x_s, fwd + bwd (E=64): "
      + ("bitwise identical on every output and grad" if not bad else f"DIFFER on {bad}"))
print("MOE SESSION PASS" if ok else "MOE SESSION FAIL")
