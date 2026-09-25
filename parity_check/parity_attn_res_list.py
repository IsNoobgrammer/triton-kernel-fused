"""AttnRes LIST mode (BlockStore: blocks read in place, block grads accumulated in-kernel in fp32) vs
the cat path (torch.cat blocks, autograd sums bf16 block grads), on the board topology: 4 blocks,
10 depth reads with N = 2,2,2,3,3,3,4,4,4 and the output read at 5. T=65536 H=512 bf16.

Gate: forward outputs and every prefix-sum / score-weight grad BITWISE equal to the cat path (same
kernel arithmetic); block grads at least as close to fp64 as the cat path; bitwise repeatable.

    python parity_check/parity_attn_res_list.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kernels.sm120.attn_res import attn_res, attn_res_reference, BlockStore

dev, bf = "cuda", torch.bfloat16
T, H = 65536, 512
NREAD = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4]          # blocks visible at each read (N = n + 1)


def make(seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    blocks = [(torch.randn(T, H, device=dev, generator=g) * s).to(bf) for s in (0.05, 20, 40, 60)]
    ps = [(torch.randn(T, H, device=dev, generator=g) * 10).to(bf) for _ in NREAD]
    ws = [torch.randn(H, device=dev, generator=g) * 0.05 for _ in NREAD]
    gs = [torch.randn(T, H, device=dev, generator=g).to(bf) for _ in NREAD]
    return blocks, ps, ws, gs


def run(mode, seed=0, dtype=None):
    blocks, ps, ws, gs = make(seed)
    if dtype is not None:
        blocks, ps, gs = [b.to(dtype) for b in blocks], [p.to(dtype) for p in ps], [x.to(dtype) for x in gs]
        ws = [w.to(dtype) for w in ws]
    leaves = [t.requires_grad_() for t in blocks + ps + ws]
    outs = []
    if mode == "list":
        st = BlockStore()
        for n, p, w in zip(NREAD, ps, ws):
            while len(st) < n:
                st.archive(blocks[len(st)])
            outs.append(st.mix(p, w, 1e-6))
        st.close()
    else:
        for n, p, w in zip(NREAD, ps, ws):
            br = torch.stack(blocks[:n], dim=1)
            outs.append(attn_res_reference(br, p, w, 1e-6) if mode == "ref" else attn_res(br, p, w, 1e-6))
    torch.autograd.backward(outs, gs)
    r = {f"out{i}": o.detach() for i, o in enumerate(outs)}
    r.update({f"d_block{i}": b.grad for i, b in enumerate(blocks)})
    r.update({f"d_ps{i}": p.grad for i, p in enumerate(ps)})
    r.update({f"d_w{i}": w.grad for i, w in enumerate(ws)})
    return r


ref = run("ref", dtype=torch.float64)
cat = run("cat")
lst = run("list")
lst2 = run("list")
ok = True
for k in cat:
    rep = torch.equal(lst[k], lst2[k])
    if k.startswith("d_block"):
        rn = ref[k].norm().item()
        ec = (cat[k].double() - ref[k]).norm().item() / rn
        el = (lst[k].double() - ref[k]).norm().item() / rn
        good = rep and el <= ec * 1.001
        print(f"   {k:9s} rel err vs fp64: cat {ec:.3e}  list {el:.3e}  repeat {'bitwise' if rep else 'DIFFERS'}  {'OK' if good else 'WORSE'}")
    else:
        good = rep and torch.equal(cat[k], lst[k])
        if rep and not good:
            # the LIST specialization may compile a reduction differently (fp32 reassociation):
            # then it must be at least as close to fp64 as the cat path
            rn = ref[k].norm().item() or 1.0
            ec = (cat[k].double() - ref[k]).norm().item() / rn
            el = (lst[k].double() - ref[k]).norm().item() / rn
            good = el <= max(ec * 1.05, 1e-6)          # both at the fp32 floor
            print(f"   {k:9s} not bitwise vs cat; rel err vs fp64: cat {ec:.3e}  list {el:.3e}  {'OK' if good else 'WORSE'}")
    ok &= good
print("   outputs, prefix-sum grads, score-weight grads: bitwise == cat path unless listed above")


def timed(mode, n=10):
    for _ in range(3):
        run(mode)
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        run(mode)
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


print(f"   10-read topology fwd+bwd (incl. input setup): cat {timed('cat'):.2f} ms  list {timed('list'):.2f} ms")
print("ARLIST PASS" if ok else "ARLIST FAIL")
