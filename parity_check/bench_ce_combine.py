"""(1) fused-CE row kernels: config sweep of _fwd_reduce_kernel (online logsumexp over V) and
_grad_logits_kernel on one real chunk (1 GB of bf16 logits, V=81920), GB/s.
(2) MoE dX deterministic combine: fp32 vs bf16 row buffer, time and error vs fp64 at the board shape.
(3) combine_gather storing bf16 directly == fp32 store + .to(bf16), bitwise.

    python parity_check/bench_ce_combine.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton

CE = importlib.import_module("kernels.sm75.cross_entropy")
FG = importlib.import_module("kernels.sm120.moe_fused_glu")
dev, bf = "cuda", torch.bfloat16


def ms(f, n=20):
    for _ in range(3):
        f()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(n):
        f()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / n


# ---------------- (1) CE
V = 81920
C = CE._chunk_rows(65536, V, 1024 * 1024 * 1024)
g = torch.Generator(device=dev).manual_seed(0)
logits = (torch.randn(C, V, device=dev, generator=g) * 3).to(bf)
labels = torch.randint(0, V, (C,), device=dev, generator=g)
lse, tgt = torch.empty(C, device=dev), torch.empty(C, device=dev)
nbytes = logits.numel() * 2
print(f"== CE fwd reduce, one chunk {C} x {V} bf16 ({nbytes / 2**30:.2f} GiB)")
ref_lse = torch.logsumexp(logits.float(), -1)
rows = []
for bv in (1024, 2048, 4096, 8192):
    for w in (4, 8, 16):
        f = lambda: CE._fwd_reduce_kernel[(C,)](logits, labels, lse, tgt, C, V, logits.stride(0),
                                                logits.stride(1), -100, BLOCK_V=bv, num_warps=w)
        t = ms(f)
        err = float((lse - ref_lse).abs().max())
        rows.append((t, bv, w, err))
for t, bv, w, err in sorted(rows)[:6]:
    print(f"   BLOCK_V={bv:5d} warps={w:2d}  {t:6.3f} ms  {nbytes / t / 1e6:6.0f} GB/s  max|lse err| {err:.1e}"
          + ("   <- current" if (bv, w) == (1024, 4) else ""))
cur = [r for r in rows if r[1:3] == (1024, 4)][0]
print(f"   current BLOCK_V=1024 warps=4  {cur[0]:6.3f} ms  {nbytes / cur[0] / 1e6:6.0f} GB/s")

print(f"== CE grad_logits (read + write in place)")
nv = torch.tensor([float(C)], device=dev)
work = logits.clone()
rows = []
for bm in (1, 2, 4, 8, 16):
    for bv in (512, 1024, 2048, 4096):
        for w in (4, 8):
            if bm * bv > 16384:
                continue
            f = lambda: CE._grad_logits_kernel[(triton.cdiv(C, bm), triton.cdiv(V, bv))](
                work, lse, labels, nv, C, V, -100, work.stride(0), work.stride(1),
                BLOCK_M=bm, BLOCK_V=bv, num_warps=w)
            rows.append((ms(f, 10), bm, bv, w))
for t, bm, bv, w in sorted(rows)[:6]:
    print(f"   BLOCK_M={bm:2d} BLOCK_V={bv:5d} warps={w}  {t:6.3f} ms  {2 * nbytes / t / 1e6:6.0f} GB/s"
          + ("   <- current" if (bm, bv, w) == (8, 1024, 4) else ""))
cur = [r for r in rows if r[1:] == (8, 1024, 4)][0]
print(f"   current BLOCK_M=8 BLOCK_V=1024 warps=4  {cur[0]:6.3f} ms  {2 * nbytes / cur[0] / 1e6:6.0f} GB/s")
del logits, work

# ---------------- (2) dX combine rows fp32 vs bf16, board MoE layer
N, K, E, H, I2 = 65536, 6, 64, 512, 1536
torch.manual_seed(0)
idx = torch.randn(N, E, device=dev).topk(K, dim=-1).indices
flat = idx.flatten()
sorted_e, order = flat.sort(stable=True)
counts = torch.bincount(sorted_e, minlength=E)
tm = FG.build_tile_map(None, counts, dev, bm=FG._GG[0], m_rows=N * K)
inv = FG.inverse_order(order)
a = (torch.randn(N * K, I2, device=dev) * 0.1).to(bf)
w = (torch.randn(E, I2, H, device=dev) * I2 ** -0.5).to(bf)
# fp64 truth: row r = a[r] @ w[e_r]; token t sums its K rows
rows64 = torch.empty(N * K, H, device=dev, dtype=torch.float64)
st = 0
for e in range(E):
    c = int(counts[e])
    rows64[st:st + c] = a[st:st + c].double() @ w[e].double()
    st += c
ref = rows64[inv.long().view(N, K)].sum(1)
print("== MoE dX deterministic combine (grouped GEMM rows -> k-way gather-sum), bf16 output")
for name, rdt in (("fp32 rows", torch.float32), ("bf16 rows", torch.bfloat16)):
    f = lambda: FG.grouped_gemm_gather(a, w, inv, tm, N, K, out_dtype=bf, rows_dtype=rdt)
    o = f()
    rel = float((o.double() - ref).norm() / ref.norm())
    print(f"   {name}:  {ms(f):6.3f} ms   rel err vs fp64 {rel:.3e}")
ro = FG.grouped_gemm_gather(a, w, inv, tm, N, K)
print(f"   (fp32 output, fp32 rows: rel err {float((ro.double() - ref).norm() / ref.norm()):.3e} -- the bf16 cast floor)")

# ---------------- (3) combine_gather bf16 store == fp32 store then cast
eo = torch.randn(N * K, H, device=dev).to(bf)
sw = torch.rand(N * K, device=dev)
x32 = FG.combine_gather(eo, inv, N, K, w=sw).to(bf)
x16 = FG.combine_gather(eo, inv, N, K, w=sw, out_dtype=bf)
print(f"== combine_gather direct bf16 store vs fp32 + .to(bf16): {'bitwise identical' if torch.equal(x32, x16) else 'DIFFERS'}")
d32 = FG.grouped_gemm_gather(a, w, inv, tm, N, K).to(bf)
d16 = FG.grouped_gemm_gather(a, w, inv, tm, N, K, out_dtype=bf)
print(f"== dX grouped_gemm_gather direct bf16 store vs fp32 + .to(bf16): {'bitwise identical' if torch.equal(d32, d16) else 'DIFFERS'}")
print("ALLDONE_CE_COMBINE")
