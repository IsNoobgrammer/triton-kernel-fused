"""Prototype: logits GEMM with the logsumexp partials in its epilogue (removes the separate fwd-reduce
read of all logits) vs today's cuBLAS mm + _fwd_reduce_kernel, one CE chunk (6553 x 81920, H=512).
lse must match the reference computed from the SAME bf16-rounded logits.

    python parity_check/bench_ce_lse_gemm.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import triton
import triton.language as tl

CE = importlib.import_module("kernels.sm75.cross_entropy")
dev, bf = "cuda", torch.bfloat16


@triton.jit
def _logits_lse_kernel(X, W, L, PM, PS, M, V, NT, K: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    # grouped launch order so a band of W tiles stays in L2 across row blocks
    width = GROUP * NT
    g = pid // width
    first = g * GROUP
    gs = tl.minimum(nm - first, GROUP)
    pm = first + (pid % width) % gs
    pn = (pid % width) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    lb = acc.to(tl.bfloat16)
    tl.store(L + rm[:, None].to(tl.int64) * V + rn[None, :], lb, mask=mm[:, None])
    xf = lb.to(tl.float32)                          # stats from the ROUNDED logits the grad pass reads
    m = tl.max(xf, axis=1)
    s = tl.sum(tl.exp(xf - m[:, None]), axis=1)
    tl.store(PM + rm * NT + pn, m, mask=mm)
    tl.store(PS + rm * NT + pn, s, mask=mm)


@triton.jit
def _lse_combine(PM, PS, LSE, M, NT, BT: tl.constexpr):
    r = tl.program_id(0)
    t = tl.arange(0, BT)
    mk = t < NT
    m = tl.load(PM + r * NT + t, mask=mk, other=-float("inf"))
    s = tl.load(PS + r * NT + t, mask=mk, other=0.0)
    mx = tl.max(m, 0)
    tl.store(LSE + r, mx + tl.log(tl.sum(s * tl.exp(m - mx), 0)))


def fused(x, w, cfg):
    BM, BN, BK, G, nw, ns = cfg
    M, K = x.shape
    V = w.shape[0]
    NT = V // BN
    L = torch.empty(M, V, device=x.device, dtype=bf)
    PM = torch.empty(M, NT, device=x.device)
    PS = torch.empty(M, NT, device=x.device)
    lse = torch.empty(M, device=x.device)
    _logits_lse_kernel[(triton.cdiv(M, BM) * NT,)](x, w, L, PM, PS, M, V, NT, K, BM, BN, BK, G,
                                                   num_warps=nw, num_stages=ns)
    _lse_combine[(M,)](PM, PS, lse, M, NT, triton.next_power_of_2(NT), num_warps=4)
    return L, lse


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


V, H = 81920, 512
C = CE._chunk_rows(65536, V, 1024 * 1024 * 1024)
g = torch.Generator(device=dev).manual_seed(0)
x = (torch.randn(C, H, device=dev, generator=g)).to(bf)
w = (torch.randn(V, H, device=dev, generator=g) * H ** -0.5 * 3).to(bf)
lab = torch.randint(0, V, (C,), device=dev, generator=g)
lse0, tgt0 = torch.empty(C, device=dev), torch.empty(C, device=dev)


def today():
    L = torch.mm(x, w.t())
    CE._fwd_reduce_kernel[(C,)](L, lab, lse0, tgt0, C, V, L.stride(0), L.stride(1), -100,
                                BLOCK_V=2048, num_warps=16)
    return L


L0 = today()
ref = torch.logsumexp(L0.float(), -1)
t_mm = ms(lambda: torch.mm(x, w.t()))
t0 = ms(today)
print(f"== chunk {C} x {V}, H={H}: cuBLAS mm {t_mm:.3f} ms, + fwd reduce = {t0:.3f} ms "
      f"({2 * C * V * H / t_mm / 1e9:.0f} TFLOPS for the mm)")
best = None
for cfg in ((128, 256, 64, 8, 8, 3), (128, 256, 32, 8, 8, 4), (128, 128, 64, 8, 4, 4), (256, 128, 64, 8, 8, 3),
            (128, 256, 64, 16, 8, 3), (64, 256, 64, 8, 4, 4), (128, 128, 64, 8, 8, 3)):
    try:
        L, lse = fused(x, w, cfg)
        t = ms(lambda: fused(x, w, cfg))
    except Exception as ex:
        print(f"   {cfg}: {type(ex).__name__} {str(ex).splitlines()[0][:70]}")
        continue
    same_logits = torch.equal(L, L0)
    err = float((lse - torch.logsumexp(L.float(), -1)).abs().max())
    print(f"   fused {cfg}: {t:.3f} ms  logits {'bitwise == cuBLAS' if same_logits else 'differ from cuBLAS'}"
          f"  max|lse - lse(own logits)| {err:.1e}")
    best = t if best is None else min(best, t)
print(f"   best fused {best:.3f} ms vs today {t0:.3f} ms -> {'WIN' if best and best < t0 else 'no win'} "
      f"({(t0 - best) * 40:.1f} ms/step at 10 chunks x 4 micro)" if best else "")
print("ALLDONE_LSE")
