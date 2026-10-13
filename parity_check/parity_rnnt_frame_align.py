"""rnnt_joint_node_logp(..., frame_targets=K): the teacher's one-token-per-frame CTC alignment
(kernels/sm120/rnnt_joint.py _frame_align_kernel) vs an fp64 reference of the same algorithm.

Reference: fp64 alpha / beta (parity_rnnt_joint.lattice) on the kernel's own node log-probs; token u's frame =
argmax_t alpha(t,u) + lpy(t,u) + beta(t,u+1); forward f(u) >= f(u-1) + 1 (+1 between identical neighbours),
backward f(u) <= f(u+1) - 1 (-1) and >= 0.
Gates:
  1. frames match the reference for >= 99% of tokens (bf16 joint: near-tied posteriors may pick a neighbour frame)
  2. the targets' greedy CTC decode (argmax per frame, collapse repeats, drop blank) == the reference tokens exactly,
     for every utterance with room (U + repeats <= T)
  3. slot 0 is the symbol with log-prob 0, slots 1..K-1 empty (-inf)
  4. kl_div_topk on these one-hot targets == cross-entropy -log_softmax(z)[target] (rel 1e-5)

    python parity_check/parity_rnnt_frame_align.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch

import parity_rnnt_joint as pj
import kernels.sm120.rnnt_joint as rj
from kernels.sm120.kl_div import kl_div_topk

ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def ref_frames(lpb, lpy, y, T):
    U = len(y)
    a, be = pj.lattice(lpb, lpy, T, U)
    want = []
    for u in range(U):
        v = a[:, u] + lpy[:, u] + be[:, u + 1]
        want.append(int(torch.argmax(v)))
    f, last, prev = [], -1, None
    for u in range(U):
        x = max(want[u], last + 1 + (1 if y[u] == prev else 0))
        f.append(x)
        last, prev = x, y[u]
    nxt, ny = T, None
    for u in range(U - 1, -1, -1):
        x = max(min(f[u], nxt - 1 - (1 if y[u] == ny else 0)), 0)
        f[u] = x
        nxt, ny = x, y[u]
    return f


def case(name, B, T, U, H, V, blank_bias=0.0, K=1):
    print(f"{name}: B {B} T {T} U {U} V {V} blank bias {blank_bias} K {K}", flush=True)
    f, g, W, b, y, tl, yl = pj.data(B, T, U, H, V, seed=3)
    b[-1] += blank_bias
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lpb, lpy, _, (idx, lp) = rj.rnnt_joint_node_logp(f, g, W, b, y, tl, yl, frame_targets=K)
    blank = V - 1
    n_tok = n_same = 0
    dec_ok = True
    off = 0
    for i in range(B):
        Ti, Ui = int(tl[i]), int(yl[i])
        n = Ti * (Ui + 1)
        pb = lpb[off:off + n].view(Ti, Ui + 1).double().cpu()
        py = lpy[off:off + n].view(Ti, Ui + 1).double().cpu()
        off += n
        yy = y[i, :Ui].tolist()
        fr = ref_frames(pb, py, yy, Ti)
        got = idx[i, :Ti, 0].tolist()
        kf = [t for t in range(Ti) if got[t] != blank]
        n_tok += Ui
        n_same += sum(1 for u in range(min(len(kf), Ui)) if kf[u] == fr[u])
        reps = sum(1 for u in range(1, Ui) if yy[u] == yy[u - 1])
        if Ui + reps <= Ti:
            out, prev = [], None
            for k in got:
                if k != prev and k != blank:
                    out.append(k)
                prev = k
            dec_ok &= out == yy
    check("1 frames == fp64 reference", n_same >= 0.99 * n_tok, f"{n_same}/{n_tok} tokens on the same frame")
    check("2 targets decode to the reference", dec_ok, "every utterance with room")
    s0 = bool((lp[..., 0] == 0).all())
    rest = bool(torch.isneginf(lp[..., 1:]).all()) if K > 1 else True
    check("3 slot layout", s0 and rest, f"slot0 logp 0: {s0}, other slots -inf: {rest}")
    z = torch.randn(B * T, V, device="cuda")
    kl = kl_div_topk(z, idx.reshape(-1, K), lp.reshape(-1, K))
    ce = -torch.log_softmax(z.double(), -1).gather(1, idx.reshape(-1, K)[:, :1].long()).squeeze(1)
    e = ((kl.double() - ce).abs().max() / ce.abs().max()).item()
    check("4 kl_div_topk(one-hot) == cross-entropy", e < 1e-5, f"max rel {e:.1e}")


def main():
    case("small", 4, 30, 10, 64, 129)
    case("confident blank", 4, 30, 10, 64, 129, blank_bias=8.0)
    case("dense (U close to T)", 3, 24, 20, 64, 129, K=4)
    case("wide", 3, 60, 16, 128, 1025)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
