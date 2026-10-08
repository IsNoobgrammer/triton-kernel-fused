"""relpos_attention (kernels/sm120/relpos_attn.py) vs NeMo RelPositionMultiHeadAttention's core math.

Reference = NeMo's own ops (rel_shift by pad/view/slice, mask from ConformerEncoder._create_masks 'chunked_limited',
masked_fill -10000, softmax, masked_fill 0, @ v) in fp64 (ground truth) and in fp32 with TF32 matmuls (what NeMo runs,
torch.set_float32_matmul_precision('high')). Every training context [70,13] [70,6] [70,1] [70,0], padded lengths,
T not a multiple of the tile.
Gates:
  1. o, dq, dk, dv, dp, du, dv_bias: ours at least as close to fp64 as NeMo-fp32 (x1.1, or rel < 2e-4)
  2. bitwise repeatable
  3. dropout: forward/backward consistent (directional derivative vs autograd, IEEE dots, rel < 2e-3) and the kept
     fraction is 1 - p
  4. padding query rows give exactly 0

    python parity_check/parity_relpos_attn.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from kernels.sm120.relpos_attn import relpos_attention

dev = "cuda"
PREC = os.environ.get("RPA_PREC", "tf32x3")
PRECG = os.environ.get("RPA_PRECG") or None
ok = True


def check(tag, cond, msg):
    global ok
    ok &= bool(cond)
    print(f"  [{'PASS' if cond else 'FAIL'}] {tag}: {msg}", flush=True)


def nemo_mask(T, lengths, left, right):
    """True = masked, exactly as ConformerEncoder._create_masks (chunked_limited) + the padding mask."""
    att = torch.ones(1, T, T, dtype=torch.bool, device=dev)
    chunk = right + 1
    lc = left // chunk
    ci = torch.div(torch.arange(T, device=dev, dtype=torch.int), chunk, rounding_mode="trunc")
    d = ci.unsqueeze(1) - ci.unsqueeze(0)
    att = att & ((d <= lc) & (d >= 0)).unsqueeze(0)
    pad = torch.arange(T, device=dev)[None, :] < lengths[:, None]
    both = pad.unsqueeze(1) & pad.unsqueeze(2)
    return ~(att & both)                                               # (B, T, T)


def rel_shift(x):
    b, h, qlen, pos_len = x.size()
    x = torch.nn.functional.pad(x, pad=(1, 0))
    x = x.view(b, h, -1, qlen)
    return x[:, :, 1:].view(b, h, qlen, pos_len)


def nemo_core(q, k, v, p, u, vb, mask):
    """NeMo RelPositionMultiHeadAttention.forward after the projections (non-sdpa path), any dtype."""
    q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)    # (B, H, T, D)
    pp = p.transpose(0, 1).unsqueeze(0)                                   # (1, H, 2T-1, D)
    qu = (q.transpose(1, 2) + u).transpose(1, 2)
    qv = (q.transpose(1, 2) + vb).transpose(1, 2)
    bd = rel_shift(torch.matmul(qv, pp.transpose(-2, -1)))
    ac = torch.matmul(qu, k.transpose(-2, -1))
    scores = (ac + bd[:, :, :, :ac.size(-1)]) / q.shape[-1] ** 0.5
    m = mask.unsqueeze(1)
    attn = torch.softmax(scores.masked_fill(m, -10000.0), dim=-1).masked_fill(m, 0.0)
    return torch.matmul(attn, v).transpose(1, 2)                         # (B, T, H, D)


def data(B, T, H, D, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    q, k, v = (torch.randn(B, T, H, D, device=dev, generator=g) for _ in range(3))
    p = torch.randn(2 * T - 1, H, D, device=dev, generator=g)
    u, vb = (0.3 * torch.randn(H, D, device=dev, generator=g) for _ in range(2))
    lengths = torch.randint(T // 2, T + 1, (B,), device=dev, generator=g)
    lengths[0] = T
    W = torch.randn(B, T, H, D, device=dev, generator=g)
    return [q, k, v, p, u, vb], lengths, W


def grads(fn, xs, W, dtype):
    xs = [x.detach().to(dtype).clone().requires_grad_() for x in xs]
    o = fn(*xs)
    (o * W.to(o.dtype)).sum().backward()
    return [o.detach()] + [x.grad for x in xs]


def rel(a, b):
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm().clamp(min=1e-300)).item()


NAMES = ["o", "dq", "dk", "dv", "dp", "du", "dv_bias"]


def main():
    B, T, H, D = 3, 150, 8, 64
    xs, lengths, W = data(B, T, H, D)
    for left, right in ((70, 13), (70, 6), (70, 1), (70, 0)):
        print(f"\n== context [{left}, {right}], B={B} T={T} H={H} D={D}, lengths {lengths.tolist()}", flush=True)
        mask = nemo_mask(T, lengths, left, right)
        gt = grads(lambda *a: nemo_core(*a, mask), xs, W, torch.float64)
        torch.backends.cuda.matmul.allow_tf32 = True
        nf = grads(lambda *a: nemo_core(*a, mask), xs, W, torch.float32)
        torch.backends.cuda.matmul.allow_tf32 = False
        ours = lambda *a: relpos_attention(*a, lengths, left, right, prec=PREC, precg=PRECG)
        ou = grads(ours, xs, W, torch.float32)
        bad = []
        for n, a, e, g in zip(NAMES, ou, nf, gt):
            if rel(a, g) > max(1.1 * rel(e, g), 2e-4):
                bad.append(f"{n} ours {rel(a, g):.2e} nemo-fp32 {rel(e, g):.2e}")
        worst = max(zip(NAMES, ou, nf, gt), key=lambda t: rel(t[1], t[3]))
        check("1 vs fp64 <= NeMo fp32", not bad, "; ".join(bad) if bad else
              f"worst {worst[0]}: ours {rel(worst[1], worst[3]):.1e} nemo-fp32 {rel(worst[2], worst[3]):.1e}")
        ou2 = grads(ours, xs, W, torch.float32)
        check("2 repeatable", all(torch.equal(a, b) for a, b in zip(ou, ou2)), "bitwise")
        padrows = torch.arange(T, device=dev)[None, :] >= lengths[:, None]
        check("4 padding rows = 0", bool((ou[0][padrows] == 0).all()), f"{int(padrows.sum())} rows")

    print("\n== dropout 0.1, context [70, 13], IEEE dots", flush=True)
    f = lambda *a: relpos_attention(*a, lengths, 70, 13, dropout=0.1, seed=99, prec="ieee")
    g0 = grads(f, xs, W, torch.float32)
    gen = torch.Generator(device=dev).manual_seed(5)
    dirs = [torch.randn(x.shape, device=dev, generator=gen) for x in xs]
    eps = 1e-3
    with torch.no_grad():
        fp = (f(*[x + eps * d for x, d in zip(xs, dirs)]) * W).sum().item()
        fm = (f(*[x - eps * d for x, d in zip(xs, dirs)]) * W).sum().item()
    fd = (fp - fm) / (2 * eps)
    an = sum((gr * d).sum().item() for gr, d in zip(g0[1:], dirs))
    check("3 dropout fwd/bwd consistent", abs(fd - an) / abs(an) < 2e-3, f"finite diff {fd:.4f} autograd {an:.4f}")
    o_nd = grads(lambda *a: relpos_attention(*a, lengths, 70, 13, prec="ieee"), xs, W, torch.float32)[0]
    check("3 dropout changes the output", rel(g0[0], o_nd) > 1e-2, f"rel {rel(g0[0], o_nd):.2e}")
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
