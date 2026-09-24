"""Fused elementwise tail of the FusedMuon step (every variant that does not own its step).

The eager step spends ~20 ms of a 71 ms BiBo-board base step outside Newton-Schulz, in separate
kernels that each re-read the same tensors:
  pre   gbuf = grad.to(bf16) ; mom.mul_(mu).add_(gbuf) ; u = gbuf.add_(mom, alpha=mu)
  post  p.mul_(1 - lr*wd)    ; p.add_(out, alpha=-lr*gain)          (out may be a transposed view)
Here each is one pass. Bit-identical by construction: every intermediate torch rounds to bf16 is
rounded here too, in the same order. FMA selects `fma(alpha, b, a)` vs `a + alpha * b` for the two
alpha-adds, whichever matches torch's compiled add on this toolchain (chosen by parity test).
"""
import torch
import triton
import triton.language as tl

FMA = True


@triton.jit
def _pre_kernel(G, M, U, N, mu, NESTEROV: tl.constexpr, USE_FMA: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    msk = offs < N
    g = tl.load(G + offs, mask=msk, other=0.0).to(M.dtype.element_ty).to(tl.float32)   # grad.to(bf16)
    m = tl.load(M + offs, mask=msk, other=0.0).to(tl.float32)
    m = (m * mu).to(M.dtype.element_ty).to(tl.float32)                                # mom.mul_(mu)
    m = (m + g).to(M.dtype.element_ty)                                                # .add_(gbuf)
    tl.store(M + offs, m, mask=msk)
    if NESTEROV:
        mf = m.to(tl.float32)
        if USE_FMA:
            u = tl.fma(mu, mf, g)
        else:
            u = g + mu * mf
        tl.store(U + offs, u.to(U.dtype.element_ty), mask=msk)                        # gbuf.add_(mom, alpha=mu)
    else:
        tl.store(U + offs, m, mask=msk)


@triton.jit
def _post_kernel(P, O, R, C, spb, spr, spc, sob, sor, soc, alpha, decay,
                 HAS_DECAY: tl.constexpr, O_T: tl.constexpr, USE_FMA: tl.constexpr,
                 BR: tl.constexpr, BC: tl.constexpr):
    pr, pc = tl.program_id(0), tl.program_id(1)
    b = tl.program_id(2).to(tl.int64)
    rr = pr * BR + tl.arange(0, BR)
    rc = pc * BC + tl.arange(0, BC)
    msk = (rr[:, None] < R) & (rc[None, :] < C)
    pptr = P + b * spb + rr[:, None] * spr + rc[None, :] * spc
    p = tl.load(pptr, mask=msk, other=0.0)
    if O_T:        # out is a transposed view (sor == 1): load the tile in its memory order, then trans
        o = tl.trans(tl.load(O + b * sob + rc[:, None] * soc + rr[None, :] * sor, mask=tl.trans(msk), other=0.0))
    else:
        o = tl.load(O + b * sob + rr[:, None] * sor + rc[None, :] * soc, mask=msk, other=0.0)
    o = o.to(tl.float32)
    if HAS_DECAY:
        p = p * decay                                                                 # p.mul_(1 - lr*wd)
    if USE_FMA:
        p = tl.fma(alpha, o, p)
    else:
        p = p + alpha * o                                                             # p.add_(out, alpha)
    tl.store(pptr, p, mask=msk)


def tail_pre(grads, gbuf, mom, momentum, nesterov):
    """grads: list of per-member grads; gbuf/mom: the chunk's (crows, r, c) buffers, members in order.
    Returns u (a view of gbuf for nesterov, else mom), exactly as the eager step."""
    off = 0
    for g in grads:
        n = g.numel()
        gd, md, ud = g.reshape(-1), mom.view(-1)[off:off + n], gbuf.view(-1)[off:off + n]
        BLOCK = 2048
        _pre_kernel[(triton.cdiv(n, BLOCK),)](gd, md, ud, n, float(momentum), NESTEROV=bool(nesterov),
                                             USE_FMA=FMA, BLOCK=BLOCK, num_warps=4)
        off += n
    return gbuf if nesterov else mom


def tail_post(p3, o3, alpha, decay):
    """p3 += alpha * o3 after p3 *= decay (decay None = no weight decay). p3 (n, R, C), o3 same shape, any strides."""
    n, R, C = p3.shape
    o_t = o3.stride(-1) != 1 and o3.stride(-2) == 1
    BR, BC = 64, 64
    grid = (triton.cdiv(R, BR), triton.cdiv(C, BC), n)
    _post_kernel[grid](p3, o3, R, C, *p3.stride(), *o3.stride(), float(alpha), float(decay or 1.0),
                       HAS_DECAY=decay is not None, O_T=o_t, USE_FMA=FMA, BR=BR, BC=BC, num_warps=4)
