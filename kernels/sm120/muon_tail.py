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

FMA = True     # torch's compiled add(alpha=) contracts to FMA: FMA=True is bit-identical 12/12, False is not


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
    # A Triton store is invisible to autograd's version counter, and kernels/sm75/moe.py:_cached_cast
    # keys its bf16 weight-cast cache on (storage, _version): without this bump the MoE kept reading
    # the STALE bf16 expert weights forever (params bit-identical to eager, forward not -- 50-step
    # board runs 0.16 worse at step 45). Views share the counter, so bumping p3 bumps the parameter.
    torch.autograd.graph.increment_version(p3)


# ---------------------------------------------------------------------------------------------
# Muown (muon_scaling.Muown, sm75 FusedMuon._muown_chunk): the whole non-NS step in two kernels.
# Each program owns BR rows of one matrix and walks its columns twice (the second walk re-reads
# from L2); the per-row scalars g / vn / m / s / dL/dg never leave registers between walks.
#   pre   u_ = W/g ; dg = sum(G*u_) ; gv = bf16((g/vn) * (G - u_*dg)) ; momentum + nesterov as above
#   post  Adam on g ; v = u_*vn + step_a*O ; vn' = ||v|| ; W = g' * v/vn' [ - lr*wd*W ; g = ||W|| ]
# Row reductions are Triton tree sums, so NOT bit-identical to torch's (last-ulp order effects);
# everything elementwise follows the eager op order, IEEE div/sqrt.

@triton.jit
def _muown_pre_kernel(W, G, GG, VN, DG, M, U, R, C, mu,
                      NESTEROV: tl.constexpr, USE_FMA: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    ridx = b * R + rows
    g = tl.load(GG + ridx, mask=rm, other=1.0)
    vn = tl.load(VN + ridx, mask=rm, other=1.0)
    # A gain of exactly 0 (the wd resync can underflow to it) must not become 0/0: that row has no
    # direction left, so u = 0. Rows with g != 0 divide by g exactly as before (bit-identical).
    gz = g == 0.0
    gs = tl.where(gz, 1.0, g)
    base = ridx[:, None] * C
    acc = tl.zeros([BR], dtype=tl.float32)
    for c0 in range(0, C, BC):
        cols = c0 + tl.arange(0, BC)
        msk = rm[:, None] & (cols[None, :] < C)
        w = tl.load(W + base + cols[None, :], mask=msk, other=0.0).to(tl.float32)
        gr = tl.load(G + base + cols[None, :], mask=msk, other=0.0).to(tl.float32)
        acc += tl.sum(gr * tl.where(gz[:, None], 0.0, tl.math.div_rn(w, gs[:, None])), axis=1)
    tl.store(DG + ridx, acc, mask=rm)
    k = tl.math.div_rn(g, vn)
    for c0 in range(0, C, BC):
        cols = c0 + tl.arange(0, BC)
        msk = rm[:, None] & (cols[None, :] < C)
        off = base + cols[None, :]
        w = tl.load(W + off, mask=msk, other=0.0).to(tl.float32)
        gr = tl.load(G + off, mask=msk, other=0.0).to(tl.float32)
        u_ = tl.where(gz[:, None], 0.0, tl.math.div_rn(w, gs[:, None]))
        gv = (k[:, None] * (gr - u_ * acc[:, None])).to(M.dtype.element_ty).to(tl.float32)
        m = tl.load(M + off, mask=msk, other=0.0).to(tl.float32)
        m = (m * mu).to(M.dtype.element_ty).to(tl.float32)
        m = (m + gv).to(M.dtype.element_ty)
        tl.store(M + off, m, mask=msk)
        if NESTEROV:
            mf = m.to(tl.float32)
            if USE_FMA:
                u = tl.fma(mu, mf, gv)
            else:
                u = gv + mu * mf
            tl.store(U + off, u.to(U.dtype.element_ty), mask=msk)
        else:
            tl.store(U + off, m, mask=msk)


@triton.jit
def _muown_post_kernel(P, O, GG, VN, MS, SS, DG, R, C, sob, sor, soc,
                       step_a, lr, b1, omb1, b2, omb2, bc1, bc2, eps, lr_wd,
                       HAS_WD: tl.constexpr, O_T: tl.constexpr, BR: tl.constexpr, BC: tl.constexpr):
    b = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * BR + tl.arange(0, BR)
    rm = rows < R
    ridx = b * R + rows
    g = tl.load(GG + ridx, mask=rm, other=1.0)
    vn = tl.load(VN + ridx, mask=rm, other=1.0)
    gz = g == 0.0                                   # see _muown_pre_kernel
    gs = tl.where(gz, 1.0, g)
    dg = tl.load(DG + ridx, mask=rm, other=0.0)
    m = tl.load(MS + ridx, mask=rm, other=0.0) * b1 + omb1 * dg
    s = tl.load(SS + ridx, mask=rm, other=0.0) * b2 + omb2 * (dg * dg)
    g_new = g - lr * tl.math.div_rn(tl.math.div_rn(m, bc1), tl.math.sqrt_rn(tl.math.div_rn(s, bc2)) + eps)
    base = ridx[:, None] * C
    obase = b * sob
    acc = tl.zeros([BR], dtype=tl.float32)
    for c0 in range(0, C, BC):
        cols = c0 + tl.arange(0, BC)
        msk = rm[:, None] & (cols[None, :] < C)
        w = tl.load(P + base + cols[None, :], mask=msk, other=0.0).to(tl.float32)
        if O_T:
            o = tl.trans(tl.load(O + obase + cols[:, None] * soc + rows[None, :] * sor, mask=tl.trans(msk), other=0.0))
        else:
            o = tl.load(O + obase + rows[:, None] * sor + cols[None, :] * soc, mask=msk, other=0.0)
        v = tl.where(gz[:, None], 0.0, tl.math.div_rn(w, gs[:, None])) * vn[:, None] + step_a * o.to(tl.float32)
        acc += tl.sum(v * v, axis=1)
    vn_new = tl.math.sqrt_rn(acc)
    acc2 = tl.zeros([BR], dtype=tl.float32)
    acc3 = tl.zeros([BR], dtype=tl.float32)        # same sum at x 2^40: underflow-safe for tiny rows
    for c0 in range(0, C, BC):
        cols = c0 + tl.arange(0, BC)
        msk = rm[:, None] & (cols[None, :] < C)
        off = base + cols[None, :]
        w = tl.load(P + off, mask=msk, other=0.0).to(tl.float32)
        if O_T:
            o = tl.trans(tl.load(O + obase + cols[:, None] * soc + rows[None, :] * sor, mask=tl.trans(msk), other=0.0))
        else:
            o = tl.load(O + obase + rows[:, None] * sor + cols[None, :] * soc, mask=msk, other=0.0)
        v = tl.where(gz[:, None], 0.0, tl.math.div_rn(w, gs[:, None])) * vn[:, None] + step_a * o.to(tl.float32)
        wn = g_new[:, None] * tl.math.div_rn(v, vn_new[:, None])
        if HAS_WD:
            wn = wn - lr_wd * w
            acc2 += tl.sum(wn * wn, axis=1)
            wb = wn * 1099511627776.0                  # 2^40, exact
            acc3 += tl.sum(wb * wb, axis=1)
        tl.store(P + off, wn.to(P.dtype.element_ty), mask=msk)
    if HAS_WD:
        # The resynced gain is the row norm. For a row near 1e-19 the squares underflow fp32 and the
        # plain sum is 0 -> g = 0 -> the next step's W/g was 0/0 (muown wd 0.1 NaN, step 884). Below
        # 2^-100 use the 2^40-scaled sum (exact power-of-two rescale); every other row keeps the
        # plain sqrt bit for bit.
        g_new = tl.where(acc2 < 7.888609052210118e-31, tl.math.sqrt_rn(acc3) * 9.094947017729282e-13,
                         tl.math.sqrt_rn(acc2))
    tl.store(GG + ridx, g_new, mask=rm)
    tl.store(VN + ridx, vn_new, mask=rm)
    tl.store(MS + ridx, m, mask=rm)
    tl.store(SS + ridx, s, mask=rm)


MUOWN_BR, MUOWN_BC = 16, 128


def muown_pre(p3, g3, st, dg, mom, u, momentum, nesterov):
    """One member (n, r, c). st: its g/vn slices (n, r); dg (n, r) fp32 out; mom/u (n, r, c) views."""
    n, R, C = p3.shape
    grid = (triton.cdiv(R, MUOWN_BR), n)
    _muown_pre_kernel[grid](p3, g3.contiguous(), st["g"], st["vn"], dg, mom, u, R, C, float(momentum),
                            NESTEROV=bool(nesterov), USE_FMA=FMA, BR=MUOWN_BR, BC=MUOWN_BC, num_warps=4)


def muown_post(p3, o3, st, dg, step_a, lr, betas, eps, t, lr_wd):
    """Adam on g, direction step, recompose, write p3 in place; updates st g/vn/m/s."""
    n, R, C = p3.shape
    b1, b2 = betas
    o_t = o3.stride(-1) != 1 and o3.stride(-2) == 1
    grid = (triton.cdiv(R, MUOWN_BR), n)
    _muown_post_kernel[grid](p3, o3, st["g"], st["vn"], st["m"], st["s"], dg, R, C, *o3.stride(),
                             float(step_a), float(lr), float(b1), float(1 - b1), float(b2), float(1 - b2), float(1 - b1 ** t),
                             float(1 - b2 ** t), float(eps), float(lr_wd),
                             HAS_WD=lr_wd != 0, O_T=o_t, BR=MUOWN_BR, BC=MUOWN_BC, num_warps=4)
    torch.autograd.graph.increment_version(p3)      # see tail_post
