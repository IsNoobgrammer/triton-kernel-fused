import torch, triton, triton.language as tl, time

@triton.jit
def _hh_fused_kernel(A, H, TAU, sb, si, sj, stb, N: tl.constexpr, BN: tl.constexpr):
    b = tl.program_id(0)
    rows = tl.arange(0, BN)
    cols = tl.arange(0, BN)
    rmask = rows < N
    cmask = cols < N
    full = rmask[:, None] & cmask[None, :]
    off = b * sb + rows[:, None] * si + cols[None, :] * sj
    a = tl.load(A + off, mask=full, other=0.0)            # [BN,BN]
    tau = tl.zeros([BN], tl.float32)
    for j in range(N):
        colj = tl.sum(tl.where(cols[None, :] == j, a, 0.0), axis=1)        # a[:,j], [BN]
        alpha = tl.sum(tl.where(rows == j, colj, 0.0))                     # scalar
        tailm = (rows > j) & rmask
        xnorm2 = tl.sum(tl.where(tailm, colj * colj, 0.0))
        sgn = tl.where(alpha >= 0.0, 1.0, -1.0)
        beta = -sgn * tl.sqrt(alpha * alpha + xnorm2)
        active = xnorm2 > 0.0
        beta_s = tl.where(beta == 0.0, 1.0, beta)
        tau_j = tl.where(active, (beta - alpha) / beta_s, 0.0)
        den = alpha - beta
        den = tl.where(den == 0.0, 1.0, den)
        v = tl.where(rows == j, 1.0, 0.0) + tl.where(tailm & active, colj / den, 0.0)   # [BN]
        beta_f = tl.where(active, beta, alpha)
        w = tl.sum(v[:, None] * a, axis=0)                                 # v^T a, [BN]
        a = tl.where(cols[None, :] >= j, a - tau_j * v[:, None] * w[None, :], a)
        # store reflector below diag of col j, and beta on diag
        a = tl.where((cols[None, :] == j) & (rows[:, None] > j) & rmask[:, None], v[:, None], a)
        a = tl.where((cols[None, :] == j) & (rows[:, None] == j), beta_f, a)
        tau = tl.where(rows == j, tau_j, tau)
    tl.store(H + off, a, mask=full)
    tl.store(TAU + b * stb + rows, tau, mask=rmask)


def qr_hh_fused(data):
    b, n, _ = data.shape
    BN = triton.next_power_of_2(n)
    H = torch.empty_like(data)
    tau = torch.empty(b, n, device=data.device, dtype=data.dtype)
    _hh_fused_kernel[(b,)](data, H, tau, data.stride(0), data.stride(1), data.stride(2),
                           tau.stride(0), N=n, BN=BN, num_warps=4)
    return H, tau


