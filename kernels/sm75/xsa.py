import torch
import triton
import triton.language as tl

__all__ = ["fused_xsa", "FusedXSA"]


def _cfgs():
    return [triton.Config({"XBLOCK": xb}, num_warps=w)
            for xb in (1, 2, 4, 8, 16, 32, 64, 128, 256) for w in (2, 4, 8)]


# key EXCLUDES S. S only sets the grid size (n_rows = B*Hkv*S); it does not change the work per
# program at all -- XBLOCK is rows-per-program and BLOCK_D covers D. Keying on it re-tunes 27
# configs for every distinct sequence length: measured 5.37 s per new S, which turned the
# variable-length eval into a 17.7 min stall vs 4.3 min without XSA (training, always S=1024,
# never showed it). B is excluded for the same reason.
@triton.autotune(configs=_cfgs(), key=["D", "H", "Hkv"])
@triton.jit
def _xsa_fwd_kernel(Y, V, Z, A, n_rows, S, D, H, Hkv, GROUP: tl.constexpr,
                    HAS_A: tl.constexpr, BLOCK_D: tl.constexpr, XBLOCK: tl.constexpr):
    rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    rmask = rows < n_rows
    offs_d = tl.arange(0, BLOCK_D)
    full = rmask[:, None] & (offs_d < D)[None, :]
    s = rows % S
    t = rows // S
    kv = t % Hkv
    b = t // Hkv
    v = tl.load(V + rows[:, None] * D + offs_d[None, :], mask=full, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=1)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    for j in range(GROUP):
        y_row = (b * H + kv * GROUP + j) * S + s
        yp = Y + y_row[:, None] * D + offs_d[None, :]
        y = tl.load(yp, mask=full, other=0.0).to(tl.float32)
        coeff = tl.sum(y * v, axis=1) * inv
        if HAS_A:
            coeff = coeff * tl.load(A + (kv * GROUP + j), mask=rmask, other=0.0).to(tl.float32)
        z = y - coeff[:, None] * v
        tl.store(Z + y_row[:, None] * D + offs_d[None, :], z.to(Z.dtype.element_ty), mask=full)


# reset_to_zero=["GA"] is REQUIRED, not tidiness: the autotuner runs this kernel once per config
# (and per warmup/rep iteration) to time it, and GA is accumulated with atomic_add. Without the
# reset, the FIRST backward for each new shape returns the alpha gradient summed over every
# autotune trial -- measured ~17000x too large -- and every later call is correct because the
# config is then cached. That is the worst shape of bug: one enormous wrong optimizer step on
# alpha, once, invisible afterwards. GY/GV are plain stores, so they are unaffected.
# key EXCLUDES S. S only sets the grid size (n_rows = B*Hkv*S); it does not change the work per
# program at all -- XBLOCK is rows-per-program and BLOCK_D covers D. Keying on it re-tunes 27
# configs for every distinct sequence length: measured 5.37 s per new S, which turned the
# variable-length eval into a 17.7 min stall vs 4.3 min without XSA (training, always S=1024,
# never showed it). B is excluded for the same reason.
@triton.autotune(configs=_cfgs(), key=["D", "H", "Hkv"], reset_to_zero=["GA"])
@triton.jit
def _xsa_bwd_kernel(GZ, Y, V, GY, GV, A, GA, n_rows, S, D, H, Hkv, GROUP: tl.constexpr,
                    HAS_A: tl.constexpr, BLOCK_D: tl.constexpr, XBLOCK: tl.constexpr):
    rows = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)
    rmask = rows < n_rows
    offs_d = tl.arange(0, BLOCK_D)
    full = rmask[:, None] & (offs_d < D)[None, :]
    s = rows % S
    t = rows // S
    kv = t % Hkv
    b = t // Hkv
    v = tl.load(V + rows[:, None] * D + offs_d[None, :], mask=full, other=0.0).to(tl.float32)
    n2 = tl.sum(v * v, axis=1)
    inv = tl.where(n2 > 0.0, 1.0 / n2, 0.0)
    gv = tl.zeros((XBLOCK, BLOCK_D), dtype=tl.float32)
    for j in range(GROUP):
        row = (b * H + kv * GROUP + j) * S + s
        bp = row[:, None] * D + offs_d[None, :]
        y = tl.load(Y + bp, mask=full, other=0.0).to(tl.float32)
        gz = tl.load(GZ + bp, mask=full, other=0.0).to(tl.float32)
        dot = tl.sum(y * v, axis=1)
        gzv = tl.sum(gz * v, axis=1)
        coeff = dot * inv
        a = tl.full((XBLOCK,), 1.0, dtype=tl.float32)
        if HAS_A:
            a = tl.load(A + (kv * GROUP + j), mask=rmask, other=0.0).to(tl.float32)
            # dL/da = -coeff * (gz.v), reduced over every row this head sees. atomic because a block
            # is not guaranteed to hold one head's rows exclusively.
            tl.atomic_add(GA + (kv * GROUP + j), -coeff * gzv, mask=rmask)
        gy = gz - (a * gzv * inv)[:, None] * v
        tl.store(GY + bp, gy.to(GY.dtype.element_ty), mask=full)
        gv += a[:, None] * ((-(gzv * inv))[:, None] * y
                            + (2.0 * dot * gzv * inv * inv)[:, None] * v - coeff[:, None] * gz)
    tl.store(GV + rows[:, None] * D + offs_d[None, :], gv.to(GV.dtype.element_ty), mask=full)


class FusedXSA(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Y, V, alpha=None):
        Y = Y.contiguous()
        V = V.contiguous()
        A = None if alpha is None else torch.tanh(alpha.float()).contiguous()
        B, H, S, D = Y.shape
        Hkv = V.shape[1]
        group = H // Hkv
        Z = torch.empty_like(Y)
        BLOCK_D = triton.next_power_of_2(D)
        n_rows = B * Hkv * S
        grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]),)
        _xsa_fwd_kernel[grid](Y, V, Z, A if A is not None else Y, n_rows, S, D, H, Hkv,
                              GROUP=group, HAS_A=A is not None, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(Y, V, A if A is not None else Y.new_zeros(0), alpha
                              if alpha is not None else Y.new_zeros(0))
        ctx.has_a = alpha is not None
        ctx.shape = (B, H, S, D, Hkv, group, BLOCK_D)
        return Z

    @staticmethod
    def backward(ctx, gZ):
        Y, V, A, alpha = ctx.saved_tensors
        B, H, S, D, Hkv, group, BLOCK_D = ctx.shape
        gZ = gZ.contiguous()
        GY = torch.empty_like(Y)
        GV = torch.empty_like(V)
        GA = torch.zeros(H, device=Y.device, dtype=torch.float32) if ctx.has_a else None
        n_rows = B * Hkv * S
        grid = lambda meta: (triton.cdiv(n_rows, meta["XBLOCK"]),)
        _xsa_bwd_kernel[grid](gZ, Y, V, GY, GV, A if ctx.has_a else Y, GA if ctx.has_a else Y,
                              n_rows, S, D, H, Hkv, GROUP=group, HAS_A=ctx.has_a, BLOCK_D=BLOCK_D)
        if not ctx.has_a:
            return GY, GV, None
        # chain through the tanh: alpha_used = tanh(theta), so dL/dtheta = dL/dalpha * (1 - a^2)
        return GY, GV, (GA * (1.0 - A * A)).to(alpha.dtype)


def fused_xsa(attn_output: torch.Tensor, value_states: torch.Tensor,
              alpha: torch.Tensor = None) -> torch.Tensor:
    """Y - a*(Y.V/|V|^2)*V. `alpha` is an optional per-head (H,) LOGIT; the applied strength is
    tanh(alpha), so 1 = full rejection, 0 = XSA off, negative = amplify the self-component. None
    keeps the original hard-coded full rejection (a = 1)."""
    return FusedXSA.apply(attn_output, value_states, alpha)
