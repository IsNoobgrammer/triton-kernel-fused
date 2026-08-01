import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["moe", "moe_per_expert", "moe_grouped", "moe_grouped_cublas", "moe_eager",
           "BatchedGLU", "GROUPED_MIN_TOKENS"]

GROUPED_MIN_TOKENS = 4096
SCHED_BLOCK_M = 64
_NS_EPS = 1e-6
_NS_BLOCK_I = 256


def _code_max(act_codes):
    m = getattr(act_codes, "_code_max_cache", None)
    if m is None:
        m = int(act_codes.max())
        try:
            act_codes._code_max_cache = m
        except Exception:
            pass
    return m


_CAST_CACHE = {}
try:
    from kernels.sm120 import moe_fused_glu as _FUSED_GLU
except Exception:
    _FUSED_GLU = None


def _cached_cast(t, dt):
    key = t.untyped_storage().data_ptr()
    hit = _CAST_CACHE.get(key)
    if hit is not None and hit[0] == t._version and hit[1] is dt and hit[2].shape == t.shape:
        return hit[2]
    c = t.to(dt)
    _CAST_CACHE[key] = (t._version, dt, c)
    if len(_CAST_CACHE) > 256:
        for k in list(_CAST_CACHE)[:128]:
            _CAST_CACHE.pop(k, None)
    return c


def _amp_cast(*ts):
    if torch.is_autocast_enabled("cuda"):
        dt = torch.get_autocast_dtype("cuda")
        return tuple((_cached_cast(t, dt) if (t.ndim == 3 and t.dtype != dt) else t.to(dt))
                     if t.is_floating_point() else t for t in ts)
    return ts


@triton.jit
def _row_rms_kernel(GateUp_ptr, Act_ptr, Rms_ptr, I, s_gu_m, s_gu_i,
                    EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=offs < I, other=0.0).to(tl.float32)
        acc += g * g
    rms = tl.sqrt(tl.sum(acc) / I + EPS)
    tl.store(Rms_ptr + row, tl.where((at == 2) | (at == 8), rms, 1.0))


@triton.jit
def _row_s_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Rms_ptr, S_ptr, I,
                  s_go_m, s_go_i, s_gu_m, s_gu_i, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    r = tl.load(Rms_ptr + row)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        m = offs < I
        go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
        up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
        gn = gate / r
        sig = 1.0 / (1.0 + tl.exp(-gn))
        acc += go * up * (sig * (1.0 + gn * (1.0 - sig))) * gn
    tl.store(S_ptr + row, tl.sum(acc))


_ROWFUSE_MAX_I = 1024


@triton.jit
def _glu_fwd_row_kernel(GateUp_ptr, Act_ptr, Alpha_ptr, Out_ptr, I,
                        s_gu_m, s_gu_i, s_o_m, s_o_i, s_ap,
                        EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I)
    msk = offs < I
    gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    r = tl.sqrt(tl.sum(gate * gate) / I + EPS)
    gn = tl.where((at == 2) | (at == 8), gate / r, gate)
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    rp = tl.exp(p8 * tl.log(r))
    z = tl.where(at == 8, gn, aa * gn)
    sig = 1.0 / (1.0 + tl.exp(-z))
    f = z * sig
    act = tl.where(at == 8, rp * f, f)
    tl.store(Out_ptr + row * s_o_m + offs * s_o_i, (act * up).to(Out_ptr.dtype.element_ty), mask=msk)


@triton.jit
def _glu_bwd_row_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Alpha_ptr,
                        GradGateUp_ptr, DA_ptr, I,
                        s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i, s_ap,
                        EPS: tl.constexpr, WANT_AP: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I)
    msk = offs < I
    go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=msk, other=0.0).to(tl.float32)
    gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    r = tl.sqrt(tl.sum(gate * gate) / I + EPS)
    is_norm = (at == 2) | (at == 8)
    gn = tl.where(is_norm, gate / r, gate)
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    lr8 = tl.log(r)
    rp = tl.exp(p8 * lr8)
    rpm1 = tl.exp((p8 - 1.0) * lr8)
    z = tl.where(at == 8, gn, aa * gn)
    sig = 1.0 / (1.0 + tl.exp(-z))
    f = z * sig
    df = sig * (1.0 + z * (1.0 - sig))
    act = tl.where(at == 8, rp * f, f)
    gu_ = go * up
    S = tl.sum(tl.where(is_norm, gu_ * df * gn, 0.0))
    T = tl.sum(tl.where(at == 8, gu_ * f, 0.0))
    grad_gate = tl.where(at == 8, rpm1 * (gu_ * df - (gn / I) * (S - p8 * T)),
                    tl.where(is_norm, aa * (gu_ * df - (S / I) * gn) / r, aa * gu_ * df))
    tl.store(GradGateUp_ptr + row * s_ggu_m + offs * s_ggu_i, grad_gate, mask=msk)
    tl.store(GradGateUp_ptr + row * s_ggu_m + (I + offs) * s_ggu_i, go * act, mask=msk)
    if WANT_AP:
        da8 = p8 * (1.0 - p8) * rp * lr8 * T
        tl.store(DA_ptr + row, tl.where(at == 8, da8, tl.sum(gu_ * df * gn)))


@triton.jit
def _glu_fwd_kernel(GateUp_ptr, Act_ptr, Rms_ptr, Alpha_ptr, Out_ptr, M, I,
                    s_gu_m, s_gu_i, s_o_m, s_o_i, s_ap,
                    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0); pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M; mask = mask_m[:, None] & (offs_i < I)[None, :]
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + offs_m, mask=mask_m, other=0)[:, None]
    r = tl.load(Rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    aa = tl.load(Alpha_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    gn = gate / r
    z = aa * gn
    sig = 1.0 / (1.0 + tl.exp(-z))
    act = z * sig
    tl.store(Out_ptr + offs_m[:, None] * s_o_m + offs_i[None, :] * s_o_i,
             (act * up).to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _glu_bwd_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Rms_ptr, S_ptr, Alpha_ptr,
                    GradGateUp_ptr, M, I,
                    s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i, s_ap,
                    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0); pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M; mask = mask_m[:, None] & (offs_i < I)[None, :]
    go = tl.load(GradOut_ptr + offs_m[:, None] * s_go_m + offs_i[None, :] * s_go_i, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + offs_m, mask=mask_m, other=0)[:, None]
    r = tl.load(Rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    sv = tl.load(S_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    aa = tl.load(Alpha_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    r = tl.load(Rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    sv = tl.load(S_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    gn = gate / r
    z = aa * gn
    sig = 1.0 / (1.0 + tl.exp(-z))
    act = z * sig
    dsilu = sig * (1.0 + z * (1.0 - sig))
    grad_gate = tl.where(at == 2, (go * up * dsilu - (sv / I) * gn) / r, go * up * dsilu) * aa
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + offs_i[None, :] * s_ggu_i, grad_gate, mask=mask)
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + (I + offs_i)[None, :] * s_ggu_i, go * act, mask=mask)


_ONES_CACHE = {}


def _ones(M, device):
    buf = _ONES_CACHE.get(device)
    if buf is None or buf.numel() < M:
        buf = torch.ones(max(M, 8192), device=device, dtype=torch.float32)
        _ONES_CACHE[device] = buf
    return buf[:M]


def _row_rms(gate_up, row_act, I):
    M = gate_up.shape[0]
    rms = torch.empty(M, device=gate_up.device, dtype=torch.float32)
    if M > 0:
        _row_rms_kernel[(M,)](gate_up, row_act, rms, I, gate_up.stride(0), gate_up.stride(1),
                              EPS=_NS_EPS, BLOCK_I=_NS_BLOCK_I)
    return rms


def _ap_stride(row_alpha):
    return 0 if (row_alpha is not None and row_alpha.numel() == 1) else 1


@triton.jit
def _glu_fwd_rowloop_kernel(GateUp_ptr, Act_ptr, Alpha_ptr, Out_ptr, I,
                            s_gu_m, s_gu_i, s_o_m, s_o_i, s_ap,
                            EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    is_norm = (at == 2) | (at == 8)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=offs < I, other=0.0).to(tl.float32)
        acc += g * g
    r = tl.sqrt(tl.sum(acc) / I + EPS)
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    rp = tl.exp(p8 * tl.log(r))
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        m = offs < I
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
        gn = tl.where(is_norm, g / r, g)
        z = tl.where(at == 8, gn, aa * gn)
        f = z * (1.0 / (1.0 + tl.exp(-z)))
        act = tl.where(at == 8, rp * f, f)
        tl.store(Out_ptr + row * s_o_m + offs * s_o_i, (act * u).to(Out_ptr.dtype.element_ty), mask=m)


@triton.jit
def _glu_bwd_rowloop_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Alpha_ptr,
                            GradGateUp_ptr, DA_ptr, I,
                            s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i, s_ap,
                            EPS: tl.constexpr, WANT_AP: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    is_norm = (at == 2) | (at == 8)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=offs < I, other=0.0).to(tl.float32)
        acc += g * g
    r = tl.sqrt(tl.sum(acc) / I + EPS)
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    lr8 = tl.log(r)
    rp = tl.exp(p8 * lr8)
    rpm1 = tl.exp((p8 - 1.0) * lr8)
    sa = tl.zeros([BLOCK_I], dtype=tl.float32)
    tt = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        m = offs < I
        go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=m, other=0.0).to(tl.float32)
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
        gn = tl.where(is_norm, g / r, g)
        z = tl.where(at == 8, gn, aa * gn)
        sig = 1.0 / (1.0 + tl.exp(-z))
        f = z * sig
        df = sig * (1.0 + z * (1.0 - sig))
        gu_ = go * u
        sa += tl.where(m, gu_ * df * gn, 0.0)
        tt += tl.where(m & (at == 8), gu_ * f, 0.0)
    SA = tl.sum(sa)
    T = tl.sum(tt)
    S = tl.where(is_norm, SA, 0.0)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        m = offs < I
        go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=m, other=0.0).to(tl.float32)
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
        u = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
        gn = tl.where(is_norm, g / r, g)
        z = tl.where(at == 8, gn, aa * gn)
        sig = 1.0 / (1.0 + tl.exp(-z))
        f = z * sig
        df = sig * (1.0 + z * (1.0 - sig))
        act = tl.where(at == 8, rp * f, f)
        gu_ = go * u
        grad_gate = tl.where(at == 8, rpm1 * (gu_ * df - (gn / I) * (S - p8 * T)),
                        tl.where(is_norm, aa * (gu_ * df - (S / I) * gn) / r, aa * gu_ * df))
        tl.store(GradGateUp_ptr + row * s_ggu_m + offs * s_ggu_i, grad_gate, mask=m)
        tl.store(GradGateUp_ptr + row * s_ggu_m + (I + offs) * s_ggu_i, go * act, mask=m)
    if WANT_AP:
        da8 = p8 * (1.0 - p8) * rp * lr8 * T
        tl.store(DA_ptr + row, tl.where(at == 8, da8, SA))


_LOOP_BLOCK_I = 1024


def _row_tiling(I):
    if I > _ROWFUSE_MAX_I:
        return True, _LOOP_BLOCK_I, 8
    if I % 256 == 0 and (I & (I - 1)) != 0:
        return True, 256, 4
    b = max(16, triton.next_power_of_2(I))
    return False, b, (8 if b >= 1024 else 4)


def _needs_row(row_act, code_hint, row_alpha):
    if row_alpha is not None:
        return True
    if code_hint is not None:
        return code_hint in (2, 8)
    return True


def _glu_fwd(gate_up, row_act, code_hint=None, row_alpha=None):
    M, twoI = gate_up.shape; I = twoI // 2
    if code_hint == 8 and row_alpha is None:
        raise ValueError(
            "act code 8 (radial NormSiLU, r^p*SiLU(g/r)) requires row_alpha -- it carries the "
            "exponent LOGIT theta, p=sigmoid(theta). With alpha absent the kernel would default to "
            "theta=1 => p=0.731 instead of the intended 0.5 init. Pass act_params.")
    ra = _ones(M, gate_up.device) if row_alpha is None else row_alpha
    out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
    looped, BLOCK_I, nw = _row_tiling(I)
    if not looped:
        if M > 0:
            _glu_fwd_row_kernel[(M,)](gate_up, row_act, ra, out, I,
                                      gate_up.stride(0), gate_up.stride(1), out.stride(0), out.stride(1),
                                      _ap_stride(row_alpha),
                                      EPS=_NS_EPS, BLOCK_I=BLOCK_I, num_warps=nw)
        return out
    if _needs_row(row_act, code_hint, row_alpha):
        if M > 0:
            _glu_fwd_rowloop_kernel[(M,)](gate_up, row_act, ra, out, I,
                                          gate_up.stride(0), gate_up.stride(1),
                                          out.stride(0), out.stride(1), _ap_stride(row_alpha),
                                          EPS=_NS_EPS, BLOCK_I=BLOCK_I, num_warps=nw)
        return out
    skip_ns = code_hint is not None and code_hint not in (2, 6, 7, 8)
    rms = _ones(M, gate_up.device) if skip_ns else _row_rms(gate_up, row_act, I)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_fwd_kernel[grid](gate_up, row_act, rms, ra, out, M, I, gate_up.stride(0), gate_up.stride(1),
                          out.stride(0), out.stride(1), _ap_stride(row_alpha),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    return out


def _glu_bwd(grad_out, gate_up, row_act, code_hint=None, row_alpha=None, want_act_grads=False):
    M, twoI = gate_up.shape; I = twoI // 2
    if code_hint == 8 and row_alpha is None:
        raise ValueError(
            "act code 8 (radial NormSiLU, r^p*SiLU(g/r)) requires row_alpha -- it carries the "
            "exponent LOGIT theta, p=sigmoid(theta). With alpha absent the kernel would default to "
            "theta=1 => p=0.731 instead of the intended 0.5 init. Pass act_params.")
    ra = _ones(M, gate_up.device) if row_alpha is None else row_alpha
    ggu = torch.empty_like(gate_up)
    looped, BLOCK_I, nw = _row_tiling(I)
    if not looped:
        if want_act_grads:
            da = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        else:
            da = gate_up
        if M > 0:
            _glu_bwd_row_kernel[(M,)](grad_out, gate_up, row_act, ra, ggu, da, I,
                                      grad_out.stride(0), grad_out.stride(1),
                                      gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                                      _ap_stride(row_alpha),
                                      EPS=_NS_EPS, WANT_AP=want_act_grads, BLOCK_I=BLOCK_I,
                                      num_warps=nw)
        return (ggu, da) if want_act_grads else ggu
    if _needs_row(row_act, code_hint, row_alpha):
        ggu = torch.empty_like(gate_up)
        if want_act_grads:
            da = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        else:
            da = gate_up
        if M > 0:
            _glu_bwd_rowloop_kernel[(M,)](grad_out, gate_up, row_act, ra, ggu, da, I,
                                          grad_out.stride(0), grad_out.stride(1),
                                          gate_up.stride(0), gate_up.stride(1),
                                          ggu.stride(0), ggu.stride(1), _ap_stride(row_alpha),
                                          EPS=_NS_EPS, WANT_AP=want_act_grads,
                                          BLOCK_I=BLOCK_I, num_warps=nw)
        return (ggu, da) if want_act_grads else ggu
    skip_ns = code_hint is not None and code_hint not in (2, 6, 7, 8)
    if skip_ns:
        rms = _ones(M, gate_up.device)
        sbuf = rms
    else:
        rms = _row_rms(gate_up, row_act, I)
        sbuf = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        if M > 0:
            _row_s_kernel[(M,)](grad_out, gate_up, row_act, rms, sbuf, I,
                                grad_out.stride(0), grad_out.stride(1),
                                gate_up.stride(0), gate_up.stride(1), BLOCK_I=_NS_BLOCK_I)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_bwd_kernel[grid](grad_out, gate_up, row_act, rms, sbuf, ra, ggu, M, I,
                          grad_out.stride(0), grad_out.stride(1),
                          gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                          _ap_stride(row_alpha),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    if want_act_grads:
        raise AssertionError("want_act_grads reached the tiled path")
    return ggu


class BatchedGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, gate_up, row_act, row_alpha=None):
        ctx.save_for_backward(gate_up, row_act,
                              row_alpha if row_alpha is not None else torch.empty(0),
                              )
        ctx.has_ap = row_alpha is not None
        return _glu_fwd(gate_up, row_act, row_alpha=row_alpha)

    @staticmethod
    def backward(ctx, grad_out):
        gate_up, row_act, row_alpha = ctx.saved_tensors
        ra = row_alpha if ctx.has_ap else None
        go = grad_out
        if ctx.has_ap and ctx.needs_input_grad[2]:
            ggu, da = _glu_bwd(go, gate_up, row_act, row_alpha=ra, want_act_grads=True)
            return ggu, None, da
        return _glu_bwd(go, gate_up, row_act, row_alpha=ra), None, None


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ], key=["K", "N"])
@triton.jit
def _grouped_mm_kernel(X_ptr, W_ptr, Out_ptr, TileExpert_ptr, TileStart_ptr, ExpertEnd_ptr,
                       K, N, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                       stride_om, stride_on,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0); pid_n = tl.program_id(1)
    e = tl.load(TileExpert_ptr + pid_t); m0 = tl.load(TileStart_ptr + pid_t); m_end = tl.load(ExpertEnd_ptr + e)
    offs_m = m0 + tl.arange(0, BLOCK_M); offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m_end; mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    W_e = W_ptr + e * stride_we
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K); mask_k = offs_k < K
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W_e + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": 64}, num_warps=4, num_stages=2),
    ], key=["N", "K"])
@triton.jit
def _grouped_wgrad_kernel(A_ptr, B_ptr, GW_ptr, ExpertStart_ptr, ExpertEnd_ptr, N, K,
                          stride_am, stride_an, stride_bm, stride_bk, stride_ge, stride_gn, stride_gk,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_e = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    m_start = tl.load(ExpertStart_ptr + pid_e); m_end = tl.load(ExpertEnd_ptr + pid_e)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N; mask_k = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    m = m_start
    while m < m_end:
        offs_m = m + tl.arange(0, BLOCK_M); mask_m = offs_m < m_end
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(tl.trans(a), b)
        m += BLOCK_M
    tl.store(GW_ptr + pid_e * stride_ge + offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk,
             acc.to(GW_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_k[None, :])


def _build_schedule(counts, bounds, E, device, block_m=SCHED_BLOCK_M):
    tile_expert, tile_start = [], []
    for e in range(E):
        for ti in range((counts[e] + block_m - 1) // block_m):
            tile_expert.append(e); tile_start.append(bounds[e] + ti * block_m)
    if not tile_expert:
        return None, None
    return (torch.tensor(tile_expert, dtype=torch.int32, device=device),
            torch.tensor(tile_start, dtype=torch.int32, device=device))


def _grouped_mm(x_sorted, W, te, ts, e_end, N, trans_w=False):
    M, K = x_sorted.shape
    out = torch.empty(M, N, device=x_sorted.device, dtype=x_sorted.dtype)
    s_we, s_wn, s_wk = (W.stride(0), W.stride(2), W.stride(1)) if trans_w else (W.stride(0), W.stride(1), W.stride(2))
    grid = lambda meta: (te.numel(), triton.cdiv(N, meta["BLOCK_N"]))
    _grouped_mm_kernel[grid](x_sorted, W, out, te, ts, e_end, K, N, x_sorted.stride(0), x_sorted.stride(1),
                             s_we, s_wn, s_wk, out.stride(0), out.stride(1))
    return out


def _grouped_wgrad(A, B, e_start, e_end, E, N, K):
    gW = torch.zeros(E, N, K, device=A.device, dtype=A.dtype)
    grid = lambda meta: (E, triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))
    _grouped_wgrad_kernel[grid](A, B, gW, e_start, e_end, N, K, A.stride(0), A.stride(1),
                                B.stride(0), B.stride(1), gW.stride(0), gW.stride(1), gW.stride(2))
    return gW


_CODES_CACHE = {}


def _codes_list(act_codes):
    key = (act_codes.data_ptr(), act_codes._version, act_codes.numel())
    ent = _CODES_CACHE.get(key)
    if ent is not None and ent[0] is act_codes:
        return ent[1]
    v = act_codes.tolist()
    if len(_CODES_CACHE) > 32:
        _CODES_CACHE.clear()
    _CODES_CACHE[key] = (act_codes, v)
    return v


def _sort_by_expert(idx, wt, E):
    ntok, top_k = idx.shape
    flat_t = torch.arange(ntok, device=idx.device).unsqueeze(1).expand_as(idx).flatten()
    sorted_e, order = idx.flatten().sort()
    counts_dev = torch.bincount(sorted_e, minlength=E)
    counts = counts_dev.tolist()
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    return flat_t[order], wt.flatten()[order], order, counts, bounds, counts_dev


class _GroupedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, wt, gate_up_proj, down_proj, act_codes):
        x, wt, gate_up_proj, down_proj = _amp_cast(x, wt, gate_up_proj, down_proj)
        ntok, H = x.shape
        top_k = idx.shape[1]; E = gate_up_proj.shape[0]; I = gate_up_proj.shape[1] // 2
        dev = x.device
        st, sw, order, counts, bounds, counts_t = _sort_by_expert(idx, wt, E)
        e_start = torch.tensor(bounds[:E], dtype=torch.int32, device=dev)
        e_end = torch.tensor(bounds[1:], dtype=torch.int32, device=dev)
        te, ts = _build_schedule(counts, bounds, E, dev)
        row_act = torch.repeat_interleave(act_codes, counts_t).to(torch.int32)
        x_s = x[st].contiguous()
        gate_up = _grouped_mm(x_s, gate_up_proj, te, ts, e_end, 2 * I)
        inter = _glu_fwd(gate_up, row_act)
        eo = _grouped_mm(inter, down_proj, te, ts, e_end, H)
        out = torch.zeros(ntok, H, device=dev, dtype=torch.float32)
        out.index_add_(0, st, (eo * sw.unsqueeze(-1)).float())
        out = out.to(x.dtype)
        ctx.save_for_backward(x_s, gate_up, inter, eo, st, sw, order, te, ts, e_start, e_end,
                              row_act, gate_up_proj, down_proj)
        ctx.shapes = (ntok, H, I, top_k, E)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x_s, gate_up, inter, eo, st, sw, order, te, ts, e_start, e_end, row_act,
         gate_up_proj, down_proj) = ctx.saved_tensors
        ntok, H, I, top_k, E = ctx.shapes
        go_s = grad_out[st].contiguous()
        grad_w_s = (go_s.float() * eo.float()).sum(-1).to(grad_out.dtype)
        grad_eo = go_s * sw.unsqueeze(-1)
        grad_inter = _grouped_mm(grad_eo, down_proj, te, ts, e_end, I, trans_w=True)
        grad_down_proj = _grouped_wgrad(grad_eo, inter, e_start, e_end, E, H, I)
        grad_gate_up = _glu_bwd(grad_inter, gate_up, row_act)
        grad_x_s = _grouped_mm(grad_gate_up, gate_up_proj, te, ts, e_end, H, trans_w=True)
        grad_gate_up_proj = _grouped_wgrad(grad_gate_up, x_s, e_start, e_end, E, 2 * I, H)
        grad_x = torch.zeros(ntok, H, device=grad_out.device, dtype=grad_out.dtype)
        grad_x.index_add_(0, st, grad_x_s)
        grad_wt = torch.zeros(ntok * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = grad_w_s
        return grad_x, None, grad_wt.view(ntok, top_k), grad_gate_up_proj, grad_down_proj, None


def moe_grouped(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    if _code_max(act_codes) > 4:
        raise ValueError("code 8 (radial) unsupported on the grouped path; use moe_per_expert(act_params=...)")
    return _GroupedMoE.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)


def moe_grouped_cublas(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm unavailable in this torch build")
    if torch.cuda.get_device_capability(hidden.device)[0] < 8:
        raise RuntimeError("torch._grouped_mm needs bf16 tensor cores (sm_80+); skipped on this GPU")
    if _code_max(act_codes) > 4:
        raise ValueError("code 8 (radial) unsupported on the grouped-cublas path; use moe_per_expert(act_params=...)")
    N, H = hidden.shape
    E = gate_up_proj.shape[0]
    flat_t = torch.arange(N, device=hidden.device).unsqueeze(1).expand_as(top_k_indices).flatten()
    sorted_e, order = top_k_indices.flatten().sort()
    st = flat_t[order]
    sw = top_k_weights.flatten()[order]
    counts = torch.bincount(sorted_e, minlength=E)
    offs = counts.cumsum(0).to(torch.int32)
    row_act = torch.repeat_interleave(act_codes, counts).to(torch.int32)
    x_s = hidden[st].contiguous()
    bf = torch.bfloat16
    gate_up = torch._grouped_mm(x_s.to(bf), gate_up_proj.transpose(-2, -1).to(bf), offs=offs).to(hidden.dtype)
    inter = BatchedGLU.apply(gate_up, row_act)
    eo = torch._grouped_mm(inter.to(bf), down_proj.transpose(-2, -1).to(bf), offs=offs).to(hidden.dtype)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)
    out.index_add_(0, st, (eo * sw.unsqueeze(-1)).float())
    return out.to(hidden.dtype)


@triton.jit
def _combine_scatter_kernel(EO_ptr, W_ptr, Tok_ptr, Out_ptr, m, H, s_eo_m, s_eo_h, s_out_n, s_out_h,
                            BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_m = tl.program_id(0); pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < m; mask = mask_m[:, None] & (offs_h < H)[None, :]
    eo = tl.load(EO_ptr + offs_m[:, None] * s_eo_m + offs_h[None, :] * s_eo_h, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    tok = tl.load(Tok_ptr + offs_m, mask=mask_m, other=0)
    tl.atomic_add(Out_ptr + tok[:, None] * s_out_n + offs_h[None, :] * s_out_h, eo * w, mask=mask)


@triton.jit
def _combine_bwd_kernel(GO_ptr, EO_ptr, W_ptr, Tok_ptr, GradEO_ptr, GradW_ptr, m, H,
                        s_go_n, s_go_h, s_eo_m, s_eo_h, s_geo_m, s_geo_h,
                        BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < m; mask = mask_m[:, None] & (offs_h < H)[None, :]
    tok = tl.load(Tok_ptr + offs_m, mask=mask_m, other=0)
    go = tl.load(GO_ptr + tok[:, None] * s_go_n + offs_h[None, :] * s_go_h, mask=mask, other=0.0).to(tl.float32)
    eo = tl.load(EO_ptr + offs_m[:, None] * s_eo_m + offs_h[None, :] * s_eo_h, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    tl.store(GradEO_ptr + offs_m[:, None] * s_geo_m + offs_h[None, :] * s_geo_h,
             (go * w).to(GradEO_ptr.dtype.element_ty), mask=mask)
    tl.store(GradW_ptr + offs_m, tl.sum(go * eo, axis=1), mask=mask_m)


def _combine_scatter(eo, w, tok, out):
    m, H = eo.shape
    BLOCK_M = max(16, min(64, triton.next_power_of_2(m))); BLOCK_H = max(16, min(128, triton.next_power_of_2(H)))
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(H, BLOCK_H))
    _combine_scatter_kernel[grid](eo, w, tok, out, m, H, eo.stride(0), eo.stride(1),
                                  out.stride(0), out.stride(1), BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H)


def _combine_bwd(grad_out, eo, w, tok):
    m, H = eo.shape
    grad_eo = torch.empty_like(eo)
    grad_w = torch.empty(m, device=eo.device, dtype=torch.float32)
    BLOCK_M = 16; BLOCK_H = triton.next_power_of_2(H)
    _combine_bwd_kernel[(triton.cdiv(m, BLOCK_M),)](grad_out, eo, w, tok, grad_eo, grad_w, m, H,
                        grad_out.stride(0), grad_out.stride(1), eo.stride(0), eo.stride(1),
                        grad_eo.stride(0), grad_eo.stride(1), BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H)
    return grad_eo, grad_w


class _PerExpertMoE(torch.autograd.Function):

    @staticmethod
    def forward(ctx, hidden, idx, wt, gate_up_proj, down_proj, act_codes, act_params=None):
        hidden, wt, gate_up_proj, down_proj = _amp_cast(hidden, wt, gate_up_proj, down_proj)
        N, H = hidden.shape
        E = act_codes.shape[0]
        codes = _codes_list(act_codes)
        top_k = idx.shape[1]; dev = hidden.device
        st, sw, order, counts, bounds, counts_t = _sort_by_expert(idx, wt, E)
        x_s = hidden.index_select(0, st)
        M_rows = idx.numel()
        row_act = torch.repeat_interleave(act_codes, counts_t, output_size=M_rows).to(torch.int32)
        ap32 = act_params.float().contiguous() if act_params is not None else None
        ap_shape = None
        if ap32 is not None:
            ap_shape = ap32.shape
            if ap32.ndim == 1:
                ap32 = ap32[:, None].contiguous()
            if ap32.shape[0] != E:
                raise ValueError(
                    f"act_params has {ap32.shape[0]} rows but act_codes has {E} experts. Rows are "
                    f"indexed by EXPERT ID, so a stack with special experts needs a row for each of "
                    f"them too (their value is unused). Pass a full (E,) or (E,2) tensor -- padding "
                    f"the specials with zeros is fine.")
        gate_up_l = [None] * E; inter_l = [None] * E
        M_tot = st.numel()
        eo_all = torch.empty(M_tot, H, device=dev, dtype=hidden.dtype)
        sw_eff = sw
        uniform = all(c <= 2 or c >= 6 for c in codes) and ap32 is None
        use_gmm = (uniform and hasattr(torch, "_grouped_mm")
                   and hidden.dtype in (torch.bfloat16, torch.float16))
        offs = counts_t.cumsum(0).to(torch.int32) if use_gmm else None
        tile_map = None; tile_map_gg = None; tile_map_bw = None
        if use_gmm:
            hint = codes[0] if len(set(codes)) == 1 else None
            if _FUSED_GLU is not None and _FUSED_GLU.tiles_supported(x_s):
                tile_map_gg = _FUSED_GLU.build_tile_map(counts, counts_t, dev,
                                                        bm=_FUSED_GLU._GG[0])
            gu_all = it_all = None
            if tile_map_gg is not None and _FUSED_GLU.gemm_supported(x_s, gate_up_proj, codes):
                tm = _FUSED_GLU.build_tile_map(counts, counts_t, dev)
                act = _FUSED_GLU.fused_supported(x_s, gate_up_proj, codes)
                gu_all, it_all = _FUSED_GLU.fused_gate_up_glu(x_s, gate_up_proj, tm, codes[0],
                                                              want_gu=True, act=act)
                if act:
                    tile_map = tm
                    tile_map_bw = _FUSED_GLU.build_tile_map(counts, counts_t, dev,
                                                            bm=_FUSED_GLU._BBM)
            if gu_all is None:
                gu_all = torch._grouped_mm(x_s, gate_up_proj.transpose(1, 2), offs=offs)
            if it_all is None:
                it_all = _glu_fwd(gu_all, row_act, code_hint=hint)
            eo_all = None
            if tile_map_gg is not None:
                eo_all = _FUSED_GLU.grouped_gemm(it_all, down_proj.transpose(1, 2).contiguous()
                                                 if not down_proj.transpose(1, 2).is_contiguous()
                                                 else down_proj.transpose(1, 2), tile_map_gg)
            if eo_all is None:
                eo_all = torch._grouped_mm(it_all, down_proj.transpose(1, 2), offs=offs)
            gate_up_l = gu_all; inter_l = it_all
        elif uniform:
            gu_all = torch.empty(M_tot, 2 * gate_up_proj.shape[1] // 2, device=dev, dtype=hidden.dtype)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en > s:
                    torch.mm(x_s[s:en], gate_up_proj[e].t(), out=gu_all[s:en])
            hint = codes[0] if len(set(codes)) == 1 else None
            it_all = _glu_fwd(gu_all, row_act, code_hint=hint)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en > s:
                    torch.mm(it_all[s:en], down_proj[e].t(), out=eo_all[s:en])
            gate_up_l = gu_all; inter_l = it_all
        else:
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en == s:
                    continue
                if codes[e] == 3 or codes[e] == 4:
                    eo_all[s:en].copy_(x_s[s:en])
                    if codes[e] == 4:
                        if sw_eff is sw:
                            sw_eff = sw.clone()
                        sw_eff[s:en].neg_()
                    continue
                gu = x_s[s:en] @ gate_up_proj[e].t()
                _has_ap = ap32 is not None
                it = _glu_fwd(gu, row_act[s:en], code_hint=codes[e],
                              row_alpha=(ap32[e, 0:1] if _has_ap else None),
                              )
                torch.mm(it, down_proj[e].t(), out=eo_all[s:en])
                gate_up_l[e] = gu; inter_l[e] = it
        out = torch.zeros(N, H, device=dev, dtype=torch.float32)
        _combine_scatter(eo_all, sw_eff, st, out)
        ctx.sw_eff = sw_eff
        ctx.save_for_backward(x_s, st, sw, order, row_act, gate_up_proj, down_proj,
                              ap32 if ap32 is not None else torch.empty(0))
        ctx.lists = (gate_up_l, inter_l, eo_all); ctx.bounds = bounds; ctx.uniform = uniform
        ctx.offs = offs; ctx.shapes = (N, H, top_k, E); ctx.tile_map = tile_map; ctx.tile_map_gg = tile_map_gg; ctx.tile_map_bw = tile_map_bw
        ctx.codes = codes; ctx.has_ap = ap32 is not None; ctx.ap_shape = ap_shape
        return out.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x_s, st, sw, order, row_act, gate_up_proj, down_proj, ap32) = ctx.saved_tensors
        gate_up_l, inter_l, eo_all = ctx.lists
        sw_eff = ctx.sw_eff
        N, H, top_k, E = ctx.shapes; bounds = ctx.bounds; codes = ctx.codes
        M = st.numel()
        if ctx.offs is not None:
            gu_all, it_all = gate_up_l, inter_l
            offs = ctx.offs
            ge_all, gw_all = _combine_bwd(grad_out, eo_all, sw, st)
            grad_down_proj = torch._grouped_mm(ge_all.t(), it_all, offs=offs)
            hint = codes[0] if len(set(codes)) == 1 else None
            if ctx.tile_map is not None:
                grad_gate_up = _FUSED_GLU.fused_dinter_glu_bwd(
                    ge_all, down_proj, gu_all, ctx.tile_map_bw, codes[0])
            else:
                grad_inter = (_FUSED_GLU.grouped_gemm(ge_all, down_proj, ctx.tile_map_gg)
                              if ctx.tile_map_gg is not None else None)
                if grad_inter is None:
                    grad_inter = torch._grouped_mm(ge_all, down_proj, offs=offs)
                grad_gate_up = _glu_bwd(grad_inter, gu_all, row_act, code_hint=hint)
            grad_gate_up_proj = torch._grouped_mm(grad_gate_up.t(), x_s, offs=offs)
            grad_hidden = None
            if ctx.tile_map_gg is not None:
                gh32 = _FUSED_GLU.grouped_gemm_scatter(grad_gate_up, gate_up_proj, st,
                                                       ctx.tile_map_gg, N)
                if gh32 is not None:
                    grad_hidden = gh32.to(grad_out.dtype)
            if grad_hidden is None:
                grad_x = torch._grouped_mm(grad_gate_up, gate_up_proj, offs=offs)
                grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
                grad_hidden.index_add_(0, st, grad_x)
            grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
            grad_wt[order] = gw_all.to(grad_out.dtype)
            return (grad_hidden, None, grad_wt.view(N, top_k), grad_gate_up_proj, grad_down_proj,
                    None, None)
        if ctx.uniform:
            gu_all, it_all = gate_up_l, inter_l
            Icols = it_all.shape[1]
            ge_all, gw_all = _combine_bwd(grad_out, eo_all, sw, st)
            grad_w_s = gw_all.to(grad_out.dtype)
            grad_inter = torch.empty(M, Icols, device=grad_out.device, dtype=grad_out.dtype)
            grad_gate_up_proj = torch.zeros_like(gate_up_proj)
            grad_down_proj = torch.zeros_like(down_proj)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en == s:
                    continue
                torch.mm(ge_all[s:en], down_proj[e], out=grad_inter[s:en])
                torch.mm(ge_all[s:en].t(), it_all[s:en], out=grad_down_proj[e])
            hint = codes[0] if len(set(codes)) == 1 else None
            grad_gate_up = _glu_bwd(grad_inter, gu_all, row_act, code_hint=hint)
            grad_x = torch.empty(M, H, device=grad_out.device, dtype=grad_out.dtype)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en == s:
                    continue
                torch.mm(grad_gate_up[s:en].t(), x_s[s:en], out=grad_gate_up_proj[e])
                torch.mm(grad_gate_up[s:en], gate_up_proj[e], out=grad_x[s:en])
            grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
            grad_hidden.index_add_(0, st, grad_x)
            grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
            grad_wt[order] = grad_w_s
            return (grad_hidden, None, grad_wt.view(N, top_k), grad_gate_up_proj, grad_down_proj,
                    None, None)
        grad_w_s = torch.zeros(M, device=grad_out.device, dtype=grad_out.dtype)
        grad_gate_up_proj = torch.zeros_like(gate_up_proj)
        grad_down_proj = torch.zeros_like(down_proj)
        grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
        want_ap = ctx.has_ap and ctx.needs_input_grad[6]
        # Must match the shape the CALLER passed, not a fixed (E,2): autograd rejects a gradient
        # whose shape differs from its input, and a (E,) act_params is documented as legal.
        grad_act_params = (torch.zeros(E, 2, device=grad_out.device, dtype=torch.float32)
                           if want_ap else None)
        for e in range(E):
            s, en = bounds[e], bounds[e + 1]
            if en == s:
                continue
            if codes[e] == 3 or codes[e] == 4:
                sgn = 1.0 if codes[e] == 3 else -1.0
                ge, gw = _combine_bwd(grad_out, x_s[s:en], sw_eff[s:en], st[s:en])
                grad_w_s[s:en].copy_(gw * sgn)
                grad_hidden.index_add_(0, st[s:en], ge)
                continue
            it = inter_l[e]
            ge, gw = _combine_bwd(grad_out, eo_all[s:en], sw[s:en], st[s:en])
            grad_w_s[s:en].copy_(gw)
            grad_inter = ge @ down_proj[e]
            torch.mm(ge.t(), it, out=grad_down_proj[e])
            if want_ap:
                grad_gate_up, da = _glu_bwd(grad_inter, gate_up_l[e], row_act[s:en],
                                                code_hint=codes[e], row_alpha=ap32[e, 0:1],
                                                want_act_grads=True)
                grad_act_params[e, 0] = da.sum()
            else:
                _has_ap = ctx.has_ap
                grad_gate_up = _glu_bwd(grad_inter, gate_up_l[e], row_act[s:en], code_hint=codes[e],
                                        row_alpha=(ap32[e, 0:1] if _has_ap else None),
                                        )
            torch.mm(grad_gate_up.t(), x_s[s:en], out=grad_gate_up_proj[e])
            grad_hidden.index_add_(0, st[s:en], grad_gate_up @ gate_up_proj[e])
        grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = grad_w_s
        if grad_act_params is not None and ctx.ap_shape is not None:
            grad_act_params = grad_act_params[:, 0] if len(ctx.ap_shape) == 1 else                 grad_act_params[:, :ctx.ap_shape[1]]
        return (grad_hidden, None, grad_wt.view(N, top_k), grad_gate_up_proj, grad_down_proj, None,
                grad_act_params)


def moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
                   act_params=None):
    return _PerExpertMoE.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj,
                               act_codes, act_params)


def moe(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    cap_major = torch.cuda.get_device_capability(hidden.device)[0]
    glu_only = _code_max(act_codes) <= 2
    if top_k_indices.numel() >= GROUPED_MIN_TOKENS and cap_major >= 8 and glu_only:
        return moe_grouped(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)
    return moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
                          act_params)


def _act_eager(gate, code, alpha=None):
    a = 1.0 if alpha is None else alpha
    if code == 0:
        return F.silu(a * gate.float()).to(gate.dtype)
    if code == 8:
        g = gate.float()
        r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
        pw = torch.sigmoid(alpha if torch.is_tensor(alpha) else torch.tensor(float(a), device=g.device))
        return (r.pow(pw) * F.silu(g / r)).to(gate.dtype)
    if code != 2:
        raise ValueError(f"unsupported act code {code}; only 0 (silu), 2 (normsilu), 8 (radial) exist")

    g = gate.float()
    g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
    return F.silu(a * g).to(gate.dtype)


def moe_eager(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
              act_params=None):
    N, H = hidden.shape
    twoI = gate_up_proj.shape[1]
    I = twoI // 2
    codes = _codes_list(act_codes)
    E = len(codes)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)
    for e in range(E):
        rows = (top_k_indices == e).any(-1)
        if not bool(rows.any()):
            continue
        w = (top_k_weights * (top_k_indices == e)).sum(-1)[rows]
        if codes[e] == 3 or codes[e] == 4:
            sgn = 1.0 if codes[e] == 3 else -1.0
            out[rows] += (hidden[rows] * (sgn * w).unsqueeze(-1)).float()
            continue
        gate_up = hidden[rows] @ gate_up_proj[e].t()
        a, g = ((act_params[e, 0], act_params[e, 1]) if codes[e] == 5 and act_params is not None
                else (1.0, 1.0))
        inter = _act_eager(gate_up[:, :I], codes[e], a) * gate_up[:, I:]
        out[rows] += ((inter @ down_proj[e].t()) * w.unsqueeze(-1)).float()
    return out.to(hidden.dtype)
