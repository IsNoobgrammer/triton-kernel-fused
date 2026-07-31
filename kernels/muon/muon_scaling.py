import torch

SCALAR_MODES = ("polar",)
PERROW_MODES = ("normuon",)
AURORA_MODES = ("aurora",)
AURORA_EMA_MODES = ("aurora_ema", "aurora_ema_v2")
ALL_MODES = SCALAR_MODES + PERROW_MODES + AURORA_MODES + AURORA_EMA_MODES
DEFAULT_MODE = "aurora"

RMS_TARGET = 0.2

PERROW_BETA2 = 0.95
PERROW_EPS = 1e-8
AURORA_K = 1
AURORA_BETA = 0.0

_REMOVED = {"moonlight": "polar", "polarexpress": "polar", "jordan": "polar",
            "unormuon": "normuon", "unormuon_spectral": "normuon"}


def is_perrow(mode):
    return mode in PERROW_MODES


def is_aurora(mode):
    return mode in AURORA_MODES


def is_aurora_ema(mode):
    return mode in AURORA_EMA_MODES


def needs_perrow_state(mode):
    return mode in PERROW_MODES or mode in AURORA_EMA_MODES


def folds_scale(mode):
    return mode in AURORA_MODES or mode in PERROW_MODES or mode in AURORA_EMA_MODES


def validate(mode):
    if mode in _REMOVED:
        raise ValueError(f"scale_mode {mode!r} was removed; use {_REMOVED[mode]!r} "
                         f"(all modes now share the AdamW LR band, update RMS {RMS_TARGET})")
    if mode not in ALL_MODES:
        raise ValueError(f"unknown scale_mode {mode!r}; choose from {ALL_MODES}")
    return mode


def scalar_scale(mode, rows, cols):
    if mode == "polar":
        return RMS_TARGET * (max(rows, cols) ** 0.5)
    raise ValueError(f"{mode!r} is not a scalar scale_mode")


def perrow_state(M, rows, device):
    return torch.zeros((M, rows), device=device, dtype=torch.float32)


def aurora_update(M, polar_fn, gain=None, K=AURORA_K, beta=AURORA_BETA, eps=PERROW_EPS):
    rows, cols = M.shape[-2], M.shape[-1]
    if gain is None:
        gain = RMS_TARGET * (max(rows, cols) ** 0.5)
    tgt = (min(rows, cols) / rows) ** 0.5
    dt = M.dtype
    if K == 1 and beta == 0.0:
        rn = torch.linalg.vector_norm(M, dim=-1, dtype=torch.float32).clamp_min(eps)
        X = polar_fn((M * (tgt / rn).unsqueeze(-1)).to(dt))
        return (X * gain).to(dt)
    X = M.float()
    fro = X.flatten(-2, -1).norm(dim=-1).clamp_min(eps)
    X = X / fro.view(*fro.shape, 1, 1)
    D = torch.ones(X.shape[:-1], device=X.device)
    for _ in range(K):
        r = X.norm(dim=-1).clamp_min(eps)
        D = D.pow(beta) * r.pow(1.0 - beta)
        X = polar_fn((tgt * (X / D.unsqueeze(-1))).to(dt)).float()
    return (gain * X).to(dt)


def aurora_ema_update(M, polar_fn, v_ema, gain=None, beta2=PERROW_BETA2, eps=PERROW_EPS):
    rows, cols = M.shape[-2], M.shape[-1]
    if gain is None:
        gain = RMS_TARGET * (max(rows, cols) ** 0.5)
    tgt = (min(rows, cols) / rows) ** 0.5
    dt = M.dtype
    rn = torch.linalg.vector_norm(M, dim=-1, dtype=torch.float32)
    fro = torch.linalg.vector_norm(rn, dim=-1).clamp_min(eps)
    row_ms = (rn / fro.unsqueeze(-1)).square() / cols
    v_ema.mul_(beta2).add_(row_ms, alpha=1.0 - beta2)
    D = v_ema.sqrt().clamp_min(eps)
    X = polar_fn((M * (tgt / (fro.unsqueeze(-1) * D)).unsqueeze(-1)).to(dt))
    return (X * gain).to(dt)


def aurora_ema_v2_update(M, polar_fn, v_ema, gain=None, K=AURORA_K, beta2=PERROW_BETA2, eps=PERROW_EPS):
    rows, cols = M.shape[-2], M.shape[-1]
    if gain is None:
        gain = RMS_TARGET * (max(rows, cols) ** 0.5)
    O = aurora_update(M, polar_fn, gain=gain, K=K).float()
    row_sq = O.mul(O).mean(dim=-1)
    v_ema.mul_(beta2).add_(row_sq, alpha=1.0 - beta2)
    Ohat = O / v_ema.sqrt().add(eps).unsqueeze(-1)
    fro = Ohat.flatten(-2).norm(dim=-1).clamp_min(1e-12)
    C = RMS_TARGET * (rows * cols) ** 0.5
    return (Ohat * (C / fro).view(*fro.shape, 1, 1)).to(M.dtype)


def apply_perrow(mode, O, v, beta2=PERROW_BETA2, eps=PERROW_EPS):
    if mode not in PERROW_MODES:
        raise ValueError(f"{mode!r} is not a per-row scale_mode")
    rows, cols = O.shape[-2], O.shape[-1]
    rn = torch.linalg.vector_norm(O, dim=-1, dtype=torch.float32)
    v.mul_(beta2).add_(rn.square() / cols, alpha=1.0 - beta2)
    inv = 1.0 / (v.sqrt() + eps)
    fro = torch.linalg.vector_norm(rn * inv, dim=-1).clamp_min(1e-12)
    C = RMS_TARGET * (rows * cols) ** 0.5
    mult = inv * (C / fro).unsqueeze(-1)
    return (O * mult.unsqueeze(-1)).to(O.dtype)


def xorth_whiten_batch(G, beta, eps=1e-6):
    C = G @ G.mT
    C = C / C.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12).view(-1, 1, 1)
    ev, V = torch.linalg.eigh(C)
    isq = V @ torch.diag_embed(ev.clamp_min(eps).rsqrt()) @ V.mT
    T = beta * isq
    T.diagonal(dim1=-2, dim2=-1).add_(1.0 - beta)
    return T @ G


def xorth_whiten_ns(G, beta, iters=18, ridge=1e-3):
    C = G @ G.mT
    C = C / C.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12).view(-1, 1, 1)
    isq = _db_isq(C, iters, ridge)
    T = beta * isq
    T.diagonal(dim1=-2, dim2=-1).add_(1.0 - beta)
    return T @ G


def _db_isq(C, iters, ridge):
    Cw = C.clone()
    Cw.diagonal(dim1=-2, dim2=-1).add_(ridge)
    s = Cw.abs().sum(-1).amax(-1).clamp_min(1e-12).view(-1, 1, 1)
    E = Cw.shape[-1]
    I = torch.eye(E, device=Cw.device, dtype=Cw.dtype).expand_as(Cw)
    Y = Cw / s
    Z = I
    for _ in range(iters):
        Mk = 1.5 * I - 0.5 * (Z @ Y)
        Y = Y @ Mk
        Z = Mk @ Z
    return Z / s.sqrt()


def xorth_whiten_gated(G, cema, beta_max, rho=0.95, gate_ref=0.3, iters=18, ridge=1e-3):
    E = G.shape[1]
    C = G @ G.mT
    C = C / C.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-12).view(-1, 1, 1)
    cema.mul_(rho).add_(C, alpha=1.0 - rho)
    off = cema.clone()
    off.diagonal(dim1=-2, dim2=-1).zero_()
    corr = off.square().sum(dim=(-2, -1)).div(max(E * E - E, 1)).sqrt()
    if gate_ref > 0:
        gate = (corr / gate_ref).clamp(0.0, 1.0)
    else:
        gate = torch.ones_like(corr)
    beta = beta_max * gate
    isq = _db_isq(cema, iters, ridge)
    T = beta.view(-1, 1, 1) * isq
    T.diagonal(dim1=-2, dim2=-1).add_((1.0 - beta).unsqueeze(-1))
    return T @ G


def xorth_whiten(O, beta, eps=1e-6):
    E = O.shape[0]
    G = O.reshape(E, -1).float().unsqueeze(0)
    return xorth_whiten_batch(G, beta, eps)[0].reshape_as(O).to(O.dtype)


def spectral_wd_mult(u, e_ema, gamma, beta=0.99, eps=1e-12):
    e_now = u.float().pow(2).mean(dim=-1)
    e_ema.mul_(beta).add_(e_now, alpha=1.0 - beta)
    mean = e_ema.mean(dim=-1, keepdim=True).clamp_min(eps)
    cov = (e_ema.std(dim=-1) / mean.squeeze(-1)).mean()
    if gamma == 0:
        return None, cov
    s = (e_ema / mean).clamp_min(eps).pow(-gamma)
    s = s / s.mean(dim=-1, keepdim=True).clamp_min(eps)
    return s.clamp(0.25, 4.0), cov


def _selfcheck():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (m, n) in [(2048, 2048), (8192, 2048), (2048, 8192)]:
        O = torch.randn(2, m, n, device=dev)
        O = O / O.flatten(1).norm(dim=1).view(-1, 1, 1) * (min(m, n) ** 0.5)
        v = perrow_state(2, m, dev)
        T = apply_perrow("normuon", O, v)
        rn = T[0].norm(dim=-1)
        cv = (rn.std() / rn.mean()).item()
        dead = (rn < 0.1 * rn.mean()).float().mean().item()
        rms_pr = T.pow(2).mean().sqrt().item()
        assert cv < 0.05 and dead == 0.0, f"normuon {m}x{n}: CV {cv:.3f} dead {dead:.0%}"
        assert abs(rms_pr - RMS_TARGET) / RMS_TARGET < 0.05, f"normuon {m}x{n}: RMS {rms_pr:.4f}"
        Q = torch.linalg.qr(torch.randn(1, max(m, n), min(m, n), device=dev))[0]
        Q = Q if m >= n else Q.transpose(-2, -1)
        rms_sc = (scalar_scale("polar", m, n) * Q).pow(2).mean().sqrt().item()
        rms_au = aurora_update(Q, lambda x: x, K=1).pow(2).mean().sqrt().item()
        for name, rms in [("polar", rms_sc), ("aurora", rms_au)]:
            assert abs(rms - RMS_TARGET) / RMS_TARGET < 0.05, f"{name} {m}x{n}: RMS {rms:.4f}"
        print(f"{m:>5}x{n:<5}  RMS  polar {rms_sc:.4f}  normuon {rms_pr:.4f}  aurora {rms_au:.4f}"
              f"  | normuon row-CV {cv:.4f} dead {dead:.0%}")
    print(f"muon_scaling self-check PASS (all modes AdamW band, RMS {RMS_TARGET})")


if __name__ == "__main__":
    _selfcheck()
