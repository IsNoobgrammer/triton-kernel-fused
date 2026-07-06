"""Shared Muon update scalings (arch-independent) — used by kernels/sm75/muon.py and sm120/muon.py.

A "scaling" maps the orthogonalized update `O = NewtonSchulz(momentum)` (shape `(M, rows, cols)`,
M stacked same-shape matrices) to the tensor added to the weight, before the outer `-lr`.

Every mode targets update RMS = RMS_TARGET (0.2) — the Moonlight / DeepSeek-V4 convention — so AdamW
LR and weight decay carry over unchanged, for every mode. The modes differ only in the ROW SHAPE of
the update:

  polar   : plain orthogonalized update, scaled by 0.2*sqrt(max(rows,cols)). On a tall matrix its row
            norms follow the leverage scores — some rows get near-zero updates ("neuron death").
  normuon : per-row EMA second-moment normalize AFTER the polar -> uniform row norms. Cheap fix for
            neuron death, but a post-hoc row rescale slightly breaks the orthogonal spectrum.
  aurora  : DEFAULT. Divide rows by their norm BEFORE the polar and re-orthogonalize (K passes), so
            the output is both leverage-balanced and orthogonal. Costs K polar solves.

Square matrices have uniform rows anyway, so all three coincide there; the modes only differ on
rectangular (tall) weights.

Self-check: python -m kernels.muon.muon_scaling
"""
import torch

SCALAR_MODES = ("polar",)
PERROW_MODES = ("normuon",)
AURORA_MODES = ("aurora",)
AURORA_EMA_MODES = ("aurora_ema", "aurora_ema_v2")                    # aurora + normuon EMA: v1 pre-polar (stays orthogonal), v2 post-polar (normuon-faithful, breaks it)
ALL_MODES = SCALAR_MODES + PERROW_MODES + AURORA_MODES + AURORA_EMA_MODES
DEFAULT_MODE = "aurora"

# Update RMS every mode targets (Moonlight convention; = DeepSeek-V4's sqrt(max(n,m))*gamma). For a
# (near-)orthogonal O, RMS(O) = 1/sqrt(max(rows,cols)), so RMS_TARGET*sqrt(max) pins RMS at 0.2.
RMS_TARGET = 0.2

PERROW_BETA2 = 0.95
PERROW_EPS = 1e-8
# aurora polar-solve iterations. K=1 (prescale + one polar) matches the paper's K=2 for our task at
# HALF the cost. K=2 = the paper's full method.
AURORA_K = 1
# aurora row-norm damping: D_k = D_{k-1}^beta * rownorm^(1-beta). 0 = full prescale (best at K=1,
# measured); the paper uses 0.5 at K=2.
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
    """Modes that keep a persistent per-row EMA buffer (normuon post-hoc; aurora_ema prescale)."""
    return mode in PERROW_MODES or mode in AURORA_EMA_MODES


def folds_scale(mode):
    """Modes that bake the update scale into the tensor (caller applies -lr, not the scalar)."""
    return mode in AURORA_MODES or mode in PERROW_MODES or mode in AURORA_EMA_MODES


def validate(mode):
    if mode in _REMOVED:
        raise ValueError(f"scale_mode {mode!r} was removed; use {_REMOVED[mode]!r} "
                         f"(all modes now share the AdamW LR band, update RMS {RMS_TARGET})")
    if mode not in ALL_MODES:
        raise ValueError(f"unknown scale_mode {mode!r}; choose from {ALL_MODES}")
    return mode


def scalar_scale(mode, rows, cols):
    """The per-matrix constant for a SCALAR mode (raises for other modes)."""
    if mode == "polar":
        return RMS_TARGET * (max(rows, cols) ** 0.5)
    raise ValueError(f"{mode!r} is not a scalar scale_mode")


def perrow_state(M, rows, device):
    """EMA second-moment buffer for `normuon`: one value per (stacked-matrix, row). Persisted in
    optimizer state so it round-trips through state_dict, exactly like the momentum buffer."""
    return torch.zeros((M, rows), device=device, dtype=torch.float32)


def aurora_update(M, polar_fn, gain=None, K=AURORA_K, beta=AURORA_BETA, eps=PERROW_EPS):
    """Aurora: iterate {leverage-prescale rows, re-orthogonalize} K times, then scale to the AdamW band.

    M: (B, rows, cols) batch of momenta. polar_fn: the arch's orthogonalizer (returns U V^T, same shape).
    Each iteration divides rows by a (damped) row-norm and re-runs the polar, so the output is a
    leverage-BALANCED near-orthogonal factor, unlike a post-hoc row rescale which breaks orthogonality.
    gain=None (default) -> RMS_TARGET*sqrt(max(rows,cols)): update RMS ~= 0.2, AdamW LR/WD reusable.
    Cost: K polar solves.
    """
    rows, cols = M.shape[-2], M.shape[-1]
    if gain is None:
        gain = RMS_TARGET * (max(rows, cols) ** 0.5)
    tgt = (min(rows, cols) / rows) ** 0.5                        # sqrt(cols/rows) tall, 1 wide
    dt = M.dtype
    X = M.float()
    fro = X.flatten(-2, -1).norm(dim=-1).clamp_min(eps)          # per-matrix Frobenius -> (B,)
    X = X / fro.view(*fro.shape, 1, 1)
    D = torch.ones(X.shape[:-1], device=X.device)                # (B, rows)
    for _ in range(K):
        r = X.norm(dim=-1).clamp_min(eps)                        # per-row L2 -> (B, rows)
        D = D.pow(beta) * r.pow(1.0 - beta)
        X = polar_fn((tgt * (X / D.unsqueeze(-1))).to(dt)).float()
    return (gain * X).to(dt)


def aurora_ema_update(M, polar_fn, v_ema, gain=None, beta2=PERROW_BETA2, eps=PERROW_EPS):
    """Aurora WITH MEMORY = best-of-both attempt. Prescale rows by their EMA 2nd-moment
    (normuon's per-row adaptivity) BEFORE the polar, then re-orthogonalize - so unlike normuon's
    post-hoc rescale, the output STAYS orthogonal (aurora's strength) while gaining cross-step
    per-row memory (normuon's strength). One polar solve (K=1).

    M: (B, rows, cols) momenta. polar_fn: the arch's orthogonalizer. v_ema: (B, rows) fp32 EMA
    buffer, MUTATED IN PLACE (persisted in optimizer state, +rows/(rows*cols) memory ~ normuon).
    """
    rows, cols = M.shape[-2], M.shape[-1]
    if gain is None:
        gain = RMS_TARGET * (max(rows, cols) ** 0.5)
    tgt = (min(rows, cols) / rows) ** 0.5                        # sqrt(cols/rows) tall, 1 wide
    dt = M.dtype
    X = M.float()
    fro = X.flatten(-2, -1).norm(dim=-1).clamp_min(eps)          # per-matrix Frobenius -> (B,)
    X = X / fro.view(*fro.shape, 1, 1)
    row_ms = X.mul(X).mean(dim=-1)                               # per-row mean-square -> (B, rows)
    v_ema.mul_(beta2).add_(row_ms, alpha=1.0 - beta2)           # EMA in place (normuon's memory)
    D = v_ema.sqrt().clamp_min(eps)                              # EMA-based row scale -> (B, rows)
    X = polar_fn((tgt * (X / D.unsqueeze(-1))).to(dt)).float()  # re-orthogonalize (output stays orthogonal)
    return (gain * X).to(dt)


def aurora_ema_v2_update(M, polar_fn, v_ema, gain=None, K=AURORA_K, beta2=PERROW_BETA2, eps=PERROW_EPS):
    """Aurora THEN normuon post-hoc (the faithful stack): run the full aurora update (prescale +
    re-orthogonalize), then rescale rows by their EMA 2nd-moment AFTER the polar - exactly where
    normuon applies it. This DOES break orthogonality (unlike aurora_ema v1's pre-polar EMA), so
    v1-vs-v2 isolates whether the per-row EMA belongs before or after the orthogonalization.

    v_ema: (B, rows) fp32 EMA buffer, MUTATED IN PLACE (post-ortho row 2nd-moment, like normuon).
    """
    rows, cols = M.shape[-2], M.shape[-1]
    if gain is None:
        gain = RMS_TARGET * (max(rows, cols) ** 0.5)
    O = aurora_update(M, polar_fn, gain=gain, K=K).float()      # aurora output: orthogonal, RMS 0.2
    row_sq = O.mul(O).mean(dim=-1)                              # per-row mean-square -> (B, rows)
    v_ema.mul_(beta2).add_(row_sq, alpha=1.0 - beta2)          # EMA in place (post-ortho, normuon-style)
    Ohat = O / v_ema.sqrt().add(eps).unsqueeze(-1)             # per-row rescale (breaks orthogonality)
    fro = Ohat.flatten(-2).norm(dim=-1).clamp_min(1e-12)       # renorm each slice to the RMS target
    C = RMS_TARGET * (rows * cols) ** 0.5
    return (Ohat * (C / fro).view(*fro.shape, 1, 1)).to(M.dtype)


def apply_perrow(mode, O, v, beta2=PERROW_BETA2, eps=PERROW_EPS):
    """NorMuon on a batch of same-shape updates, scaled to the AdamW band (update RMS 0.2).

    O : (M, rows, cols) orthogonalized updates (M stacked matrices).
    v : (M, rows) fp32 EMA second-moment buffer, MUTATED IN PLACE.
    Returns the tensor T (same shape/dtype as O) to add as `weight -= lr * T` — i.e. lr is folded
    OUT (the caller applies `-lr`), the RMS_TARGET and Frobenius factors are folded IN. Frobenius
    norm is per-slice (each stacked matrix scaled by its own ||Ohat||).
    """
    if mode not in PERROW_MODES:
        raise ValueError(f"{mode!r} is not a per-row scale_mode")
    rows, cols = O.shape[-2], O.shape[-1]
    Of = O.float()
    row_sq = Of.mul(Of).mean(dim=-1)                              # mean_cols(O*O) -> (M, rows)
    v.mul_(beta2).add_(row_sq, alpha=1.0 - beta2)                 # EMA in place
    Ohat = Of / v.sqrt().add(eps).unsqueeze(-1)                   # per-row RMS normalize -> (M, rows, cols)
    fro = Ohat.flatten(-2).norm(dim=-1).clamp_min(1e-12)          # per-slice Frobenius -> (M,)
    C = RMS_TARGET * (rows * cols) ** 0.5                         # ||T||_F -> 0.2*sqrt(m*n) => RMS 0.2
    return (Ohat * (C / fro).view(-1, 1, 1)).to(O.dtype)


def xorth_whiten(O, beta, eps=1e-6):
    """AFTER-NS cross-expert decorrelation: damped E x E whitening of the ORTHOGONALIZED update.

    O    : (E, rows, cols) - the E experts' post-NS updates for one weight (E stacked matrices).
    beta : strength (0 = off, 1 = full whiten). Damped: T = (1-beta) I + beta * C^{-1/2}.
    Decorrelates the CLEAN uniform-spectrum updates (well-conditioned gram) instead of the noisy
    raw gradient (pre-NS). Cost: mixing experts by T breaks each expert's individual orthogonality
    (the post-polar trade, cf. normuon). Returns T @ O along the E axis, same shape/dtype as O.
    """
    E = O.shape[0]
    G = O.reshape(E, -1).float()                                  # (E, D) flatten rows*cols
    C = G @ G.mT                                                  # (E, E) cross-expert gram
    C = C / C.diagonal().mean().clamp_min(1e-12)                  # normalize (diag ~1)
    ev, V = torch.linalg.eigh(C)
    isq = V @ torch.diag_embed(ev.clamp_min(1e-6).rsqrt()) @ V.mT # C^{-1/2}
    T = (1 - beta) * torch.eye(E, device=O.device, dtype=C.dtype) + beta * isq
    return (T @ G).reshape_as(O).to(O.dtype)


def spectral_wd_mult(u, e_ema, gamma, beta=0.99, eps=1e-12):
    """Spectral weight decay: REDISTRIBUTE decoupled wd across rows by their accumulated momentum energy.

    u     : (M, rows, cols) pre-orthogonalization momentum (the NS input).
    e_ema : (M, rows) fp32 per-row energy EMA, MUTATED IN PLACE.
    gamma : redistribution strength (0 = uniform = standard wd).
    Returns (mult, cov):
      mult : (M, rows) per-row multiplier on the decay, mean 1 per slice (so AVG decay == wd; only the
             DISTRIBUTION changes). Low-energy (stale) rows -> mult > 1 (decayed harder); active < 1.
      cov  : mean per-slice coefficient-of-variation of e_ema (the 5-min gate: if ~0, rows are uniform
             and spectral wd is a no-op regardless of gamma).
    """
    e_now = u.float().pow(2).mean(dim=-1)                         # (M, rows) per-row energy this step
    e_ema.mul_(beta).add_(e_now, alpha=1.0 - beta)               # EMA in place
    mean = e_ema.mean(dim=-1, keepdim=True).clamp_min(eps)        # per-slice mean -> (M,1)
    cov = (e_ema.std(dim=-1) / mean.squeeze(-1)).mean()          # gate diagnostic (scalar)
    if gamma == 0:
        return None, cov
    s = (e_ema / mean).clamp_min(eps).pow(-gamma)                # low energy -> large multiplier
    s = s / s.mean(dim=-1, keepdim=True).clamp_min(eps)          # renormalize: mean 1 per slice
    return s.clamp(0.25, 4.0), cov


def _selfcheck():                                                # pragma: no cover
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
        # polar scalar and aurora both pin RMS at RMS_TARGET on a (near-)orthogonal input
        Q = torch.linalg.qr(torch.randn(1, max(m, n), min(m, n), device=dev))[0]
        Q = Q if m >= n else Q.transpose(-2, -1)
        rms_sc = (scalar_scale("polar", m, n) * Q).pow(2).mean().sqrt().item()
        rms_au = aurora_update(Q, lambda x: x, K=1).pow(2).mean().sqrt().item()
        for name, rms in [("polar", rms_sc), ("aurora", rms_au)]:
            assert abs(rms - RMS_TARGET) / RMS_TARGET < 0.05, f"{name} {m}x{n}: RMS {rms:.4f}"
        print(f"{m:>5}x{n:<5}  RMS  polar {rms_sc:.4f}  normuon {rms_pr:.4f}  aurora {rms_au:.4f}"
              f"  | normuon row-CV {cv:.4f} dead {dead:.0%}")
    print(f"muon_scaling self-check PASS (all modes AdamW band, RMS {RMS_TARGET})")


if __name__ == "__main__":                                       # pragma: no cover
    _selfcheck()
