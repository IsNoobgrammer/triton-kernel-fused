"""Muon variants: WHAT the orthogonalized step does to a matrix, one class per optimizer.

    FusedMuon(params, variant="aurora", scale="adam", ns_coeffs="ns8")

variant   polar | aurora | normuon | muown        (or an instance, e.g. Aurora(k=2), Muown(betas=...))
scale     "adam" -> every variant's update has RMS 0.2 (Moonlight / DeepSeek-V4 convention), so the
                    AdamW lr and wd carry over unchanged. This is 0.2*sqrt(max(rows, cols)) on an
                    orthogonal update; normuon renormalizes to exactly that RMS.
          "none" -> the raw orthogonalized update (unit singular values). The lr must be retuned.
ns_coeffs a preset name from NS_PRESETS or an explicit tuple of (a, b, c) per iteration.

Every variant runs the same momentum, the same Newton-Schulz and the same decoupled wd; the only
thing that differs is the row handling below, so an A/B between two variants changes one thing.
"""
import torch

RMS_TARGET = 0.2
SCALES = ("adam", "none")

_KJ = (3.4445, -4.7750, 2.0315)      # Keller Jordan quintic: fast growth of small singular values
_PIN = (2.0, -1.5, 0.5)              # finishing step: pulls the singular values onto 1
NS_PRESETS = {
    "ns8": (_KJ,) * 6 + (_PIN,) * 2,                 # BiBo board default
    "dsv4": (_KJ,) * 8 + (_PIN,) * 2,                # DeepSeek-V4 10-step schedule
    "quintic5": (_KJ,) * 5,                          # Keller Jordan / Muown reference
    "pe8": (
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
        (1.8750014808534479, -1.2500016453999487, 0.3750001645474248),
        (1.875, -1.25, 0.375),
    ),                                               # Polar Express 8-step
}

PERROW_BETA2 = 0.95
PERROW_EPS = 1e-8

# Removed names -> what to use. aurora_ema / aurora_ema_v2 were closed twice (137M four-way tie,
# MNIST-1D five-way tie incl. polar): deleted Sep 24 2026, recoverable from git before that date.
_REMOVED = {"moonlight": "polar", "polarexpress": "polar", "jordan": "polar",
            "unormuon": "normuon", "unormuon_spectral": "normuon",
            "aurora_ema": "aurora", "aurora_ema_v2": "aurora"}


def ns_coeffs(spec):
    if isinstance(spec, str):
        if spec not in NS_PRESETS:
            raise ValueError(f"unknown ns_coeffs preset {spec!r}; choose from {tuple(NS_PRESETS)}")
        return NS_PRESETS[spec]
    return tuple(tuple(c) for c in spec)


def validate_scale(scale):
    if scale not in SCALES:
        raise ValueError(f"unknown scale {scale!r}; choose from {SCALES}")
    return scale


def gain(scale, rows, cols):
    """Multiplier that takes an orthogonal (rows, cols) update to the scale's RMS."""
    return RMS_TARGET * (max(rows, cols) ** 0.5) if scale == "adam" else 1.0


class Variant:
    """Base = polar. Subclasses override `direction`, or own the whole chunk step (Muown).

    folds_gain: False -> the optimizer applies `gain` through the lr (alpha = -lr * gain), which
                keeps polar bit-identical to the pre-refactor path; True -> direction() returns
                an already-scaled update and alpha = -lr.
    graphable:  the CUDA-graph fast path replays a fixed kernel sequence with no per-row state.
    owns_step:  the variant replaces momentum + write-back for its chunk (Muown).
    needs_weights: init_state reads the weight values (Muown); otherwise only the shape is used,
                so the optimizer never materializes a copy of the weights for it.
    """
    name = "polar"
    folds_gain = False
    graphable = True
    owns_step = False
    needs_weights = False

    def init_state(self, shape, device, W=None):
        """shape (M, r, c) of a bucket; W its weights iff needs_weights -> state dict, or None."""
        return None

    def direction(self, u, polar, state, scale, rows, cols):
        return polar(u)

    def __repr__(self):
        return f"{type(self).__name__}()"


class Polar(Variant):
    pass


class Aurora(Variant):
    """Divide each momentum row by its norm, THEN orthogonalize (k passes). Uniform rows, orthogonal."""
    name = "aurora"
    folds_gain = True
    graphable = False

    def __init__(self, k=1):
        self.k = int(k)

    def direction(self, u, polar, state, scale, rows, cols):
        return aurora_update(u, polar, gain=gain(scale, rows, cols), K=self.k)

    def __repr__(self):
        return f"Aurora(k={self.k})"


class NorMuon(Variant):
    """Orthogonalize, then divide rows by an EMA of their mean square; renormalize the whole update."""
    name = "normuon"
    folds_gain = True
    graphable = False

    def __init__(self, beta2=PERROW_BETA2, eps=PERROW_EPS):
        self.beta2, self.eps = float(beta2), float(eps)

    def init_state(self, shape, device, W=None):
        return {"v": torch.zeros(shape[:-1], device=device, dtype=torch.float32)}

    def direction(self, u, polar, state, scale, rows, cols):
        fro = RMS_TARGET * (rows * cols) ** 0.5 if scale == "adam" else min(rows, cols) ** 0.5
        return apply_perrow(polar(u), state["v"], fro, self.beta2, self.eps)

    def __repr__(self):
        return f"NorMuon(beta2={self.beta2:g})"


class Muown(Variant):
    """W_i = g_i * v_i/||v_i||: Muon on the direction v, Adam on the per-row gain g, same lr.
    arXiv 2605.10797; reference github.com/kcc-lion/muown optim/muown.py @3bd0c05."""
    name = "muown"
    folds_gain = True
    graphable = False
    owns_step = True
    needs_weights = True

    def __init__(self, betas=None, eps=None):
        self.betas = tuple(betas) if betas is not None else MUOWN_BETAS
        self.eps = float(eps) if eps is not None else MUOWN_EPS

    def init_state(self, shape, device, W=None):
        return muown_state(W)

    def __repr__(self):
        return f"Muown(betas={self.betas})"


VARIANTS = {"polar": Polar, "aurora": Aurora, "normuon": NorMuon, "muown": Muown}
DEFAULT_VARIANT = "aurora"


def make_variant(spec):
    if isinstance(spec, Variant):
        return spec
    if spec in _REMOVED:
        raise ValueError(f"variant {spec!r} was removed; use {_REMOVED[spec]!r}")
    if spec not in VARIANTS:
        raise ValueError(f"unknown variant {spec!r}; choose from {tuple(VARIANTS)}")
    return VARIANTS[spec]()


def slice_state(state, start, n):
    """Per-row state tensors are stacked over a bucket's matrices; take one chunk's rows."""
    return None if state is None else {k: v[start:start + n] for k, v in state.items()}


def aurora_update(M, polar_fn, gain=None, K=1, beta=0.0, eps=PERROW_EPS):
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


def apply_perrow(O, v, fro, beta2=PERROW_BETA2, eps=PERROW_EPS):
    """NorMuon row normalize: rows / sqrt(EMA of row mean-square), then Frobenius -> `fro`."""
    rows, cols = O.shape[-2], O.shape[-1]
    rn = torch.linalg.vector_norm(O, dim=-1, dtype=torch.float32)
    v.mul_(beta2).add_(rn.square() / cols, alpha=1.0 - beta2)
    inv = 1.0 / (v.sqrt() + eps)
    nrm = torch.linalg.vector_norm(rn * inv, dim=-1).clamp_min(1e-12)
    mult = inv * (fro / nrm).unsqueeze(-1)
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


# Muown (arXiv 2605.10797, reference github.com/kcc-lion/muown optim/muown.py @3bd0c05).
# Every 2D row is W_i = g_i * v_i / ||v_i||. Muon moves the direction v, Adam moves the gain g,
# at the SAME lr (Muon's step carries the 0.2*sqrt(max) AdamW-RMS match). All math is fp32 on
# (n, r, c) stacks; g / vn / m / s are (n, r). Reference names: v_norm -> vn, m_g -> m, v_g -> s.
MUOWN_BETAS = (0.9, 0.95)
MUOWN_EPS = 1e-8


def muown_state(W):
    rn = torch.linalg.vector_norm(W.float(), dim=-1)
    z = torch.zeros_like(rn)
    return {"g": rn.clone(), "vn": rn.clone(), "m": z, "s": z.clone()}


def muown_split(W, G, g, vn):
    """(W, dL/dW) -> (v, dL/dg, dL/dv). Reference `_wn_pre_ns`, same op order."""
    u = W / g.unsqueeze(-1)
    grad_g = (G * u).sum(dim=-1)
    grad_v = (g / vn).unsqueeze(-1) * (G - u * grad_g.unsqueeze(-1))
    return u * vn.unsqueeze(-1), grad_g, grad_v


def muown_adam_g(g, m, s, grad_g, lr, t, betas=MUOWN_BETAS, eps=MUOWN_EPS):
    b1, b2 = betas
    m.mul_(b1).add_(grad_g, alpha=1 - b1)
    s.mul_(b2).addcmul_(grad_g, grad_g, value=1 - b2)
    g.addcdiv_(m / (1 - b1 ** t), (s / (1 - b2 ** t)).sqrt().add_(eps), value=-lr)


def muown_compose(v_new, g):
    """W = g * v_new / ||v_new||; returns (W, new vn). Reference `_wn_recompose`."""
    vn = torch.linalg.vector_norm(v_new, dim=-1)
    return g.unsqueeze(-1) * (v_new / vn.unsqueeze(-1)), vn


def _selfcheck():
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (m, n) in [(2048, 2048), (8192, 2048), (2048, 8192)]:
        O = torch.randn(2, m, n, device=dev)
        O = O / O.flatten(1).norm(dim=1).view(-1, 1, 1) * (min(m, n) ** 0.5)
        nm = NorMuon()
        T = nm.direction(O, lambda x: x, nm.init_state(O.shape, O.device), "adam", m, n)
        rn = T[0].norm(dim=-1)
        cv = (rn.std() / rn.mean()).item()
        dead = (rn < 0.1 * rn.mean()).float().mean().item()
        rms_pr = T.pow(2).mean().sqrt().item()
        assert cv < 0.05 and dead == 0.0, f"normuon {m}x{n}: CV {cv:.3f} dead {dead:.0%}"
        assert abs(rms_pr - RMS_TARGET) / RMS_TARGET < 0.05, f"normuon {m}x{n}: RMS {rms_pr:.4f}"
        Q = torch.linalg.qr(torch.randn(1, max(m, n), min(m, n), device=dev))[0]
        Q = Q if m >= n else Q.transpose(-2, -1)
        rms_sc = (gain("adam", m, n) * Q).pow(2).mean().sqrt().item()
        rms_au = Aurora().direction(Q, lambda x: x, None, "adam", m, n).pow(2).mean().sqrt().item()
        for name, rms in [("polar", rms_sc), ("aurora", rms_au)]:
            assert abs(rms - RMS_TARGET) / RMS_TARGET < 0.05, f"{name} {m}x{n}: RMS {rms:.4f}"
        print(f"{m:>5}x{n:<5}  RMS  polar {rms_sc:.4f}  normuon {rms_pr:.4f}  aurora {rms_au:.4f}"
              f"  | normuon row-CV {cv:.4f} dead {dead:.0%}")
    print(f"muon_scaling self-check PASS (scale='adam' -> RMS {RMS_TARGET} for every variant)")


if __name__ == "__main__":
    _selfcheck()
