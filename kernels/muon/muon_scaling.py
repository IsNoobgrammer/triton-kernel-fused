"""Shared Muon update scalings (arch-independent) — used by kernels/sm75/muon.py and sm120/muon.py.

A "scaling" maps the orthogonalized update `O = NewtonSchulz(momentum)` (shape `(M, rows, cols)`,
M stacked same-shape matrices) to the tensor added to the weight, before the outer `-lr`. Two families:

SCALAR (stateless) — one per-matrix constant `s`, applied as `alpha = -lr*s` on the raw NS output:
  polarexpress / jordan : s = max(1, rows/cols)**0.5        aspect-ratio scale (Keller-Jordan / Polar-Express)
  moonlight             : s = 0.2*sqrt(max(rows, cols))     consistent-RMS, AdamW-band LR

PER-ROW (stateful, leverage-aware) — fixes the non-uniform row norms Muon produces on RECTANGULAR
matrices (the Tilde/Aurora "neuron death" finding: on a tall matrix the update's row norms follow the
leverage scores and can starve rows). Each output row (neuron) is normalized by an EMA second moment,
then a global Frobenius factor sets the overall scale. Faithful to the Tilde algorithm boxes:
  normuon           : Ohat=O/(sqrt(v)+eps), scale rows -> 0.2*sqrt(rows*cols)/||Ohat||_F  (unit RMS;
                      MIS-SCALES tall: magnitude flat in the aspect ratio)
  unormuon          : same Ohat, scale -> 0.2*min(rows,cols)/||Ohat||_F   (leverage-CORRECT target;
                      faithful to the Tilde box, BUT its magnitude ~ 0.2*sqrt(rows)*sqrt(cols/rows)
                      GROWS with model width sqrt(rows) -> not scale-invariant, and large)
  unormuon_spectral : same Ohat, scale -> k*sqrt(min(rows,cols))/||Ohat||_F. Same uniform leverage-
                      correct rows, SCALE-INVARIANT gain: per-row norm = k*sqrt(cols/rows), update
                      spectral norm ~= k = SPECTRAL_GAIN. Post-hoc row-rescale of ONE polar, so it
                      BREAKS orthogonality/spectrum (SV diff ~1e-2 vs Aurora).
      where  v_t = beta2*v_{t-1} + (1-beta2)*mean_cols(O*O)   (per-row EMA second moment)

AURORA family (an orthogonalization METHOD, not a post-scale: it re-runs the polar) — see aurora_update:
  aurora            : DEFAULT. Iterate {divide rows by (damped) row-norm, re-orthogonalize} K times,
                      then scale to spectral gain k. Because it PRESCALES the input (not the output),
                      the polar rebuilds orthogonality -> a leverage-balanced near-orthogonal factor
                      (this is exactly the prescale mechanism NorMuon/U-NorMuon skip). K = AURORA_K:
                      K=1 (one polar) matches the paper's K=2 for our task at HALF the cost; K=2 = the
                      paper's full method. Magnitude = RMS_TARGET*sqrt(max(rows,cols)) -> update RMS 0.2,
                      AdamW band (same convention as moonlight/normuon and DeepSeek-V4's Muon).

`aurora` (K=1) is the DEFAULT: Aurora-quality leverage correction at ONE polar solve, in the AdamW LR
band — moonlight, normuon and aurora all target update RMS 0.2 so AdamW LR/WD carry over with no retune.
The Muon-LR-band conventions remain by design: polarexpress/jordan (paper-faithful aspect scale; their
AdamW-band twin IS moonlight) and unormuon_spectral (spectral-gain k; its AdamW-band twin IS normuon).

Aurora's *iterative* variant (interleave rescale + re-orthogonalize, K polar solves) is NOT here — it
replaces the Newton-Schulz itself rather than post-scaling its output, so it belongs with newton_schulz.

Self-check: python -m kernels.muon.muon_scaling
"""
import torch

SCALAR_MODES = ("polarexpress", "jordan", "moonlight")
PERROW_MODES = ("normuon", "unormuon", "unormuon_spectral")
AURORA_MODES = ("aurora",)
ALL_MODES = SCALAR_MODES + PERROW_MODES + AURORA_MODES
DEFAULT_MODE = "aurora"

PERROW_BETA2 = 0.95
PERROW_EPS = 1e-8
# AdamW-band magnitude: update RMS ~= 0.2 (Moonlight convention, = DeepSeek-V4's sqrt(max(n,m))*gamma).
# For a (near-)orthogonal O, RMS(O) = 1/sqrt(max(rows,cols)), so RMS_TARGET*sqrt(max) pins RMS at 0.2
# and AdamW LR/WD carry over unchanged. moonlight, normuon and aurora (default) all use this band.
RMS_TARGET = 0.2
# Target spectral norm of the update (scale-invariant gain) for `unormuon_spectral` ONLY — the one
# remaining Muon-LR-band mode by design. ~k*sqrt(cols/rows) per-row at projection aspect ratios.
SPECTRAL_GAIN = 4.5
# aurora polar-solve iterations. K=1 (prescale + one polar) matches the paper's K=2 for our task at
# HALF the cost (measured: closer to K=2 than K=2's own K=3 iterate). K=2 = the paper's full method.
AURORA_K = 1
# aurora row-norm damping: D_k = D_{k-1}^beta * rownorm^(1-beta). 0 = full prescale (best at K=1,
# measured); the paper uses 0.5 at K=2.
AURORA_BETA = 0.0


def is_perrow(mode):
    return mode in PERROW_MODES


def validate(mode):
    if mode not in ALL_MODES:
        raise ValueError(f"unknown scale_mode {mode!r}; choose from {ALL_MODES}")
    return mode


def scalar_scale(mode, rows, cols):
    """The per-matrix constant `s` for a SCALAR mode (raises for per-row modes)."""
    if mode == "moonlight":
        return 0.2 * (max(rows, cols) ** 0.5)
    if mode in ("polarexpress", "jordan"):
        return max(1.0, rows / cols) ** 0.5
    raise ValueError(f"{mode!r} is not a scalar scale_mode (use apply_perrow)")


def perrow_state(M, rows, device):
    """EMA second-moment buffer for a per-row mode: one value per (stacked-matrix, row). Persisted in
    optimizer state so it round-trips through state_dict, exactly like the momentum buffer."""
    return torch.zeros((M, rows), device=device, dtype=torch.float32)


def is_aurora(mode):
    return mode in AURORA_MODES


def aurora_update(M, polar_fn, gain=None, K=AURORA_K, beta=AURORA_BETA, eps=PERROW_EPS):
    """Aurora: iterate {leverage-prescale rows, re-orthogonalize} K times, then scale to the AdamW band.

    M: (B, rows, cols) batch of momenta. polar_fn: the arch's orthogonalizer (returns U V^T, same shape).
    Each iteration divides rows by a (damped) row-norm and re-runs the polar, so the output is a
    leverage-BALANCED near-orthogonal factor (rows ~ sqrt(cols/rows)), unlike a post-hoc row rescale
    which breaks orthogonality. K=1 (one polar) matches the paper's K=2 for our task; K=2 = the paper.
    gain=None (default) -> RMS_TARGET*sqrt(max(rows,cols)): update RMS ~= 0.2, AdamW LR/WD reusable
    (same band as moonlight/normuon). Pass an explicit gain for a spectral-norm convention instead.
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


def apply_perrow(mode, O, v, beta2=PERROW_BETA2, eps=PERROW_EPS):
    """Faithful NorMuon / U-NorMuon on a batch of same-shape updates.

    O : (M, rows, cols) orthogonalized updates (M stacked matrices).
    v : (M, rows) fp32 EMA second-moment buffer, MUTATED IN PLACE.
    Returns the tensor T (same shape/dtype as O) to add as `weight -= lr * T` — i.e. lr is folded
    OUT (the caller applies `-lr`), the 0.2*... and Frobenius factors are folded IN. Frobenius norm
    is per-slice (each stacked matrix scaled by its own ||Ohat||), matching the single-matrix boxes.
    """
    rows, cols = O.shape[-2], O.shape[-1]
    Of = O.float()
    row_sq = Of.mul(Of).mean(dim=-1)                              # mean_cols(O*O) -> (M, rows)
    v.mul_(beta2).add_(row_sq, alpha=1.0 - beta2)                 # EMA in place
    Ohat = Of / v.sqrt().add(eps).unsqueeze(-1)                   # per-row RMS normalize -> (M, rows, cols)
    fro = Ohat.flatten(-2).norm(dim=-1).clamp_min(1e-12)         # per-slice Frobenius -> (M,)
    mn = float(min(rows, cols))
    if mode == "normuon":
        C = 0.2 * (rows * cols) ** 0.5                            # rows -> unit RMS (0.2*sqrt(m*n))
    elif mode == "unormuon":
        C = 0.2 * mn                                              # leverage-correct, moonlight-band (grows w/ sqrt(rows))
    else:                                                         # unormuon_spectral: scale-invariant gain k
        C = SPECTRAL_GAIN * mn ** 0.5                             # per-row norm -> k*sqrt(min/rows); spectral norm ~= k
    factor = C / fro                                             # (M,)  [lr folded out]
    return (Ohat * factor.view(-1, 1, 1)).to(O.dtype)


def _selfcheck():                                                # pragma: no cover
    torch.manual_seed(0)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for (m, n) in [(2048, 2048), (8192, 2048), (2048, 8192)]:
        # a skewed near-orthogonal batch (2 stacked) to exercise leverage
        O = torch.randn(2, m, n, device=dev)
        O = O / O.flatten(1).norm(dim=1).view(-1, 1, 1) * (min(m, n) ** 0.5)
        for mode in PERROW_MODES:
            v = perrow_state(2, m, dev)
            T = apply_perrow(mode, O, v)
            rn = T[0].norm(dim=-1)
            cv = (rn.std() / rn.mean()).item()
            dead = (rn < 0.1 * rn.mean()).float().mean().item()
            assert cv < 0.05, f"{mode} {m}x{n}: CV {cv:.3f} not uniform"
            assert dead == 0.0, f"{mode} {m}x{n}: {dead:.0%} dead rows"
            print(f"{mode:>17} {m:>5}x{n:<5}  row-norm CV {cv:.4f}  dead {dead:.0%}  mean {rn.mean():.3f}")
        s_ml = scalar_scale("moonlight", m, n)
        s_pe = scalar_scale("polarexpress", m, n)
        print(f"          scalar  moonlight {s_ml:.3f}  polarexpress {s_pe:.3f}")
        # aurora AdamW-band check: identity polar on an already-orthogonal input -> update RMS ~= 0.2
        Q = torch.linalg.qr(torch.randn(1, max(m, n), min(m, n), device=dev))[0][:, :m, :] \
            if m >= n else torch.linalg.qr(torch.randn(1, n, m, device=dev))[0].transpose(-2, -1)
        T = aurora_update(Q, lambda x: x, K=1)
        rms = T.pow(2).mean().sqrt().item()
        assert abs(rms - RMS_TARGET) / RMS_TARGET < 0.05, f"aurora {m}x{n}: RMS {rms:.4f} != {RMS_TARGET}"
        print(f"           aurora {m:>5}x{n:<5}  update RMS {rms:.4f} (AdamW band, target {RMS_TARGET})")
    print("muon_scaling self-check PASS")


if __name__ == "__main__":                                       # pragma: no cover
    _selfcheck()
