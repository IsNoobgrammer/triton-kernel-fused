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
  unormuon_aurora   : DEFAULT. `leverage_prescale` the momentum (divide each row by its L2 norm)
                      BEFORE Newton-Schulz, then the SAME spectral post-scale. The single polar then
                      orthogonalizes an already leverage-balanced matrix -> a genuinely near-orthogonal,
                      row-uniform factor = Aurora's K=2 output, at ONE polar (Aurora's iterative = K
                      polars). Measured: closer to Aurora(K=2) than Aurora's own K=3 iterate on tall
                      skewed inputs; SV spectrum preserved (diff ~1e-4). Same magnitude as
                      unormuon_spectral (k), so no LR change vs it. Prescale is a no-op for rows<=cols.
      where  v_t = beta2*v_{t-1} + (1-beta2)*mean_cols(O*O)   (per-row EMA second moment)

`unormuon_aurora` is the DEFAULT: Aurora-quality leverage correction at one polar solve + one row
reduction + one divide. Switching the default from the old `polarexpress` changes the effective LR band,
so retune LR (k=SPECTRAL_GAIN is the one knob — it IS the target spectral norm of the update).

Aurora's *iterative* variant (interleave rescale + re-orthogonalize, K polar solves) is NOT here — it
replaces the Newton-Schulz itself rather than post-scaling its output, so it belongs with newton_schulz.

Self-check: python -m kernels.muon.muon_scaling
"""
import torch

SCALAR_MODES = ("polarexpress", "jordan", "moonlight")
PERROW_MODES = ("normuon", "unormuon", "unormuon_spectral", "unormuon_aurora")
ALL_MODES = SCALAR_MODES + PERROW_MODES
DEFAULT_MODE = "unormuon_aurora"

PERROW_BETA2 = 0.95
PERROW_EPS = 1e-8
# Target spectral norm of the update for `unormuon_spectral` (scale-invariant gain). ~k*sqrt(cols/rows)
# per-row -> ~2-3 at projection aspect ratios (rows/cols ~ 2.5-4). The one LR-band knob; tune per run.
SPECTRAL_GAIN = 4.5


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


def prescale_needed(mode, rows, cols):
    """Whether to leverage-prescale the momentum before Newton-Schulz (aurora mode, tall only)."""
    return mode == "unormuon_aurora" and rows > cols


def leverage_prescale(u, eps=PERROW_EPS):
    """Aurora(K=1, beta=0): divide each row of the momentum by its L2 norm BEFORE Newton-Schulz, so the
    single polar solve produces a leverage-balanced near-orthogonal factor (matches Aurora's K=2 output
    within its own fp16 convergence noise) at ZERO extra polar cost. u: (M, rows, cols). fp32 reduction
    (fp16 sum-of-squares overflows). Caller gates on prescale_needed (tall matrices only)."""
    r = u.float().norm(dim=-1, keepdim=True).clamp_min(eps)
    return (u.float() / r).to(u.dtype)


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
    print("muon_scaling self-check PASS")


if __name__ == "__main__":                                       # pragma: no cover
    _selfcheck()
