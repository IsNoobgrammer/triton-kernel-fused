"""ManasOptimizer — aurora-K1 Muon (NS-8) + rolling-probe gradient alignment (Nexus-adapted).

Base step = FusedMuon with the ns8 Newton-Schulz schedule (KJ x6 + pinned x2) and the fused
aurora K=1 leverage prescale — the exp_kappa 'ns8' arm. On top rides a ROLLING PROBE adapted
from Nexus (arXiv:2604.09258, "common minima via gradient similarity"):

    d_{t+1} = rho * d_t - (gamma / ||g||_global) * g_t          # short-memory NORMALIZED heavy ball
    g_t     = grad evaluated AT THE PROBE POINT theta_t + d_t   # via `with opt.probe(): fwd/bwd`

Mechanism (as MEASURED, autoresearch manas round — .autoresearch/manas/): the original
Nexus-mapping theory (probe assembles the pairwise gradient-alignment force via g(theta+d)
~= g + H*d cross-terms) is NOT what carries the win. Direct measurement (a7_mechanism.py)
shows the alignment force injects most cleanly at rho=0 — which is downstream-INERT — while
the winning config injects it barely at all; and the trajectory moves 20-160x farther than
||d|| over the memory window, so the same-point assumption behind that theory is dead on
arrival. What the ablations pin down instead: the gain needs (a) LONG memory (rho ~0.98,
~50 batches; rho=0 pure extragradient does nothing), (b) the DESCENT sign (+probe inert),
(c) the momentum-FREE normalized-gradient direction (probing along Muon's own momentum is
inert — Nesterov already covers that direction). Working model: trajectory-level lookahead
along a variance-reduced descent direction distinct from the update direction. Composition
with Muon still rests on the polar-only-rotates argument: the probe changes the gradient's
DIRECTION, never hands Muon magnitude structure to flatten (the paper's reported failure).

Validated (MNIST-1D heterogeneous-sources protocol, paired seeds, held-out gated):
gamma 8e-2 / rho 0.98 full-d gives OOD-acc-at-matched-train-loss +0.0245 (5 sigma) and
positive peak-OOD-acc vs base Muon, training slightly faster; dose optimum gamma 0.08-0.12,
rho saturates at 0.98. gamma is the scale-sensitive knob — re-sweep it per task/model scale.

Usage (training loop):
    opt = ManasOptimizer(params, lr=3e-4, probe_gamma=0.08, probe_rho=0.98)
    with opt.probe():          # forward/backward run at theta + d
        loss = model(x).loss
        loss.backward()
    opt.step(); opt.zero_grad()

Probe memory, two modes:
  probe_rank=None (full)   : one model-shaped fp32 buffer `manas_d` per 2D/3D param.
  probe_rank=r  (low-rank) : per matrix, d = Q @ C with Q (.., m, r) orthonormal and C (.., r, n)
      — r(m+n)/(m*n) of the full buffer (~3% at r=8, 512x1536). The basis is a randomized range
      of the current gradient, refreshed every `probe_refresh` probe updates (GaLore-style); on
      refresh the old d is re-projected into the new basis (no discontinuity). Probe apply
      materializes Q@C per param (transient), remove recomputes the same product (deterministic
      GEMM -> bit-identical subtract).

Safety, all sync-free (no .item()/host branch):
  * increments are GLOBALLY normalized (gamma/||g||) -> ||d|| <= gamma/(1-rho) by construction;
    no overflow regardless of gradient scale.
  * inv = gamma/||g|| is zeroed via torch.where when ||g|| is 0/inf/nan -> a bad step only
    DECAYS d (never poisons it); training recovers on the next finite step.
  * step() raises if called while the probe offset is applied (update would land on theta+d).
  * scope: these guards protect the PROBE state (d stays finite+bounded through inf/nan/1e8-scale
    grads and recovers on the next finite step). The BASE Muon step consuming a nonfinite gradient
    into the weights is — as with any optimizer — the grad scaler's job to skip (bench/train.py's
    fp16 GradScaler does); Manas does not second-guess it.
"""
from contextlib import contextmanager

import torch

from kernels.sm75.muon import FusedMuon

__all__ = ["ManasOptimizer", "NS8_COEFFS"]

_KJ = (3.4445, -4.7750, 2.0315)
_PIN = (2.0, -1.5, 0.5)
NS8_COEFFS = (_KJ,) * 6 + (_PIN,) * 2          # exp_kappa 'ns8': compressed KJ x6 + pinned polish x2


class ManasOptimizer(FusedMuon):
    def __init__(self, params, lr=3e-4, probe_gamma=0.08, probe_rho=0.98,
                 probe_rank=None, probe_refresh=200, coeffs=NS8_COEFFS,
                 scale_mode="aurora", aurora_k=1, **kw):
        super().__init__(params, lr=lr, coeffs=coeffs, scale_mode=scale_mode,
                         aurora_k=aurora_k, **kw)
        if not (0.0 <= probe_rho < 1.0):
            raise ValueError(f"probe_rho must be in [0, 1), got {probe_rho}")
        self.probe_gamma = float(probe_gamma)
        self.probe_rho = float(probe_rho)
        # probe_rank: None (full-d) | int (fixed rank) | float in (0,1) (fraction of min(m,n),
        # size-adaptive per matrix). The per-param resolved rank is clamped to [1, min(m,n)].
        self.probe_rank = None if probe_rank is None else probe_rank
        self.probe_refresh = int(probe_refresh)
        self._probe_on = False
        self._probe_updates = 0

    # ---------------- probe state ----------------
    def _probe_params(self):
        return [p for g in self.param_groups for p in g["params"] if p.ndim in (2, 3)]

    def _full_d(self, p):
        st = self.state[p]
        if "manas_d" not in st:
            st["manas_d"] = torch.zeros_like(p, dtype=torch.float32)
        return st["manas_d"]

    def _rank_for(self, m, n):
        r = self.probe_rank
        r = max(1, round(r * min(m, n))) if isinstance(r, float) else int(r)   # float => fraction
        return min(r, m, n)

    def _lowrank_qc(self, p):
        st = self.state[p]
        if "manas_q" not in st:
            m, n = p.shape[-2], p.shape[-1]
            r = self._rank_for(m, n)
            lead = p.shape[:-2]
            # deterministic orthonormal init (refreshed from real gradients later)
            g = torch.Generator(device="cpu").manual_seed(0x9A5 + p.numel())
            q = torch.linalg.qr(torch.randn(*lead, m, r, generator=g).to(p.device))[0]
            st["manas_q"] = q.to(torch.float32)
            st["manas_c"] = torch.zeros(*lead, r, n, device=p.device, dtype=torch.float32)
        return st["manas_q"], st["manas_c"]

    def _d_of(self, p):
        """Dense probe offset for one param (view for full mode, materialized for low-rank)."""
        if self.probe_rank is None:
            return self._full_d(p)
        q, c = self._lowrank_qc(p)
        return q @ c

    # ---------------- probe application ----------------
    @contextmanager
    def probe(self):
        """Run the enclosed forward/backward at theta + d. Nested use / step() inside raise."""
        self.apply_probe()
        try:
            yield
        finally:
            self.remove_probe()

    def _shift(self, sign):
        ps = self._probe_params()
        if self.probe_rank is None and all(p.dtype == torch.float32 for p in ps):
            # fused fast path (fp32 masters, full-d): ONE foreach over all params
            if sign > 0:
                torch._foreach_add_(ps, [self._full_d(p) for p in ps])
            else:
                torch._foreach_sub_(ps, [self._full_d(p) for p in ps])
        else:
            for p in ps:
                p.add_(self._d_of(p).to(p.dtype), alpha=sign)

    @torch.no_grad()
    def apply_probe(self):
        if self._probe_on:
            raise RuntimeError("probe already applied")
        self._shift(+1)
        self._probe_on = True

    @torch.no_grad()
    def remove_probe(self):
        if not self._probe_on:
            raise RuntimeError("probe not applied")
        self._shift(-1)                          # deterministic recompute -> exact inverse of apply
        self._probe_on = False

    # ---------------- step ----------------
    @torch.no_grad()
    def step(self, closure=None):
        if self._probe_on:
            raise RuntimeError("remove_probe() (or exit the probe() context) before step()")
        loss = super().step(closure)             # fused aurora-K1 Muon on the probe-point grads
        self._update_probe()
        return loss

    @torch.no_grad()
    def _update_probe(self):
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        # global L2 of the (2D/3D) gradient vector, fp32-accumulated; tiny (len(ps),) stack
        gn = torch.linalg.vector_norm(
            torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float32) for p in ps]))
        inv = self.probe_gamma / gn
        # sync-free guard: zero/inf/nan gradient norm -> inv = 0 -> this step only DECAYS d
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        # refresh fires on the FIRST update too (C is still zero -> re-projection lossless), so the
        # basis aligns with real gradients immediately instead of projecting through the random init
        refresh = (self.probe_rank is not None
                   and self._probe_updates % max(self.probe_refresh, 1) == 0)
        self._probe_updates += 1
        # nan_to_num: with a nonfinite gn, inv is already 0, but inf*0 = NaN elementwise — sanitize
        # the OPERAND so the (zero-scaled) increment is 0, not NaN. gn itself sees the raw grads
        # (an inf entry MUST zero inv). Only bites on bad steps; a fresh tensor either way.
        if self.probe_rank is None:
            ds = [self._full_d(p) for p in ps]
            torch._foreach_mul_(ds, self.probe_rho)                  # ONE fused decay sweep
            for p, d in zip(ps, ds):
                g32 = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
                d.addcmul_(g32, inv, value=-1.0)                     # d -= (gamma/||g||) * g
            return
        for p in ps:
            q, c = self._lowrank_qc(p)
            gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if refresh:
                # randomized range of the CURRENT gradient (GaLore-style basis refresh),
                # then re-project the old offset so d is continuous across the swap
                r = q.shape[-1]
                omega = torch.randn(*p.shape[:-2], p.shape[-1], r, device=p.device)
                q_new = torch.linalg.qr(gf @ omega)[0]
                c.copy_((q_new.mT @ q) @ c)
                q.copy_(q_new)
            c.mul_(self.probe_rho)
            c.addcmul_(q.mT @ gf, inv, value=-1.0)                   # project increment into basis
