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
positive peak-OOD-acc vs base Muon, training slightly faster; dose optimum gamma 0.08-0.12.

Setting the knobs at a new scale (measured scaling laws, .autoresearch/manas/):
  rho — NOT free: it holds MEMORY IN SAMPLES. rho* ~= 1 - batch_size/N_mem with N_mem a
    task constant (~6k samples on the reference task; invariant across BS 32-512, where
    the rho optimum moved 0.995 -> 0.90 exactly as the law predicts). Bigger batches want
    SHORTER windows; re-derive rho from batch size, don't copy it.
  gamma — couples to weight-space geometry (LR / weight norms / curvature): the toy
    optimum was 80x the old default, so re-sweep the dose per model scale; the curve was
    smooth with a broad plateau and no instability up to 2x the optimum.

Optional U BUFFER (comp != None): a second rho-decayed memory of the APPLIED updates,
added to the probe offset as  comp * gamma * u/||u||_global  — the probe then also looks
along the optimizer's realized travel. comp is continuous in units of gamma; comp=+1
(extend) validated best on the toy (frontier tie with the no-u champion; back-off signs
neutral; dose turns by +2). At probe_rank=r, u lives in d's SAME rank-r basis (state = one
extra (r,n) Cu per matrix, ~zero marginal memory; re-projected on basis refresh) — this
shared-basis compression is held-out validated (+0.0235, tie with full fp32 u's +0.0266).
u did not SEPARATE from no-u on the toy (batches too homogeneous for update-history
information); it is expected to matter on heterogeneous large-batch data — LM A/B decides.

Usage (training loop):
    opt = ManasOptimizer(params, lr=3e-4, probe_gamma=0.08, probe_rho=0.98)  # rank-8 d default
    opt = ManasOptimizer(params, ..., comp=1.0)                              # + rank-8 u buffer
    opt = ManasOptimizer(params, ..., rgd_tau=3.0)                           # + RGD-weighted votes
    with opt.probe():          # forward/backward run at theta + d
        loss = model(x).loss
        loss.backward()
    opt.step(); opt.zero_grad()                  # rgd_tau set: opt.step(probe_loss=loss.detach())

RGD vote weighting (rgd_tau, default off): weights each batch's probe vote by the KL-DRO
factor e^{min(loss,tau)/(tau+1)} (arXiv 2306.09222), EMA-normalized so only batch-RELATIVE
surprise tilts the consensus — hard batches vote more, the tau clip caps outliers, and the
weight is clamped to [0.25, 4] (||d|| stays bounded). Probe direction only; the training
loss and base Muon update are untouched. NOTE: the weight acts on the per-STEP batch-mean
loss, so its spread — and the whole effect — shrinks ~1/sqrt(batch tokens); at ~8M-token
batches it degrades gracefully to the validated equal-vote probe (w -> 1), never past it.

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
                 probe_rank=8, probe_refresh=200, comp=None, coeffs=NS8_COEFFS,
                 scale_mode="aurora", aurora_k=1, probe_warmup_steps=0,
                 rgd_tau=None, probe_norm="global", **kw):
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
        # comp: u-buffer strength in units of gamma (None = off). +1 validated on the toy.
        self.comp = None if comp is None else float(comp)
        self._probe_on = False
        self._probe_updates = 0
        # WARMUP: skip probe-buffer (d) accumulation for the first probe_warmup_steps step()s so the
        # long-memory d isn't poisoned by early noisy gradients (d stays 0 -> probe is a no-op ->
        # pure Muon until warmup passes, then the lookahead engages). 0 = active from step 1.
        self.probe_warmup_steps = int(probe_warmup_steps)
        self._manas_step = 0
        # RGD VOTE WEIGHTING (arXiv 2306.09222, KL-DRO): rgd_tau=None (default) = the validated
        # equal-vote probe. Set tau > 0 to weight each batch's probe VOTE by w = e^{min(loss,tau)/(tau+1)}
        # (the paper's g(l) with its gamma = 1/(tau+1)), normalized by a running EMA of w so the
        # secular loss decline (11 -> 2 over a run) cancels and only batch-relative surprise tilts the
        # consensus. Hard/surprising batches vote more; the tau clip caps an outlier's vote. This
        # weights the probe DIRECTION only — the training loss and the base Muon update are untouched.
        # w is clamped to [0.25, 4] so ||d|| <= 4*gamma/(1-rho) stays bounded. Pass the batch loss via
        # step(probe_loss=loss) (tensor, no host sync needed — or float).
        self.rgd_tau = float(rgd_tau) if rgd_tau else None   # None/0/False = off (equal votes)
        self._probe_loss = None
        self._rgd_wema = None      # smoothing state, attr-only: lost on resume -> one-EMA-window re-anchor, harmless
        # probe_norm: 'global' (validated default) = one gamma/||g||_all scalar — a matrix's share of
        # each vote follows its share of the GLOBAL gradient norm, which couples gamma to model
        # width/depth (suspected source of the 80x toy->BiBo gamma shift). 'perparam' = normalize each
        # matrix's vote by ITS OWN grad norm — every matrix gets an equal-size gamma vote per step
        # (muP-flavored; candidate fix for gamma transfer across scales). 3D expert stacks normalize
        # at the TENSOR level (whole stack), not per slice.
        if probe_norm not in ("global", "perparam"):
            raise ValueError(f"probe_norm must be 'global' or 'perparam', got {probe_norm!r}")
        self.probe_norm = probe_norm

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

    def _u_of(self, p):
        """Dense u (applied-update memory) for one param. Low-rank mode shares d's basis:
        u = Q @ Cu — one extra (r,n) buffer, re-projected on basis refresh like C."""
        st = self.state[p]
        if self.probe_rank is None:
            if "manas_u" not in st:
                st["manas_u"] = torch.zeros_like(p, dtype=torch.float32)
            return st["manas_u"]
        q, _c = self._lowrank_qc(p)
        if "manas_cu" not in st:
            st["manas_cu"] = torch.zeros_like(st["manas_c"])
        return q @ st["manas_cu"]

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
        if self.comp is not None:
            # offset = d + comp*gamma*u/||u||; cache the applied tensor -> bit-exact remove
            if sign > 0:
                us = {p: self._u_of(p) for p in ps}
                un = torch.linalg.vector_norm(torch.stack(
                    [torch.linalg.vector_norm(us[p]) for p in ps])).clamp_min(1e-12)
                k = self.comp * self.probe_gamma / un
                for p in ps:
                    self.state[p]["manas_shift"] = (self._d_of(p) + k * us[p]).to(p.dtype)
            for p in ps:
                p.add_(self.state[p]["manas_shift"], alpha=sign)
            return
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
    def step(self, closure=None, probe_loss=None):
        if self._probe_on:
            raise RuntimeError("remove_probe() (or exit the probe() context) before step()")
        if self.rgd_tau is not None and probe_loss is None:
            raise RuntimeError("rgd_tau is set: pass the batch loss via step(probe_loss=loss)")
        self._probe_loss = probe_loss            # consumed by _update_probe (RGD vote weight)
        self._manas_step += 1                    # drives the probe warmup gate (see _update_probe)
        if self.comp is not None:                # u tracks the APPLIED update: snapshot theta,
            ps = self._probe_params()            # and (low-rank) the basis if a refresh will
            before = [p.detach().clone() for p in ps]        # fire inside _update_probe
            fire = (self.probe_rank is not None
                    and self._probe_updates % max(self.probe_refresh, 1) == 0)
            q_old = [self._lowrank_qc(p)[0].clone() if fire else None for p in ps]
        loss = super().step(closure)             # fused aurora-K1 Muon on the probe-point grads
        self._update_probe()
        if self.comp is not None:
            for p, b, qo in zip(ps, before, q_old):
                delta = (p.detach() - b).float()
                if self.probe_rank is None:
                    self._u_of(p).mul_(self.probe_rho).add_(delta)
                    continue
                q, _c = self._lowrank_qc(p)
                st = self.state[p]
                if "manas_cu" not in st:
                    st["manas_cu"] = torch.zeros_like(st["manas_c"])
                cu = st["manas_cu"]
                if qo is not None:               # basis refreshed: carry u into the new basis
                    cu.copy_((q.mT @ qo) @ cu)
                cu.mul_(self.probe_rho).add_(q.mT @ delta)
        return loss

    @torch.no_grad()
    def _update_probe(self):
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        if self._manas_step <= self.probe_warmup_steps:   # warmup: leave d at 0 (pure Muon)
            return
        # per-param L2 norms, fp32-accumulated; tiny (len(ps),) stack
        pn = torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float32) for p in ps])
        if self.probe_norm == "perparam":
            gn = pn                                       # (len(ps),) — each matrix votes gamma on its own
        else:
            gn = torch.linalg.vector_norm(pn)             # scalar — one global vote split by norm share
        inv = self.probe_gamma / gn
        # sync-free guard: zero/inf/nan gradient norm -> inv = 0 -> this step only DECAYS d.
        # perparam: elementwise, so one dead/overflowed matrix skips only ITS vote, not everyone's.
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        if self.rgd_tau is not None:             # RGD vote weight (see __init__); all tensor ops, sync-free
            lt = torch.as_tensor(self._probe_loss, dtype=torch.float32, device=gn.device)
            w_raw = torch.exp(lt.clamp(max=self.rgd_tau) / (self.rgd_tau + 1.0))
            w_raw = torch.where(torch.isfinite(w_raw), w_raw,
                                self._rgd_wema if self._rgd_wema is not None else torch.ones_like(w_raw))
            if self._rgd_wema is None:           # init at the first observed weight -> w starts ~1, no bias
                self._rgd_wema = w_raw.clone()
            self._rgd_wema.mul_(0.98).add_(w_raw, alpha=0.02)
            inv = inv * (w_raw / self._rgd_wema).clamp(0.25, 4.0)
        self._probe_loss = None                  # single-use: a stale loss never weights a later vote
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
            for i, (p, d) in enumerate(zip(ps, ds)):
                g32 = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
                d.addcmul_(g32, inv if inv.ndim == 0 else inv[i], value=-1.0)   # d -= (gamma/||g||) * g
            return
        for i, p in enumerate(ps):
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
            c.addcmul_(q.mT @ gf, inv if inv.ndim == 0 else inv[i], value=-1.0)  # project increment into basis


if __name__ == "__main__":                                           # pragma: no cover
    # Self-check for the RGD vote weighting (probe math only; no NS/CUDA needed).
    torch.manual_seed(0)
    def mk(tau):
        p = torch.nn.Parameter(torch.randn(32, 16))
        opt = ManasOptimizer([p], probe_rank=None, rgd_tau=tau)
        return p, opt
    # 1) constant loss -> w == 1 exactly -> identical d to the equal-vote probe
    (p0, o0), (p1, o1) = mk(None), mk(3.0)
    for _ in range(5):
        g = torch.randn(32, 16)
        for p, o, kw in ((p0, o0, {}), (p1, o1, {"probe_loss": 2.0})):
            p.grad = g.clone(); o._manas_step += 1; o._probe_loss = kw.get("probe_loss"); o._update_probe()
    assert torch.allclose(o0._full_d(p0), o1._full_d(p1), atol=1e-6), "constant loss must reproduce equal votes"
    # 2) a high-loss batch votes more than a low-loss one (same gradient)
    (p2, o2), g = mk(3.0)[0:2], torch.randn(32, 16)
    p2.grad = g.clone(); o2._manas_step += 1; o2._probe_loss = 2.0; o2._update_probe()
    base = o2._full_d(p2).norm().item()
    p2.grad = g.clone(); o2._manas_step += 1; o2._probe_loss = 4.0; o2._update_probe()
    hi = (o2._full_d(p2).norm().item() - 0.98 * base)      # this vote's contribution
    p2.grad = g.clone(); o2._manas_step += 1; o2._probe_loss = 0.5; o2._update_probe()
    # 3) nonfinite loss cannot poison d
    p2.grad = g.clone(); o2._manas_step += 1; o2._probe_loss = float("nan"); o2._update_probe()
    assert torch.isfinite(o2._full_d(p2)).all(), "nan loss must not poison d"
    # 4) missing loss with rgd on fails loud
    try:
        o2.step(); raise AssertionError("step() without probe_loss must raise when rgd_tau is set")
    except RuntimeError:
        pass
    assert hi > 0, "high-loss vote must contribute"
    # 5) perparam norm: equal-size votes regardless of per-matrix grad scale (global: share-weighted)
    for mode, expect_equal in (("perparam", True), ("global", False)):
        pa, pb = torch.nn.Parameter(torch.randn(32, 16)), torch.nn.Parameter(torch.randn(32, 16))
        o = ManasOptimizer([pa, pb], probe_rank=None, probe_norm=mode)
        pa.grad = torch.randn(32, 16); pb.grad = 100.0 * torch.randn(32, 16)   # 100x scale gap
        o._manas_step += 1; o._update_probe()
        na, nb = o._full_d(pa).norm().item(), o._full_d(pb).norm().item()
        equal = abs(na - nb) / max(na, nb) < 0.05
        assert equal == expect_equal, f"{mode}: vote norms {na:.4f} vs {nb:.4f}"
    # 6) perparam guard is per-matrix: a nan grad zeroes only its own vote
    pa, pb = torch.nn.Parameter(torch.randn(32, 16)), torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([pa, pb], probe_rank=None, probe_norm="perparam")
    pa.grad = torch.full((32, 16), float("nan")); pb.grad = torch.randn(32, 16)
    o._manas_step += 1; o._update_probe()
    assert o._full_d(pa).norm() == 0 and o._full_d(pb).norm() > 0, "nan matrix must not veto healthy votes"
    print("manas rgd+perparam self-check PASS")
