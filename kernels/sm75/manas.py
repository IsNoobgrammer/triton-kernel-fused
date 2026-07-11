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

U BUFFER: DEPRECATED AND REMOVED (2026-07-11). The applied-update memory failed to separate
from no-u at BOTH scales it was hypothesized to help: toy (tie within noise) and the BiBo
137M comp {0.5, 1, 2} sweep (spread 0.0035, non-monotone = jitter), while costing ~1.2% tps
plus a full param clone per step and a dense per-param shift cache. Working explanation:
u ~= the post-polar momentum direction, and momentum-direction probing is a measured inert
control. comp= is accepted but IGNORED (DeprecationWarning); code lives in git <= e85af8b.

VOTE-WEIGHTING KNOBS: DEPRECATED AND REMOVED (2026-07-11, BiBo 137M knob bench). Both
measured against the equal-vote champion (rho.88/g.06/r8, -0.030 vs muon) and lost:
  rgd_tau  (KL-DRO loss-surprise votes, arXiv 2306.09222): tau {1, 3} within +0.004 of
           equal votes = no-op, as predicted — batch-MEAN loss spread is too small at 65k
           tokens and shrinks ~1/sqrt(batch tokens) beyond.
  cos_beta (cos(g,d) geometry votes): +0.5 (sharpen) neutral, -0.5 (novelty) LOSES HALF the
           manas gain — direct confirmation the mechanism is variance reduction (re-adding
           the noise the averaging removed is immediately billed). Equal vote is optimal.
Both args are accepted but IGNORED (DeprecationWarning); code lives in git <= db41f11.

Usage (training loop):
    opt = ManasOptimizer(params, lr=3e-4, probe_gamma=0.08, probe_rho=0.98)  # rank-8 d default
    opt = ManasOptimizer(params, ..., micro_vote=True)                       # per-micro-batch votes
    opt = ManasOptimizer(params, ..., micro_vote=True, nexus_gamma=0.03)     # + common-basin walker
    opt = ManasOptimizer(params, ..., micro_vote=True,                       # TWO-CLOCK decay:
                         probe_rho=1.0, probe_rho_step=0.9)                  # equal votes in-step,
                                                                             # memory in STEPS

    with opt.probe():          # forward/backward run at theta + d
        loss = model(x).loss
        loss.backward()
    opt.step(); opt.zero_grad()

Micro-vote loop (gradient accumulation; vote() is a no-op when micro_vote=False):
    for micro in range(accum):
        with opt.probe():
            (loss / accum).backward()
        opt.vote()
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
import warnings
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
                 rgd_tau=None, probe_norm="global", cos_beta=0.0,
                 micro_vote=False, nexus_gamma=0.0, probe_rho_step=None, **kw):
        super().__init__(params, lr=lr, coeffs=coeffs, scale_mode=scale_mode,
                         aurora_k=aurora_k, **kw)
        # TWO-CLOCK DECAY (probe_rho_step, micro_vote only): probe_rho becomes the WITHIN-step
        # per-vote decay (1.0 = all micro-batch votes weighted equally — micro order is arbitrary,
        # so recency inside a step is pure ordering noise at high accum) and probe_rho_step decays
        # the whole accumulated c once per optimizer step. A vote s steps ago, j votes back within
        # its step, weighs rho_step^s * rho^j — memory is fixed IN STEPS regardless of accum count.
        # Votes are additionally scaled by 1/(last step's vote count), so gamma is a PER-STEP
        # budget and reach = gamma/(1-rho_step) is batch-invariant (first step: k unknown, uses 1 —
        # one-step overshoot transient). None = the validated per-vote clock (rho decays per vote).
        self.probe_rho_step = None if probe_rho_step is None else float(probe_rho_step)
        if self.probe_rho_step is not None:
            if not micro_vote:
                raise ValueError("probe_rho_step requires micro_vote=True")
            if not (0.0 < self.probe_rho_step < 1.0):
                raise ValueError(f"probe_rho_step must be in (0, 1), got {probe_rho_step}")
        rho_hi = 1.0 if self.probe_rho_step is not None else 1.0 - 1e-12
        if not (0.0 <= probe_rho <= rho_hi):
            raise ValueError(f"probe_rho must be in [0, 1) (or [0, 1] with probe_rho_step), "
                             f"got {probe_rho}")
        self.probe_gamma = float(probe_gamma)
        self.probe_rho = float(probe_rho)
        # probe_rank: None (full-d) | int (fixed rank) | float in (0,1) (fraction of min(m,n),
        # size-adaptive per matrix). The per-param resolved rank is clamped to [1, min(m,n)].
        self.probe_rank = None if probe_rank is None else probe_rank
        self.probe_refresh = int(probe_refresh)
        # comp / U BUFFER: DEPRECATED 2026-07-11, ignored. The u buffer (rho-decayed memory of
        # APPLIED updates added to the probe offset) failed to separate from no-u TWICE: toy
        # (homogeneous-batch tie, +0.0235 vs +0.0266 within noise) AND the BiBo 137M comp sweep
        # {0.5, 1, 2} at rho.88/g.06 (spread 0.0035, NON-monotone = jitter) — while costing ~1.2%
        # tps plus a full param clone per step and a dense manas_shift cache in state. Suspected
        # mechanism: u ~= the momentum direction post-polar, and momentum-direction probing is a
        # measured inert control (Nesterov already covers it). Code removed; see git history
        # (<= e85af8b) to resurrect for a long-horizon heterogeneous test.
        if comp is not None:
            warnings.warn("ManasOptimizer(comp=...) is deprecated and IGNORED (u buffer removed: "
                          "no effect at toy or BiBo scale, ~1.2% tps + clone/shift memory cost; "
                          "see manas.py comment / git history <= e85af8b)", DeprecationWarning,
                          stacklevel=2)
        self.comp = None
        self._probe_on = False
        self._probe_updates = 0
        # WARMUP: skip probe-buffer (d) accumulation for the first probe_warmup_steps step()s so the
        # long-memory d isn't poisoned by early noisy gradients (d stays 0 -> probe is a no-op ->
        # pure Muon until warmup passes, then the lookahead engages). 0 = active from step 1.
        self.probe_warmup_steps = int(probe_warmup_steps)
        self._manas_step = 0
        # rgd_tau / cos_beta VOTE WEIGHTING: DEPRECATED 2026-07-11, ignored (see module docstring:
        # rgd_tau measured no-op at BiBo; cos_beta +0.5 neutral, -0.5 loses half the gain — the
        # equal vote IS the mechanism). Code lives in git <= db41f11.
        if rgd_tau:
            warnings.warn("ManasOptimizer(rgd_tau=...) is deprecated and IGNORED (measured no-op "
                          "at BiBo: batch-mean loss spread too small; git <= db41f11)",
                          DeprecationWarning, stacklevel=2)
        if cos_beta:
            warnings.warn("ManasOptimizer(cos_beta=...) is deprecated and IGNORED (BiBo: sharpen "
                          "neutral, novelty loses half the manas gain; git <= db41f11)",
                          DeprecationWarning, stacklevel=2)
        self.rgd_tau, self.cos_beta = None, 0.0
        # probe_norm: 'global' (validated default) = one gamma/||g||_all scalar — a matrix's share of
        # each vote follows its share of the GLOBAL gradient norm, which couples gamma to model
        # width/depth (suspected source of the 80x toy->BiBo gamma shift). 'perparam' = normalize each
        # matrix's vote by ITS OWN grad norm — every matrix gets an equal-size gamma vote per step
        # (muP-flavored; candidate fix for gamma transfer across scales). 3D expert stacks normalize
        # at the TENSOR level (whole stack), not per slice.
        if probe_norm not in ("global", "perparam"):
            raise ValueError(f"probe_norm must be 'global' or 'perparam', got {probe_norm!r}")
        self.probe_norm = probe_norm
        # MICRO-BATCH VOTING (micro_vote=True): with gradient accumulation, d gets one vote per
        # MICRO-batch (call opt.vote() after each backward, OUTSIDE the probe context) instead of
        # one per optimizer step from the accumulated mean. The votes are recovered as rank-space
        # DELTAS of the accumulating p.grad (prev-projection diff — no model-sized snapshot), so
        # this is low-rank-only. Point: the rho-batch law then binds to the MICRO batch size —
        # rho stays useful at any global batch (step-voting degenerates to extragradient once
        # effective batch ~ N_mem samples). Votes are normalized by their rank-space norm (the
        # in-subspace component; rank-1-sufficiency says that's where the signal lives).
        #
        # NEXUS WALKER (nexus_gamma > 0, requires micro_vote): a SECOND rank-space offset s,
        # zeroed each step, accumulating each micro's normalized direction UNdecayed; micro i
        # probes at theta + d + s_{i-1}. The accumulated grad sum then carries the Taylor
        # cross-terms g_j^T H g_i = the gradient of pairwise cosine similarity across micros
        # (Nexus, arXiv 2604.09258, common-minima objective) — WITHOUT the paper's inner model:
        # the probe shift IS the inner walk. Base Muon consumes mean-grad + alignment force.
        # Walker ceiling = nexus_gamma * (micros per step); keep it in the reach band (~0.2-0.5).
        self.micro_vote = bool(micro_vote)
        self.nexus_gamma = float(nexus_gamma)
        if self.micro_vote and self.probe_rank is None:
            raise ValueError("micro_vote=True requires a low-rank probe (probe_rank is None)")
        if self.nexus_gamma and not self.micro_vote:
            raise ValueError("nexus_gamma requires micro_vote=True")
        self._votes_cast = 0
        self._votes_last = 1        # last step's vote count (two-clock gamma normalization)

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
        """Dense probe offset for one param (view for full mode, materialized for low-rank).
        With the nexus walker, the offset is q @ (c + cs): consensus + intra-step alignment walk."""
        if self.probe_rank is None:
            return self._full_d(p)
        q, c = self._lowrank_qc(p)
        cs = self.state[p].get("manas_cs") if self.nexus_gamma else None
        return q @ (c + cs) if cs is not None else q @ c

    def _micro_state(self, p):
        """(prev_proj, cs) rank-space buffers for micro voting; zero-init, reset each step."""
        st = self.state[p]
        _q, c = self._lowrank_qc(p)
        if "manas_prev_proj" not in st:
            st["manas_prev_proj"] = torch.zeros_like(c)
        if self.nexus_gamma and "manas_cs" not in st:
            st["manas_cs"] = torch.zeros_like(c)
        return st["manas_prev_proj"], st.get("manas_cs")

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

    # ---------------- micro voting ----------------
    @torch.no_grad()
    def vote(self):
        """Cast one micro-batch vote (micro_vote mode; no-op otherwise, so loops can be
        mode-agnostic). Call AFTER each micro-batch's backward, OUTSIDE the probe context:

            with opt.probe():
                (loss / accum).backward()
            opt.vote()

        Reads each param's rank-space projection of the ACCUMULATING p.grad and diffs it
        against the previous call — the delta IS this micro-batch's gradient, in basis, with
        no model-sized snapshot. Votes d (rho-decayed consensus) and, if nexus_gamma, the
        undecayed intra-step walker cs. A missed vote() degrades gracefully (the next delta
        just spans two micros); voting inside the probe context raises (d would desync the
        exact probe remove)."""
        if not self.micro_vote:
            return
        if self._probe_on:
            raise RuntimeError("vote() must be called outside the probe() context")
        if self._manas_step < self.probe_warmup_steps:   # warmup: probe stays a no-op
            return
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        self._votes_cast += 1
        deltas, norms = [], []
        for p in ps:
            q, _c = self._lowrank_qc(p)
            prev, _cs = self._micro_state(p)
            gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            proj = q.mT @ gf                              # projection of the accumulated sum
            deltas.append(proj - prev)
            prev.copy_(proj)
            norms.append(torch.linalg.vector_norm(deltas[-1]))
        pn = torch.stack(norms)
        gn = pn if self.probe_norm == "perparam" else torch.linalg.vector_norm(pn)
        inv = self.probe_gamma / gn
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        if self.probe_rho_step is not None:
            inv = inv / max(self._votes_last, 1)   # gamma = per-STEP budget, split across votes
        nxs = (self.nexus_gamma / self.probe_gamma) if self.nexus_gamma else 0.0
        for i, p in enumerate(ps):
            _c = self.state[p]["manas_c"]
            iv = inv if inv.ndim == 0 else inv[i]
            _c.mul_(self.probe_rho).addcmul_(deltas[i], iv, value=-1.0)   # consensus vote
            if nxs:
                _prev, cs = self._micro_state(p)
                cs.addcmul_(deltas[i], iv * nxs, value=-1.0)              # walker: undecayed, in-step

    # ---------------- step ----------------
    @torch.no_grad()
    def step(self, closure=None, probe_loss=None):
        # probe_loss: accepted for back-compat with the removed rgd_tau wiring; unused.
        if self._probe_on:
            raise RuntimeError("remove_probe() (or exit the probe() context) before step()")
        self._manas_step += 1                    # drives the probe warmup gate (see _update_probe)
        loss = super().step(closure)             # fused aurora-K1 Muon on the probe-point grads
        if self.micro_vote:
            self._finish_micro_step()
        else:
            self._update_probe()
        return loss

    @torch.no_grad()
    def _finish_micro_step(self):
        """Step-boundary bookkeeping for micro_vote mode: basis refresh (needs the FULL
        accumulated grad, so it can only fire here — a mid-step refresh would desync prev_proj
        and the walker), then reset the per-step buffers. Zero votes cast -> fall back to the
        step-vote so the probe never silently dies (warned once)."""
        if self._votes_cast == 0:
            if not getattr(self, "_warned_no_votes", False):
                warnings.warn("micro_vote=True but no vote() was cast this step; falling back "
                              "to step-voting (call opt.vote() after each micro-batch backward)")
                self._warned_no_votes = True
            self._update_probe()
        else:
            refresh = self._probe_updates % max(self.probe_refresh, 1) == 0
            self._probe_updates += 1
            for p in self._probe_params():
                if refresh and p.grad is not None:
                    q, c = self._lowrank_qc(p)
                    gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
                    r = q.shape[-1]
                    omega = torch.randn(*p.shape[:-2], p.shape[-1], r, device=p.device)
                    q_new = torch.linalg.qr(gf @ omega)[0]
                    c.copy_((q_new.mT @ q) @ c)
                    q.copy_(q_new)
                st = self.state[p]
                if "manas_prev_proj" in st:
                    st["manas_prev_proj"].zero_()     # p.grad restarts from zero next step
                if "manas_cs" in st:
                    st["manas_cs"].zero_()            # the walker never crosses a step boundary
            self._votes_last = self._votes_cast       # gamma split for next step's votes
        if self.probe_rho_step is not None:           # step clock: whole block ages one step
            for p in self._probe_params():
                st = self.state[p]
                if "manas_c" in st:
                    st["manas_c"].mul_(self.probe_rho_step)
        self._votes_cast = 0

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
    # Self-check (probe math only; no NS/CUDA needed).
    torch.manual_seed(0)
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
    # 7) deprecated knobs (comp / rgd_tau / cos_beta): each warns and is ignored
    for kwargs, attr, off in ((dict(comp=1.0), "comp", None),
                              (dict(rgd_tau=3.0), "rgd_tau", None),
                              (dict(cos_beta=0.5), "cos_beta", 0.0)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o = ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], **kwargs)
        assert getattr(o, attr) == off and any(issubclass(w.category, DeprecationWarning) for w in caught), \
            f"{kwargs} must warn DeprecationWarning and be ignored"
    o.step(probe_loss=2.0)   # legacy rgd wiring stays callable (no grads -> no work)
    # 9) micro voting: deltas of the accumulating grad, walker accumulates in-step, resets at boundary
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, nexus_gamma=0.04)
    p.grad = torch.randn(32, 16); o.vote()
    p.grad = p.grad + torch.randn(32, 16); o.vote()            # accumulation continues
    assert o._votes_cast == 2
    q, c = o._lowrank_qc(p); cs = o.state[p]["manas_cs"]
    assert c.norm() > 0 and cs.norm() > 0, "both consensus and walker must receive votes"
    assert torch.allclose(o._d_of(p), q @ (c + cs)), "offset must include the walker"
    c_before = c.clone()
    o.vote()                                                    # grad unchanged -> delta 0 -> decay only
    assert torch.allclose(c, 0.98 * c_before, atol=1e-7), "zero delta must only decay the consensus"
    o._finish_micro_step()
    assert cs.norm() == 0 and o.state[p]["manas_prev_proj"].norm() == 0 and o._votes_cast == 0, \
        "walker and prev-projection must reset at the step boundary"
    try:
        o.apply_probe(); o.vote(); raise AssertionError("vote() inside probe must raise")
    except RuntimeError:
        o.remove_probe()
    # 10) two-clock decay: equal votes within a step (rho=1), boundary decay rho_step,
    #     gamma split by last step's vote count
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0, probe_rho_step=0.9)
    p.grad = torch.randn(32, 16); o.vote()
    p.grad = p.grad + torch.randn(32, 16); o.vote()
    c = o.state[p]["manas_c"]; c_end = c.clone()
    o._probe_updates = 1                                        # dodge the boundary basis refresh
    o._finish_micro_step()
    assert torch.allclose(c, 0.9 * c_end, atol=1e-7), "boundary must decay the whole block by rho_step"
    assert o._votes_last == 2, "vote count must carry to the next step's gamma split"
    p.grad = torch.randn(32, 16); o.vote()                      # next step: gamma/2 per vote
    inc = (c - 0.9 * c_end).norm().item()
    assert abs(inc - o.probe_gamma / 2) < 1e-5, f"vote must carry gamma/k_last, got {inc}"
    for bad in (dict(probe_rho=1.0),                            # rho=1 needs the step clock
                dict(probe_rho_step=0.9),                       # step clock needs micro_vote
                dict(probe_rank=4, micro_vote=True, probe_rho_step=1.0),
                dict(probe_rank=None, micro_vote=True), dict(nexus_gamma=0.1)):
        try:
            ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], **bad)
            raise AssertionError(f"constructor must reject {bad}")
        except ValueError:
            pass
    print("manas perparam+microvote+nexus self-check PASS")
