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
    opt = ManasOptimizer(params, ..., micro_vote=True, probe_rho=1.0,        # + FRESH-BLOCK boost:
                         probe_rho_step=0.9, probe_gamma=0.02,               # this step's votes at
                         probe_gamma_intra=0.06)                             # gamma_intra, folded to
                                                                             # history at gamma/vote

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
      of the current gradient, refreshed every `probe_refresh` probe updates (GaLore-style;
      default AUTO = 2/(1-rho) from the active clock — window as fresh as its oldest vote); on
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
                 probe_rank=8, probe_refresh=None, comp=None, coeffs=NS8_COEFFS,
                 scale_mode="aurora", aurora_k=1, probe_warmup_steps=0,
                 rgd_tau=None, probe_norm="global", cos_beta=0.0,
                 micro_vote=False, nexus_gamma=0.0, probe_rho_step=None,
                 probe_gamma_intra=None, probe_sketch_rho=0.90, probe_sketch_votes=None,
                 probe_sketch_min_votes=None, **kw):
        super().__init__(params, lr=lr, coeffs=coeffs, scale_mode=scale_mode,
                         aurora_k=aurora_k, **kw)
        # TWO-CLOCK / FRESH-BLOCK MODE (probe_rho_step, micro_vote only): probe_rho_step is the
        # ONLY decay — memory in STEPS, invariant to accum count. ALL stored state is RAW unit
        # votes: within a step they accumulate undecayed (probe_rho pinned 1.0 in spirit — micro
        # order is arbitrary, in-step recency is ordering noise) into the FRESH BLOCK manas_cnow;
        # at the boundary the block folds into history at COEFFICIENT 1 (history = rho_step *
        # history + block) — only rho_step ever touches storage. Both doses are applied at PROBE
        # time: d = q @ (gamma*history + gamma_intra*block). Nothing is baked in, so either dose
        # can change mid-run and rescales its buffer uniformly/retroactively; gamma_intra=0 is
        # legal (pure-history probing). gamma_intra defaults to gamma.
        # GAMMA IS PER-VOTE here (no per-step split): measured invariant across ga — the b32
        # two-rho grid winners were 0.02/vote at ga1 AND 0.1/step = 0.025/vote at ga4.
        # None = the validated per-vote rho clock (rho decays per vote, gamma per vote).
        self.probe_rho_step = None if probe_rho_step is None else float(probe_rho_step)
        if self.probe_rho_step is not None:
            if not micro_vote:
                raise ValueError("probe_rho_step requires micro_vote=True")
            if not (0.0 < self.probe_rho_step < 1.0):
                raise ValueError(f"probe_rho_step must be in (0, 1), got {probe_rho_step}")
        if probe_gamma_intra is not None and self.probe_rho_step is None:
            raise ValueError("probe_gamma_intra requires probe_rho_step (fresh-block mode)")
        # gamma_intra: probe-time dose of the CURRENT step's raw block (see mode comment above);
        # fully independent of gamma, 0 legal (pure-history probing: fresh votes enter the
        # consensus at the boundary but never steer a live probe). UNSET -> TRACKS gamma live
        # (schedule gamma and the fresh dose follows); set explicitly -> independent dial.
        self._probe_gamma_intra = None if probe_gamma_intra is None else float(probe_gamma_intra)
        if self._probe_gamma_intra is not None and self._probe_gamma_intra < 0:
            raise ValueError(f"probe_gamma_intra must be >= 0, got {probe_gamma_intra}")
        rho_hi = 1.0 if self.probe_rho_step is not None else 1.0 - 1e-12
        if not (0.0 <= probe_rho <= rho_hi):
            raise ValueError(f"probe_rho must be in [0, 1) (or [0, 1] with probe_rho_step), "
                             f"got {probe_rho}")
        self.probe_gamma = float(probe_gamma)
        self.probe_rho = float(probe_rho)
        # probe_rank: None (full-d) | int (fixed rank) | float in (0,1) (fraction of min(m,n),
        # size-adaptive per matrix). The per-param resolved rank is clamped to [1, min(m,n)].
        self.probe_rank = None if probe_rank is None else probe_rank
        # probe_refresh: basis rebuild cadence in steps. UNSET -> AUTO from the memory horizon:
        # 2/(1-rho) with the active rho clock (step clock if set, else the per-vote clock) —
        # the window should be as fresh as the oldest vote it hosts (rho 0.96 -> 50, 0.98 -> 100).
        # The old fixed default was 200 = 8 memory-lifetimes stale. c=2 pending the refresh sweep.
        self._probe_refresh = None if probe_refresh is None else int(probe_refresh)
        # THE WINDOW, one sentence: Q is the QR of an EMA of unit vote directions.
        # Per vote, the micro-gradient's ambient sketch (p.grad@omega differenced — the same
        # telescoping trick vote() uses) is UNIT-NORMALIZED into a raw per-step pad Y_now; at
        # the boundary Y = rho_q*Y + Y_now and the window is developed on the spot: Q = QR(Y),
        # EVERY boundary — "refresh" is not a concept here; the window IS the current evidence
        # (consecutive Y's overlap heavily, so the re-projection crop is slivers — measured free
        # at 5x cadence). The normalization is the point: raw deltas telescope back to the
        # boundary SUM (spread destroyed); unit increments give each micro DIRECTION equal
        # voice, so Y covers the vote distribution, not its collapsed mean (union-vs-mean
        # measured 3x at k=2). rho_q = the aim's memory in boundaries (vote-clock hypothesis:
        # rho_q = 1 - k/20). GA1 GATE (hard, measured): a step with < 2 votes has no spread to
        # cover (one vote/step = the info cap; sketch lag is pure cost there) — its boundary
        # falls back to cadence-based snapshot refresh. None = legacy snapshot cadence only.
        # Full-rank probes have no window; micro_vote=False casts no votes, so the sketch is
        # inert there by design (snapshot cadence applies).
        self.probe_sketch_rho = None if probe_sketch_rho is None else float(probe_sketch_rho)
        if self.probe_sketch_rho is not None and probe_rank is None:
            self.probe_sketch_rho = None                  # full-d: silently no window/no sketch
        if self.probe_sketch_rho is not None:
            if not (0.0 < self.probe_sketch_rho < 1.0):
                raise ValueError(f"probe_sketch_rho must be in (0, 1), got {probe_sketch_rho}")
        # 2026-07-13 mode-zoo collapse: boundary-sum sketch DELETED (dominated 3x by the vote
        # union at its first head-to-head); ga1 gate hardcoded at 2 (min_votes=1 arm measured
        # null — info cap confirmed). Both args accepted and IGNORED for harness back-compat.
        if probe_sketch_votes is not None or probe_sketch_min_votes is not None:
            warnings.warn("probe_sketch_votes / probe_sketch_min_votes are deprecated and "
                          "IGNORED: the vote-union sketch is THE sketch (boundary-sum deleted), "
                          "gate fixed at 2 votes (measured)", DeprecationWarning, stacklevel=2)
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

    @property
    def probe_refresh(self):
        """Basis rebuild cadence (steps). Unset -> 2/(1-rho) from the active clock, live."""
        if self._probe_refresh is not None:
            return self._probe_refresh
        rho = self.probe_rho_step if self.probe_rho_step is not None else self.probe_rho
        return max(2, round(2.0 / max(1.0 - rho, 1e-6)))

    @probe_refresh.setter
    def probe_refresh(self, v):
        self._probe_refresh = None if v is None else int(v)

    @property
    def probe_gamma_intra(self):
        """Fresh-block probe dose. Unset -> tracks probe_gamma live; explicit -> independent."""
        return self.probe_gamma if self._probe_gamma_intra is None else self._probe_gamma_intra

    @probe_gamma_intra.setter
    def probe_gamma_intra(self, v):
        self._probe_gamma_intra = None if v is None else float(v)

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

    def _cnow(self, p):
        """Fresh-block rank-space buffer (this step's votes, RAW unit weights); zeroed at fold."""
        st = self.state[p]
        if "manas_cnow" not in st:
            _q, c = self._lowrank_qc(p)
            st["manas_cnow"] = torch.zeros_like(c)
        return st["manas_cnow"]

    def _d_of(self, p):
        """Dense probe offset for one param (view for full mode, materialized for low-rank).
        Low-rank: q @ (history + fresh block if two-clock + walker cs if nexus)."""
        if self.probe_rank is None:
            return self._full_d(p)
        q, c = self._lowrank_qc(p)
        if self.probe_rho_step is not None:
            # fresh-block mode: ALL state is raw (unit votes; only rho_step ever touches
            # history). Both doses are applied HERE, at probe time: d = gamma*history +
            # gamma_intra*block. Nothing is baked into storage, so either dose can change
            # mid-run and rescales its buffer uniformly and retroactively.
            total = self.probe_gamma * c
            if "manas_cnow" in self.state[p]:
                total = total + self.probe_gamma_intra * self.state[p]["manas_cnow"]
            cs = self.state[p].get("manas_cs") if self.nexus_gamma else None
            return q @ (total + cs) if cs is not None else q @ total
        total = c
        cs = self.state[p].get("manas_cs") if self.nexus_gamma else None
        if cs is not None:
            total = total + cs
        return q @ total

    def _sketch_state(self, p):
        """omega (fixed, seeded per-param) + Y + the per-step pad Y_now / prev_yp."""
        st = self.state[p]
        if "manas_omega" not in st:
            q, _c = self._lowrank_qc(p)
            r = q.shape[-1]
            g = torch.Generator(device="cpu").manual_seed(0x51E7C4 ^ p.numel())
            st["manas_omega"] = torch.randn(*p.shape[:-2], p.shape[-1], r, generator=g).to(
                device=p.device, dtype=torch.float32)
            st["manas_y"] = torch.zeros_like(q)
            st["manas_ynow"] = torch.zeros_like(q)
            st["manas_prev_yp"] = torch.zeros_like(q)
        return st

    def _sketch_vote(self, p, gf):
        """Two-clock window, per-vote: unit-normalized micro sketch delta into Y_now (raw,
        coefficient 1 — the same telescoping-difference trick vote() uses for C)."""
        st = self._sketch_state(p)
        yp = gf @ st["manas_omega"]
        delta = yp - st["manas_prev_yp"]
        st["manas_prev_yp"].copy_(yp)
        n = torch.linalg.vector_norm(delta)
        inv = torch.where(torch.isfinite(n) & (n > 0), 1.0 / n, torch.zeros_like(n))
        st["manas_ynow"].add_(delta * inv)

    def _sketch_fold(self, p):
        """Two-clock window, boundary: Y = rho_q*Y + Y_now; per-step buffers reset."""
        st = self._sketch_state(p)
        st["manas_y"].mul_(self.probe_sketch_rho).add_(st["manas_ynow"])
        st["manas_ynow"].zero_()
        st["manas_prev_yp"].zero_()          # p.grad restarts from zero next step
        return st["manas_y"]

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
            if self.probe_sketch_rho is not None:
                self._sketch_vote(p, gf)                  # window hears each micro DIRECTION
            proj = q.mT @ gf                              # projection of the accumulated sum
            deltas.append(proj - prev)
            prev.copy_(proj)
            norms.append(torch.linalg.vector_norm(deltas[-1]))
        pn = torch.stack(norms)
        gn = pn if self.probe_norm == "perparam" else torch.linalg.vector_norm(pn)
        inv = self.probe_gamma / gn
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        fresh = self.probe_rho_step is not None
        if fresh:
            inv = inv / self.probe_gamma       # RAW unit votes into the block (gammas applied
                                               # at probe/fold time, not at storage)
        nxs = (self.nexus_gamma / self.probe_gamma) if self.nexus_gamma else 0.0
        for i, p in enumerate(ps):
            _c = self._cnow(p) if fresh else self.state[p]["manas_c"]   # fresh block vs history
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
            if self.probe_rho_step is not None:       # keep bounded even with probe_rho=1.0
                for p in self._probe_params():
                    if "manas_c" in self.state[p]:
                        self.state[p]["manas_c"].mul_(self.probe_rho_step)
        else:
            refresh = self._probe_updates % max(self.probe_refresh, 1) == 0
            self._probe_updates += 1
            for p in self._probe_params():
                stf = self.state[p]
                if self.probe_rho_step is not None and "manas_c" in stf:
                    c = stf["manas_c"]
                    c.mul_(self.probe_rho_step)              # history ages one step
                    if "manas_cnow" in stf:                  # raw block folds in at coeff 1
                        c.add_(stf["manas_cnow"])            # (BEFORE refresh: same basis)
                        stf["manas_cnow"].zero_()
                sk = self.probe_sketch_rho is not None
                if p.grad is not None and (refresh or sk):
                    q, c = self._lowrank_qc(p)
                    if sk and self._votes_cast >= 2:
                        # THE window path: fold the pad, develop the evidence. Every boundary.
                        y = self._sketch_fold(p)
                        q_new = torch.linalg.qr(y)[0]
                        c.copy_((q_new.mT @ q) @ c)
                        q.copy_(q_new)
                    else:
                        if sk:
                            self._sketch_fold(p)          # keep Y warm through gated steps
                        if refresh:                       # ga1 gate / no sketch: snapshot cadence
                            gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0,
                                                  neginf=0.0).to(torch.float32)
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
        if self.probe_rho_step is not None:
            inv = inv / self.probe_gamma       # fresh-block mode: history is RAW (gamma at probe)
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
            # step-vote mode (and the zero-vote fallback) is one vote per step by definition:
            # no in-step spread to cover -> snapshot aim, no sketch bookkeeping (GA1 GATE)
            y = None
            if refresh:
                # window re-aim: EMA sketch (consensus-aimed) if enabled, else randomized range
                # of the CURRENT gradient (GaLore-style); re-project the old offset either way
                # so d is continuous across the swap
                if y is None:
                    r = q.shape[-1]
                    omega = torch.randn(*p.shape[:-2], p.shape[-1], r, device=p.device)
                    y = gf @ omega
                q_new = torch.linalg.qr(y)[0]
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
    # 10) fresh-block two-clock, RAW storage: unit votes land in cnow (coeff 1), gamma_intra
    #     scales the block only at PROBE time, gamma only at FOLD time; history decays rho_step
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.08, probe_gamma_intra=0.16, probe_sketch_rho=None)
    p.grad = torch.randn(32, 16); o.vote()
    c, cn = o.state[p]["manas_c"], o.state[p]["manas_cnow"]
    assert c.norm() == 0 and abs(cn.norm().item() - 1.0) < 1e-5, \
        "block stores RAW unit votes; history untouched mid-step"
    p.grad = p.grad + torch.randn(32, 16); o.vote()
    q, _ = o._lowrank_qc(p)
    assert torch.allclose(o._d_of(p), q @ (0.08 * c + 0.16 * cn)), \
        "probe must apply BOTH doses at probe time: gamma*history + gamma_intra*block"
    cn_end = cn.clone()
    o._probe_updates = 1                                        # dodge the boundary basis refresh
    o._finish_micro_step()
    assert cn.norm() == 0 and torch.allclose(c, cn_end, atol=1e-7), \
        "boundary must fold the raw block at coeff 1 and zero it"
    c_hist = c.clone()
    p.grad = torch.randn(32, 16); o.vote()                      # next step, then boundary again
    cn2 = cn.clone()
    o._finish_micro_step()
    assert torch.allclose(c, 0.9 * c_hist + cn2, atol=1e-7), \
        "second boundary: history decays rho_step, new raw block folds at coeff 1"
    # 10s) EMA-sketch window: Y accumulates G@omega with FIXED omega every boundary; refresh
    #      takes Q from QR(Y); omega is deterministic (seeded per-param)
    # 10w) THE WINDOW: Q = QR(EMA of unit vote directions), developed EVERY boundary
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0, probe_rho_step=0.9,
                       probe_sketch_rho=0.8, probe_refresh=1000)   # cadence irrelevant when sk on
    g1 = torch.randn(32, 16); p.grad = g1.clone(); o.vote()
    g2 = torch.randn(32, 16); p.grad = g1 + g2; o.vote()
    om = o.state[p]["manas_omega"]; yn = o.state[p]["manas_ynow"]
    u1 = (g1 @ om) / (g1 @ om).norm(); u2 = (g2 @ om) / (g2 @ om).norm()
    assert torch.allclose(yn, u1 + u2, atol=1e-5), "Y_now must sum UNIT micro sketches"
    assert not torch.allclose(yn, ((g1 + g2) @ om) / ((g1 + g2) @ om).norm(), atol=1e-2), \
        "unit increments must NOT telescope to the normalized boundary sum"
    o._probe_updates = 1
    o._finish_micro_step()
    y = o.state[p]["manas_y"]; q, _ = o._lowrank_qc(p)
    assert torch.allclose(y, u1 + u2, atol=1e-5) and yn.norm() == 0 \
        and o.state[p]["manas_prev_yp"].norm() == 0, "boundary: fold Y_now, reset per-step buffers"
    assert torch.allclose(q.mT @ q, torch.eye(4), atol=1e-5), "developed Q orthonormal"
    assert torch.allclose(q @ (q.mT @ y), y, atol=1e-3), \
        "window develops EVERY boundary (no cadence): Q must span the evidence"
    g3 = torch.randn(32, 16); g4 = torch.randn(32, 16)
    p.grad = g3.clone(); o.vote()
    p.grad = g3 + g4; o.vote()
    o._finish_micro_step()
    u3 = (g3 @ om) / (g3 @ om).norm(); u4 = (g4 @ om) / (g4 @ om).norm()
    assert torch.allclose(y, 0.8 * (u1 + u2) + u3 + u4, atol=1e-4), \
        "Y must EMA across boundaries (rho_q * old + new pad)"
    # GA1 GATE: a 1-vote step must NOT develop from Y (falls to snapshot cadence; here cadence
    # never fires, so Q stays exactly where the last 2-vote boundary left it)
    q_before = q.clone()
    p.grad = torch.randn(32, 16); o.vote()
    o._finish_micro_step()
    assert torch.allclose(o._lowrank_qc(p)[0], q_before), \
        "1-vote step: gate holds, window untouched (Y stays warm)"
    # omega deterministic per shape (seeded)
    o2 = ManasOptimizer([torch.nn.Parameter(torch.randn(32, 16))], probe_rank=4, micro_vote=True,
                        probe_rho=1.0, probe_rho_step=0.9, probe_sketch_rho=0.8)
    p2 = o2._probe_params()[0]
    p2.grad = torch.randn(32, 16); o2.vote()
    p2.grad = p2.grad + torch.randn(32, 16); o2.vote()
    o2._finish_micro_step()
    assert torch.allclose(o2.state[p2]["manas_omega"], om), \
        "omega must be deterministic per shape (seeded)"
    # deprecated sketch args: warn + ignored
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], probe_rank=4, micro_vote=True,
                       probe_rho=1.0, probe_rho_step=0.9, probe_sketch_votes=True,
                       probe_sketch_min_votes=1)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
        "probe_sketch_votes/min_votes must warn DeprecationWarning"
    # 10a) refresh auto-couples to the active rho clock; explicit value and live rho both honored
    o_auto = ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], probe_rank=4,
                            micro_vote=True, probe_rho=1.0, probe_rho_step=0.96)
    assert o_auto.probe_refresh == 50, f"rho_step 0.96 must auto-refresh at 50, got {o_auto.probe_refresh}"
    o_auto.probe_rho_step = 0.98
    assert o_auto.probe_refresh == 100, "auto refresh must track rho live"
    assert ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))],
                          probe_rho=0.98).probe_refresh == 100, "per-vote clock: 2/(1-rho)"
    assert ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))],
                          probe_refresh=200).probe_refresh == 200, "explicit refresh must win"
    # 10b) gamma_intra=0: pure-history probing (block never steers a live probe, still folds)
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.08, probe_gamma_intra=0.0, probe_sketch_rho=None)
    p.grad = torch.randn(32, 16); o.vote()
    q, c = o._lowrank_qc(p)
    assert o.state[p]["manas_cnow"].norm() > 0 and torch.allclose(o._d_of(p), q @ (0.08 * c)), \
        "gamma_intra=0 must probe history-only while the block still accumulates"
    o._probe_updates = 1; o._finish_micro_step()
    assert c.norm() > 0, "gamma_intra=0 block must still fold into history at gamma"
    for bad in (dict(probe_rho=1.0),                            # rho=1 needs the step clock
                dict(probe_rho_step=0.9),                       # step clock needs micro_vote
                dict(probe_gamma_intra=0.1),                    # intra needs the step clock
                dict(probe_rank=4, micro_vote=True, probe_rho_step=1.0),
                dict(probe_rank=None, micro_vote=True), dict(nexus_gamma=0.1)):
        try:
            ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], **bad)
            raise AssertionError(f"constructor must reject {bad}")
        except ValueError:
            pass
    print("manas perparam+microvote+nexus self-check PASS")
