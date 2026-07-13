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

WHEN TO USE (measured recommendation): ga1 -> use plain FusedMuon; ga>=2 -> use Manas.
At one vote/step the consensus has no spread to work with (info cap): the loss edge is
~0-0.02 while the probe costs 5-10% tps - wall-clock negative. From 2 votes up the edge
(-0.03 to -0.145, growing with votes) dwarfs the overhead. Slicing a batch into micro
batches is FLOPs-free, so "ga1" is almost always a choice, not a constraint.

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
      With micro_vote + probe_rho_step (RECOMMENDED at 137M+): 3 fp32 buffers per param
      (D/Dnow/prev_g), no sketch/QR/GEMMs — full-rank unit votes. Motivation: the BiBo
      rank ladder was monotone (8 < 32 <= 64 < 512 on train and bpb; r512 beat muon bpb
      at every checkpoint but cost 34% tps in QR) — the sketch was the bottleneck; full
      rank is the limit of the trend at ~zero compute.
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
                 probe_gamma_intra=None, probe_sketch_rho=0.96, probe_sketch_votes=None,
                 probe_sketch_min_votes=None, probe_min_votes=2, **kw):
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
        self._shift_on = False       # weights currently hold theta + d (lazy shift)
        self._lazy = None            # fp32-weights check, cached on first apply
        self._pg = None              # persistent shape-grouped backing tensors (built lazily)
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
        # FULL-RANK MICRO-VOTE (probe_rank=None + micro_vote, 2026-07-14): the BiBo 137M rank
        # ladder was MONOTONE (8 < 32 <= 64 < 512 on train AND bpb; r512 the first config to
        # beat muon bpb at every checkpoint) -> the low-rank sketch itself was the bottleneck,
        # not vote count (k=8 == k=4 at fixed rank). Full rank is the limit of that trend and
        # DELETES the machinery: no Q/omega/QR/sketch — state is 3 fp32 model-shaped buffers
        # (manas_d history, manas_dnow fresh block, manas_prev_g snapshot), the vote is a
        # telescoping full-grad delta, unit-normalized, raw storage, both doses at probe time:
        # shift = gamma*D + gamma_intra*Dnow. Two-clock only (probe_rho_step required);
        # fp32 weights only (lazy shift). Costs ~5 elementwise passes/micro, no GEMMs.
        # probe_min_votes (micro_vote mode): the probe ENGAGES only when the previous step
        # cast at least this many votes. Default 2 = the measured recommendation baked in:
        # at ga1 Manas IS aurora Muon - no probe shifts, no vote GEMMs, no buffers, zero
        # overhead (edge ~0-0.02 at one vote/step vs 5-10% tps: wall-clock negative).
        # Slicing to ga>=2 engages the full stack automatically on the next step. Set 1 to
        # force the probe at ga1 (legacy behavior; expect task-specific tuning).
        self.probe_min_votes = int(probe_min_votes)
        if self.probe_min_votes < 1:
            raise ValueError(f"probe_min_votes must be >= 1, got {probe_min_votes}")
        self._votes_last = 0                 # previous step's vote count (0 = not yet seen)
        self.nexus_gamma = float(nexus_gamma)
        if self.micro_vote and self.probe_rank is None:
            if self.probe_rho_step is None:
                raise ValueError("full-rank micro_vote requires probe_rho_step (two-clock mode)")
            if self.nexus_gamma:
                raise ValueError("nexus_gamma requires a low-rank probe")
        if self.nexus_gamma and not self.micro_vote:
            raise ValueError("nexus_gamma requires micro_vote=True")
        self._votes_cast = 0
        # internal, not a constructor arg: min votes/step for the window to develop from Y
        # (the ga1 gate). Default 2 rests on OLD-dynamics (cadence-QR) measurements; set to 1
        # on an instance to test sketch aim at ga1 under QR-every-boundary dynamics.
        self._sketch_gate = 2

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

    def _ensure_groups(self):
        """Persistent SHAPE-GROUPED + FLAT backing, three layers of the same trick:
          1. RESTACKED PARAMS: same-shape params' p.data become views into one contiguous
             (G, *shape) tensor, so the probe shift/restore is a single in-place baddbmm_
             per group - no materialized d, no unbind, no foreach over G tensors.
          2. PINNED GRADS (micro_vote, fp32): p.grad is assigned a view into a matching
             (G, *shape) stack before backward (_pin_grads via apply_probe), so vote()
             reads the stacked gradient with ZERO copies. Anything may replace p.grad
             (harness-style synthetic grads); vote() detects that and falls back to the
             stack-copy path - the pin is a fast path, never a correctness dependency.
          3. FLAT RANK-SPACE ARENAS: every rank-space buffer kind (c/cnow/prev_proj/
             applied/cs and y/ynow/prev_yp + GEMM scratch) is ONE flat fp32 tensor with
             per-group views into it, so cross-group elementwise work (fold, decay, vote
             accumulate, resets) is a single launch instead of foreach over ~140 views.
        Each per-param self.state[p]["manas_X"] stays a VIEW into the group buffer, so
        every legacy per-param code path keeps working."""
        if self._pg is not None:
            return self._pg
        from collections import OrderedDict
        groups = OrderedDict()
        for p in self._probe_params():
            groups.setdefault(tuple(p.shape), []).append(p)
        fresh = self.probe_rho_step is not None
        sk = self.probe_sketch_rho is not None
        nexus = bool(self.nexus_gamma)
        dev = next(iter(groups.values()))[0].device
        metas, c_total, y_total = [], 0, 0
        for shape, grp in groups.items():
            p0 = grp[0]; G = len(grp)
            m, n = p0.shape[-2], p0.shape[-1]
            r = self._rank_for(m, n); lead = tuple(p0.shape[:-2])
            L = 1
            for s in lead:
                L *= s
            metas.append((shape, grp, G, m, n, r, lead, L, c_total, y_total))
            c_total += G * L * r * n
            y_total += G * L * m * r
        fl = {k: torch.zeros(c_total, device=dev, dtype=torch.float32)
              for k in ("c", "prev_proj", "applied", "projpad")}
        if fresh:
            fl["cnow"] = torch.zeros(c_total, device=dev, dtype=torch.float32)
        if nexus:
            fl["cs"] = torch.zeros(c_total, device=dev, dtype=torch.float32)
        if sk:
            for k in ("y", "ynow", "prev_yp", "ypad"):
                fl[k] = torch.zeros(y_total, device=dev, dtype=torch.float32)
        self._flat = fl
        pg = []
        for shape, grp, G, m, n, r, lead, L, co, yo in metas:
            p0 = grp[0]; numel = p0.numel()
            csz, ysz = G * L * r * n, G * L * m * r
            def cview(k): return fl[k][co:co + csz].view(G, *lead, r, n)
            def yview(k): return fl[k][yo:yo + ysz].view(G, *lead, m, r)
            # q init is per-param but shape (hence numel) is identical across the group ->
            # identical seed -> identical q; reproduce it once and stack (bit-for-bit parity).
            gq = torch.Generator(device="cpu").manual_seed(0x9A5 + numel)
            q0 = torch.linalg.qr(torch.randn(*lead, m, r, generator=gq).to(dev))[0].to(torch.float32)
            q_b = q0.unsqueeze(0).expand(G, *q0.shape).contiguous()
            # RESTACK params: same values, storage now one contiguous (G, *shape) tensor
            pstk = torch.empty(G, *shape, device=dev, dtype=p0.dtype)
            for i, p in enumerate(grp):
                pstk[i].copy_(p.data)
                p.data = pstk[i]
            buf = dict(shape=shape, params=grp, G=G, r=r, lead=lead, co=co, csz=csz,
                       q=q_b, c=cview("c"), prev_proj=cview("prev_proj"),
                       applied=cview("applied"), projpad=cview("projpad"),
                       pstk=pstk, p3=pstk.view(G * L, m, n), q3=q_b.view(G * L, m, r),
                       a3=fl["applied"][co:co + csz].view(G * L, r, n),
                       pp3=fl["projpad"][co:co + csz].view(G * L, r, n))
            if fresh:
                buf["cnow"] = cview("cnow")
            if nexus:
                buf["cs"] = cview("cs")
            if self.micro_vote and p0.dtype == torch.float32:
                buf["gstk"] = torch.zeros(G, *shape, device=dev, dtype=torch.float32)
                buf["gviews"] = list(buf["gstk"].unbind(0))
                buf["g3"] = buf["gstk"].view(G * L, m, n)
            if sk:
                go = torch.Generator(device="cpu").manual_seed(0x51E7C4 ^ numel)
                o0 = torch.randn(*lead, n, r, generator=go).to(dev).to(torch.float32)
                buf["omega"] = o0.unsqueeze(0).expand(G, *o0.shape).contiguous()
                buf["omega3"] = buf["omega"].view(G * L, n, r)
                buf["y"] = yview("y"); buf["ynow"] = yview("ynow")
                buf["prev_yp"] = yview("prev_yp"); buf["ypad"] = yview("ypad")
                buf["yp3"] = fl["ypad"][yo:yo + ysz].view(G * L, m, r)
            for i, p in enumerate(grp):
                st = self.state[p]
                st["manas_q"] = q_b[i]; st["manas_c"] = buf["c"][i]
                st["manas_prev_proj"] = buf["prev_proj"][i]; st["manas_applied"] = buf["applied"][i]
                if fresh: st["manas_cnow"] = buf["cnow"][i]
                if sk:
                    st["manas_omega"] = buf["omega"][i]; st["manas_y"] = buf["y"][i]
                    st["manas_ynow"] = buf["ynow"][i]; st["manas_prev_yp"] = buf["prev_yp"][i]
                if nexus: st["manas_cs"] = buf["cs"][i]
            pg.append(buf)
        self._pg = pg
        return pg

    @staticmethod
    def _cholqr2(y):
        """Batched CholeskyQR2 orthonormalization of tall-skinny (..., m, r) stacks.
        Basis-equivalent to torch.linalg.qr(y)[0] - downstream math depends only on the
        PROJECTOR span(y), never on which orthonormal basis represents it - but pure
        GEMM + tiny r x r Cholesky/trsm instead of cuSOLVER Householder (measured 11.4ms
        -> <1ms per boundary on bibo-min shapes). Pass 1 carries a relative ridge so a
        rank-deficient window still factors (deficient directions get an arbitrary
        completion, exactly like QR); pass 2 polishes orthogonality to ~machine eps."""
        r = y.shape[-1]
        eye = torch.eye(r, device=y.device, dtype=y.dtype)
        q = y
        for ridge in (1e-6, 1e-7):
            g = q.mT @ q
            scale = g.diagonal(dim1=-2, dim2=-1).mean(-1)[..., None, None]
            L = torch.linalg.cholesky_ex(g + (ridge * scale + 1e-30) * eye)[0]
            q = torch.linalg.solve_triangular(L, q.mT, upper=False).mT
        return q

    def _pin_grads(self, pg):
        """Pin p.grad as views into the group grad stack BEFORE backward, so autograd
        accumulates straight into the stacked buffer and vote() is copy-free. Only fills
        missing (None) grads - an existing foreign p.grad is left alone and vote() copies."""
        for buf in pg:
            gv = buf.get("gviews")
            if gv is None:
                continue
            miss = [i for i, p in enumerate(buf["params"]) if p.grad is None]
            if not miss:
                continue
            if len(miss) == len(buf["params"]):
                buf["gstk"].zero_()
            else:
                for i in miss:
                    gv[i].zero_()
            for i in miss:
                buf["params"][i].grad = gv[i]

    def _full_d(self, p):
        st = self.state[p]
        if "manas_d" not in st:
            st["manas_d"] = torch.zeros_like(p, dtype=torch.float32)
        return st["manas_d"]

    def _full_state(self, p):
        """Full-rank micro-vote buffers: manas_d = history D (raw unit votes, rho_step-decayed
        at boundaries), manas_dnow = this step's fresh block, manas_prev_g = accumulating-grad
        snapshot for the telescoping vote delta. All fp32, model-shaped."""
        st = self.state[p]
        if "manas_dnow" not in st:
            self._full_d(p)
            st["manas_dnow"] = torch.zeros_like(p, dtype=torch.float32)
            st["manas_prev_g"] = torch.zeros_like(p, dtype=torch.float32)
        return st

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

    def _coef_of(self, p):
        """Rank-space coefficient total the probe applies: d = q @ coef. Fresh-block mode:
        ALL state is raw (unit votes; only rho_step ever touches history) and both doses are
        applied HERE, at probe time: coef = gamma*history + gamma_intra*block. Nothing is
        baked into storage, so either dose can change mid-run and rescales retroactively."""
        _q, c = self._lowrank_qc(p)
        if self.probe_rho_step is not None:
            total = self.probe_gamma * c
            if "manas_cnow" in self.state[p]:
                total = total + self.probe_gamma_intra * self.state[p]["manas_cnow"]
        else:
            total = c
        cs = self.state[p].get("manas_cs") if self.nexus_gamma else None
        return total + cs if cs is not None else total

    def _d_of(self, p):
        """Dense probe offset for one param (view for full mode, materialized for low-rank)."""
        if self.probe_rank is None:
            if self.micro_vote:                  # full-rank two-clock: raw storage, dosed here
                st = self._full_state(p)
                return self.probe_gamma * st["manas_d"] \
                    + self.probe_gamma_intra * st["manas_dnow"]
            return self._full_d(p)
        q, _c = self._lowrank_qc(p)
        return q @ self._coef_of(p)

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

    def _micro_state(self, p):
        """(prev_proj, cs) rank-space buffers for micro voting; zero-init, reset each step."""
        st = self.state[p]
        _q, c = self._lowrank_qc(p)
        if "manas_prev_proj" not in st:
            st["manas_prev_proj"] = torch.zeros_like(c)
        if self.nexus_gamma and "manas_cs" not in st:
            st["manas_cs"] = torch.zeros_like(c)
        return st["manas_prev_proj"], st.get("manas_cs")

    def _probe_engaged(self):
        """micro_vote mode self-gates to pure aurora Muon below probe_min_votes/step
        (measured: at 1 vote/step the probe is wall-clock negative). Step-vote mode
        (micro_vote=False) is an explicit legacy choice and always engages."""
        return (not self.micro_vote) or self._votes_last >= self.probe_min_votes

    # ---------------- probe application (LAZY SHIFT) ----------------
    # Between micros the weights STAY at theta + d: remove_probe() only clears the context
    # flag, apply_probe() adjusts the weights by the DELTA of the rank-space coefficients
    # since the last apply (tracked in manas_applied, r x n per param), and step() restores
    # theta exactly before the base update. This halves the full-tensor passes per step
    # (2k apply/remove -> k deltas + 1 restore). fp32 weights only (delta adds don't cancel
    # bit-exactly in low precision); non-fp32 falls back to the exact apply/remove pair.
    # Consequence to know: code that reads the model BETWEEN micro-batches sees theta + d;
    # anything after step() (evals, checkpoints) sees clean theta as always.
    @contextmanager
    def probe(self):
        """Run the enclosed forward/backward at theta + d. Nested use / step() inside raise."""
        self.apply_probe()
        try:
            yield
        finally:
            self.remove_probe()

    def _lazy_ok(self):
        if self._lazy is None:
            self._lazy = all(p.dtype == torch.float32 for p in self._probe_params())
        return self._lazy

    @torch.no_grad()
    def apply_probe(self):
        if self._probe_on:
            raise RuntimeError("probe already applied")
        if not self._probe_engaged():           # ga1 self-gate: pure Muon, no shift at all
            self._probe_on = True
            return
        ps = self._probe_params()
        if self.probe_rank is None:
            if self.micro_vote:
                # full-rank two-clock: first apply of the step shifts by the dosed total;
                # every vote() then mirrors its own block increment straight onto theta
                # (lazy sync), so later applies are no-ops. Restore recomputes the same
                # total at step() — value-exact, fp32-rounding drift only (same class the
                # low-rank lazy path accepts). Don't change gamma between micros.
                if not self._lazy_ok():
                    raise RuntimeError("full-rank micro_vote requires fp32 weights (lazy shift)")
                if not self._shift_on:
                    sps = [p for p in ps if "manas_dnow" in self.state[p]]
                    if sps:
                        torch._foreach_add_(sps, [self.state[p]["manas_d"] for p in sps],
                                            alpha=self.probe_gamma)
                        torch._foreach_add_(sps, [self.state[p]["manas_dnow"] for p in sps],
                                            alpha=self.probe_gamma_intra)
                    self._shift_on = True
            elif not self._shift_on:             # full-d step-vote: all-or-nothing (d fixed)
                ds = [self._full_d(p) for p in ps]
                if self._lazy_ok():
                    torch._foreach_add_(ps, ds)
                else:
                    for p, d in zip(ps, ds):
                        p.add_(d.to(p.dtype))
                self._shift_on = True
        elif self._lazy_ok():
            # low-rank lazy: coefficient delta on the FLAT arena (a few tiny launches for
            # ALL groups), then ONE in-place baddbmm_ per shape group writes the shift
            # straight into the restacked weights - no materialized d, no foreach.
            pg = self._ensure_groups()
            if self.micro_vote:
                self._pin_grads(pg)          # backward lands in the vote()-ready stack
            fl = self._flat
            if self.probe_rho_step is not None:
                coef = self.probe_gamma * fl["c"]
                if "cnow" in fl:
                    coef.add_(fl["cnow"], alpha=self.probe_gamma_intra)
            else:
                coef = fl["c"].clone()
            if "cs" in fl:
                coef.add_(fl["cs"])
            dcoef = coef - fl["applied"]
            fl["applied"].copy_(coef)
            for buf in pg:
                d3 = dcoef[buf["co"]:buf["co"] + buf["csz"]].view_as(buf["a3"])
                buf["p3"].baddbmm_(buf["q3"], d3)            # theta += q @ dcoef, in place
            self._shift_on = True
        else:
            for p in ps:                         # exact pair path (non-fp32 weights)
                p.add_(self._d_of(p).to(p.dtype))
            self._shift_on = True
        self._probe_on = True

    @torch.no_grad()
    def remove_probe(self):
        if not self._probe_on:
            raise RuntimeError("probe not applied")
        self._probe_on = False
        if self._shift_on and self.probe_rank is not None and not self._lazy_ok():
            # non-fp32 low-rank: exact inverse now (deterministic recompute, bit-identical)
            for p in self._probe_params():
                p.sub_(self._d_of(p).to(p.dtype))
            self._shift_on = False
        # full-d (any dtype: same d tensor adds/subs bit-identically) and fp32 low-rank:
        # LAZY - theta restored at step()

    @torch.no_grad()
    def _restore_theta(self):
        """Exactly restore clean theta (called by step() before the base update)."""
        if not self._shift_on:
            return
        ps = self._probe_params()
        if self.probe_rank is None:
            if self.micro_vote:                  # subtract the dosed total (see apply_probe)
                sps = [p for p in ps if "manas_dnow" in self.state[p]]
                if sps:
                    torch._foreach_add_(sps, [self.state[p]["manas_d"] for p in sps],
                                        alpha=-self.probe_gamma)
                    torch._foreach_add_(sps, [self.state[p]["manas_dnow"] for p in sps],
                                        alpha=-self.probe_gamma_intra)
                self._shift_on = False
                return
            ds = [self._full_d(p) for p in ps]
            if self._lazy_ok():
                torch._foreach_sub_(ps, ds)
            else:
                for p, d in zip(ps, ds):
                    p.sub_(d.to(p.dtype))
        elif self._lazy_ok():
            for buf in self._ensure_groups():        # ONE in-place baddbmm_ per group
                buf["p3"].baddbmm_(buf["q3"], buf["a3"], alpha=-1.0)
            self._flat["applied"].zero_()            # one flat launch for all groups
        else:
            for p in ps:
                ap = self.state[p].get("manas_applied")
                if ap is not None:
                    q, _c = self._lowrank_qc(p)
                    p.sub_(q @ ap)
                    ap.zero_()
        self._shift_on = False

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
        if not self._probe_engaged():            # ga1 self-gate: count only (so a switch to
            self._votes_cast += 1                # ga>=2 engages the probe on the next step)
            return
        if self._manas_step < self.probe_warmup_steps:   # warmup: probe stays a no-op
            return
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        self._votes_cast += 1
        if self.probe_rank is None:
            # ---- FULL-RANK vote: telescoping full-grad delta, unit-normalized, raw into
            # Dnow; theta (already shifted, lazy) mirrors the increment so the next micro
            # probes at the CURRENT consensus without a re-apply. ~5 passes, no GEMMs. ----
            deltas, dnows, norms, sync = [], [], [], []
            for p in ps:
                st = self._full_state(p)
                delta = torch.nan_to_num_(p.grad.to(torch.float32) - st["manas_prev_g"],
                                          nan=0.0, posinf=0.0, neginf=0.0)
                st["manas_prev_g"].copy_(p.grad)
                deltas.append(delta); dnows.append(st["manas_dnow"]); sync.append(p)
                norms.append(torch.linalg.vector_norm(delta))
            pn = torch.stack(norms)
            gn = pn if self.probe_norm == "perparam" else torch.linalg.vector_norm(pn)
            inv = torch.where(torch.isfinite(gn) & (gn > 0), 1.0 / gn, torch.zeros_like(gn))
            invs = [inv] * len(ps) if inv.ndim == 0 else list(inv.unbind())
            torch._foreach_addcmul_(dnows, deltas, invs, value=-1.0)     # raw unit vote
            if self._shift_on and self.probe_gamma_intra:
                torch._foreach_addcmul_(sync, deltas, invs, value=-self.probe_gamma_intra)
            return
        fresh = self.probe_rho_step is not None
        sk = self.probe_sketch_rho is not None
        nxs = (self.nexus_gamma / self.probe_gamma) if self.nexus_gamma else 0.0
        if len(ps) == len(self._probe_params()):
            # ============ BATCHED per shape group + FLAT tail (launch-bound fix) ============
            # Per group: sanitize the PINNED grad stack in place-free out= (zero copies when
            # backward accumulated into our views) and run the two rank-r GEMMs with out=
            # into the flat scratch arenas. Then the whole vote tail (delta, norm, decay,
            # accumulate) is a handful of launches on the FLAT tensors, all groups at once.
            # Bit-parity with the per-param path (same elementwise math; only the reduction
            # ORDER of the global grad-norm differs, ~1e-7).
            pg = self._ensure_groups()
            fl = self._flat
            pnorms = []
            for buf in pg:
                gv = buf.get("gviews")
                if gv is not None and all(p.grad is gv[i]
                                          for i, p in enumerate(buf["params"])):
                    # zero-copy: GEMM straight off the RAW pinned grad stack. The
                    # nonfinite guard moves to RANK SPACE (delta/dy sanitized below) -
                    # bit-identical on finite grads, and skips a full read+write pass.
                    # Degradation on a nonfinite micro: its projection columns zero out
                    # and the NEXT micro's delta is zeroed too (prev buffers hold the
                    # poison until then); boundary resets fully - same recovery contract.
                    gf3 = buf["g3"]
                else:                        # foreign grads (synthetic/harness): copy path
                    g_stk = torch.stack([p.grad for p in buf["params"]])
                    gf = torch.nan_to_num(g_stk, nan=0.0, posinf=0.0,
                                          neginf=0.0).to(torch.float32)
                    gf3 = gf.view_as(buf["p3"])
                if sk:
                    torch.matmul(gf3, buf["omega3"], out=buf["yp3"])
                    dy = torch.nan_to_num_(buf["ypad"] - buf["prev_yp"],
                                           nan=0.0, posinf=0.0, neginf=0.0)
                    buf["prev_yp"].copy_(buf["ypad"])
                    ny = torch.linalg.vector_norm(dy, dim=tuple(range(1, dy.ndim)))
                    invy = torch.where(torch.isfinite(ny) & (ny > 0), 1.0 / ny, torch.zeros_like(ny))
                    buf["ynow"].add_(dy * invy.view(-1, *([1] * (dy.ndim - 1))))
                torch.matmul(buf["q3"].mT, gf3, out=buf["pp3"])
            delta = torch.nan_to_num_(fl["projpad"] - fl["prev_proj"],
                                      nan=0.0, posinf=0.0, neginf=0.0)   # one flat launch
            fl["prev_proj"].copy_(fl["projpad"])
            for buf in pg:                                       # per-matrix norms (ragged)
                dv = delta[buf["co"]:buf["co"] + buf["csz"]].view(buf["G"], -1)
                pnorms.append(torch.linalg.vector_norm(dv, dim=1))
            if self.probe_norm != "perparam":                    # global (default): flat tail
                gn = torch.linalg.vector_norm(torch.cat(pnorms))
                iv0 = self.probe_gamma / gn
                iv0 = torch.where(torch.isfinite(iv0) & (gn > 0), iv0, torch.zeros_like(iv0))
                if fresh:
                    iv0 = iv0 / self.probe_gamma                 # raw unit votes (fresh)
                tgt = fl["cnow"] if fresh else fl["c"]
                if self.probe_rho != 1.0:
                    tgt.mul_(self.probe_rho)
                tgt.addcmul_(delta, iv0, value=-1.0)             # consensus vote
                if nxs:
                    fl["cs"].addcmul_(delta, iv0 * nxs, value=-1.0)   # walker
                return
            for j, buf in enumerate(pg):                         # perparam: ragged per group
                pn = pnorms[j]
                iv = self.probe_gamma / pn
                iv = torch.where(torch.isfinite(iv) & (pn > 0), iv, torch.zeros_like(iv))
                if fresh:
                    iv = iv / self.probe_gamma
                dg = delta[buf["co"]:buf["co"] + buf["csz"]].view_as(buf["c"])
                ivb = iv.view(-1, *([1] * (dg.ndim - 1)))
                tgt = buf["cnow"] if fresh else buf["c"]
                if self.probe_rho != 1.0:
                    tgt.mul_(self.probe_rho)
                tgt.addcmul_(dg, ivb, value=-1.0)
                if nxs:
                    buf["cs"].addcmul_(dg, ivb * nxs, value=-1.0)
            return
        # ================= fallback: per-param (partial-grad step) =================
        deltas, norms = [], []
        for p in ps:
            q, _c = self._lowrank_qc(p)
            prev, _cs = self._micro_state(p)
            gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            if sk:
                self._sketch_vote(p, gf)
            proj = q.mT @ gf
            deltas.append(proj - prev)
            prev.copy_(proj)
            norms.append(torch.linalg.vector_norm(deltas[-1]))
        pn = torch.stack(norms)
        gn = pn if self.probe_norm == "perparam" else torch.linalg.vector_norm(pn)
        inv = self.probe_gamma / gn
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        if fresh:
            inv = inv / self.probe_gamma
        for i, p in enumerate(ps):
            _c = self._cnow(p) if fresh else self.state[p]["manas_c"]
            iv = inv if inv.ndim == 0 else inv[i]
            _c.mul_(self.probe_rho).addcmul_(deltas[i], iv, value=-1.0)
            if nxs:
                _prev, cs = self._micro_state(p)
                cs.addcmul_(deltas[i], iv * nxs, value=-1.0)

    # ---------------- step ----------------
    @torch.no_grad()
    def step(self, closure=None, probe_loss=None):
        # probe_loss: accepted for back-compat with the removed rgd_tau wiring; unused.
        if self._probe_on:
            raise RuntimeError("remove_probe() (or exit the probe() context) before step()")
        self._manas_step += 1                    # drives the probe warmup gate (see _update_probe)
        self._restore_theta()                    # lazy shift: clean theta before the base update
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
        if not self._probe_engaged():
            # ga1 self-gate: pure-Muon step; record the count so ga>=2 engages next step
            self._votes_last = self._votes_cast
            self._votes_cast = 0
            return
        self._votes_last = self._votes_cast
        if self.probe_rank is None:
            # full-rank boundary: history ages one step, raw block folds at coeff 1,
            # per-step buffers reset. Zero votes -> decay only (probe never poisons).
            sts = [self.state[p] for p in self._probe_params()
                   if "manas_dnow" in self.state[p]]
            if sts:
                torch._foreach_mul_([s["manas_d"] for s in sts], self.probe_rho_step)
                if self._votes_cast:
                    torch._foreach_add_([s["manas_d"] for s in sts],
                                        [s["manas_dnow"] for s in sts])
                torch._foreach_zero_([s["manas_dnow"] for s in sts])
                torch._foreach_zero_([s["manas_prev_g"] for s in sts])
            self._votes_cast = 0
            return
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
            ps = self._probe_params()
            sk = self.probe_sketch_rho is not None
            # ---- BATCHED BOUNDARY: folds/zeros are single launches on the FLAT arenas, ----
            # ---- QR + re-projection run straight off the group buffers (no re-stack)  ----
            fl = getattr(self, "_flat", None) if self._pg is not None else None
            if self.probe_rho_step is not None:
                if fl is not None and "cnow" in fl:
                    fl["c"].mul_(self.probe_rho_step)       # history ages one step
                    fl["c"].add_(fl["cnow"])                # raw block folds, coeff 1
                    fl["cnow"].zero_()                      # (BEFORE develop: same basis)
                else:
                    cs_fold = [self.state[p]["manas_c"] for p in ps if "manas_c" in self.state[p]
                               and "manas_cnow" in self.state[p]]
                    cn_fold = [self.state[p]["manas_cnow"] for p in ps if "manas_c" in self.state[p]
                               and "manas_cnow" in self.state[p]]
                    if cs_fold:
                        torch._foreach_mul_(cs_fold, self.probe_rho_step)
                        torch._foreach_add_(cs_fold, cn_fold)
                        torch._foreach_zero_(cn_fold)
            develop = sk and self._votes_cast >= self._sketch_gate
            gps = [p for p in ps if p.grad is not None and "manas_omega" in self.state[p]] \
                if sk else []
            full = sk and fl is not None and "y" in fl and len(gps) == len(ps)
            if full:
                fl["y"].mul_(self.probe_sketch_rho)         # Y = rho_q*Y + Y_now, one launch
                fl["y"].add_(fl["ynow"])
                fl["ynow"].zero_()
                fl["prev_yp"].zero_()
            elif sk and gps:
                ys = [self.state[p]["manas_y"] for p in gps]
                yn = [self.state[p]["manas_ynow"] for p in gps]
                torch._foreach_mul_(ys, self.probe_sketch_rho)
                torch._foreach_add_(ys, yn)
                torch._foreach_zero_(yn)
                torch._foreach_zero_([self.state[p]["manas_prev_yp"] for p in gps])
            if develop and full:
                for buf in self._pg:                        # already stacked: GEMM-only QR
                    q_new = self._cholqr2(buf["y"])
                    c_new = (q_new.mT @ buf["q"]) @ buf["c"]
                    buf["c"].copy_(c_new)
                    buf["q"].copy_(q_new)
            elif develop and gps:
                groups = {}
                for p in gps:                            # group by FULL param shape: q (m,r)
                    groups.setdefault(p.shape, []).append(p)   # AND c (r,n) must both stack
                for shape, grp in groups.items():
                    y_stk = torch.stack([self.state[p]["manas_y"] for p in grp])
                    q_stk = torch.stack([self.state[p]["manas_q"] for p in grp])
                    c_stk = torch.stack([self.state[p]["manas_c"] for p in grp])
                    q_new = torch.linalg.qr(y_stk)[0]                    # ONE batched QR
                    c_new = (q_new.mT @ q_stk) @ c_stk                   # ONE batched re-projection
                    for i, p in enumerate(grp):
                        self.state[p]["manas_c"].copy_(c_new[i])
                        self.state[p]["manas_q"].copy_(q_new[i])
            elif refresh:                        # ga1 gate / no sketch: snapshot cadence
                for p in ps:
                    if p.grad is None:
                        continue
                    q, c = self._lowrank_qc(p)
                    gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0,
                                          neginf=0.0).to(torch.float32)
                    r = q.shape[-1]
                    omega = torch.randn(*p.shape[:-2], p.shape[-1], r, device=p.device)
                    q_new = torch.linalg.qr(gf @ omega)[0]
                    c.copy_((q_new.mT @ q) @ c)
                    q.copy_(q_new)
            if fl is not None:
                fl["prev_proj"].zero_()          # p.grad restarts; walker never crosses
                if "cs" in fl:
                    fl["cs"].zero_()
            else:
                resets = [self.state[p][k] for p in ps for k in ("manas_prev_proj", "manas_cs")
                          if k in self.state[p]]
                if resets:
                    torch._foreach_zero_(resets)
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
    # tests exercise the probe machinery directly -> pre-engage the ga1 self-gate
    # (real training engages after the first step's votes are counted)
    _orig_init = ManasOptimizer.__init__
    def _test_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self._votes_last = 9
    ManasOptimizer.__init__ = _test_init
    # 0) GA1 SELF-GATE: below probe_min_votes/step, manas IS aurora muon (no shift, no state)
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0, probe_rho_step=0.9)
    o._votes_last = 0                                           # fresh-run condition
    p0 = p.detach().clone()
    o._manas_step = 1
    o.apply_probe()
    assert torch.allclose(p, p0) and not o._shift_on, "disengaged probe must not touch theta"
    p.grad = torch.randn(32, 16); o.remove_probe(); o.vote()
    assert "manas_q" not in o.state[p], "disengaged vote must not allocate probe state"
    o._finish_micro_step()
    assert o._votes_last == 1 and not o._probe_engaged(), "ga1: stays pure muon"
    o.apply_probe(); p.grad = torch.randn(32, 16); o.remove_probe(); o.vote()
    p.grad = p.grad + torch.randn(32, 16); o.vote()             # 2 votes this step
    o._finish_micro_step()
    assert o._votes_last == 2 and o._probe_engaged(), "ga>=2 must engage on the next step"
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
    # 10m) MIXED SHAPES: same (m, r) window shape but different n must not cross-stack
    pmx_a = torch.nn.Parameter(torch.randn(32, 16))
    pmx_b = torch.nn.Parameter(torch.randn(32, 24))         # same m=32 -> same q shape, n differs
    omx = ManasOptimizer([pmx_a, pmx_b], probe_rank=4, micro_vote=True, probe_rho=1.0,
                         probe_rho_step=0.9, probe_sketch_rho=0.8)
    for _ in range(2):
        pmx_a.grad = (torch.randn(32, 16) if pmx_a.grad is None
                      else pmx_a.grad + torch.randn(32, 16))
        pmx_b.grad = (torch.randn(32, 24) if pmx_b.grad is None
                      else pmx_b.grad + torch.randn(32, 24))
        omx.vote()
    omx._finish_micro_step()                                # batched develop must group correctly
    for pp, nn in ((pmx_a, 16), (pmx_b, 24)):
        qx, cx = omx._lowrank_qc(pp)
        assert qx.shape == (32, 4) and cx.shape == (4, nn) and \
            torch.allclose(qx.mT @ qx, torch.eye(4), atol=1e-5), "mixed-shape develop broken"
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
    # 10L) LAZY SHIFT: weights stay at theta+d between micros (delta-adjusted per apply);
    #      step-time restore returns clean theta; probes see the CURRENT d each micro
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.08)
    p0 = p.detach().clone()
    o._manas_step = 1                                           # past warmup gate
    o.apply_probe()                                             # d=0: no-op shift
    assert torch.allclose(p, p0), "empty consensus: probe must not move theta"
    p.grad = torch.randn(32, 16); o.remove_probe(); o.vote()
    o.apply_probe()                                             # now shifted by gamma_i*v1
    d1 = o._d_of(p)
    assert torch.allclose(p - p0, d1, atol=1e-6), "weights must hold theta + current d"
    p.grad = p.grad + torch.randn(32, 16); o.remove_probe(); o.vote()
    o.apply_probe()                                             # delta-adjusted to new d
    assert torch.allclose(p - p0, o._d_of(p), atol=1e-6), "lazy delta must track d exactly"
    o.remove_probe()
    o._restore_theta()
    assert torch.allclose(p, p0, atol=1e-5), "restore must return clean theta"
    assert o.state[p]["manas_applied"].norm() == 0 and not o._shift_on
    # 10F) FULL-RANK micro-vote == low-rank at COMPLETE rank (m <= n, r = m: q is m x m
    #      orthogonal, so projection is lossless and the two modes must agree exactly)
    pf = torch.nn.Parameter(torch.randn(4, 16))
    pl = torch.nn.Parameter(pf.detach().clone())
    of = ManasOptimizer([pf], probe_rank=None, micro_vote=True, probe_rho=1.0,
                        probe_rho_step=0.9, probe_gamma=0.08)
    ol = ManasOptimizer([pl], probe_rank=4, micro_vote=True, probe_rho=1.0,
                        probe_rho_step=0.9, probe_gamma=0.08, probe_sketch_rho=None)
    of._probe_updates = ol._probe_updates = 1                   # dodge boundary refresh
    gseq = [torch.randn(4, 16) for _ in range(3)]
    for o_, p_ in ((of, pf), (ol, pl)):
        p_.grad = gseq[0].clone(); o_.vote()
        p_.grad = p_.grad + gseq[1]; o_.vote()
    assert torch.allclose(of._d_of(pf), ol._d_of(pl), atol=1e-5), \
        "full-rank must equal low-rank at complete rank (in-step)"
    of._finish_micro_step(); ol._finish_micro_step()
    for o_, p_ in ((of, pf), (ol, pl)):
        p_.grad = gseq[2].clone(); o_.vote()
    assert torch.allclose(of._d_of(pf), ol._d_of(pl), atol=1e-5), \
        "full-rank must equal low-rank across the boundary (fold + decay parity)"
    st_f = of.state[pf]
    assert st_f["manas_dnow"].norm() > 0 and st_f["manas_d"].norm() > 0, \
        "full-rank: history and fresh block both live"
    # 10G) full-rank lazy sync: theta tracks gamma*D + gamma_i*Dnow through votes;
    #      restore returns clean theta; boundary resets per-step buffers
    p = torch.nn.Parameter(torch.randn(6, 8))
    o = ManasOptimizer([p], probe_rank=None, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.05)
    p0 = p.detach().clone()
    o._manas_step = 1
    o.apply_probe()
    assert torch.allclose(p, p0) and o._shift_on, "empty consensus: shift flagged, theta still"
    p.grad = torch.randn(6, 8); o.remove_probe(); o.vote()
    assert torch.allclose(p - p0, o._d_of(p), atol=1e-6), "vote must sync theta to current d"
    o.apply_probe()                                             # no-op (already synced)
    p.grad = p.grad + torch.randn(6, 8); o.remove_probe(); o.vote()
    assert torch.allclose(p - p0, o._d_of(p), atol=1e-6), "second vote must keep theta synced"
    o._restore_theta()
    assert torch.allclose(p, p0, atol=1e-5) and not o._shift_on, "restore must return clean theta"
    o._finish_micro_step()
    st = o.state[p]
    assert st["manas_dnow"].norm() == 0 and st["manas_prev_g"].norm() == 0 \
        and st["manas_d"].norm() > 0, "boundary: block folds into history, snapshots reset"
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
                dict(probe_rank=None, micro_vote=True),         # full micro needs the step clock
                dict(probe_rank=None, micro_vote=True, probe_rho=1.0,
                     probe_rho_step=0.9, nexus_gamma=0.1),      # walker is rank-space only
                dict(nexus_gamma=0.1)):
        try:
            ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], **bad)
            raise AssertionError(f"constructor must reject {bad}")
        except ValueError:
            pass
    print("manas perparam+microvote+nexus self-check PASS")
