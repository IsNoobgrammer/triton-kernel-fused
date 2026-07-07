"""ExpManas — experimental Manas sandbox (user free-hand). NOT the shipped optimizer.

Extends kernels/sm75/manas.ManasOptimizer with switchable mechanism variants, one issue each:

  ref_mode   'theta' (shipped) | 'ema'  [A1] — probe evaluates at theta_bar + d where
             theta_bar is an EMA of theta (beta = probe_rho). All gradients in the rho-window
             are then measured near a COMMON slow-moving reference, approximating the paper's
             same-point requirement instead of a point that moves at full Muon speed.
             (The training gradient is then slightly lagged: probe point = theta_bar + d
             rather than theta + d — the ablation decides whether that trade pays.)
  norm_mode  'global' (shipped) | 'permat'  [B10] — per-matrix normalized increments,
             matching Muon's per-matrix geometry; a bad layer no longer suppresses all probes.
  sign       -1.0 (shipped, descent/extragradient) | +1.0 (SAM-like ascent)  [A10]
  trust      None (shipped) | c > 0  [A4/A1] — cap ||d|| <= c * EMA(||Muon step motion||):
             ties the probe scale to the trajectory scale so the probe cannot dwarf (or
             vanish next to) the actual weight motion as training progresses.

All variants preserve the shipped invariants: sync-free guards, momentum-free probe chain,
bounded d, probe removed before step(). Full-d only (low-rank stays in the shipped class).
"""
import torch

from kernels.sm75.manas import ManasOptimizer


class ExpManas(ManasOptimizer):
    """probe_src 'buffer' (shipped d) | 'muonmom' [A7 wave-4 hypothesis] — probe along Muon's
    OWN momentum direction, d = -gamma * M/||M||_global, recomputed each step: ZERO extra
    probe memory (kills B5 by construction) and deliberately momentum-IMPURE — the A7
    measurement says the paper's purity requirement is not where the gain comes from."""

    def __init__(self, params, ref_mode="theta", norm_mode="global", sign=-1.0,
                 trust=None, probe_src="buffer", comp=None, **kw):
        # low-rank d is inherited from the parent; only trust/ema/permat/sign paths
        # still require full-d (they touch _full_d / raw grads directly)
        if kw.get("probe_rank") is not None:
            assert trust is None and ref_mode == "theta" and norm_mode == "global" \
                and sign == -1.0, "low-rank ExpManas supports comp/muonmom variants only"
        super().__init__(params, **kw)
        assert ref_mode in ("theta", "ema") and norm_mode in ("global", "permat")
        assert probe_src in ("buffer", "muonmom")
        self.ref_mode, self.norm_mode = ref_mode, norm_mode
        self.sign, self.trust = float(sign), (None if trust is None else float(trust))
        self.probe_src = probe_src
        # comp [user's update-history idea]: keep a rho-decayed buffer u of the APPLIED
        # updates (post-polar Muon steps) and add kappa * gamma * u/||u|| to the probe
        # offset — the probe then knows how much of its direction the weights have already
        # realized. kappa < 0 = back off by realized travel; kappa > 0 = extend along it.
        # Differs from momdir (inert): u is post-NS and CORRECTS d instead of replacing it.
        self.comp = None if comp is None else float(comp)
        self._motion_ema = None                                  # EMA of ||theta step motion||

    def _u_of(self, p):
        """Dense u for one param. Low-rank mode: u = Q @ Cu in d's SAME rank-r basis —
        zero extra basis state; the smoothing thesis says the lossy projection keeps the
        signal (this is exactly the test). Cu re-projected on basis refresh like C."""
        st = self.state[p]
        if self.probe_rank is None:
            if "exp_u" not in st:
                st["exp_u"] = torch.zeros_like(p, dtype=torch.float32)
            return st["exp_u"]
        q, _c = self._lowrank_qc(p)
        if "exp_cu" not in st:
            st["exp_cu"] = torch.zeros_like(st["manas_c"])
        return q @ st["exp_cu"]

    def _offsets(self):
        """Dense probe offsets per param (cached-apply path: ema and/or comp active)."""
        ps = self._probe_params()
        offs = {p: ManasOptimizer._d_of(self, p).clone() for p in ps}
        if self.ref_mode == "ema":
            for p in ps:
                offs[p] += self._ema_ref(p) - p.detach().float()
        if self.comp is not None:
            us = {p: self._u_of(p) for p in ps}
            un = torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm(us[p]) for p in ps])).clamp_min(1e-12)
            k = self.comp * self.probe_gamma / un
            for p in ps:
                offs[p] += k * us[p]
        return {p: offs[p].to(p.dtype) for p in ps}

    # ---- probe_src 'muonmom': offset from the batched muon_mom buffers, no d state ----
    def _mom_views(self):
        views = {}
        for group in self.param_groups:
            params = [p for p in group["params"] if p.ndim in (2, 3)]
            if not params:
                continue
            for g in self._plan(group, params):
                mom = self.state[g["anchor"]]["muon_mom"]
                for members, start, crows in g["chunks"]:
                    for p, o, n in members:
                        views[p] = mom[start + o:start + o + n].reshape(p.shape)
        return views

    def _mom_offsets(self):
        views = self._mom_views()
        ps = self._probe_params()
        gn = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(views[p].float()) for p in ps])).clamp_min(1e-12)
        scale = float(self.sign) * self.probe_gamma / gn
        return {p: (views[p].float() * scale).to(p.dtype) for p in ps}

    # ---- ref_mode 'ema': probe offset = (theta_bar - theta) + d, still one add/sub ----
    def _ema_ref(self, p):
        st = self.state[p]
        if "exp_ref" not in st:
            st["exp_ref"] = p.detach().clone().to(torch.float32)
        return st["exp_ref"]

    def _d_of(self, p):
        d = super()._d_of(p)
        if self.ref_mode == "ema":
            return d + (self._ema_ref(p) - p.detach().float())   # theta + off = theta_bar + d
        return d

    def _shift(self, sign):
        if self.probe_src == "muonmom":
            ps = self._probe_params()
            if not getattr(self, "_plan_cache", None):           # step 0: no momentum yet
                return
            key = "exp_shift"
            offs = self._mom_offsets() if sign > 0 else None     # cache -> bit-exact remove
            for p in ps:
                st = self.state[p]
                if sign > 0:
                    st[key] = offs[p]
                p.add_(st[key], alpha=sign)
            return
        if self.ref_mode == "ema" or self.comp is not None:      # offsets depend on p / u;
            ps = self._probe_params()                            # cache the applied offset so
            key = "exp_shift"                                    # remove is bit-exact
            offs = self._offsets() if sign > 0 else None
            for p in ps:
                st = self.state[p]
                if sign > 0:
                    st[key] = offs[p]
                p.add_(st[key], alpha=sign)
            return
        super()._shift(sign)

    @torch.no_grad()
    def step(self, closure=None):
        track = self.trust is not None or self.ref_mode == "ema" or self.comp is not None
        if track:
            before = [p.detach().clone() for p in self._probe_params()]
        if self.comp is not None and self.probe_rank is not None:
            # parent refreshes Q IN-PLACE inside _update_probe; predict the fire (same
            # condition it uses, pre-increment) and snapshot the old basis for Cu re-projection
            fire = self._probe_updates % max(self.probe_refresh, 1) == 0
            self._q_before = [self._lowrank_qc(p)[0].clone() if fire else None
                              for p in self._probe_params()]
        loss = super().step(closure)                             # Muon step + _update_probe
        ps = self._probe_params()
        if track:
            motion = torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm((p - b).float()) for p, b in zip(ps, before)]))
            m = self._motion_ema
            self._motion_ema = motion if m is None else 0.95 * m + 0.05 * motion
        if self.comp is not None:                                # decayed applied-update buffer
            if self.probe_rank is None:
                for p, b in zip(ps, before):
                    self._u_of(p).mul_(self.probe_rho).add_((p.detach() - b).float())
            else:
                for p, b, qo in zip(ps, before, self._q_before):
                    q, _c = self._lowrank_qc(p)
                    st = self.state[p]
                    if "exp_cu" not in st:
                        st["exp_cu"] = torch.zeros_like(st["manas_c"])
                    cu = st["exp_cu"]
                    if qo is not None:                           # basis refreshed this step
                        cu.copy_((q.mT @ qo) @ cu)
                    cu.mul_(self.probe_rho).add_(q.mT @ (p.detach() - b).float())
        if self.ref_mode == "ema":                               # theta_bar tracks theta at rho
            for p in ps:
                self._ema_ref(p).lerp_(p.detach().float(), 1.0 - self.probe_rho)
        if self.trust is not None:                               # cap ||d|| <= trust * motion EMA
            dn = torch.linalg.vector_norm(torch.stack(
                [torch.linalg.vector_norm(self._full_d(p)) for p in ps]))
            cap = self.trust * self._motion_ema
            scale = torch.clamp(cap / dn.clamp_min(1e-12), max=1.0)
            for p in ps:
                self._full_d(p).mul_(scale)
        return loss

    # ---- probe update with sign / per-matrix norm variants ----
    @torch.no_grad()
    def _update_probe(self):
        if self.probe_src == "muonmom":                          # no probe state to update
            return
        if self.norm_mode == "global" and self.sign == -1.0:
            return super()._update_probe()
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        self._probe_updates += 1
        if self.norm_mode == "permat":
            for p in ps:
                d = self._full_d(p)
                gn = torch.linalg.vector_norm(p.grad, dtype=torch.float32)
                inv = self.probe_gamma / gn
                inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
                g32 = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).float()
                d.mul_(self.probe_rho).addcmul_(g32, inv, value=self.sign)
            return
        gn = torch.linalg.vector_norm(torch.stack(
            [torch.linalg.vector_norm(p.grad, dtype=torch.float32) for p in ps]))
        inv = self.probe_gamma / gn
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        for p in ps:
            g32 = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).float()
            self._full_d(p).mul_(self.probe_rho).addcmul_(g32, inv, value=self.sign)
