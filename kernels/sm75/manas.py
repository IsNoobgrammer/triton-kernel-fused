import warnings
from contextlib import contextmanager

import torch

from kernels.sm75.muon import FusedMuon

__all__ = ["ManasOptimizer", "NS8_COEFFS"]

_KJ = (3.4445, -4.7750, 2.0315)
_PIN = (2.0, -1.5, 0.5)
NS8_COEFFS = (_KJ,) * 6 + (_PIN,) * 2


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
        self.probe_rho_step = None if probe_rho_step is None else float(probe_rho_step)
        if self.probe_rho_step is not None:
            if not micro_vote:
                raise ValueError("probe_rho_step requires micro_vote=True")
            if not (0.0 < self.probe_rho_step < 1.0):
                raise ValueError(f"probe_rho_step must be in (0, 1), got {probe_rho_step}")
        if probe_gamma_intra is not None and self.probe_rho_step is None:
            raise ValueError("probe_gamma_intra requires probe_rho_step (fresh-block mode)")
        self._probe_gamma_intra = None if probe_gamma_intra is None else float(probe_gamma_intra)
        if self._probe_gamma_intra is not None and self._probe_gamma_intra < 0:
            raise ValueError(f"probe_gamma_intra must be >= 0, got {probe_gamma_intra}")
        rho_hi = 1.0 if self.probe_rho_step is not None else 1.0 - 1e-12
        if not (0.0 <= probe_rho <= rho_hi):
            raise ValueError(f"probe_rho must be in [0, 1) (or [0, 1] with probe_rho_step), "
                             f"got {probe_rho}")
        self.probe_gamma = float(probe_gamma)
        self.probe_rho = float(probe_rho)
        self.probe_rank = None if probe_rank is None else probe_rank
        self._probe_refresh = None if probe_refresh is None else int(probe_refresh)
        self.probe_sketch_rho = None if probe_sketch_rho is None else float(probe_sketch_rho)
        if self.probe_sketch_rho is not None and probe_rank is None:
            self.probe_sketch_rho = None
        if self.probe_sketch_rho is not None:
            if not (0.0 < self.probe_sketch_rho < 1.0):
                raise ValueError(f"probe_sketch_rho must be in (0, 1), got {probe_sketch_rho}")
        if probe_sketch_votes is not None or probe_sketch_min_votes is not None:
            warnings.warn("probe_sketch_votes / probe_sketch_min_votes are deprecated and "
                          "IGNORED: the vote-union sketch is THE sketch (boundary-sum deleted), "
                          "gate fixed at 2 votes (measured)", DeprecationWarning, stacklevel=2)
        if comp is not None:
            warnings.warn("ManasOptimizer(comp=...) is deprecated and IGNORED (u buffer removed: "
                          "no effect at toy or BiBo scale, ~1.2% tps + clone/shift memory cost; "
                          "see manas.py comment / git history <= e85af8b)", DeprecationWarning,
                          stacklevel=2)
        self.comp = None
        self._probe_on = False
        self._shift_on = False
        self._lazy = None
        self._pg = None
        self._probe_updates = 0
        self.probe_warmup_steps = int(probe_warmup_steps)
        self._manas_step = 0
        if rgd_tau:
            warnings.warn("ManasOptimizer(rgd_tau=...) is deprecated and IGNORED (measured no-op "
                          "at BiBo: batch-mean loss spread too small; git <= db41f11)",
                          DeprecationWarning, stacklevel=2)
        if cos_beta:
            warnings.warn("ManasOptimizer(cos_beta=...) is deprecated and IGNORED (BiBo: sharpen "
                          "neutral, novelty loses half the manas gain; git <= db41f11)",
                          DeprecationWarning, stacklevel=2)
        self.rgd_tau, self.cos_beta = None, 0.0
        if probe_norm not in ("global", "perparam"):
            raise ValueError(f"probe_norm must be 'global' or 'perparam', got {probe_norm!r}")
        self.probe_norm = probe_norm
        self.micro_vote = bool(micro_vote)
        self.probe_min_votes = int(probe_min_votes)
        if self.probe_min_votes < 1:
            raise ValueError(f"probe_min_votes must be >= 1, got {probe_min_votes}")
        self._votes_last = 0
        self.nexus_gamma = float(nexus_gamma)
        if self.micro_vote and self.probe_rank is None:
            if self.probe_rho_step is None:
                raise ValueError("full-rank micro_vote requires probe_rho_step (two-clock mode)")
            if self.nexus_gamma:
                raise ValueError("nexus_gamma requires a low-rank probe")
            if self._probe_gamma_intra is not None:
                raise ValueError("full-rank micro_vote has no separate fresh block: "
                                 "gamma_intra == gamma (leave probe_gamma_intra unset)")
        if self.nexus_gamma and not self.micro_vote:
            raise ValueError("nexus_gamma requires micro_vote=True")
        self._votes_cast = 0
        self._sketch_gate = 2

    @property
    def probe_refresh(self):
        if self._probe_refresh is not None:
            return self._probe_refresh
        rho = self.probe_rho_step if self.probe_rho_step is not None else self.probe_rho
        return max(2, round(2.0 / max(1.0 - rho, 1e-6)))

    @probe_refresh.setter
    def probe_refresh(self, v):
        self._probe_refresh = None if v is None else int(v)

    @property
    def probe_gamma_intra(self):
        return self.probe_gamma if self._probe_gamma_intra is None else self._probe_gamma_intra

    @probe_gamma_intra.setter
    def probe_gamma_intra(self, v):
        self._probe_gamma_intra = None if v is None else float(v)

    def _probe_params(self):
        return [p for g in self.param_groups for p in g["params"] if p.ndim in (2, 3)]

    def _ensure_groups(self):
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
            gq = torch.Generator(device="cpu").manual_seed(0x9A5 + numel)
            q0 = torch.linalg.qr(torch.randn(*lead, m, r, generator=gq).to(dev))[0].to(torch.float32)
            q_b = q0.unsqueeze(0).expand(G, *q0.shape).contiguous()
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
        st = self.state[p]
        if "manas_prev_g" not in st:
            st["manas_d"] = torch.zeros_like(p, dtype=torch.bfloat16)
            st["manas_prev_g"] = torch.zeros_like(p, dtype=torch.bfloat16)
        return st

    def _rank_for(self, m, n):
        r = self.probe_rank
        r = max(1, round(r * min(m, n))) if isinstance(r, float) else int(r)
        return min(r, m, n)

    def _lowrank_qc(self, p):
        st = self.state[p]
        if "manas_q" not in st:
            m, n = p.shape[-2], p.shape[-1]
            r = self._rank_for(m, n)
            lead = p.shape[:-2]
            g = torch.Generator(device="cpu").manual_seed(0x9A5 + p.numel())
            q = torch.linalg.qr(torch.randn(*lead, m, r, generator=g).to(p.device))[0]
            st["manas_q"] = q.to(torch.float32)
            st["manas_c"] = torch.zeros(*lead, r, n, device=p.device, dtype=torch.float32)
        return st["manas_q"], st["manas_c"]

    def _cnow(self, p):
        st = self.state[p]
        if "manas_cnow" not in st:
            _q, c = self._lowrank_qc(p)
            st["manas_cnow"] = torch.zeros_like(c)
        return st["manas_cnow"]

    def _coef_of(self, p):
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
        if self.probe_rank is None:
            if self.micro_vote:
                return self.probe_gamma * self._full_state(p)["manas_d"].to(torch.float32)
            return self._full_d(p)
        q, _c = self._lowrank_qc(p)
        return q @ self._coef_of(p)

    def _sketch_state(self, p):
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
        st = self._sketch_state(p)
        yp = gf @ st["manas_omega"]
        delta = yp - st["manas_prev_yp"]
        st["manas_prev_yp"].copy_(yp)
        n = torch.linalg.vector_norm(delta)
        inv = torch.where(torch.isfinite(n) & (n > 0), 1.0 / n, torch.zeros_like(n))
        st["manas_ynow"].add_(delta * inv)

    def _micro_state(self, p):
        st = self.state[p]
        _q, c = self._lowrank_qc(p)
        if "manas_prev_proj" not in st:
            st["manas_prev_proj"] = torch.zeros_like(c)
        if self.nexus_gamma and "manas_cs" not in st:
            st["manas_cs"] = torch.zeros_like(c)
        return st["manas_prev_proj"], st.get("manas_cs")

    def _probe_engaged(self):
        return (not self.micro_vote) or self._votes_last >= self.probe_min_votes

    @contextmanager
    def probe(self):
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
        if not self._probe_engaged():
            self._probe_on = True
            return
        ps = self._probe_params()
        if self.probe_rank is None:
            if self.micro_vote:
                if not self._lazy_ok():
                    raise RuntimeError("full-rank micro_vote requires fp32 weights (lazy shift)")
                if not self._shift_on:
                    for p in ps:
                        if "manas_prev_g" in self.state[p]:
                            p.add_(self.state[p]["manas_d"], alpha=self.probe_gamma)
                    self._shift_on = True
            elif not self._shift_on:
                ds = [self._full_d(p) for p in ps]
                if self._lazy_ok():
                    torch._foreach_add_(ps, ds)
                else:
                    for p, d in zip(ps, ds):
                        p.add_(d.to(p.dtype))
                self._shift_on = True
        elif self._lazy_ok():
            pg = self._ensure_groups()
            if self.micro_vote:
                self._pin_grads(pg)
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
                buf["p3"].baddbmm_(buf["q3"], d3)
            self._shift_on = True
        else:
            for p in ps:
                p.add_(self._d_of(p).to(p.dtype))
            self._shift_on = True
        self._probe_on = True

    @torch.no_grad()
    def remove_probe(self):
        if not self._probe_on:
            raise RuntimeError("probe not applied")
        self._probe_on = False
        if self._shift_on and self.probe_rank is not None and not self._lazy_ok():
            for p in self._probe_params():
                p.sub_(self._d_of(p).to(p.dtype))
            self._shift_on = False

    @torch.no_grad()
    def _restore_theta(self):
        if not self._shift_on:
            return
        ps = self._probe_params()
        if self.probe_rank is None:
            if self.micro_vote:
                for p in ps:
                    if "manas_prev_g" in self.state[p]:
                        p.add_(self.state[p]["manas_d"], alpha=-self.probe_gamma)
                self._shift_on = False
                return
            ds = [self._full_d(p) for p in ps]
            if self._lazy_ok():
                torch._foreach_sub_(ps, ds)
            else:
                for p, d in zip(ps, ds):
                    p.sub_(d.to(p.dtype))
        elif self._lazy_ok():
            for buf in self._ensure_groups():
                buf["p3"].baddbmm_(buf["q3"], buf["a3"], alpha=-1.0)
            self._flat["applied"].zero_()
        else:
            for p in ps:
                ap = self.state[p].get("manas_applied")
                if ap is not None:
                    q, _c = self._lowrank_qc(p)
                    p.sub_(q @ ap)
                    ap.zero_()
        self._shift_on = False

    @torch.no_grad()
    def vote(self):
        if not self.micro_vote:
            return
        if self._probe_on:
            raise RuntimeError("vote() must be called outside the probe() context")
        if not self._probe_engaged():
            self._votes_cast += 1
            return
        if self._manas_step < self.probe_warmup_steps:
            return
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        self._votes_cast += 1
        if self.probe_rank is None:
            deltas, dhist, norms, sync = [], [], [], []
            for p in ps:
                st = self._full_state(p)
                delta = torch.nan_to_num_(p.grad.to(torch.float32)
                                          - st["manas_prev_g"].to(torch.float32),
                                          nan=0.0, posinf=0.0, neginf=0.0)
                st["manas_prev_g"].copy_(p.grad)
                deltas.append(delta); dhist.append(st["manas_d"]); sync.append(p)
                norms.append(torch.linalg.vector_norm(delta))
            pn = torch.stack(norms)
            gn = pn if self.probe_norm == "perparam" else torch.linalg.vector_norm(pn)
            inv = torch.where(torch.isfinite(gn) & (gn > 0), 1.0 / gn, torch.zeros_like(gn))
            invs = [inv] * len(ps) if inv.ndim == 0 else list(inv.unbind())
            for d, delta, iv in zip(dhist, deltas, invs):
                d.sub_((delta * iv).to(torch.bfloat16))
            if self._shift_on and self.probe_gamma:
                torch._foreach_addcmul_(sync, deltas, invs, value=-self.probe_gamma)
            return
        fresh = self.probe_rho_step is not None
        sk = self.probe_sketch_rho is not None
        nxs = (self.nexus_gamma / self.probe_gamma) if self.nexus_gamma else 0.0
        if len(ps) == len(self._probe_params()):
            pg = self._ensure_groups()
            fl = self._flat
            pnorms = []
            for buf in pg:
                gv = buf.get("gviews")
                if gv is not None and all(p.grad is gv[i]
                                          for i, p in enumerate(buf["params"])):
                    gf3 = buf["g3"]
                else:
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
                                      nan=0.0, posinf=0.0, neginf=0.0)
            fl["prev_proj"].copy_(fl["projpad"])
            for buf in pg:
                dv = delta[buf["co"]:buf["co"] + buf["csz"]].view(buf["G"], -1)
                pnorms.append(torch.linalg.vector_norm(dv, dim=1))
            if self.probe_norm != "perparam":
                gn = torch.linalg.vector_norm(torch.cat(pnorms))
                iv0 = self.probe_gamma / gn
                iv0 = torch.where(torch.isfinite(iv0) & (gn > 0), iv0, torch.zeros_like(iv0))
                if fresh:
                    iv0 = iv0 / self.probe_gamma
                tgt = fl["cnow"] if fresh else fl["c"]
                if self.probe_rho != 1.0:
                    tgt.mul_(self.probe_rho)
                tgt.addcmul_(delta, iv0, value=-1.0)
                if nxs:
                    fl["cs"].addcmul_(delta, iv0 * nxs, value=-1.0)
                return
            for j, buf in enumerate(pg):
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

    @torch.no_grad()
    def step(self, closure=None, probe_loss=None):
        if self._probe_on:
            raise RuntimeError("remove_probe() (or exit the probe() context) before step()")
        self._manas_step += 1
        self._restore_theta()
        loss = super().step(closure)
        if self.micro_vote:
            self._finish_micro_step()
        else:
            self._update_probe()
        return loss

    @torch.no_grad()
    def _finish_micro_step(self):
        if not self._probe_engaged():
            self._votes_last = self._votes_cast
            self._votes_cast = 0
            return
        self._votes_last = self._votes_cast
        if self.probe_rank is None:
            sts = [self.state[p] for p in self._probe_params()
                   if "manas_prev_g" in self.state[p]]
            if sts:
                torch._foreach_mul_([s["manas_d"] for s in sts], self.probe_rho_step)
                torch._foreach_zero_([s["manas_prev_g"] for s in sts])
            self._votes_cast = 0
            return
        if self._votes_cast == 0:
            if not getattr(self, "_warned_no_votes", False):
                warnings.warn("micro_vote=True but no vote() was cast this step; falling back "
                              "to step-voting (call opt.vote() after each micro-batch backward)")
                self._warned_no_votes = True
            self._update_probe()
            if self.probe_rho_step is not None:
                for p in self._probe_params():
                    if "manas_c" in self.state[p]:
                        self.state[p]["manas_c"].mul_(self.probe_rho_step)
        else:
            refresh = self._probe_updates % max(self.probe_refresh, 1) == 0
            self._probe_updates += 1
            ps = self._probe_params()
            sk = self.probe_sketch_rho is not None
            fl = getattr(self, "_flat", None) if self._pg is not None else None
            if self.probe_rho_step is not None:
                if fl is not None and "cnow" in fl:
                    fl["c"].mul_(self.probe_rho_step)
                    fl["c"].add_(fl["cnow"])
                    fl["cnow"].zero_()
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
                fl["y"].mul_(self.probe_sketch_rho)
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
                for buf in self._pg:
                    q_new = self._cholqr2(buf["y"])
                    c_new = (q_new.mT @ buf["q"]) @ buf["c"]
                    buf["c"].copy_(c_new)
                    buf["q"].copy_(q_new)
            elif develop and gps:
                groups = {}
                for p in gps:
                    groups.setdefault(p.shape, []).append(p)
                for shape, grp in groups.items():
                    y_stk = torch.stack([self.state[p]["manas_y"] for p in grp])
                    q_stk = torch.stack([self.state[p]["manas_q"] for p in grp])
                    c_stk = torch.stack([self.state[p]["manas_c"] for p in grp])
                    q_new = torch.linalg.qr(y_stk)[0]
                    c_new = (q_new.mT @ q_stk) @ c_stk
                    for i, p in enumerate(grp):
                        self.state[p]["manas_c"].copy_(c_new[i])
                        self.state[p]["manas_q"].copy_(q_new[i])
            elif refresh:
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
                fl["prev_proj"].zero_()
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
        if self._manas_step <= self.probe_warmup_steps:
            return
        pn = torch.stack([torch.linalg.vector_norm(p.grad, dtype=torch.float32) for p in ps])
        if self.probe_norm == "perparam":
            gn = pn
        else:
            gn = torch.linalg.vector_norm(pn)
        inv = self.probe_gamma / gn
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        if self.probe_rho_step is not None:
            inv = inv / self.probe_gamma
        refresh = (self.probe_rank is not None
                   and self._probe_updates % max(self.probe_refresh, 1) == 0)
        self._probe_updates += 1
        if self.probe_rank is None:
            ds = [self._full_d(p) for p in ps]
            torch._foreach_mul_(ds, self.probe_rho)
            for i, (p, d) in enumerate(zip(ps, ds)):
                g32 = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
                d.addcmul_(g32, inv if inv.ndim == 0 else inv[i], value=-1.0)
            return
        for i, p in enumerate(ps):
            q, c = self._lowrank_qc(p)
            gf = torch.nan_to_num(p.grad, nan=0.0, posinf=0.0, neginf=0.0).to(torch.float32)
            y = None
            if refresh:
                if y is None:
                    r = q.shape[-1]
                    omega = torch.randn(*p.shape[:-2], p.shape[-1], r, device=p.device)
                    y = gf @ omega
                q_new = torch.linalg.qr(y)[0]
                c.copy_((q_new.mT @ q) @ c)
                q.copy_(q_new)
            c.mul_(self.probe_rho)
            c.addcmul_(q.mT @ gf, inv if inv.ndim == 0 else inv[i], value=-1.0)


if __name__ == "__main__":
    torch.manual_seed(0)
    _orig_init = ManasOptimizer.__init__
    def _test_init(self, *a, **k):
        _orig_init(self, *a, **k)
        self._votes_last = 9
    ManasOptimizer.__init__ = _test_init
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0, probe_rho_step=0.9)
    o._votes_last = 0
    p0 = p.detach().clone()
    o._manas_step = 1
    o.apply_probe()
    assert torch.allclose(p, p0) and not o._shift_on, "disengaged probe must not touch theta"
    p.grad = torch.randn(32, 16); o.remove_probe(); o.vote()
    assert "manas_q" not in o.state[p], "disengaged vote must not allocate probe state"
    o._finish_micro_step()
    assert o._votes_last == 1 and not o._probe_engaged(), "ga1: stays pure muon"
    o.apply_probe(); p.grad = torch.randn(32, 16); o.remove_probe(); o.vote()
    p.grad = p.grad + torch.randn(32, 16); o.vote()
    o._finish_micro_step()
    assert o._votes_last == 2 and o._probe_engaged(), "ga>=2 must engage on the next step"
    for mode, expect_equal in (("perparam", True), ("global", False)):
        pa, pb = torch.nn.Parameter(torch.randn(32, 16)), torch.nn.Parameter(torch.randn(32, 16))
        o = ManasOptimizer([pa, pb], probe_rank=None, probe_norm=mode)
        pa.grad = torch.randn(32, 16); pb.grad = 100.0 * torch.randn(32, 16)
        o._manas_step += 1; o._update_probe()
        na, nb = o._full_d(pa).norm().item(), o._full_d(pb).norm().item()
        equal = abs(na - nb) / max(na, nb) < 0.05
        assert equal == expect_equal, f"{mode}: vote norms {na:.4f} vs {nb:.4f}"
    pa, pb = torch.nn.Parameter(torch.randn(32, 16)), torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([pa, pb], probe_rank=None, probe_norm="perparam")
    pa.grad = torch.full((32, 16), float("nan")); pb.grad = torch.randn(32, 16)
    o._manas_step += 1; o._update_probe()
    assert o._full_d(pa).norm() == 0 and o._full_d(pb).norm() > 0, "nan matrix must not veto healthy votes"
    for kwargs, attr, off in ((dict(comp=1.0), "comp", None),
                              (dict(rgd_tau=3.0), "rgd_tau", None),
                              (dict(cos_beta=0.5), "cos_beta", 0.0)):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            o = ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], **kwargs)
        assert getattr(o, attr) == off and any(issubclass(w.category, DeprecationWarning) for w in caught), \
            f"{kwargs} must warn DeprecationWarning and be ignored"
    o.step(probe_loss=2.0)
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, nexus_gamma=0.04)
    p.grad = torch.randn(32, 16); o.vote()
    p.grad = p.grad + torch.randn(32, 16); o.vote()
    assert o._votes_cast == 2
    q, c = o._lowrank_qc(p); cs = o.state[p]["manas_cs"]
    assert c.norm() > 0 and cs.norm() > 0, "both consensus and walker must receive votes"
    assert torch.allclose(o._d_of(p), q @ (c + cs)), "offset must include the walker"
    c_before = c.clone()
    o.vote()
    assert torch.allclose(c, 0.98 * c_before, atol=1e-7), "zero delta must only decay the consensus"
    o._finish_micro_step()
    assert cs.norm() == 0 and o.state[p]["manas_prev_proj"].norm() == 0 and o._votes_cast == 0, \
        "walker and prev-projection must reset at the step boundary"
    try:
        o.apply_probe(); o.vote(); raise AssertionError("vote() inside probe must raise")
    except RuntimeError:
        o.remove_probe()
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
    o._probe_updates = 1
    o._finish_micro_step()
    assert cn.norm() == 0 and torch.allclose(c, cn_end, atol=1e-7), \
        "boundary must fold the raw block at coeff 1 and zero it"
    c_hist = c.clone()
    p.grad = torch.randn(32, 16); o.vote()
    cn2 = cn.clone()
    o._finish_micro_step()
    assert torch.allclose(c, 0.9 * c_hist + cn2, atol=1e-7), \
        "second boundary: history decays rho_step, new raw block folds at coeff 1"
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0, probe_rho_step=0.9,
                       probe_sketch_rho=0.8, probe_refresh=1000)
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
    pmx_a = torch.nn.Parameter(torch.randn(32, 16))
    pmx_b = torch.nn.Parameter(torch.randn(32, 24))
    omx = ManasOptimizer([pmx_a, pmx_b], probe_rank=4, micro_vote=True, probe_rho=1.0,
                         probe_rho_step=0.9, probe_sketch_rho=0.8)
    for _ in range(2):
        pmx_a.grad = (torch.randn(32, 16) if pmx_a.grad is None
                      else pmx_a.grad + torch.randn(32, 16))
        pmx_b.grad = (torch.randn(32, 24) if pmx_b.grad is None
                      else pmx_b.grad + torch.randn(32, 24))
        omx.vote()
    omx._finish_micro_step()
    for pp, nn in ((pmx_a, 16), (pmx_b, 24)):
        qx, cx = omx._lowrank_qc(pp)
        assert qx.shape == (32, 4) and cx.shape == (4, nn) and \
            torch.allclose(qx.mT @ qx, torch.eye(4), atol=1e-5), "mixed-shape develop broken"
    q_before = q.clone()
    p.grad = torch.randn(32, 16); o.vote()
    o._finish_micro_step()
    assert torch.allclose(o._lowrank_qc(p)[0], q_before), \
        "1-vote step: gate holds, window untouched (Y stays warm)"
    o2 = ManasOptimizer([torch.nn.Parameter(torch.randn(32, 16))], probe_rank=4, micro_vote=True,
                        probe_rho=1.0, probe_rho_step=0.9, probe_sketch_rho=0.8)
    p2 = o2._probe_params()[0]
    p2.grad = torch.randn(32, 16); o2.vote()
    p2.grad = p2.grad + torch.randn(32, 16); o2.vote()
    o2._finish_micro_step()
    assert torch.allclose(o2.state[p2]["manas_omega"], om), \
        "omega must be deterministic per shape (seeded)"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], probe_rank=4, micro_vote=True,
                       probe_rho=1.0, probe_rho_step=0.9, probe_sketch_votes=True,
                       probe_sketch_min_votes=1)
    assert any(issubclass(w.category, DeprecationWarning) for w in caught), \
        "probe_sketch_votes/min_votes must warn DeprecationWarning"
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.08)
    p0 = p.detach().clone()
    o._manas_step = 1
    o.apply_probe()
    assert torch.allclose(p, p0), "empty consensus: probe must not move theta"
    p.grad = torch.randn(32, 16); o.remove_probe(); o.vote()
    o.apply_probe()
    d1 = o._d_of(p)
    assert torch.allclose(p - p0, d1, atol=1e-6), "weights must hold theta + current d"
    p.grad = p.grad + torch.randn(32, 16); o.remove_probe(); o.vote()
    o.apply_probe()
    assert torch.allclose(p - p0, o._d_of(p), atol=1e-6), "lazy delta must track d exactly"
    o.remove_probe()
    o._restore_theta()
    assert torch.allclose(p, p0, atol=1e-5), "restore must return clean theta"
    assert o.state[p]["manas_applied"].norm() == 0 and not o._shift_on
    pf = torch.nn.Parameter(torch.randn(4, 16))
    pl = torch.nn.Parameter(pf.detach().clone())
    of = ManasOptimizer([pf], probe_rank=None, micro_vote=True, probe_rho=1.0,
                        probe_rho_step=0.9, probe_gamma=0.08)
    ol = ManasOptimizer([pl], probe_rank=4, micro_vote=True, probe_rho=1.0,
                        probe_rho_step=0.9, probe_gamma=0.08 * 0.9, probe_gamma_intra=0.08,
                        probe_sketch_rho=None)
    of._probe_updates = ol._probe_updates = 1
    gseq = [torch.randn(4, 16) for _ in range(3)]
    for o_, p_ in ((of, pf), (ol, pl)):
        p_.grad = gseq[0].clone(); o_.vote()
        p_.grad = p_.grad + gseq[1]; o_.vote()
    assert torch.allclose(of._d_of(pf), ol._d_of(pl), rtol=0.02, atol=1e-3), \
        "full-rank must equal dose-renormalized low-rank at complete rank (in-step)"
    of._finish_micro_step(); ol._finish_micro_step()
    for o_, p_ in ((of, pf), (ol, pl)):
        p_.grad = gseq[2].clone(); o_.vote()
    assert torch.allclose(of._d_of(pf), ol._d_of(pl), rtol=0.02, atol=1e-3), \
        "full-rank must equal dose-renormalized low-rank across the boundary"
    st_f = of.state[pf]
    assert st_f["manas_d"].dtype == torch.bfloat16 \
        and st_f["manas_prev_g"].dtype == torch.bfloat16 and st_f["manas_d"].norm() > 0, \
        "full-rank: D live, both buffers bf16"
    p = torch.nn.Parameter(torch.randn(6, 8))
    o = ManasOptimizer([p], probe_rank=None, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.05)
    p0 = p.detach().clone()
    o._manas_step = 1
    o.apply_probe()
    assert torch.allclose(p, p0) and o._shift_on, "empty consensus: shift flagged, theta still"
    p.grad = torch.randn(6, 8); o.remove_probe(); o.vote()
    assert torch.allclose(p - p0, o._d_of(p), atol=1e-3), "vote must sync theta to current d"
    o.apply_probe()
    p.grad = p.grad + torch.randn(6, 8); o.remove_probe(); o.vote()
    assert torch.allclose(p - p0, o._d_of(p), atol=1e-3), "second vote must keep theta synced"
    o._restore_theta()
    assert torch.allclose(p, p0, atol=1e-3) and not o._shift_on, "restore must return clean theta"
    d_pre = o.state[p]["manas_d"].clone()
    o._finish_micro_step()
    st = o.state[p]
    assert st["manas_prev_g"].norm() == 0 \
        and torch.allclose(st["manas_d"].float(), 0.9 * d_pre.float(), rtol=0.01), \
        "boundary: D decays rho_step, snapshot resets"
    o_auto = ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], probe_rank=4,
                            micro_vote=True, probe_rho=1.0, probe_rho_step=0.96)
    assert o_auto.probe_refresh == 50, f"rho_step 0.96 must auto-refresh at 50, got {o_auto.probe_refresh}"
    o_auto.probe_rho_step = 0.98
    assert o_auto.probe_refresh == 100, "auto refresh must track rho live"
    assert ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))],
                          probe_rho=0.98).probe_refresh == 100, "per-vote clock: 2/(1-rho)"
    assert ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))],
                          probe_refresh=200).probe_refresh == 200, "explicit refresh must win"
    p = torch.nn.Parameter(torch.randn(32, 16))
    o = ManasOptimizer([p], probe_rank=4, micro_vote=True, probe_rho=1.0,
                       probe_rho_step=0.9, probe_gamma=0.08, probe_gamma_intra=0.0, probe_sketch_rho=None)
    p.grad = torch.randn(32, 16); o.vote()
    q, c = o._lowrank_qc(p)
    assert o.state[p]["manas_cnow"].norm() > 0 and torch.allclose(o._d_of(p), q @ (0.08 * c)), \
        "gamma_intra=0 must probe history-only while the block still accumulates"
    o._probe_updates = 1; o._finish_micro_step()
    assert c.norm() > 0, "gamma_intra=0 block must still fold into history at gamma"
    for bad in (dict(probe_rho=1.0),
                dict(probe_rho_step=0.9),
                dict(probe_gamma_intra=0.1),
                dict(probe_rank=4, micro_vote=True, probe_rho_step=1.0),
                dict(probe_rank=None, micro_vote=True),
                dict(probe_rank=None, micro_vote=True, probe_rho=1.0,
                     probe_rho_step=0.9, nexus_gamma=0.1),
                dict(probe_rank=None, micro_vote=True, probe_rho=1.0,
                     probe_rho_step=0.9, probe_gamma_intra=0.2),
                dict(nexus_gamma=0.1)):
        try:
            ManasOptimizer([torch.nn.Parameter(torch.randn(8, 4))], **bad)
            raise AssertionError(f"constructor must reject {bad}")
        except ValueError:
            pass
    print("manas perparam+microvote+nexus self-check PASS")
