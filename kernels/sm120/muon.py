import torch

from kernels.sm75.muon import newton_schulz, _PE_COEFFS, _DSV4_COEFFS
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul
from kernels.sm120.newton_schulz_gram import newton_schulz_gram
from kernels.muon import muon_scaling as _scaling

NS_BATCH_ELEMS = 8 * 1024 * 1024


class FusedMuon(_FusedMuon75):

    DEFAULT_NS_DTYPE = torch.bfloat16

    def __init__(self, *args, use_symmul=True, use_gram=True, gram_restarts=None, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)
        self.use_symmul = use_symmul
        self.use_gram = use_gram
        self.gram_restarts = gram_restarts

    def _ns(self, u, force_eager=False):
        if self.use_gram:
            kw = {} if self.gram_restarts is None else {"restart_at": self.gram_restarts}
            return newton_schulz_gram(u, self.coeffs, self.ns_dtype, force_eager=force_eager, **kw)
        return newton_schulz_symmul(u, self.coeffs, self.ns_dtype, force_eager=force_eager)

    def _polar(self, u):
        return self._ns(u)

    @torch.no_grad()
    def step(self, closure=None):
        _eager_mode = (_scaling.is_perrow(self.scale_mode) or _scaling.is_aurora(self.scale_mode)
                       or _scaling.is_aurora_ema(self.scale_mode))
        _sm75_only = self.spectral_wd > 0
        if not self.use_symmul or _sm75_only or (self.use_graph and not _eager_mode):
            return super().step(closure)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._xorth_step += 1
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            plan = self._plan(group, params)
            cautious = self.cautious_decay and wd != 0
            if wd != 0 and not cautious:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            perrow = _scaling.is_perrow(self.scale_mode)
            aurora = _scaling.is_aurora(self.scale_mode)
            aurora_ema = _scaling.is_aurora_ema(self.scale_mode)
            xp = group["xorth_post"]
            do_xorth = xp > 0 and self._xorth_step > self.xorth_warmup_steps
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                v_all = self.state[g["anchor"]].get("scale_v")
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:
                    mom_c = mom[start:start + crows]
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c
                    if do_xorth and self.xorth_where == "pre":
                        if not nesterov:
                            u = u.clone()
                        self._whiten_chunk(u, members, r, c, xp)
                    if aurora:
                        out = _scaling.aurora_update(u, self._polar, K=self.aurora_k)
                    elif aurora_ema:
                        v_c = v_all[start:start + crows]
                        if self.scale_mode == "aurora_ema_v2":
                            out = _scaling.aurora_ema_v2_update(u, self._polar, v_c, K=self.aurora_k)
                        else:
                            out = _scaling.aurora_ema_update(u, self._polar, v_c)
                    else:
                        out = self._ns(u)
                        if perrow:
                            out = _scaling.apply_perrow(self.scale_mode, out, v_all[start:start + crows])
                    if do_xorth and self.xorth_where == "post":
                        self._whiten_chunk(out, members, r, c, xp)
                    _pl = [p for p, _, _ in members]
                    _ul = [out[o:o + n].reshape(p.shape) for p, o, n in members]
                    if cautious:
                        for _p, _u in zip(_pl, _ul):
                            _p.sub_(_p * ((_u * _p) < 0), alpha=lr * wd)
                    torch._foreach_add_(_pl, _ul,
                                        alpha=(-lr if (perrow or aurora or aurora_ema) else alpha))
        return loss

    def _compute(self, work, decay):
        if not self.use_symmul:
            return super()._compute(work, decay)
        for params, f in decay:
            torch._foreach_mul_(params, f)
        for w in work:
            mom_c, gbuf = w["mom_c"], w["gbuf"]
            mom_c.mul_(w["momentum"]).add_(gbuf)
            u = gbuf.add_(mom_c, alpha=w["momentum"]) if w["nesterov"] else mom_c
            out = self._ns(u, force_eager=True)
            torch._foreach_add_(w["out_params"],
                                [out[o:o + n].reshape(p.shape) for p, o, n in w["members"]], alpha=w["alpha"])


class DistributedMuon(_DistributedMuon75):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)


AmalgamatedMuon = FusedMuon
