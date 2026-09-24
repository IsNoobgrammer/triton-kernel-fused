import torch

from kernels.sm75.muon import newton_schulz, _PE_COEFFS, _DSV4_COEFFS
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul
from kernels.sm120.newton_schulz_gram import newton_schulz_gram

NS_BATCH_ELEMS = 8 * 1024 * 1024


class FusedMuon(_FusedMuon75):
    """sm75 FusedMuon with the Blackwell Newton-Schulz backends. Only `_polar` changes: the step
    loop, every variant and every decay mode are the shared sm75 code."""

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
        # use_symmul=False is the documented "no Triton at all" switch, so it must reach every
        # variant (it used to leave aurora on the gram kernel).
        if not self.use_symmul:
            return newton_schulz(u, self.coeffs, self.ns_dtype)
        return self._ns(u)

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
