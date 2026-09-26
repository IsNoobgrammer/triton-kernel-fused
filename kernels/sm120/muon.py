import torch

from kernels.sm75.muon import newton_schulz, _PE_COEFFS, _DSV4_COEFFS
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75
from kernels.sm120.ns_router import NSRouter, FAMILIES as NS_BACKENDS
from kernels.sm120.newton_schulz_gram import GRAM_RESTART_AT
from kernels.sm120.muon_tail import tail_pre, tail_post, muown_pre, muown_post
from kernels.muon import muon_scaling as _scaling

NS_BATCH_ELEMS = 8 * 1024 * 1024


class FusedMuon(_FusedMuon75):
    """sm75 FusedMuon with the Blackwell Newton-Schulz backends. Only `_polar` changes: the step
    loop, every variant and every decay mode are the shared sm75 code.

    ns_backend  "auto" (default): per shape bucket, the fastest backend whose error vs an fp32 NS is
                within ns_tol x the best exact backend's, probed over the first ns_probe_steps (10)
                optimizer steps -- training applies the cuBLAS result meanwhile, i.e. exactly the
                pre-router numerics -- including every gram restart placement; see ns_router.py.
                Or force one everywhere: "cublas" (no Triton) | "epi" | "symepi" | "symmul" | "gram".
    gram_restarts  auto: None = search all placements, or pin one; forced gram: None = (4, 6).
    ns_tol      auto only: allowed error ratio vs cuBLAS. 1.05 keeps cuBLAS-level accuracy (gram is
                rejected at small shapes); raise it to let gram in where it is faster.
    ns_pinned   auto only: {shape: backend} overrides, e.g. to reproduce a logged run exactly.
    """

    DEFAULT_NS_DTYPE = torch.bfloat16

    def __init__(self, *args, ns_backend="auto", gram_restarts=None, ns_tol=1.05, ns_pinned=None,
                 ns_probe_steps=10, fused_tail=True, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)
        self._fused_tail = bool(fused_tail)     # muon_tail.py: bit-identical to the eager tail (12/12 configs)
        if ns_backend != "auto" and ns_backend not in NS_BACKENDS:
            raise ValueError(f"ns_backend must be 'auto' or one of {NS_BACKENDS}, got {ns_backend!r}")
        self.ns_backend = ns_backend
        cands = NS_BACKENDS if ns_backend == "auto" else (ns_backend,)
        if ns_backend == "gram" and gram_restarts is None:
            gram_restarts = GRAM_RESTART_AT
        self.ns_router = NSRouter(self.coeffs, self.ns_dtype, candidates=cands, tol=ns_tol,
                                  gram_restarts=gram_restarts, pinned=ns_pinned, probe_steps=ns_probe_steps)
        self._ns_fixed = None if ns_backend == "auto" else self.ns_router.fixed(ns_backend)

    @staticmethod
    def _tail_pre(grads, gbuf, mom, momentum, nesterov):
        return tail_pre(grads, gbuf, mom, momentum, nesterov)

    @staticmethod
    def _tail_post(p3, o3, alpha, decay):
        tail_post(p3, o3, alpha, decay)

    def _muown_chunk(self, g, members, start, crows, mom_c, lr, momentum, nesterov, wd):
        # Same math as the sm75 eager chunk in two Triton passes around NS (muon_tail.py).
        if not self._fused_tail:
            return super()._muown_chunk(g, members, start, crows, mom_c, lr, momentum, nesterov, wd)
        r, c = g["r"], g["c"]
        var = self.variant
        st = _scaling.slice_state(self.state[g["anchor"]]["variant"], start, crows)
        dg = torch.empty((crows, r), device=mom_c.device, dtype=torch.float32)
        gbuf = torch.empty((crows, r, c), device=mom_c.device, dtype=self.ns_dtype)
        for p, o, n in members:
            muown_pre(p.view(n, r, c), p.grad.reshape(n, r, c), _scaling.slice_state(st, o, n), dg[o:o + n],
                      mom_c[o:o + n], gbuf[o:o + n], momentum, nesterov)
        out = self._polar(gbuf if nesterov else mom_c)
        step_a = -lr * _scaling.RMS_TARGET * max(r, c) ** 0.5 if self.scale == "adam" else -lr
        for p, o, n in members:
            muown_post(p.view(n, r, c), out[o:o + n], _scaling.slice_state(st, o, n), dg[o:o + n], step_a,
                       lr * _scaling.MUOWN_GAIN_LR_MULT,
                       var.betas, var.eps, self._step_count, lr * wd)

    def _polar(self, u):
        if self._ns_fixed is not None:
            return self._ns_fixed(u)
        self.ns_router.step = self._step_count
        return self.ns_router(u)

    def _compute(self, work, decay):
        for params, f in decay:
            torch._foreach_mul_(params, f)
        for w in work:
            mom_c, gbuf = w["mom_c"], w["gbuf"]
            mom_c.mul_(w["momentum"]).add_(gbuf)
            u = gbuf.add_(mom_c, alpha=w["momentum"]) if w["nesterov"] else mom_c
            out = self._polar(u)
            torch._foreach_add_(w["out_params"],
                                [out[o:o + n].reshape(p.shape) for p, o, n in w["members"]], alpha=w["alpha"])


class DistributedMuon(_DistributedMuon75):

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)


AmalgamatedMuon = FusedMuon
