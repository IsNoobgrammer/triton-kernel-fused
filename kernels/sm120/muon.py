import torch

from kernels.sm75.muon import newton_schulz, _PE_COEFFS, _DSV4_COEFFS
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75
from kernels.sm120.ns_router import NSRouter, ORDER as NS_BACKENDS

NS_BATCH_ELEMS = 8 * 1024 * 1024


class FusedMuon(_FusedMuon75):
    """sm75 FusedMuon with the Blackwell Newton-Schulz backends. Only `_polar` changes: the step
    loop, every variant and every decay mode are the shared sm75 code.

    ns_backend  "auto" (default): per shape bucket, the fastest backend whose error vs an fp32 NS is
                within ns_tol x cuBLAS's, decided once on the first step (see ns_router.py).
                Or force one everywhere: "cublas" (no Triton) | "epi" | "symmul" | "gram".
    ns_tol      auto only: allowed error ratio vs cuBLAS. 1.05 keeps cuBLAS-level accuracy (gram is
                rejected at small shapes); raise it to let gram in where it is faster.
    ns_pinned   auto only: {shape: backend} overrides, e.g. to reproduce a logged run exactly.
    """

    DEFAULT_NS_DTYPE = torch.bfloat16

    def __init__(self, *args, ns_backend="auto", gram_restarts=None, ns_tol=1.05, ns_pinned=None, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)
        if ns_backend != "auto" and ns_backend not in NS_BACKENDS:
            raise ValueError(f"ns_backend must be 'auto' or one of {NS_BACKENDS}, got {ns_backend!r}")
        self.ns_backend = ns_backend
        cands = NS_BACKENDS if ns_backend == "auto" else (ns_backend,)
        self.ns_router = NSRouter(self.coeffs, self.ns_dtype, candidates=cands, tol=ns_tol,
                                  gram_restarts=gram_restarts, pinned=ns_pinned)
        self._ns_fixed = None if ns_backend == "auto" else self.ns_router.fns[ns_backend]

    def _polar(self, u):
        return self._ns_fixed(u) if self._ns_fixed is not None else self.ns_router(u)

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
