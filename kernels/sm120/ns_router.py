"""Per-shape Newton-Schulz backend routing, decided once on real momentum.

No backend wins every shape (measured on an RTX PRO 6000, ns8 bf16):
  small side 512, ratio <= 1.5     epi       (cuBLAS X X^T + fused-epilogue Triton GEMMs)
  small side >= 1024, square       symmul    (symmetric X X^T, half the FLOPs)
  small side >= 2048, ratio >= 1.5 gram      (Gram-space NS: ~4 large GEMMs instead of 16)
and gram's ERROR also depends on the shape (2x cuBLAS's at 512, equal at 4096). So FusedMuon routes
per shape bucket: the first time a (shape, dtype) reaches `NSRouter.__call__`, every candidate runs
on that actual momentum chunk, is timed, and is scored against an fp32 NS of the same input. The
fastest one whose error is within `tol` x cuBLAS's wins, and the choice is cached for the rest of the
run. A training run therefore mixes backends: epi on the MoE stacks, gram on a big tall matrix,
cuBLAS on tiny ones.

Numerics are only a function of the choice, so the decision rule is made hard to flip:
  - candidates within `margin` of the fastest tie, and a tie goes to the EARLIEST in `order` --
    cublas, epi, symmul are listed before gram, and cublas/epi are bit-identical;
  - `pinned` forces a backend for a shape (e.g. to reproduce a previous run exactly);
  - `table` holds every decision with its timings and errors for logging.
"""
import torch

from kernels.sm75.muon import newton_schulz as _cublas
from kernels.sm120.newton_schulz_epi import newton_schulz_epi
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul
from kernels.sm120.newton_schulz_gram import newton_schulz_gram

ORDER = ("cublas", "epi", "symmul", "gram")


def _backends(coeffs, ns_dtype, gram_restarts):
    rs = {} if gram_restarts is None else {"restart_at": gram_restarts}
    return {
        "cublas": lambda u: _cublas(u, coeffs, ns_dtype),
        "epi": lambda u: newton_schulz_epi(u, coeffs, ns_dtype),
        "symmul": lambda u: newton_schulz_symmul(u, coeffs, ns_dtype, force_eager=True, min_dim=0),
        "gram": lambda u: newton_schulz_gram(u, coeffs, ns_dtype, force_eager=True, min_dim=0,
                                             min_ratio=0.0, **rs),
    }


def _time(fn, u, reps=5, warm=2):
    for _ in range(warm):
        fn(u)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(u); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    ts.sort()
    return ts[reps // 2]


class NSRouter:
    def __init__(self, coeffs, ns_dtype, candidates=ORDER, tol=1.05, margin=0.03, gram_restarts=None,
                 pinned=None, verbose=True):
        bad = set(candidates) - set(ORDER)
        if bad:
            raise ValueError(f"unknown NS backend(s) {sorted(bad)}; choose from {ORDER}")
        self.fns = _backends(coeffs, ns_dtype, gram_restarts)
        self.coeffs, self.candidates = coeffs, [c for c in ORDER if c in candidates]
        self.tol, self.margin, self.verbose = float(tol), float(margin), verbose
        self.pinned = dict(pinned or {})
        self.choice, self.table = {}, {}

    def __call__(self, u):
        key = (tuple(u.shape), u.dtype)
        c = self.choice.get(key)
        if c is None:
            c = self.choice[key] = self._decide(u, key)
        return self.fns[c](u)

    @torch.no_grad()
    def _decide(self, u, key):
        shape = key[0]
        if shape in self.pinned:
            self.table[key] = {"choice": self.pinned[shape], "pinned": True}
            return self.pinned[shape]
        o32 = _cublas(u.float(), self.coeffs, torch.float32)
        n32 = o32.norm()
        ref = self.fns["cublas"](u)
        rows = {}
        for name in self.candidates:
            try:
                o = self.fns[name](u)
                rows[name] = {"ms": _time(self.fns[name], u),
                              "rel": ((o.float() - o32).norm() / n32).item(),
                              "bit": bool(torch.equal(o, ref))}
            except Exception as ex:                    # a candidate that cannot run here just loses
                rows[name] = {"ms": float("inf"), "rel": float("inf"), "bit": False, "err": repr(ex)[:80]}
        base = rows["cublas"]["rel"] if "cublas" in rows else min(r["rel"] for r in rows.values())
        ok = [n for n in self.candidates if rows[n]["rel"] <= self.tol * base]
        fastest = min(rows[n]["ms"] for n in ok)
        pick = next(n for n in ok if rows[n]["ms"] <= fastest * (1 + self.margin))
        self.table[key] = {"choice": pick, **{n: rows[n] for n in rows}}
        if self.verbose:
            cells = "  ".join(f"{n} {r['ms']:.2f}ms rel {r['rel']:.2e}{' =cuBLAS' if r['bit'] else ''}"
                              f"{'' if n in ok else ' REJECTED'}" for n, r in rows.items())
            print(f"[ns-router] {shape} -> {pick}   | {cells}", flush=True)
        return pick
