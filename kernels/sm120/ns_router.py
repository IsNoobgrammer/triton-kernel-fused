"""Per-shape Newton-Schulz backend routing, decided over the first few steps on real momentum.

No backend wins every shape (measured on an RTX PRO 6000, ns8 bf16):
  small side 512, ratio <= 1.5     epi       (cuBLAS X X^T + fused-epilogue Triton GEMMs)
  small side >= 1024, square       symmul    (symmetric X X^T, half the FLOPs)
  small side >= 2048, ratio >= 1.5 gram      (Gram-space NS: ~4 large GEMMs instead of 16)
and gram's ERROR depends on the shape and on WHERE its restarts sit. So FusedMuon routes per shape
bucket. For the first `probe_steps` times a (shape, dtype) reaches the router, the cuBLAS result is
what the optimizer applies (so probing never changes training), and every candidate -- including
every gram restart placement in GRAM_PLACEMENTS -- is timed and scored against an fp32 NS of that
step's actual momentum. After the window, the fastest candidate whose mean error is within `tol` x
cuBLAS's is locked in for the rest of the run. A run therefore mixes backends per shape.

Decisions are made hard to flip, because numerics depend on them:
  - candidates within `margin` of the fastest tie, and a tie goes to the EARLIEST in ORDER
    (cublas / epi / symepi / symmul are bit-identical to cuBLAS where they are supported; gram is not);
  - `pinned` forces a backend for a shape (e.g. to reproduce a logged run exactly);
  - `table` holds every decision with its timings and errors for logging.
"""
import torch

from kernels.sm75.muon import newton_schulz as _cublas
from kernels.sm120.newton_schulz_epi import newton_schulz_epi
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul, SYMMUL_MIN_DIM
from kernels.sm120.newton_schulz_gram import newton_schulz_gram

FAMILIES = ("cublas", "epi", "symepi", "symmul", "gram")
# gram restart placements tried by "gram" (1-based iteration after which to re-orthogonalize; () = none)
GRAM_PLACEMENTS = ((), (2,), (3,), (4,), (5,), (6,), (2, 4), (3, 5), (4, 6), (2, 5), (3, 6), (4, 5), (2, 4, 6))


def _name(family, restarts=None):
    return family if family != "gram" else "gram@" + ",".join(map(str, restarts))


def _supported(name, shape):
    # symmul's own small-shape fallback used to drop out= (garbage below 2048, fixed Sep 24 2026);
    # its Triton kernels are correct at 512 now, but it only ever competes where it was validated.
    return not (name == "symmul" and min(shape[-2], shape[-1]) < SYMMUL_MIN_DIM)


def _backends(coeffs, ns_dtype, families, placements):
    fns = {}
    if "cublas" in families:
        fns["cublas"] = lambda u: _cublas(u, coeffs, ns_dtype)
    if "epi" in families:
        fns["epi"] = lambda u: newton_schulz_epi(u, coeffs, ns_dtype)
    if "symepi" in families:
        fns["symepi"] = lambda u: newton_schulz_epi(u, coeffs, ns_dtype, sym=True)
    if "symmul" in families:
        fns["symmul"] = lambda u: newton_schulz_symmul(u, coeffs, ns_dtype, force_eager=True, min_dim=0)
    if "gram" in families:
        for rs in placements:
            fns[_name("gram", rs)] = (lambda r: lambda u: newton_schulz_gram(
                u, coeffs, ns_dtype, force_eager=True, min_dim=0, min_ratio=0.0, restart_at=r))(rs)
    return fns


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
    def __init__(self, coeffs, ns_dtype, candidates=FAMILIES, tol=1.05, margin=0.03, gram_restarts=None,
                 pinned=None, probe_steps=3, verbose=True):
        bad = set(candidates) - set(FAMILIES)
        if bad:
            raise ValueError(f"unknown NS backend(s) {sorted(bad)}; choose from {FAMILIES}")
        placements = GRAM_PLACEMENTS if gram_restarts is None else (tuple(gram_restarts),)
        self.fns = _backends(coeffs, ns_dtype, set(candidates) | {"cublas"}, placements)
        self.order = [n for n in self.fns if n.split("@")[0] in candidates]
        self.coeffs, self.tol, self.margin = coeffs, float(tol), float(margin)
        self.probe_steps, self.verbose = max(1, int(probe_steps)), verbose
        self.pinned = dict(pinned or {})
        self.choice, self.table, self._stats = {}, {}, {}
        self.step = 0          # set by the optimizer each step: probing counts STEPS, not calls (many
                               # layers share a shape, so one step makes many calls per key)

    def fixed(self, family):
        """The single function a forced ns_backend uses (gram: its one configured placement)."""
        return next(self.fns[n] for n in self.fns if n.split("@")[0] == family)

    def __call__(self, u):
        key = (tuple(u.shape), u.dtype)
        c = self.choice.get(key)
        if c is not None:
            return self.fns[c](u)
        if key[0] in self.pinned:
            self.choice[key] = self.pinned[key[0]]
            self.table[key] = {"choice": self.pinned[key[0]], "pinned": True}
            return self.fns[self.choice[key]](u)
        seen = self._stats.setdefault(key, {}).setdefault("_seen", set())
        if self.step in seen:                        # this shape was already probed this step
            return self.fns["cublas"](u)
        seen.add(self.step)
        out = self._probe(u, key)                   # cuBLAS result during the probe window
        if len(seen) >= self.probe_steps:
            self.choice[key] = self._decide(key)
        return out

    @torch.no_grad()
    def _probe(self, u, key):
        st = self._stats.setdefault(key, {})
        o32 = _cublas(u.float(), self.coeffs, torch.float32)
        n32 = o32.norm()
        ref = self.fns["cublas"](u)
        for name in self.order:
            if name not in self.fns or not _supported(name, key[0]):
                continue
            s = st.setdefault(name, {"ms": [], "rel": [], "bit": True})
            try:
                o = ref if name == "cublas" else self.fns[name](u)
                s["rel"].append(((o.float() - o32).norm() / n32).item())
                s["bit"] &= bool(torch.equal(o, ref))
                s["ms"].append(_time(self.fns[name], u))
            except Exception as ex:                    # a candidate that cannot run here just loses
                s["rel"].append(float("inf")); s["ms"].append(float("inf")); s["bit"] = False
                s["err"] = repr(ex)[:80]
        return ref

    def _decide(self, key):
        rows = {n: {"ms": min(s["ms"]), "rel": sum(s["rel"]) / len(s["rel"]), "bit": s["bit"]}
                for n, s in self._stats.pop(key).items() if n != "_seen"}
        # tol is relative to the BEST of the exact-arithmetic backends, not to cuBLAS alone: on a single
        # (1, 2048, 6144) matrix cuBLAS's own error was 1.25e-2 against epi's 5.49e-3, which loosened the
        # bar enough to admit a 1.35e-2 gram. gram is the only candidate that changes the algorithm.
        base = min(r["rel"] for n, r in rows.items() if not n.startswith("gram"))
        ok = [n for n in self.order if n in rows and rows[n]["rel"] <= self.tol * base]
        fastest = min(rows[n]["ms"] for n in ok)
        pick = next(n for n in ok if rows[n]["ms"] <= fastest * (1 + self.margin))
        self.table[key] = {"choice": pick, **rows}
        if self.verbose:
            gram = {n: r for n, r in rows.items() if n.startswith("gram")}
            bestg = min(gram, key=lambda n: rows[n]["ms"] if n in ok else float("inf")) if gram else None
            show = [n for n in rows if not n.startswith("gram")] + ([bestg] if bestg else [])
            if gram and bestg not in ok:
                bestg = min(gram, key=lambda n: rows[n]["rel"])
                show[-1] = bestg
            cells = "  ".join(f"{n} {rows[n]['ms']:.2f}ms rel {rows[n]['rel']:.2e}{' =cuBLAS' if rows[n]['bit'] else ''}"
                              f"{'' if n in ok else ' REJ'}" for n in show)
            gok = sum(1 for n in gram if n in ok)
            print(f"[ns-router] {key[0]} -> {pick}   | {cells}   (gram placements within tol: {gok}/{len(gram)})",
                  flush=True)
        return pick
