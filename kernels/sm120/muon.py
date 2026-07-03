"""sm120 Muon for Blackwell — FusedMuon defaults to the GRAM-SPACE Newton-Schulz (restart@3).

Gram NS (kernels/sm120/newton_schulz_gram.py): every NS iterate is a polynomial in G = X X^T,
so the loop runs on the n x n Gram (R <- C^2 R, Q <- C Q, ALL products symmetric-halved by the
symmul kernels) with ONE rectangular apply X = Q X at the end — the five B@X GEMMs are gone.
Dao-style restart@3 re-anchors to X mid-run: parity matches the cuBLAS champion to the 4th
digit at kappa 1e2..1e6 (fp32 alone does NOT stabilize — measured). Gates: gram needs
r = m/n >= 1.5 and dim >= 2048, else symmul NS, else cuBLAS — no regime regresses.
MEASURED (RTX PRO 6000): NS-level 1.81x vs symmul at (2048,8192), 2.24x batched; end-to-end
step (3 layers d=4096) 118ms vs symmul 145ms vs cuBLAS 190ms, param parity 1.8e-4, peak mem
696 vs 718MB. `use_gram=False` restores the symmul default described below.

On Blackwell the symmul NS strictly dominates the old pure-cuBLAS path with NO precision tradeoff, so
it is now the FusedMuon default (the symmetric trick + our foreach/batched-state levers, composed):
the two SYMMETRIC NS GEMMs (X Xᵀ, A·A) compute one triangle and mirror it -> ~half the GEMM FLOPs.

  FusedMuon            : symmul NS by default. 1.28-1.43x faster than the cuBLAS NS on large matrices
                         (gram >= 2048), 1.31-3.49x vs torch.compile, beats flash-muon's exact impl
                         1.10-1.17x, mem <= compiled, scale-invariant 1B->2.6B params.
  FusedMuon(use_symmul=False) : the pure-cuBLAS champion step (no Triton launched) — for a Triton-free
                         environment or an exact-reference run.
  AmalgamatedMuon      : back-compat alias of FusedMuon.

PRECISION IS NOT A TRADEOFF: identical fp16 NS with fp32 accumulate; the symmetric kernel only changes
float rounding ORDER (triangle+mirror vs full bmm), ~1e-4..1e-3 -> parity 5.9e-3 (< 2e-2 NS tolerance),
SV ~0.98 (same orthogonalization). It self-gates below the gram knee (min(rows,cols) < SYMMUL_MIN_DIM)
to the EXACT cuBLAS NS, so small matrices are bit-for-bit the champion. Only the default `ns_batch_elems`
(8M) differs from sm75 (the Blackwell mem knee). The opt-in CUDA-graph path still uses the cuBLAS NS.
"""
import torch

from kernels.sm75.muon import newton_schulz, _PE_COEFFS, _DSV4_COEFFS  # noqa: F401
from kernels.sm75.muon import FusedMuon as _FusedMuon75, DistributedMuon as _DistributedMuon75
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul
from kernels.sm120.newton_schulz_gram import newton_schulz_gram
from kernels.muon import muon_scaling as _scaling

# Blackwell mem-gated knee (peak<=baseline at both sizes); sm75 uses 4M. Callers can still override.
NS_BATCH_ELEMS = 8 * 1024 * 1024


class FusedMuon(_FusedMuon75):
    """sm120 FusedMuon — DEFAULTS to gram NS (restart@3) -> symmul -> cuBLAS by shape gates.

    use_gram=False falls back to the symmul-default behavior documented below.

    Same foreach + batched same-shape state + Blackwell mem knee (8M) as before, but the two SYMMETRIC
    NS GEMMs (X Xᵀ, A·A) run on the Triton symmul kernel (compute one triangle, mirror it -> ~half the
    GEMM FLOPs). Measured: 1.28-1.43x faster than the pure-cuBLAS NS on large matrices (gram >= 2048),
    1.31-3.49x vs torch.compile, beats flash-muon's exact impl 1.10-1.17x, mem <= compiled, scale-
    invariant 1B->2.6B params. NO precision tradeoff: identical fp16 / fp32-accumulate NS, parity 5.9e-3
    (< 2e-2 tol), SV ~0.98; it self-gates to the EXACT cuBLAS champion NS below the gram knee, so small
    matrices are bit-for-bit unchanged.

    `use_symmul=False` -> the pure-cuBLAS champion step (no Triton kernels launched) for a Triton-free
    environment or an exact-reference run. With `use_graph=True` the captured body ALSO runs symmul (forced
    to its eager Triton path, which is CUDA-graph-capturable); the graph is a speed wash on a compute-bound
    step but lowers peak memory, so symmul-in-graph = symmul speed + the graph's smaller footprint (useful
    on a memory-constrained GPU). State dict / constructor are identical across all modes.
    """

    def __init__(self, *args, use_symmul=True, use_gram=True, gram_restarts=None, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)
        self.use_symmul = use_symmul
        self.use_gram = use_gram
        # restart iteration(s) for the gram NS; None = the tuned default (GRAM_RESTART_AT).
        # Custom `coeffs` may want their own placement: kernels.sm120.newton_schulz_gram
        # `autotune_restarts(coeffs)` finds it.
        self.gram_restarts = gram_restarts

    def _ns(self, u, force_eager=False):
        """The NS this optimizer runs: gram (default) -> symmul -> cuBLAS, by shape gates."""
        if self.use_gram:
            kw = {} if self.gram_restarts is None else {"restart_at": self.gram_restarts}
            return newton_schulz_gram(u, self.coeffs, self.ns_dtype, force_eager=force_eager, **kw)
        return newton_schulz_symmul(u, self.coeffs, self.ns_dtype, force_eager=force_eager)

    def _polar(self, u):
        """Orthogonalizer aurora iterates on — the sm120 gram/symmul path (overrides sm75's cuBLAS)."""
        return self._ns(u)

    @torch.no_grad()
    def step(self, closure=None):
        # per-row / aurora modes always take the eager symmul/gram apply below (not the captured graph)
        _eager_mode = _scaling.is_perrow(self.scale_mode) or _scaling.is_aurora(self.scale_mode)
        if not self.use_symmul or (self.use_graph and not _eager_mode):
            return super().step(closure)                  # pure-cuBLAS champion (or the graph path)
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            plan = self._plan(group, params)
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            perrow = _scaling.is_perrow(self.scale_mode)
            aurora = _scaling.is_aurora(self.scale_mode)
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                v_all = self.state[g["anchor"]].get("scale_v")   # (M,r) EMA state for per-row modes
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:
                    mom_c = mom[start:start + crows]
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c
                    if aurora:                            # iterative prescale + re-orthogonalize (K gram/symmul solves)
                        out = _scaling.aurora_update(u, self._polar, K=self.aurora_k)
                    else:
                        out = self._ns(u)                 # gram NS (default) or symmul NS
                        if perrow:                        # leverage-aware per-row rescale (scale folded into out)
                            out = _scaling.apply_perrow(self.scale_mode, out, v_all[start:start + crows])
                    torch._foreach_add_([p for p, _, _ in members],
                                        [out[o:o + n].reshape(p.shape) for p, o, n in members],
                                        alpha=(-lr if (perrow or aurora) else alpha))
        return loss

    def _compute(self, work, decay):
        """The CUDA-graph-captured body (used when use_graph=True). Runs symmul forced to its eager
        Triton path (capturable) when use_symmul; else defers to the cuBLAS compute."""
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
    """sm75 DistributedMuon with `ns_batch_elems` defaulting to the Blackwell knee (8M)."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("ns_batch_elems", NS_BATCH_ELEMS)
        super().__init__(*args, **kwargs)


# Back-compat alias: FusedMuon now IS the amalgamated (symmul) optimizer on sm120.
AmalgamatedMuon = FusedMuon
