"""Fused Muon optimizer step — Polar Express Newton-Schulz, launch-overhead-fused, cuBLAS-bound.

Baseline = the Polar-Express Muon from nprime06/parameter-golf (track_10min_16mb winner): 5 *per-iteration*
Newton-Schulz coefficient tuples (aggressive first step, settling after) instead of a fixed quintic, bf16
NS, Keller-Jordan aspect-ratio scale `max(1, rows/cols)**0.5`, Nesterov momentum, decoupled weight decay.
The original pipelines a distributed reduce-scatter/all-gather; this is the single-GPU step (T4/BiBo).

This is NOT a Triton kernel. The step is GEMM-bound (~70%): Newton-Schulz is 3 matmuls/iter and that count
is the algorithmic floor — a Triton `tl.dot` NS loses to cuBLAS (proven 3x here) and a 512^2 tile won't fit
T4 SRAM. The wins are launch-overhead, not a hand-written GEMM:
  1. `torch._foreach_*` collapses the per-param momentum/nesterov sweeps from N*(several launches) to a few.
  2. `baddbmm` folds each NS axpy (`b*A + c*(A@A)`, `a*X + B@X`) into the cuBLAS call — no pointwise kernels.
  3. `ns_dtype`: **fp16 is the T4 default** — it engages T4's fp16 tensor cores, the dominant lever
     where fp32 runs on slow CUDA cores and bf16 has NO tensor cores at all on sm_75 (and torch.compile
     skips bf16 there). Use fp32 for a numerically-safe fallback, bf16 only on Ampere/Hopper.
"""
from collections import defaultdict

import torch
import torch.optim as optim

from kernels.muon import muon_scaling as _scaling

# Polar-Express per-iteration NS coefficients (arXiv 2505.16932, Alg. 1, l0=1e-3). 8 tuples = 8 NS steps;
# tuple i is used at iteration i. The first is aggressive (expand small singular values fast), then settle
# to the FIXED-POINT tail (1.875,-1.25,0.375), whose f(1)=1.875-1.25+0.375=1.0 exactly -> converges to
# kappa 1 (the old 5-tuple tail had f(1)=1.06 and could not reach 1). Repeat the last tuple for >8 steps.
# Kept as the reference minimax schedule; no longer the default (see _DSV4_COEFFS).
_PE_COEFFS = (
    (8.28721201814563,   -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106,  0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,   0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,   0.51004894012372),
    (2.300652019954817,   -1.6689039845747493,  0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868,  0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487,  0.3750001645474248),
    (1.875,               -1.25,                0.375),
)

# DEFAULT: DeepSeek-V4's hybrid schedule (arXiv 2606.19348) — 8x Keller-Jordan quintic (a=3.44 lifts tiny
# singular values ~3.4x/iter, ~2x faster than the PE tail's 1.875) + 2x pinned polish (f(1)=1, f'(1)=0)
# that locks the KJ band [0.68,1.13] onto 1. Chosen for square (r=1) matrices, whose sigma_min sits below
# PE-8's l0=1e-3 design floor (hard edge ~1/(2n)): measured kappa at r=1 decay=2, n=2048 — PE-8 5160 vs
# this 446 (aurora_k1: 470 vs 40.5; aurora_k2: 3.45 vs 1.00). r>=2 identical (kappa 1.00 both). Cost:
# 10 NS iters vs PE-8's 8 (+25%).
_DSV4_COEFFS = ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2

# Per-arch default NS iteration dtype (single source of truth; FusedMuon exposes it as the
# class attr DEFAULT_NS_DTYPE so sm120 can override it). sm75 = Turing/T4 -> fp16 (Turing has
# fp16 tensor cores but NO bf16). The norm reduction is ALWAYS fp32 regardless (see newton_schulz).
_NS_DTYPE = torch.float16


def newton_schulz(G, coeffs=_DSV4_COEFFS, ns_dtype=_NS_DTYPE, eps=1e-7):
    """Orthogonalize G (drive singular values -> 1) via Polar-Express Newton-Schulz (per-iteration coeffs).

    2D weights are unsqueezed to (1,A,B); 3D stacked experts (E,A,B) batch over E. Normalization is fp32
    (an fp16 sum-of-squares of a ~unit 512^2 matrix overflows; fp32 is also strictly better than bf16's) —
    the norm is the one place low precision hurts (iteration stability needs the initial spectral norm
    <= 1), so it stays fp32 regardless of ns_dtype. The iteration GEMMs run in `ns_dtype`: fp16 here
    (sm75 = Turing/T4, fp16 tensor cores, no bf16; cuBLAS still accumulates in fp32) — sm120/Blackwell
    overrides to bf16 (wider range, device-portable). baddbmm folds each iteration's axpy into the GEMM.

    NOTE (Round-4 T4): each baddbmm emits a cuBLAS bias DtoD memcpy (~11% of the step, DtoD count ~=
    baddbmm count). Replacing it with bmm + in-place axpy DOES kill the memcpy but trades it for two extra
    elementwise passes + double the bmm launches -> 5% SLOWER on the compute-bound T4 step. So the fold is
    the cheaper option and stays; the memcpy is the lesser evil, not waste. (bmm-variant refuted, in git.)
    """
    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G                             # (1,A,B) — unify 2D + (E,A,B)
    # Per-slice Frobenius norm with fp32 ACCUMULATION (no full fp32 copy of X — an fp16 sum-of-squares
    # would overflow, but vector_norm(dtype=fp32) accumulates in fp32 while reading fp16). This is the
    # whole copy-elimination: for the fp16-momentum->fp16-NS path there is now ZERO dtype round-trip.
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)                              # iterate on the smaller Gram
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype) / nrm.to(ns_dtype)                           # normalize; one contiguous ns_dtype tensor
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))                          # XX^T  (transpose is a free cuBLAS flag)
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)                  # b*A + c*(A@A)  — axpy folded into the GEMM
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)                # a*X + B@X
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)                                          # no-op when orig_dtype == ns_dtype


class FusedMuon(optim.Optimizer):
    """Polar-Express Muon with foreach + baddbmm + configurable NS dtype (fp16 on Turing/sm75; sm120 overrides to bf16).

    Only 2D and 3D params with a grad are stepped (3D experts orthogonalized per slice); route 1D params
    and conv kernels to AdamW upstream. `scale_mode` sets the POST-Newton-Schulz update scaling; it does
    NOT touch the NS iteration or its coefficients. Every mode targets update RMS 0.2 (the Moonlight /
    DeepSeek-V4 convention) so AdamW LR and weight decay carry over unchanged; the modes differ only in
    the update's ROW SHAPE (see kernels/muon/muon_scaling.py):
      'polar'   : plain orthogonalized update x 0.2*sqrt(max(rows,cols)). Tall matrices get leverage-
                  skewed row norms ("neuron death").
      'normuon' : per-row EMA normalize AFTER the polar (uniform rows, slightly breaks orthogonality;
                  EMA state round-trips in state_dict).
      'aurora'  : DEFAULT — prescale rows BEFORE the polar and re-orthogonalize (`aurora_k` passes,
                  K=1 matches paper Aurora's K=2 at half cost): uniform rows AND orthogonal.
    normuon/aurora use the eager apply (not the CUDA-graph capture path).
    """

    # Per-arch NS iteration dtype, overridable by subclasses (sm120 -> bf16). Single source of
    # truth for the optimizer default; ns_dtype=None in __init__ resolves to this.
    DEFAULT_NS_DTYPE = _NS_DTYPE

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, weight_decay=0.0,
                 coeffs=_DSV4_COEFFS, ns_dtype=None, scale_mode=_scaling.DEFAULT_MODE,
                 ns_batch_elems=4 * 1024 * 1024, use_graph=False, graph_warmup=3, aurora_k=None,
                 spectral_wd=0.0, swd_beta=0.99, xorth_post=0.0):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.coeffs = coeffs
        self.ns_dtype = ns_dtype if ns_dtype is not None else self.DEFAULT_NS_DTYPE
        self.scale_mode = _scaling.validate(scale_mode)
        self.aurora_k = _scaling.AURORA_K if aurora_k is None else aurora_k
        # SPECTRAL WEIGHT DECAY (idea: route accumulated per-row update energy into the DECAY, not the
        # update). gamma=spectral_wd; 0 = standard uniform decoupled wd. gamma>0 REDISTRIBUTES the same
        # total decay by row staleness: rows with LOW EMA'd momentum energy (stale, optimizer not
        # maintaining them) get decayed HARDER; active rows lighter. Acts on W, sidesteps the polar
        # entirely (no orthogonality break, no re-flatten). Mean-normalized so avg decay == wd (isolates
        # redistribution). swd_beta = energy-EMA momentum. See kernels/muon/muon_scaling.spectral_wd_mult.
        self.spectral_wd = float(spectral_wd)
        self.swd_beta = float(swd_beta)
        self._swd_cov = None                                          # last-step row-energy coeff-of-variation (gate diagnostic)
        # AFTER-NS cross-expert decorrelation (post-polar xorth): damped E x E whitening of the
        # ORTHOGONALIZED update, per expert-stack (n>1). Cleaner gram than pre-NS grad xorth (uniform
        # spectrum) but breaks per-expert orthogonality. 0 = off. See muon_scaling.xorth_whiten.
        self.xorth_post = float(xorth_post)
        # Cap rows*r*c per batched Newton-Schulz call: batches the many small 2D params while row-chunking
        # the big expert stacks so the per-step transient stays BELOW the baseline's peak (a hard eval
        # gate). Bigger caps run bigger GEMMs (faster: 64M hits 1.16x/1.10x) but blow past baseline mem
        # (1155 vs 891 MB @48t) — so 4M is the default knee: 1.09x/1.05x at 862/3180 MB, both UNDER the
        # baseline-mixed peak (891/3222 MB). Raise it only when you have VRAM headroom over the baseline.
        self.ns_batch_elems = ns_batch_elems
        # CUDA-graph capture (opt-in, OFF by default): captures momentum->NS->scatter as ONE graph and
        # replays it, with the grad-gather kept EAGER (reads current p.grad -> robust to grads rebinding
        # under zero_grad(set_to_none=True)). It collapses launches (T4: 1768->830) but is a WALL-CLOCK
        # WASH ON T4 (Round 3): the step is GPU-COMPUTE-bound on the small fp16 GEMMs, not CPU-launch-
        # bound — the profiler's "Command Buffer Full" was CPU submission OVERLAPPED behind a saturated
        # GPU, not idle stall. Kept as opt-in for a genuinely launch-bound host (slow CPU / many tiny
        # params) where it would pay off. ASSUMES STATIC HYPERPARAMS (lr/wd/momentum baked into the
        # captured ops) and stable shapes — call set_graph(None) to recapture after an LR-schedule change.
        self.use_graph = use_graph
        self.graph_warmup = graph_warmup
        self._graph = None
        self._gwork = None
        self._gstep = 0
        self._graph_failed = False

    def _scale(self, p):
        return _scaling.scalar_scale(self.scale_mode, p.shape[-2], p.shape[-1])

    def _polar(self, u):
        """The raw orthogonalizer aurora iterates on (overridden on sm120 to gram/symmul)."""
        return newton_schulz(u, self.coeffs, self.ns_dtype)

    def _plan(self, group, params):
        """Build (once, cached) the same-shape grouping + one persistent (M,r,c) batched momentum buffer
        per group. Momentum lives under self.state[anchor] so it round-trips through state_dict."""
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = self._plan_cache = {}
        key = id(group)
        if key in cache:
            return cache[key]
        buckets = defaultdict(list)
        for p in params:
            buckets[(p.shape[-2], p.shape[-1])].append(p)
        plan = []
        for (r, c), ps in buckets.items():
            members, off = [], 0
            for p in ps:
                n = p.numel() // (r * c)                              # 1 for 2D, E for a 3D expert tensor
                members.append((p, off, n)); off += n
            M = off
            anchor = ps[0]
            if "muon_mom" not in self.state[anchor]:                 # don't clobber a loaded checkpoint
                self.state[anchor]["muon_mom"] = torch.zeros((M, r, c), device=anchor.device, dtype=self.ns_dtype)
            if _scaling.needs_perrow_state(self.scale_mode) and "scale_v" not in self.state[anchor]:
                self.state[anchor]["scale_v"] = _scaling.perrow_state(M, r, anchor.device)  # per-row EMA (normuon post / aurora_ema pre); round-trips in state_dict
            if self.spectral_wd > 0 and "swd_e" not in self.state[anchor]:
                self.state[anchor]["swd_e"] = torch.zeros((M, r), device=anchor.device)  # per-row momentum-energy EMA for spectral wd (fp32; round-trips)
            # aurora/per-row fold their scale into the update tensor, so the scalar is unused (1.0)
            scale = (_scaling.scalar_scale(self.scale_mode, r, c)
                     if self.scale_mode in _scaling.SCALAR_MODES else 1.0)
            # split members into row-chunks bounded by ns_batch_elems (params kept whole; ≥1 param/chunk)
            row_cap = max(1, self.ns_batch_elems // (r * c))
            chunks, cur, cur_rows, start = [], [], 0, 0
            for p, _o, n in members:
                if cur and cur_rows + n > row_cap:
                    chunks.append((cur, start, cur_rows)); start += cur_rows; cur, cur_rows = [], 0
                cur.append((p, cur_rows, n)); cur_rows += n          # offset relative to the chunk
            if cur:
                chunks.append((cur, start, cur_rows))
            plan.append({"r": r, "c": c, "M": M, "chunks": chunks, "anchor": anchor, "scale": scale})
        cache[key] = plan
        return plan

    def _build_graph_work(self):
        """Build (once, cached) the persistent per-chunk staging buffers + views the captured compute
        reads/writes. `gbuf` is allocated ONCE here (not per-step), so its address is stable for capture;
        `mom_c`/scatter views index into the persistent momentum/master buffers. `out_members` is kept so
        the eager gather can re-fetch the CURRENT p.grad each step (grads may rebind between steps)."""
        if self._gwork is not None:
            return self._gwork
        work, decay = [], []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            plan = self._plan(group, params)
            if wd != 0:
                decay.append((params, 1.0 - lr * wd))
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    work.append({"gbuf": gbuf, "dst": [gbuf[o:o + n] for _, o, n in members],
                                 "mom_c": mom[start:start + crows], "momentum": momentum,
                                 "nesterov": nesterov, "alpha": alpha, "members": members,
                                 "out_params": [p for p, _, _ in members], "r": r, "c": c})
        self._gwork = (work, decay)
        return self._gwork

    def _gather(self, work):
        """EAGER, every step: copy the current p.grad (fp32 master) into the persistent fp16 staging
        buffers. This is the only op that touches p.grad, so capture/replay never sees a rebound grad."""
        for w in work:
            r, c = w["r"], w["c"]
            torch._foreach_copy_(w["dst"], [p.grad.reshape(n, r, c) for p, o, n in w["members"]])

    def _compute(self, work, decay):
        """The capturable body: decoupled WD, batched momentum/Nesterov, Newton-Schulz, scatter — all on
        persistent buffers (gbuf/mom/params). Identical math to the eager path; only the buffers persist."""
        for params, f in decay:
            torch._foreach_mul_(params, f)
        for w in work:
            mom_c, gbuf = w["mom_c"], w["gbuf"]
            mom_c.mul_(w["momentum"]).add_(gbuf)
            u = gbuf.add_(mom_c, alpha=w["momentum"]) if w["nesterov"] else mom_c
            out = newton_schulz(u, self.coeffs, self.ns_dtype)
            r, c = w["r"], w["c"]
            torch._foreach_add_(w["out_params"],
                                [out[o:o + n].reshape(p.shape) for p, o, n in w["members"]],
                                alpha=w["alpha"])

    @torch.no_grad()
    def _graph_step(self):
        """Launch-bound killer: warm a few eager steps, capture the compute body into a CUDA graph, then
        replay it (one launch) every subsequent step. Falls back to permanent eager on any capture error
        so a graph-unfriendly environment degrades to the fused-mixed champion instead of crashing."""
        work, decay = self._build_graph_work()
        self._gather(work)                                    # always eager — reads current grads
        if self._graph is not None:
            self._graph.replay()
            return
        self._gstep += 1
        if self._gstep <= self.graph_warmup or self._graph_failed:
            self._compute(work, decay)                        # eager warmup (materialize state, warm cuBLAS)
            return
        try:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                self._compute(work, decay)                    # records ops; does NOT execute them
            self._graph = g
            self._graph.replay()                              # execute this step's work once
        except Exception as ex:                               # noqa: BLE001 — degrade, don't crash the bench
            self._graph, self._graph_failed = None, True
            print(f"  (FusedMuon CUDA-graph capture failed: {type(ex).__name__}: "
                  f"{str(ex).splitlines()[0]} — falling back to eager)")
            self._compute(work, decay)

    def set_graph(self, enabled):
        """Toggle graph capture and drop any captured graph (call after an LR-schedule change so the
        next step recaptures with the new hyperparams)."""
        self.use_graph = enabled
        self._graph, self._gwork, self._gstep, self._graph_failed = None, None, 0, False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if (self.use_graph and self.spectral_wd == 0 and self.xorth_post == 0
                and not (_scaling.is_perrow(self.scale_mode) or _scaling.is_aurora(self.scale_mode))):
            self._graph_step()                                    # captured graph supports scalar scales only
            return loss

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])

            # BATCHED-STATE, SAME-SHAPE: momentum is ONE (M,r,c) fp16 buffer per shape group (the lever
            # compile can't reach — it runs NS per param). Grads are gathered directly into the batched
            # layout in a single fp32->fp16 copy (no separate .to + no torch.cat), the whole group's
            # momentum + Newton-Schulz run batched, and the update is scattered back. Mixed-precision
            # (AdamW-8bit norm): fp32 master weights, fp32 grads, fp16 momentum state + NS.
            plan = self._plan(group, params)
            spectral = self.spectral_wd > 0 and wd != 0
            if wd != 0 and not spectral:
                torch._foreach_mul_(params, 1.0 - lr * wd)            # decoupled weight decay (fp32 master)
            perrow = _scaling.is_perrow(self.scale_mode)
            aurora = _scaling.is_aurora(self.scale_mode)
            aurora_ema = _scaling.is_aurora_ema(self.scale_mode)
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]             # (M,r,c) fp16, persistent
                v_all = self.state[g["anchor"]].get("scale_v")       # (M,r) EMA state for per-row modes
                e_all = self.state[g["anchor"]].get("swd_e") if spectral else None  # (M,r) energy EMA for spectral wd
                alpha = -lr * g["scale"]                              # scalar modes fold the scale here
                for members, start, crows in g["chunks"]:            # bounded row-chunks (memory cap)
                    mom_c = mom[start:start + crows]                  # view into the persistent buffer
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    # gather this chunk's grads into the batched layout in ONE foreach (fp32->fp16)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)                   # buf = momentum*buf + grad
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c   # reuse gbuf as NS input
                    if spectral:                                     # per-row decay from accumulated momentum energy (skips scalar decay above)
                        s, cov = _scaling.spectral_wd_mult(u, e_all[start:start + crows], self.spectral_wd, self.swd_beta)
                        self._swd_cov = float(cov)
                        for p, o, n in members:                     # weight[row] *= (1 - lr*wd*s[row]); s mean 1 -> avg == wd
                            m = (s[o:o + n] if s is not None else 1.0)
                            pv = p.view(n, r, c)
                            pv.mul_(1.0 - lr * wd * (m.unsqueeze(-1) if s is not None else 1.0))
                    if aurora:                                        # iterative prescale+re-orthogonalize (K polars)
                        out = _scaling.aurora_update(u, self._polar, K=self.aurora_k)
                    elif aurora_ema:                                  # aurora + normuon per-row EMA memory
                        v_c = v_all[start:start + crows]
                        if self.scale_mode == "aurora_ema_v2":        # EMA AFTER polar (normuon-faithful; breaks orthogonality)
                            out = _scaling.aurora_ema_v2_update(u, self._polar, v_c, K=self.aurora_k)
                        else:                                         # v1: EMA in the prescale (stays orthogonal)
                            out = _scaling.aurora_ema_update(u, self._polar, v_c)
                    else:
                        out = newton_schulz(u, self.coeffs, self.ns_dtype)
                        if perrow:                                    # leverage-aware per-row rescale (scale folded into out)
                            out = _scaling.apply_perrow(self.scale_mode, out, v_all[start:start + crows])
                    if self.xorth_post > 0:                            # AFTER-NS cross-expert whiten, per expert-stack (n>1)
                        for p, o, n in members:
                            if n > 1:
                                out[o:o + n] = _scaling.xorth_whiten(out[o:o + n], self.xorth_post)
                    # scatter the scaled update to the fp32 masters in ONE foreach (fp16->fp32 upcast)
                    torch._foreach_add_([p for p, _, _ in members],
                                        [out[o:o + n].reshape(p.shape) for p, o, n in members],
                                        alpha=(-lr if _scaling.folds_scale(self.scale_mode) else alpha))

        return loss


class DistributedMuon(FusedMuon):
    """Option B — exact whole-param round-robin Muon for DDP (2x T4). Each rank computes Newton-Schulz
    for a FLOP-balanced subset of the params and broadcasts its packed updates; every rank applies the
    full update set. **Bit-identical to the replicated FusedMuon** (same all-reduced grads in -> same
    weights out), but each rank does ~1/world_size of the NS work and stores momentum only for its OWNED
    params (less optimizer-state memory). Comm = `world_size` broadcasts of packed update blobs — same
    total volume as one all-gather, on top of DDP's existing grad all-reduce.

    Assumes grads are already all-reduced across ranks (DDP default) and params share one dtype.
    """

    def __init__(self, params, *, process_group=None, **kwargs):
        super().__init__(params, **kwargs)
        if _scaling.is_aurora_ema(self.scale_mode):                   # per-row EMA prescale not wired for the round-robin path
            raise NotImplementedError("scale_mode 'aurora_ema' is only supported by FusedMuon, not DistributedMuon")
        self.pg = process_group
        self._owner = None                                            # owner[i] = rank that does param i

    def _ordered(self):
        out = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim in (2, 3):
                    out.append((p, g))
        return out

    def _plan(self, ordered, ws):
        load = [0] * ws                                               # greedy least-loaded by numel (FLOP proxy)
        owner = []
        for p, _ in ordered:
            r = min(range(ws), key=lambda i: load[i])
            owner.append(r); load[r] += p.numel()
        return owner

    @torch.no_grad()
    def step(self, closure=None):
        import torch.distributed as dist
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ws, rank = dist.get_world_size(self.pg), dist.get_rank(self.pg)
        ordered = self._ordered()
        if self._owner is None or len(self._owner) != len(ordered):
            self._owner = self._plan(ordered, ws)                     # deterministic — same on every rank

        # 1) each rank computes the orthogonalized update for ITS owned params (momentum is owner-local).
        upd = {}
        for i, (p, g) in enumerate(ordered):
            if self._owner[i] != rank or p.grad is None:
                continue
            gr = p.grad.to(self.ns_dtype)
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(gr)
            buf = st["momentum_buffer"]
            buf.mul_(g["momentum"]).add_(gr)
            u = gr.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
            if _scaling.is_aurora(self.scale_mode):                  # aurora bakes the full scaled update (gain incl.)
                upd[i] = _scaling.aurora_update(u.unsqueeze(0) if u.ndim == 2 else u,
                                                self._polar, K=self.aurora_k).reshape(p.shape).to(p.dtype)
            else:
                upd[i] = newton_schulz(u, self.coeffs, self.ns_dtype).to(p.dtype)

        # 2) one broadcast per source rank of its packed owned-updates blob; every rank then applies.
        for src in range(ws):
            idxs = [i for i in range(len(ordered))
                    if self._owner[i] == src and ordered[i][0].grad is not None]
            if not idxs:
                continue
            sizes = [ordered[i][0].numel() for i in idxs]
            ref = ordered[idxs[0]][0]
            if src == rank:
                blob = torch.cat([upd[i].reshape(-1) for i in idxs])
            else:
                blob = torch.empty(sum(sizes), device=ref.device, dtype=ref.dtype)
            dist.broadcast(blob, src=src, group=self.pg)
            off = 0
            perrow = _scaling.is_perrow(self.scale_mode)
            aurora = _scaling.is_aurora(self.scale_mode)
            for i, n in zip(idxs, sizes):
                p, g = ordered[i]
                u = blob[off:off + n].view_as(p); off += n
                lr, wd = g["lr"], g["weight_decay"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                if aurora:                                            # blob is already the fully scaled update
                    p.add_(u, alpha=-lr)
                elif perrow:                                          # per-row rescale (identical on every rank -> still bit-consistent)
                    uu = u.unsqueeze(0) if u.ndim == 2 else u
                    st = self.state[p]
                    if "scale_v" not in st:
                        st["scale_v"] = _scaling.perrow_state(uu.shape[0], uu.shape[1], p.device)
                    p.add_(_scaling.apply_perrow(self.scale_mode, uu, st["scale_v"]).reshape(p.shape), alpha=-lr)
                else:
                    p.add_(u, alpha=-lr * self._scale(p))
        return loss
