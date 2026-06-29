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

# Polar-Express per-iteration NS coefficients (nprime06/parameter-golf, verbatim). 5 tuples = 5 NS steps;
# tuple i is used at iteration i. The first is aggressive (expand small singular values fast), then settle.
_PE_COEFFS = (
    (8.156554524902461,  -22.48329292557795,  15.878769915207462),
    (4.042929935166739,   -2.808917465908714,   0.5000178451051316),
    (3.8916678022926607,  -2.772484153217685,   0.5060648178503393),
    (3.285753657755655,   -2.3681294933425376,  0.46449024233003106),
    (2.3465413258596377,  -1.7097828382687081,  0.42323551169305323),
)


def newton_schulz(G, coeffs=_PE_COEFFS, ns_dtype=torch.float16, eps=1e-7):
    """Orthogonalize G (drive singular values -> 1) via Polar-Express Newton-Schulz (per-iteration coeffs).

    2D weights are unsqueezed to (1,A,B); 3D stacked experts (E,A,B) batch over E. Normalization is fp32
    (an fp16 sum-of-squares of a ~unit 512^2 matrix overflows; fp32 is also strictly better than bf16's).
    The iteration GEMMs run in `ns_dtype` — bf16 (baseline) or fp16 (T4 tensor cores; cuBLAS accumulates
    in fp32). baddbmm folds each iteration's axpy into the GEMM (3 GEMMs/iter, 0 pointwise kernels).
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
    """Polar-Express Muon with foreach + baddbmm + configurable NS dtype (bf16 baseline / fp16 for T4).

    Only 2D and 3D params with a grad are stepped (3D experts orthogonalized per slice); route 1D params
    and conv kernels to AdamW upstream. `scale_mode`: 'jordan' = max(1, rows/cols)**0.5 (the PE baseline);
    'moonlight' = 0.2*sqrt(max(rows,cols)) (BiBo's current consistent-RMS scale, AdamW-band LR).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0,
                 coeffs=_PE_COEFFS, ns_dtype=torch.float16, scale_mode="jordan",
                 ns_batch_elems=4 * 1024 * 1024):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.coeffs = coeffs
        self.ns_dtype = ns_dtype
        self.scale_mode = scale_mode
        # Cap rows*r*c per batched Newton-Schulz call: batches the many small 2D params (the launch win)
        # while chunking the large expert stacks so their transients (gbuf/X/A) stay bounded. The
        # persistent momentum buffer is unchanged (= baseline size); only the per-step transient shrinks.
        self.ns_batch_elems = ns_batch_elems

    def _scale(self, p):
        r, c = p.shape[-2], p.shape[-1]
        if self.scale_mode == "moonlight":
            return 0.2 * (max(r, c) ** 0.5)
        return max(1, r / c) ** 0.5                                   # jordan (PE baseline)

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
            scale = 0.2 * (max(r, c) ** 0.5) if self.scale_mode == "moonlight" else max(1, r / c) ** 0.5
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

    @torch.no_grad()
    def step(self, closure=None):
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

            # BATCHED-STATE, SAME-SHAPE: momentum is ONE (M,r,c) fp16 buffer per shape group (the lever
            # compile can't reach — it runs NS per param). Grads are gathered directly into the batched
            # layout in a single fp32->fp16 copy (no separate .to + no torch.cat), the whole group's
            # momentum + Newton-Schulz run batched, and the update is scattered back. Mixed-precision
            # (AdamW-8bit norm): fp32 master weights, fp32 grads, fp16 momentum state + NS.
            plan = self._plan(group, params)
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)            # decoupled weight decay (fp32 master)
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]             # (M,r,c) fp16, persistent
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:            # bounded row-chunks (memory cap)
                    mom_c = mom[start:start + crows]                  # view into the persistent buffer
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    # gather this chunk's grads into the batched layout in ONE foreach (fp32->fp16)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)                   # buf = momentum*buf + grad
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c   # reuse gbuf as NS input
                    out = newton_schulz(u, self.coeffs, self.ns_dtype)
                    # scatter the scaled update to the fp32 masters in ONE foreach (fp16->fp32 upcast)
                    torch._foreach_add_([p for p, _, _ in members],
                                        [out[o:o + n].reshape(p.shape) for p, o, n in members], alpha=alpha)

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
            for i, n in zip(idxs, sizes):
                p, g = ordered[i]
                u = blob[off:off + n].view_as(p); off += n
                lr, wd = g["lr"], g["weight_decay"]
                if wd != 0:
                    p.mul_(1.0 - lr * wd)
                p.add_(u, alpha=-lr * self._scale(p))
        return loss
