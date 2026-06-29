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
  3. `ns_dtype`: bf16 matches the baseline (Ampere+ tensor cores); fp16 engages T4's fp16 tensor cores —
     the only lever on the dominant GEMM cost where bf16 has none on sm_75. Opt-in; verify stability.
"""
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


def newton_schulz(G, coeffs=_PE_COEFFS, ns_dtype=torch.bfloat16, eps=1e-7):
    """Orthogonalize G (drive singular values -> 1) via Polar-Express Newton-Schulz (per-iteration coeffs).

    2D weights are unsqueezed to (1,A,B); 3D stacked experts (E,A,B) batch over E. Normalization is fp32
    (an fp16 sum-of-squares of a ~unit 512^2 matrix overflows; fp32 is also strictly better than bf16's).
    The iteration GEMMs run in `ns_dtype` — bf16 (baseline) or fp16 (T4 tensor cores; cuBLAS accumulates
    in fp32). baddbmm folds each iteration's axpy into the GEMM (3 GEMMs/iter, 0 pointwise kernels).
    """
    orig_dtype = G.dtype
    X = G.float()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)                                            # (1,A,B) — unify 2D + (E,A,B)
    X = X / (X.flatten(1).norm(dim=1).clamp_min(eps).view(-1, 1, 1))  # per-slice Frobenius (fp32)
    transposed = X.size(1) > X.size(2)                               # iterate on the smaller Gram
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype).contiguous()
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))                          # XX^T
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)                  # b*A + c*(A@A)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)                # a*X + B@X
    X = X.float()
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


class FusedMuon(optim.Optimizer):
    """Polar-Express Muon with foreach + baddbmm + configurable NS dtype (bf16 baseline / fp16 for T4).

    Only 2D and 3D params with a grad are stepped (3D experts orthogonalized per slice); route 1D params
    and conv kernels to AdamW upstream. `scale_mode`: 'jordan' = max(1, rows/cols)**0.5 (the PE baseline);
    'moonlight' = 0.2*sqrt(max(rows,cols)) (BiBo's current consistent-RMS scale, AdamW-band LR).
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, weight_decay=0.0,
                 coeffs=_PE_COEFFS, ns_dtype=torch.bfloat16, scale_mode="jordan"):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.coeffs = coeffs
        self.ns_dtype = ns_dtype
        self.scale_mode = scale_mode

    def _scale(self, p):
        r, c = p.shape[-2], p.shape[-1]
        if self.scale_mode == "moonlight":
            return 0.2 * (max(r, c) ** 0.5)
        return max(1, r / c) ** 0.5                                   # jordan (PE baseline)

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

            # momentum in ns_dtype (matches the bf16 baseline); buffer persists across steps.
            grads = [p.grad.to(self.ns_dtype) for p in params]
            bufs = []
            for p, g in zip(params, grads):
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                bufs.append(st["momentum_buffer"])

            torch._foreach_mul_(bufs, momentum)                       # buf = momentum*buf + grad
            torch._foreach_add_(bufs, grads)
            gs = list(torch._foreach_add(grads, bufs, alpha=momentum)) if nesterov else list(bufs)

            # Newton-Schulz per param (3D experts batch over the expert dim inside NS), then scale + step.
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)            # decoupled weight decay (all params)
            for i, p in enumerate(params):
                u = newton_schulz(gs[i], self.coeffs, self.ns_dtype)
                p.add_(u.to(p.dtype), alpha=-lr * self._scale(p))     # scaled orthogonal step

        return loss
