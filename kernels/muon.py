"""Fused Muon optimizer step (Moonlight/Kimi recipe) — launch-overhead-fused, cuBLAS-bound.

This is NOT a Triton kernel. The Muon step is GEMM-bound (~70%): Newton-Schulz is 3 matmuls/iter and
that count is the algorithmic floor — a Triton `tl.dot` NS loses to cuBLAS (proven 3× in this repo) and
a 512² tile won't fit T4 SRAM. So the wins here are launch-overhead, not a hand-written GEMM:

  1. `torch._foreach_*` collapses the per-param momentum / scale / weight-decay / update sweeps from
     N×(several launches) into a handful.
  2. `baddbmm` folds the NS axpy epilogues into the cuBLAS call: `b·A + c·(A@A)` and `a·X + B@X` each
     become ONE GEMM with a beta term — no separate pointwise kernels, no extra memory.
  3. `ns_dtype=fp16` runs the NS GEMMs on T4's fp16 tensor cores instead of fp32 CUDA cores — the only
     lever on the dominant 71% GEMM cost. Opt-in: normalize/accumulate keep fp32; verify parity +
     training stability before trusting it (the recipe's 0.2·√max scale assumes SVs≈1).

The recipe matches BiBo/bench/optim.py exactly: momentum on the raw grad → Newton-Schulz orthogonalize
(Nesterov) → 0.2·√max(A,B) consistent-RMS scale → decoupled (AdamW-style) weight decay. 3D stacked MoE
expert tensors (E,A,B) are orthogonalized per-expert-slice (NS batches over the expert dim).
"""
import torch
import torch.optim as optim

# Tuned quintic Newton-Schulz coeffs (Keller Jordan / modded-nanogpt; Moonlight/Kimi use the same).
_NS_COEFFS = (3.4445, -4.7750, 2.0315)


def newton_schulz(G, num_iters=5, coeffs=_NS_COEFFS, ns_dtype=torch.float32):
    """Orthogonalize G (drive singular values → 1) via the tuned quintic Newton-Schulz.

    2D weights are unsqueezed to (1,A,B); 3D stacked experts (E,A,B) batch over E. Normalization is
    always fp32 (an fp16 sum-of-squares of a ~unit 512² matrix overflows). The iteration GEMMs run in
    `ns_dtype` — fp32 (default, bit-matches eager) or fp16 (T4 tensor cores; cuBLAS accumulates in fp32).
    baddbmm folds each iteration's axpy into the GEMM (3 GEMMs/iter, 0 pointwise kernels).
    """
    a, b, c = coeffs
    orig_dtype = G.dtype
    X = G.float()
    squeeze = X.ndim == 2
    if squeeze:
        X = X.unsqueeze(0)                                            # (1,A,B) — unify 2D + (E,A,B)
    X = X / X.flatten(1).norm(dim=1).clamp_min(1e-7).view(-1, 1, 1)   # per-slice Frobenius (fp32)
    transposed = X.size(1) > X.size(2)                               # iterate on the smaller Gram
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype).contiguous()
    for _ in range(num_iters):
        A = torch.bmm(X, X.transpose(1, 2))                          # XXᵀ
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)                  # b·A + c·(A@A)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)                # a·X + B@X
    X = X.float()
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


class FusedMuon(optim.Optimizer):
    """Muon (Moonlight/Kimi recipe) with foreach + baddbmm + optional fp16-tensor-core Newton-Schulz.

    Drop-in for BiBo's `Muon`: same defaults, same per-param math, bit-parity in the fp32 path. Set
    `ns_dtype=torch.float16` to run NS on T4 fp16 tensor cores (opt-in; verify stability first).
    Only 2D and 3D params with a grad are stepped; route 1D params / conv kernels to AdamW upstream.
    """

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True,
                 weight_decay=0.0, num_iters=5, ns_dtype=torch.float32):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.num_iters = num_iters
        self.ns_dtype = ns_dtype

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

            grads = [p.grad for p in params]
            bufs = []
            for p in params:
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(p.grad)
                bufs.append(st["momentum_buffer"])

            # momentum on the raw grad: buf = momentum·buf + grad   (foreach: 2 launches for all params)
            torch._foreach_mul_(bufs, momentum)
            torch._foreach_add_(bufs, grads)
            gs = list(torch._foreach_add(grads, bufs, alpha=momentum)) if nesterov else list(bufs)

            # Newton-Schulz per param (3D experts batch over the expert dim inside NS); collect scales.
            scales = []
            for i, p in enumerate(params):
                gs[i] = newton_schulz(gs[i], self.num_iters, ns_dtype=self.ns_dtype)
                scales.append(0.2 * (max(p.shape[-2], p.shape[-1]) ** 0.5))

            torch._foreach_mul_(gs, scales)                          # consistent-RMS scale (per-param)
            if wd != 0:
                torch._foreach_mul_(params, 1.0 - lr * wd)           # decoupled weight decay
            torch._foreach_add_(params, gs, alpha=-lr)               # the scaled orthogonal step

        return loss
