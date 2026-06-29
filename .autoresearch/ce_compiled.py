"""Inductor's compiled cross-entropy, expressed as raw Triton — the BASELINE for the CE loop.

torch.compile turns `F.cross_entropy((h @ w.T).float(), labels)` into: ONE cuBLAS GEMM that
materializes the full (N,V) fp16 logits, a fused online-softmax reduction (its `prepare_softmax_online`
prim) that streams those logits to per-row lse, and a backward that REUSES the saved logits in place
(no recompute) + 2 cuBLAS GEMMs. This module reproduces that structure as plain Triton so it runs
WITHOUT torch.compile (broken on our local box) — letting the optimization loop bench against a
faithful "compiled CE" baseline locally.

Profile = compiled CE's: FAST (no backward recompute) but HEAVY (the full (N,V) logits live from
forward to backward). Contrast with `cross_entropy.fused_linear_cross_entropy`, which never
materializes (N,V) (chunked, recomputes in backward) — cheap memory, higher latency. The loop's job
is to close that latency gap while keeping the low memory.

Drop-in identical signature:
    from kernels.ce_compiled import compiled_cross_entropy
    loss = compiled_cross_entropy(hidden, lm_head.weight, labels)
"""
import torch
import triton
import triton.language as tl

# reuse the same online-softmax reduction + grad-logit kernels as the chunked kernel (single source)
from kernels.sm75.cross_entropy import _grad_logits_inplace, _fwd_reduce_kernel

__all__ = ["compiled_cross_entropy"]


class _CECompiled(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index):
        N, Hd = hidden.shape
        V = weight.shape[0]
        logits = torch.mm(hidden, weight.t())                 # cuBLAS, full (N,V) fp16 — MATERIALIZED
        lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
        tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
        BLOCK_V = 1024
        _fwd_reduce_kernel[(N,)](logits, labels, lse, tgt, N, V, logits.stride(0), logits.stride(1),
                                 ignore_index, BLOCK_V=BLOCK_V)
        valid = labels != ignore_index
        n_valid = valid.sum().clamp(min=1)
        loss = ((lse - tgt) * valid).sum() / n_valid
        # logits is stashed on ctx (NOT save_for_backward) so we can mutate it in place in backward
        # without tripping autograd's saved-tensor version checker — this is inductor's buf-reuse.
        ctx.logits = logits
        ctx.save_for_backward(lse, labels, weight, hidden, n_valid)
        ctx.ignore_index = ignore_index
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        lse, labels, weight, hidden, n_valid = ctx.saved_tensors
        logits = ctx.logits; ctx.logits = None
        sc = grad_out / n_valid
        # in-place: logits -> grad_logits = (softmax - onehot), unscaled; scale applied post-GEMM
        g = _grad_logits_inplace(logits, lse, labels, 1.0, ctx.ignore_index)
        grad_h = torch.mm(g, weight)                          # (N,H)  cuBLAS
        grad_w = torch.mm(g.t(), hidden.float()).to(weight.dtype) if hidden.dtype == torch.float32 \
            else torch.mm(g.t(), hidden)                      # (V,H)  cuBLAS
        return (grad_h * sc.to(grad_h.dtype)), (grad_w * sc).to(weight.dtype), None, None


def compiled_cross_entropy(hidden, weight, labels, ignore_index=-100):
    """hidden (N,H), weight (V,H), labels (N,) -> mean CE. Materializes (N,V) once, no bwd recompute
    (the compiled-CE profile). Use as the latency/memory BASELINE vs the chunked fused_linear CE."""
    return _CECompiled.apply(hidden, weight, labels, ignore_index)
