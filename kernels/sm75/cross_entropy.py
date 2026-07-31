import torch
import triton
import triton.language as tl

__all__ = ["fused_linear_cross_entropy"]

_BWD_LOGITS_BUDGET = 192 * 1024 * 1024


@triton.jit
def _grad_logits_kernel(L_ptr, Lse_ptr, Lab_ptr, Nv_ptr, M, Vv, ignore_index,
                        s_lm, s_lv, BLOCK_M: tl.constexpr, BLOCK_V: tl.constexpr):
    pid_m = tl.program_id(0)
    pid_v = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_v = pid_v * BLOCK_V + tl.arange(0, BLOCK_V)
    mask_m = offs_m < M
    mask_v = offs_v < Vv
    mask = mask_m[:, None] & mask_v[None, :]
    scale = 1.0 / tl.load(Nv_ptr)
    lse = tl.load(Lse_ptr + offs_m, mask=mask_m, other=0.0)
    lab = tl.load(Lab_ptr + offs_m, mask=mask_m, other=ignore_index)
    lptr = L_ptr + offs_m[:, None] * s_lm + offs_v[None, :] * s_lv
    logit = tl.load(lptr, mask=mask, other=0.0).to(tl.float32)
    p = tl.exp(logit - lse[:, None])
    g = (p - tl.where(offs_v[None, :] == lab[:, None], 1.0, 0.0)) * scale
    g = tl.where(lab[:, None] != ignore_index, g, 0.0)
    tl.store(lptr, g.to(L_ptr.dtype.element_ty), mask=mask)


def _grad_logits_inplace(logits, lse, labels, nv, ignore_index):
    M, Vv = logits.shape
    BLOCK_M, BLOCK_V = 8, 1024
    _grad_logits_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(Vv, BLOCK_V))](
        logits, lse, labels, nv, M, Vv, ignore_index,
        logits.stride(0), logits.stride(1), BLOCK_M=BLOCK_M, BLOCK_V=BLOCK_V, num_warps=4)
    return logits


@triton.jit
def _fwd_reduce_kernel(L_ptr, Lab_ptr, Lse_ptr, Tgt_ptr, M, V, s_n, s_v, ignore_index,
                       BLOCK_V: tl.constexpr):
    row = tl.program_id(0)
    lab = tl.load(Lab_ptr + row)
    m = -float("inf")
    s = 0.0
    for v0 in range(0, V, BLOCK_V):
        offs = v0 + tl.arange(0, BLOCK_V)
        x = tl.load(L_ptr + row * s_n + offs * s_v, mask=offs < V, other=-float("inf")).to(tl.float32)
        m_new = tl.maximum(m, tl.max(x, 0))
        s = s * tl.exp(m - m_new) + tl.sum(tl.exp(x - m_new), 0)
        m = m_new
    tl.store(Lse_ptr + row, m + tl.log(s))
    safe_lab = tl.where(lab == ignore_index, 0, lab)
    tl.store(Tgt_ptr + row, tl.load(L_ptr + row * s_n + safe_lab * s_v).to(tl.float32))


def _chunk_rows(N, V, budget=None):
    return max(512, min(N, (budget or _BWD_LOGITS_BUDGET) // (V * 2)))


class _CEFusedFwdBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hidden, weight, labels, ignore_index, budget):
        if torch.is_autocast_enabled("cuda"):
            dt = torch.get_autocast_dtype("cuda")
            hidden, weight = hidden.to(dt), weight.to(dt)
        N, Hd = hidden.shape
        V = weight.shape[0]
        C = _chunk_rows(N, V, budget)
        need_gh, need_gw = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        need_grad = need_gh or need_gw
        valid = labels != ignore_index
        n_valid = valid.sum().clamp(min=1)
        nv = n_valid.to(torch.float32)
        lse = torch.empty(N, device=hidden.device, dtype=torch.float32)
        tgt = torch.empty(N, device=hidden.device, dtype=torch.float32)
        gh = torch.empty(N, Hd, device=hidden.device, dtype=hidden.dtype) if need_gh else None
        gw = torch.zeros_like(weight) if need_gw else None
        for i in range(0, N, C):
            cl = min(C, N - i)
            hc = hidden[i:i + C]
            logits = torch.mm(hc, weight.t())
            _fwd_reduce_kernel[(cl,)](logits, labels[i:i + C], lse[i:i + C], tgt[i:i + C],
                                      cl, V, logits.stride(0), logits.stride(1), ignore_index,
                                      BLOCK_V=1024)
            if need_grad:
                _grad_logits_inplace(logits, lse[i:i + C], labels[i:i + C], nv, ignore_index)
                if need_gh:
                    torch.mm(logits, weight, out=gh[i:i + C])
                if need_gw:
                    gw.addmm_(logits.t(), hc)
        loss = ((lse - tgt) * valid).sum() / n_valid
        ctx.save_for_backward(gh, gw)
        return loss

    @staticmethod
    def backward(ctx, grad_out):
        gh, gw = ctx.saved_tensors
        return (gh * grad_out.to(gh.dtype) if gh is not None else None,
                gw * grad_out.to(gw.dtype) if gw is not None else None, None, None, None)


def fused_linear_cross_entropy(hidden, weight, labels, ignore_index=-100, bwd_logits_budget=None):
    return _CEFusedFwdBwd.apply(hidden, weight, labels, ignore_index, bwd_logits_budget)
