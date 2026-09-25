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
    # swept on the RTX PRO 6000 at a 6553 x 81920 chunk (bench_ce_combine.py): 1.571 ms vs 1.624 for
    # the old 8/1024/4w. Elementwise, so the config never changes a bit.
    BLOCK_M, BLOCK_V = 1, 512
    _grad_logits_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(Vv, BLOCK_V))](
        logits, lse, labels, nv, M, Vv, ignore_index,
        logits.stride(0), logits.stride(1), BLOCK_M=BLOCK_M, BLOCK_V=BLOCK_V, num_warps=8)
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


# LOGITS GEMM WITH THE LOGSUMEXP IN ITS EPILOGUE. The fwd-reduce pass above re-read every logit
# (1 GB per chunk) just to get the row logsumexp; here each output tile also writes its row
# (max, sum exp) and _lse_combine folds the 320 tile partials. The logits come out BITWISE equal
# to cuBLAS (checked in bench_ce_lse_gemm.py), so the grad pass is unchanged; lse differs from the
# old single-pass reduce only in summation order. 2.00 vs 2.70 ms per 6553 x 81920 chunk.
# TKF_CE_LSE_GEMM=0 restores cuBLAS + _fwd_reduce_kernel.
import os as _os
LSE_GEMM = _os.environ.get("TKF_CE_LSE_GEMM", "1") != "0"
_LSE_CFG = (128, 256, 64, 8, 8, 3)       # BM, BN, BK, GROUP, warps, stages


@triton.jit
def _logits_lse_kernel(X, W, L, PM, PS, M, V, NT, K: tl.constexpr,
                       BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
    pid = tl.program_id(0)
    nm = tl.cdiv(M, BM)
    # grouped launch order so a band of W tiles stays in L2 across row blocks
    width = GROUP * NT
    g = pid // width
    first = g * GROUP
    gs = tl.minimum(nm - first, GROUP)
    pm = first + (pid % width) % gs
    pn = (pid % width) // gs
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm = rm < M
    acc = tl.zeros((BM, BN), tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        x = tl.load(X + rm[:, None] * K + rk[None, :], mask=mm[:, None], other=0.0)
        w = tl.load(W + rn[:, None] * K + rk[None, :])
        acc = tl.dot(x, tl.trans(w), acc)
    lb = acc.to(tl.bfloat16)
    tl.store(L + rm[:, None].to(tl.int64) * V + rn[None, :], lb, mask=mm[:, None])
    xf = lb.to(tl.float32)                          # stats from the ROUNDED logits the grad pass reads
    m = tl.max(xf, axis=1)
    s = tl.sum(tl.exp(xf - m[:, None]), axis=1)
    tl.store(PM + rm * NT + pn, m, mask=mm)
    tl.store(PS + rm * NT + pn, s, mask=mm)


@triton.jit
def _lse_combine(PM, PS, LSE, M, NT, BT: tl.constexpr):
    r = tl.program_id(0)
    t = tl.arange(0, BT)
    mk = t < NT
    m = tl.load(PM + r * NT + t, mask=mk, other=-float("inf"))
    s = tl.load(PS + r * NT + t, mask=mk, other=0.0)
    mx = tl.max(m, 0)
    tl.store(LSE + r, mx + tl.log(tl.sum(s * tl.exp(m - mx), 0)))



def _logits_and_lse(hc, weight, lse_out):
    """logits (bf16, cuBLAS-identical) and the row logsumexp into lse_out; None if not tileable."""
    BM, BN, BK, G, nw, ns = _LSE_CFG
    M, K = hc.shape
    V = weight.shape[0]
    if (not LSE_GEMM or hc.dtype != torch.bfloat16 or weight.dtype != torch.bfloat16 or V % BN
            or K % BK or not hc.is_contiguous() or not weight.is_contiguous()):
        return None
    NT = V // BN
    L = torch.empty(M, V, device=hc.device, dtype=hc.dtype)
    PM = torch.empty(M, NT, device=hc.device, dtype=torch.float32)
    PS = torch.empty(M, NT, device=hc.device, dtype=torch.float32)
    _logits_lse_kernel[(triton.cdiv(M, BM) * NT,)](hc, weight, L, PM, PS, M, V, NT, K, BM, BN, BK, G,
                                                   num_warps=nw, num_stages=ns)
    _lse_combine[(M,)](PM, PS, lse_out, M, NT, triton.next_power_of_2(NT), num_warps=4)
    return L


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
            logits = _logits_and_lse(hc, weight, lse[i:i + C])
            if logits is not None:
                lab = labels[i:i + C]
                safe = torch.where(lab == ignore_index, 0, lab)
                tgt[i:i + C] = logits.gather(1, safe[:, None]).squeeze(1).float()
            else:
                logits = torch.mm(hc, weight.t())
                _fwd_reduce_kernel[(cl,)](logits, labels[i:i + C], lse[i:i + C], tgt[i:i + C],
                                          cl, V, logits.stride(0), logits.stride(1), ignore_index,
                                          BLOCK_V=2048, num_warps=16)   # swept: 0.832 vs 0.901 ms/chunk
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
