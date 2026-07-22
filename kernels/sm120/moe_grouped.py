"""sm120 grouped PolyGLU MoE — one cuBLAS grouped GEMM over all routed GLU tokens (torch._grouped_mm),
correct on the BiBo special-experts stack, and a per-arch win over the per-expert path at scale.

Why this exists (Blackwell sm_120 only)
---------------------------------------
The shipped per-expert path (`kernels.sm75.moe.moe_per_expert`) sorts tokens by expert and fires ONE
cuBLAS GEMM PER EXPERT. That is the right call when each expert gets many tokens (its GEMMs are large
and the Blackwell tensor cores are already near-peak). But real MoEs route few tokens to many experts,
so the per-expert loop degenerates into a long tail of tiny, tensor-core-starved GEMMs wrapped in a
Python loop with a DtoD memcpy per expert (measured: ~1313 CUDA launches / ~510 DtoD copies per step at
E=32). A single *grouped* GEMM over all sorted tokens removes the loop, the copies, and the tiny-GEMM
inefficiency (~851 launches / ~35 copies).

`torch._grouped_mm` (PyTorch >= ~2.5, needs sm_80+ and 16-byte-aligned strides) gives us a cuBLAS
grouped GEMM that is **autograd-native** (grad_x AND grad_w both flow) in bf16 and fp16 — so this whole
path is plain autograd composition with NO hand-written backward; only the PolyGLU activation carries a
custom Function (`BatchedGLU`). All four MoE GEMMs map onto it directly:
    fwd   gate_up = _grouped_mm(x,            gate_up_proj^T, offs)
          eo      = _grouped_mm(inter,        down_proj^T,    offs)
    bwd   (handled by autograd through the same two ops)

PolyGLU + special experts
-------------------------
The GLU experts (act codes 0/1/2) own weight slots 0..E_glu-1 and form a CONTIGUOUS PREFIX once tokens
are sorted by expert id, so exactly those rows go through the grouped GEMM with per-row activation codes
(PolyGLU). The special experts live at the tail: Identity (code 3) = weighted passthrough handled by a
cheap scatter; Zero (code 4) = skip. They have no weight GEMM, so they never enter the grouped call.
This is what the old GLU-only `kernels.sm75.moe.moe_grouped` lacked (it ran GLU over every expert and
produced grad rel ~5.7 on a stack containing codes 3/4). This path is correct on that stack (grad PASS).

Performance (RTX PRO 6000 Blackwell, bf16, fwd+bwd, vs the per-expert champion)
-------------------------------------------------------------------------------
The win is governed by TOKENS-PER-EXPERT (= routed_tokens / E_glu), not N or E alone:
    tok/expert ~496  -> 3.0x and LESS memory      ~963  -> 2.1x, ~equal memory
    tok/expert ~1820 -> 1.6x, ~equal memory       ~3277 (BiBo E=9) -> 1.17x (marginal)
    tok/expert high (per-expert GEMMs already big) -> grouped loses; use per-expert.
Memory note: grouped uses bf16-accumulate scatter (top_k terms per row stay within grad tol) so it is
at or below per-expert memory up to ~2k tokens/expert; the per-expert loop is more memory-frugal at high
top_k. There is a real speed/memory Pareto frontier here (see .autoresearch/moe_grouped_reflections.md):
this module ships the balanced point. `moe()` dispatches to it only inside its win regime.
"""
import torch
from kernels.sm75.moe import BatchedGLU

__all__ = ["moe_grouped_cublas_polyglu", "grouped_supported", "prefer_grouped",
           "GROUPED_TOKENS_PER_EXPERT_MAX"]

# Above this tokens-per-expert, per-expert GEMMs are large enough to be efficient and the per-expert
# path is both faster AND more memory-frugal — so do NOT pick grouped. Below it, grouped wins on time
# and stays at/under per-expert memory. Conservative on purpose (keeps the <= memory invariant).
GROUPED_TOKENS_PER_EXPERT_MAX = 2048


def grouped_supported(hidden, gate_up_proj, down_proj):
    """True iff torch._grouped_mm can run this on this device/dtype/shape (sm_80+, bf16/fp16, 16B-aligned).
    The grouped GEMM requires 16-byte-aligned strides -> last dims a multiple of 8 elements in bf16/fp16."""
    if not hasattr(torch, "_grouped_mm"):
        return False
    if hidden.device.type != "cuda" or torch.cuda.get_device_capability(hidden.device)[0] < 8:
        return False
    if hidden.dtype not in (torch.bfloat16, torch.float16):
        return False
    H = hidden.shape[1]
    two_i = gate_up_proj.shape[1]
    inner = down_proj.shape[2]
    return (H % 8 == 0) and (two_i % 8 == 0) and (inner % 8 == 0)


def prefer_grouped(top_k_indices, gate_up_proj):
    """Heuristic: pick grouped only when tokens-per-expert is low enough that grouped wins on BOTH
    time and memory. Caller still gates on grouped_supported()."""
    e_glu = gate_up_proj.shape[0]
    routed = top_k_indices.numel()
    return (routed / max(e_glu, 1)) <= GROUPED_TOKENS_PER_EXPERT_MAX


def moe_grouped_cublas_polyglu(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Grouped PolyGLU MoE via torch._grouped_mm. Autograd-native (no custom backward).

    hidden (N,H); top_k_indices/top_k_weights (N,k); gate_up_proj (E_glu,2I,H); down_proj (E_glu,H,I);
    act_codes (E,) int32 where E = E_glu + n_special. GLU codes (0/1/2) index weight slots 0..E_glu-1;
    Identity (3) / Zero (4) are param-free and handled on the sorted tail. Returns (N,H).
    """
    from kernels.sm75.moe import _code_max
    if _code_max(act_codes) > 4:   # cached: one host sync per act_codes tensor lifetime
        raise ValueError("code 5 (SiTU) unsupported on the grouped path; use moe_per_expert(act_params=...)")
    N, H = hidden.shape
    e_glu = gate_up_proj.shape[0]
    e_total = act_codes.shape[0]
    dev = hidden.device
    dt = hidden.dtype

    flat_t = torch.arange(N, device=dev).unsqueeze(1).expand_as(top_k_indices).reshape(-1)
    sorted_e, order = top_k_indices.reshape(-1).sort()
    st = flat_t[order]
    sw = top_k_weights.reshape(-1)[order]
    counts = torch.bincount(sorted_e, minlength=e_total)
    glu_counts = counts[:e_glu]
    offs = glu_counts.cumsum(0).to(torch.int32)          # end-exclusive offsets within the GLU block
    n_glu = int(offs[-1].item())                          # one host sync (GLU token count)
    row_act = torch.repeat_interleave(act_codes[:e_glu], glu_counts).to(torch.int32)

    st_glu = st[:n_glu]
    sw_glu = sw[:n_glu]
    x_glu = hidden.index_select(0, st_glu).contiguous()
    gate_up = torch._grouped_mm(x_glu, gate_up_proj.transpose(-2, -1), offs=offs)   # (n_glu, 2I)
    inter = BatchedGLU.apply(gate_up, row_act)                                      # (n_glu, I)
    eo = torch._grouped_mm(inter, down_proj.transpose(-2, -1), offs=offs)           # (n_glu, H)

    out = torch.zeros(N, H, device=dev, dtype=dt)         # bf16/fp16 accumulate (k terms/row; grad PASS)
    out.index_add_(0, st_glu, eo * sw_glu.unsqueeze(-1))

    n_routed = st.shape[0]
    if n_glu < n_routed:                                   # special experts on the sorted tail
        tail_e = sorted_e[n_glu:]
        tail_codes = act_codes.index_select(0, tail_e)
        id_sel = (tail_codes == 3).nonzero(as_tuple=True)[0]   # Identity = weighted passthrough; Zero = skip
        if id_sel.numel() > 0:
            id_tok = st[n_glu:].index_select(0, id_sel)
            id_w = sw[n_glu:].index_select(0, id_sel)
            out.index_add_(0, id_tok, hidden.index_select(0, id_tok) * id_w.unsqueeze(-1))
    return out
