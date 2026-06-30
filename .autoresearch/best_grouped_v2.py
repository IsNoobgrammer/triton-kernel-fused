# CHAMPION (run 3): cuBLAS grouped MoE on Blackwell sm_120, PolyGLU + special experts, autograd-native.
# Beats moe_per_expert 1.6x (mid) to 3.0x (high-E) at <= or ~equivalent memory; grad PASS on the
# special-experts stack. Governing variable: tokens-per-expert (lower -> bigger grouped win).
# Develop/measured on RTX PRO 6000 Blackwell via marimo. To ship: fold into kernels/sm120/moe.py and
# wire moe() to dispatch here by tokens-per-expert. Requires torch._grouped_mm (sm_80+, 16B-aligned).
from kernels.sm75.moe import BatchedGLU
import torch

def grouped_v2(hidden, idx, wt, gate_up_proj, down_proj, act_codes):
    N, H = hidden.shape
    E_glu = gate_up_proj.shape[0]          # GLU experts have weight slots 0..E_glu-1 (contiguous prefix)
    I = gate_up_proj.shape[1] // 2
    E = act_codes.shape[0]                 # includes special experts (Identity=3, Zero=4) at the tail
    dev = hidden.device; dt = hidden.dtype
    flat_t = torch.arange(N, device=dev).unsqueeze(1).expand_as(idx).reshape(-1)
    sorted_e, order = idx.reshape(-1).sort()
    st = flat_t[order]; sw = wt.reshape(-1)[order]
    counts = torch.bincount(sorted_e, minlength=E)
    glu_counts = counts[:E_glu]
    offs = glu_counts.cumsum(0).to(torch.int32)          # end-exclusive within GLU block, len E_glu
    n_glu = int(offs[-1].item())                          # 1 host sync
    row_act = torch.repeat_interleave(act_codes[:E_glu], glu_counts).to(torch.int32)
    st_glu = st[:n_glu]; sw_glu = sw[:n_glu]
    x_glu = hidden.index_select(0, st_glu).contiguous()
    gate_up = torch._grouped_mm(x_glu, gate_up_proj.transpose(-2, -1), offs=offs)    # (n_glu, 2I)
    inter = BatchedGLU.apply(gate_up, row_act)                                       # (n_glu, I)
    eo = torch._grouped_mm(inter, down_proj.transpose(-2, -1), offs=offs)            # (n_glu, H)
    out = torch.zeros(N, H, device=dev, dtype=dt)         # bf16 accumulate (k terms/row; grad PASS)
    out.index_add_(0, st_glu, eo * sw_glu.unsqueeze(-1))
    M = st.shape[0]
    if n_glu < M:                                          # special experts on the sorted tail
        tail_e = sorted_e[n_glu:]
        tail_codes = act_codes.index_select(0, tail_e)
        id_sel = (tail_codes == 3).nonzero(as_tuple=True)[0]   # Identity = weighted passthrough; Zero = skip
        if id_sel.numel() > 0:
            id_tok = st[n_glu:].index_select(0, id_sel)
            id_w = sw[n_glu:].index_select(0, id_sel)
            out.index_add_(0, id_tok, hidden.index_select(0, id_tok) * id_w.unsqueeze(-1))
    return out
