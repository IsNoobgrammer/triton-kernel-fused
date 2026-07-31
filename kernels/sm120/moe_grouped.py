import torch
from kernels.sm75.moe import BatchedGLU

__all__ = ["moe_grouped_cublas_polyglu", "grouped_supported", "prefer_grouped",
           "GROUPED_TOKENS_PER_EXPERT_MAX"]

GROUPED_TOKENS_PER_EXPERT_MAX = 2048


def grouped_supported(hidden, gate_up_proj, down_proj):
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
    e_glu = gate_up_proj.shape[0]
    routed = top_k_indices.numel()
    return (routed / max(e_glu, 1)) <= GROUPED_TOKENS_PER_EXPERT_MAX


def moe_grouped_cublas_polyglu(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    from kernels.sm75.moe import _code_max
    if _code_max(act_codes) > 4:
        raise ValueError("code 8 (radial) unsupported on the grouped path; use moe_per_expert(act_params=...)")
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
    offs = glu_counts.cumsum(0).to(torch.int32)
    n_glu = int(offs[-1].item())
    row_act = torch.repeat_interleave(act_codes[:e_glu], glu_counts).to(torch.int32)

    st_glu = st[:n_glu]
    sw_glu = sw[:n_glu]
    x_glu = hidden.index_select(0, st_glu).contiguous()
    gate_up = torch._grouped_mm(x_glu, gate_up_proj.transpose(-2, -1), offs=offs)
    inter = BatchedGLU.apply(gate_up, row_act)
    eo = torch._grouped_mm(inter, down_proj.transpose(-2, -1), offs=offs)

    out = torch.zeros(N, H, device=dev, dtype=dt)
    out.index_add_(0, st_glu, eo * sw_glu.unsqueeze(-1))

    n_routed = st.shape[0]
    if n_glu < n_routed:
        tail_codes = act_codes.index_select(0, sorted_e[n_glu:])
        sgn = torch.where(tail_codes == 3, 1.0, -1.0).to(sw.dtype)
        tok = st[n_glu:]
        out.index_add_(0, tok, hidden.index_select(0, tok) * (sw[n_glu:] * sgn).unsqueeze(-1))
    return out
