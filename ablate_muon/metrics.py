"""Expert utilization + specialization metrics for the MoE ablations.

Replaces the noisy single-number minload and the top-1-only MI:
- load_stats: effective-experts exp(H(load)) and normalized load entropy from the FULL
  load distribution (over real, non-pad tokens). eff=E means perfectly balanced.
- soft_mi: mutual information between the ACTUAL top-k routing (both selected experts,
  weighted by combine weight) and the task label, in bits, plus the specialization
  fraction (MI / ceiling) so it is comparable across layers/runs.
"""
import math

import torch


def load_stats(load):
    """load: (E,) accumulated routing counts. -> dict(eff, norm_ent, minload)."""
    E = load.numel()
    p = load / load.sum().clamp_min(1e-20)
    nz = p[p > 0]
    H = float(-(nz * nz.log()).sum()) if nz.numel() else 0.0
    return dict(eff=math.exp(H), norm_ent=H / math.log(E), minload=float(p.min()))


def soft_mi(idx, w, labels, E, n_lab):
    """idx,w: (N,k) top-k experts + combine weights; labels: (N,). Weighted MI both experts.
    Returns (mi_bits, spec_fraction)."""
    joint = torch.zeros(E, n_lab, device=idx.device)
    lab = labels.unsqueeze(1).expand_as(idx).reshape(-1)
    joint.index_put_((idx.reshape(-1), lab), w.reshape(-1).float(), accumulate=True)
    joint = joint / joint.sum().clamp_min(1e-20)
    pe, pl = joint.sum(1, keepdim=True), joint.sum(0, keepdim=True)
    nz = joint > 0
    mi = float((joint[nz] * (joint[nz] / (pe @ pl)[nz]).log2()).sum())
    he = float(-(pe[pe > 0] * pe[pe > 0].log2()).sum())           # expert entropy
    hl = float(-(pl[pl > 0] * pl[pl > 0].log2()).sum())           # label entropy
    ceil = min(he, hl)
    return mi, (mi / ceil if ceil > 1e-9 else 0.0)
