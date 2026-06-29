"""Examples: how to use the PolyGLU MoE kernel.

    python examples/moe_usage.py        # runs all examples on CUDA

The router stays in YOUR model — these kernels take the router's top-k indices/weights and the
stacked expert weights, and return the combined per-token output. Expert weight layout:
    gate_up_proj : (E, 2*I, H)     # fused gate+up projection per expert
    down_proj    : (E, H, I)       # down projection per expert
    act_codes    : (E,) int32      # per-expert activation: 0=SiLU, 1=ReLU², 2=Tanh
"""
import torch
import torch.nn as nn

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kernels.sm75 import moe, moe_per_expert
from kernels.sm75.moe import moe_grouped, GROUPED_MIN_TOKENS   # advanced: explicit grouped path (sm_80+)

DEV, DT = "cuda", torch.float16


def make_experts(E, H, I):
    """Stacked PolyGLU expert weights + groups-of-3 activation codes [SiLU, ReLU², Tanh, ...]."""
    gate_up_proj = (torch.randn(E, 2 * I, H, device=DEV, dtype=DT) * 0.02).requires_grad_(True)
    down_proj = (torch.randn(E, H, I, device=DEV, dtype=DT) * 0.02).requires_grad_(True)
    act_codes = torch.tensor([e % 3 for e in range(E)], device=DEV, dtype=torch.int32)  # PolyGLU triples
    return gate_up_proj, down_proj, act_codes


def route(hidden, gate, top_k):
    """A plain top-k softmax router (this is the part that lives in your model)."""
    probs = torch.softmax(gate(hidden).float(), dim=-1)        # (N, E)
    top_w, top_idx = torch.topk(probs, top_k, dim=-1)          # (N, k)
    return top_idx, (top_w / top_w.sum(-1, keepdim=True)).to(hidden.dtype)


# ── Example 1: one-call auto-dispatch (grouped if rows >= GROUPED_MIN_TOKENS, else per-expert) ──
def example_auto():
    N, H, I, E, k = 4096, 512, 768, 9, 2
    hidden = torch.randn(N, H, device=DEV, dtype=DT)
    gate = nn.Linear(H, E, bias=False).to(DEV, DT)
    gup, dwn, act = make_experts(E, H, I)
    idx, w = route(hidden, gate, k)
    out = moe(hidden, idx, w, gup, dwn, act)                   # auto picks the path
    rows = N * k
    print(f"[auto] N={N} rows={rows} -> {'grouped' if rows >= GROUPED_MIN_TOKENS else 'per-expert'} | out {tuple(out.shape)}")


# ── Example 2: force a path (per-expert at small N, grouped at large N) ──
def example_force_path():
    H, I, E, k = 512, 768, 8, 2
    gup, dwn, act = make_experts(E, H, I)
    gate = nn.Linear(H, E, bias=False).to(DEV, DT)
    for N, fn, label in [(256, moe_per_expert, "per-expert (small N)"),
                         (8192, moe_grouped, "grouped (large N)")]:
        hidden = torch.randn(N, H, device=DEV, dtype=DT)
        idx, w = route(hidden, gate, k)
        out = fn(hidden, idx, w, gup, dwn, act)
        print(f"[force] {label}: N={N} -> out {tuple(out.shape)}")


# ── Example 3: a full MoE layer as an nn.Module (router + experts + training step) ──
class MoELayer(nn.Module):
    def __init__(self, H, I, E, top_k=2):
        super().__init__()
        self.gate = nn.Linear(H, E, bias=False)
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * I, H) * (H ** -0.5))
        self.down_proj = nn.Parameter(torch.randn(E, H, I) * (I ** -0.5))
        self.register_buffer("act_codes", torch.tensor([e % 3 for e in range(E)], dtype=torch.int32))
        self.top_k = top_k

    def forward(self, x):                                       # x: (B, S, H)
        B, S, H = x.shape
        flat = x.reshape(B * S, H)
        idx, w = route(flat, self.gate, self.top_k)
        out = moe(flat, idx, w, self.gate_up_proj, self.down_proj, self.act_codes)
        return out.reshape(B, S, H)


def example_training_step():
    B, S, H, I, E = 8, 512, 512, 768, 9
    layer = MoELayer(H, I, E).to(DEV, DT)
    x = torch.randn(B, S, H, device=DEV, dtype=DT, requires_grad=True)
    out = layer(x)
    loss = out.float().pow(2).mean()
    loss.backward()                                            # grads flow to gate, experts, x
    gnorm = layer.gate_up_proj.grad.norm().item()
    print(f"[layer] out {tuple(out.shape)} | loss {loss.item():.4f} | grad on experts? {gnorm > 0}")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    example_auto()
    example_force_path()
    example_training_step()
    print("OK")
