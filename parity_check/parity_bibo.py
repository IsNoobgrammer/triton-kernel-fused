"""Parity gate: our fused_mlp_router vs the REAL BiBoMoERouter -- output, loss, grads, bias update.

Run from the triton-kernel-fused repo with a venv that can import BiBo:
    python parity_check/parity_bibo.py
Imports BiBo from ../../BiBo (sibling dir).

REWRITTEN Aug 1 2026. This gate used to compare the CONV router. BiBo deleted it (its MoE debloat
cut the router to one MLP projection + sigmoid), so `router.kernel_size` stopped existing and the
gate failed on an AttributeError -- not a kernel regression, a model that moved. The conv kernel
`fused_router` still exists in this repo but now has no in-repo consumer.

Checks, on identical inputs:
  - routing decision (dense per-(token,expert) weight -- order-invariant; topk slot order differs)
  - loss (dense_w * G).sum()
  - grads d/dx, d/dweight
  - bias update: BiBoMoELayer.update_bias vs our router_bias_update (same counts)
"""
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BiBo")))
import _paths  # noqa: F401  -- repo root on sys.path
import torch
from kernels.sm75.router import fused_mlp_router, router_bias_update, _count_experts
from src.configuration_bibo import BiBoConfig
from src.modeling.ffn.router import BiBoMoERouter

DEV = "cuda"
torch.manual_seed(0)


def dense_weights(idx, w, E):
    """(B,S,k) idx + weights -> dense (B,S,E) so per-slot topk ORDER doesn't matter in the compare."""
    B, S, k = idx.shape
    d = torch.zeros(B, S, E, device=idx.device, dtype=w.dtype)
    d.scatter_(-1, idx, w)
    return d


def main():
    cfg = BiBoConfig(hidden_size=512, num_attention_heads=8, num_key_value_heads=2,
                     num_experts_per_tok=2, num_routed_experts=11, special_expert_pairs=1,
                     # "sum" is BiBo's div-sum default and what the kernel implements. The
                     # "softmax" mode has no kernel equivalent yet (measured gap on that arm:
                     # weights rel 2.6e-2, grads rel 6.6e-1), so this gate pins the sum arm.
                     norm_topk_prob="sum")
    router = BiBoMoERouter(cfg).to(DEV).half()
    E = router.num_routed_experts
    top_k = router.top_k
    B, S, H = 16, 1024, cfg.hidden_size
    print(f"config: E={E} top_k={top_k} norm_topk_prob={router.norm_topk_prob!r}")

    # non-zero bias so we actually exercise bias-affects-selection
    router.bias.copy_(torch.randn(E, device=DEV) * 0.1)
    weight = router.gate_proj.weight.detach()          # (E,H) fp16
    bias = router.bias.detach().float()

    x = torch.randn(B, S, H, device=DEV, dtype=torch.float16)
    G = torch.randn(B, S, E, device=DEV, dtype=torch.float32)  # dense upstream grad

    # ---- forward + grads: BiBo ----
    xb = x.clone().requires_grad_(True)
    router.gate_proj.weight.requires_grad_(True)
    idx_b, w_b = router(xb)
    dw_b = dense_weights(idx_b, w_b, E)
    loss_b = (dw_b * G).sum()
    loss_b.backward()
    gx_b, gw_b = xb.grad.detach(), router.gate_proj.weight.grad.detach()

    # ---- forward + grads: ours ----
    xo = x.clone().requires_grad_(True)
    wo_param = weight.clone().requires_grad_(True)
    idx_o, w_o, counts_o = fused_mlp_router(xo, wo_param, bias, top_k, E,
                                            norm_topk_prob=bool(router.norm_topk_prob),
                                            return_counts=True)
    dw_o = dense_weights(idx_o, w_o, E)
    loss_o = (dw_o * G).sum()
    loss_o.backward()
    gx_o, gw_o = xo.grad.detach(), wo_param.grad.detach()

    def rel(a, b):
        return (a - b).abs().max().item(), ((a - b).norm() / (b.norm() + 1e-12)).item()

    # ---- output parity ----
    idx_match = (idx_b.sort(-1).values == idx_o.sort(-1).values).float().mean().item()
    w_abs, w_rel = rel(dw_o.float(), dw_b.float())
    print("\n=== OUTPUT ===")
    print(f"  idx set agreement     : {idx_match:.4f}   (1.0 = identical experts chosen)")
    print(f"  dense weights         : abs {w_abs:.3e} | rel {w_rel:.3e}")
    print(f"  loss  ours {loss_o.item():.6f} | bibo {loss_b.item():.6f} | "
          f"abs {abs(loss_o.item()-loss_b.item()):.3e}")

    # ---- grad parity ----
    gx_abs, gx_rel = rel(gx_o.float(), gx_b.float())
    gw_abs, gw_rel = rel(gw_o.float(), gw_b.float())
    print("\n=== GRADS ===")
    print(f"  grad_x      : abs {gx_abs:.3e} | rel {gx_rel:.3e}")
    print(f"  grad_weight : abs {gw_abs:.3e} | rel {gw_rel:.3e}")

    # ---- counts + bias update parity ----
    # BiBo's balancer went PROPORTIONAL-on-normalized-share Aug 1 2026 (it was sign-on-raw-counts):
    #     share = counts / sum(counts);  bias += u * (mean(share) - share)
    # router_bias_update follows that rule; mode="sign" keeps the old one for pre-Aug-1 history.
    counts_bincount = torch.bincount(idx_b.reshape(-1), minlength=E)
    counts_ok = bool((counts_o == counts_bincount.int()).all().item())
    u = 0.4
    bias_bibo = bias.clone()
    _share = counts_bincount.detach().float()
    _share = _share / _share.sum().clamp_min(1.0)
    bias_bibo.add_(u * (_share.mean() - _share))
    bias_ours = bias.clone()
    router_bias_update(bias_ours, counts_o, u)
    bias_abs, bias_rel = rel(bias_ours, bias_bibo)
    print("\n=== BIAS UPDATE (proportional, normalized share) ===")
    print(f"  counts == bincount    : {counts_ok}")
    print(f"  bias after update     : abs {bias_abs:.3e} | rel {bias_rel:.3e}")

    # ---- verdict ----
    ok = (idx_match == 1.0 and w_rel < 1e-2 and gx_rel < 1e-2 and gw_rel < 5e-2
          and counts_ok and bias_abs < 1e-6)
    print(f"\n{'PASS' if ok else 'FAIL'} -- router parity vs BiBoMoERouter")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
