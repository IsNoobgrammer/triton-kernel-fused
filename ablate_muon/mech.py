"""Optimizer-side mechanism knobs, shared so olm can re-bench what grok screened.

Each is a no-op when its cfg value is 0/absent. `pre_step` acts on .grad before
opt.step(); `post_step` acts on weights after. `state` is a per-run dict the caller
creates once ({}) and threads through every step (holds EMA/slow/prev/power-iter buffers).

(grok_moe.py still has inline copies from the earlier waves; this is the reusable version
the online harness uses. Verdicts on grok live in ideas_todo.md — re-tested here because
grok was the wrong regime.)
"""
import torch
import torch.nn.functional as F


def pre_step(cfg, params, expert_ws, mblocks, state):
    if cfg.get("decor", 0) > 0:                                   # subtract shared expert grad
        for w in expert_ws:
            w.grad -= cfg["decor"] * w.grad.mean(0, keepdim=True)
    if cfg.get("grad_rep", 0) > 0:                                # amplify each expert's deviation
        for w in expert_ws:
            w.grad += cfg["grad_rep"] * (w.grad - w.grad.mean(0, keepdim=True))
    if cfg.get("xorth", 0):                                       # whiten expert grads along E axis
        for w in expert_ws:
            G = w.grad.reshape(w.shape[0], -1).float()
            C = G @ G.mT
            C = C / C.diagonal().mean().clamp_min(1e-12)
            ev, V = torch.linalg.eigh(C)
            isq = V @ torch.diag(ev.clamp_min(1e-6).rsqrt()) @ V.mT
            w.grad.copy_((isq @ G).reshape_as(w.grad).to(w.grad.dtype))
    if cfg.get("niche", 0) > 0:                                   # fitness-sharing lr (inverse load)
        for b in mblocks:
            f = (b.moe.load + 1) / (b.moe.load.sum() + b.moe.E)
            s = (1.0 / (b.moe.E * f)).pow(cfg["niche"]).clamp(0.5, 2.0)
            b.moe.w1.grad *= s.view(-1, 1, 1)
            b.moe.w2.grad *= s.view(-1, 1, 1)
    if cfg.get("grokfast", 0) > 0:                                # amplify slow/shared grad component
        gf = state.setdefault("gf_ema", {})
        a = cfg.get("gf_alpha", 0.98)
        for q in params:
            if q.grad is not None:
                e = gf.setdefault(q, torch.zeros_like(q.grad))
                e.lerp_(q.grad, 1 - a)
                q.grad += cfg["grokfast"] * e


def post_step(cfg, params, hidden, expert_ws, mblocks, state, dev, muon_lr, step):
    if cfg.get("repulse", 0) > 0:                                 # weight repulsion (PSO anti-avg)
        for w in expert_ws:
            w.add_(w - w.mean(0, keepdim=True), alpha=cfg["repulse"])
    if cfg.get("lookahead", 0):                                   # slow/fast weight averaging
        slow = state.setdefault("slow", {q: q.detach().clone() for q in params})
        if step % cfg["lookahead"] == 0:
            for q in params:
                slow[q].lerp_(q, cfg.get("la_beta", 0.5))
                q.copy_(slow[q])
    if cfg.get("scap", 0) > 0:                                    # cap only the top singular value
        sv = state.setdefault("scap_v", {})
        acc = []
        for q in hidden:
            W = q if q.ndim == 3 else q.unsqueeze(0)
            v = sv.get(q)
            if v is None:                                         # persistent power-iteration vector
                v = F.normalize(torch.randn(W.shape[0], W.shape[2], device=dev), dim=-1)
                sv[q] = v
            u = torch.einsum("brc,bc->br", W.float(), v)
            v.copy_(F.normalize(torch.einsum("brc,br->bc", W.float(), u), dim=-1))
            s = torch.einsum("brc,bc->br", W.float(), v).norm(dim=-1)
            acc.append(s.mean().item())
            W *= (cfg["scap"] / s.clamp_min(1e-12)).clamp(max=1.0).view(-1, 1, 1).to(W.dtype)
        state["scap_smax"] = sum(acc) / len(acc)                  # so caller can see if the cap binds
    if cfg.get("cautious", 0) > 0:                                # decay only where it doesn't fight
        prev = state.setdefault("prev", {q: q.detach().clone() for q in hidden})
        for q in hidden:
            u = q - prev[q]
            q.sub_(muon_lr * cfg["cautious"] * q * (u * q < 0))
            prev[q].copy_(q)
