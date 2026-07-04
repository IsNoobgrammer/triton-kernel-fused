"""Cheap alternatives to Muon's Newton-Schulz, for the 'same quality, less compute' arm.

- SinkGD (arXiv 2502.06742): Sinkhorn = alternating row/col RMS normalization of the
  momentum. ZERO NS-style GEMMs (reductions only) -> the cheapest matrix optimizer. Also
  the concrete row/col-norm optimizer the SinkGD/LEO family research grouped together.
- Dion (arXiv 2504.05295, Microsoft): low-rank orthonormalization via amortized power
  iteration (warm-started across steps) + error feedback on the momentum buffer. Cost
  O(m*n*r) vs NS O(m*n*min(m,n)*iters); rank_frac r trades quality for compute. r=1.0
  should approach Muon.
- LEO (github.com/vukrosic/leo-optimizer): Lion-style double-EMA momentum + ONE-SHOT
  element-wise row/col normalization (D/row_norm + D/col_norm), RMS-scaled to align_const.
  No matmuls; element-wise only. Paper defaults lr 0.01, betas (0.9,0.99), align 0.3.

Both handle 2D and stacked-3D (E, m, n) params (FusedMuon batches the latter natively).
Updates are scaled to RMS 0.2 so lr is shared with Muon/AdamW in the olm harness.
"""
import torch
from torch.optim import Optimizer

RMS = 0.2                                                              # update RMS band (Muon convention)


def _as3d(x):
    return x.unsqueeze(0) if x.ndim == 2 else x


def _scale_rms(U):                                                     # per-matrix -> global RMS = 0.2
    r = U.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
    return U * (RMS / r)


class SinkGD(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0, iters=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay, iters=iters))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mu, wd, it = g["lr"], g["momentum"], g["weight_decay"], g["iters"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "mom" not in st:
                    st["mom"] = torch.zeros_like(p)
                buf = st["mom"]
                buf.mul_(mu).add_(p.grad)
                U = _as3d(buf).float().clone()
                m, n = U.shape[-2], U.shape[-1]
                for _ in range(it):                                    # Sinkhorn: row then col RMS
                    U = U / (U.norm(dim=-1, keepdim=True) / (n ** 0.5)).clamp_min(1e-12)
                    U = U / (U.norm(dim=-2, keepdim=True) / (m ** 0.5)).clamp_min(1e-12)
                U = _scale_rms(U).reshape_as(p).to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(U, alpha=-lr)


class LEO(Optimizer):
    def __init__(self, params, lr=1e-2, betas=(0.9, 0.99), weight_decay=0.0, align=0.3):
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay, align=align))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, (b1, b2), wd, al = g["lr"], g["betas"], g["weight_decay"], g["align"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                if "m" not in st:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                c = m.mul(b1).add(p.grad, alpha=1 - b1)            # Lion-style interp direction
                m.mul_(b2).add_(p.grad, alpha=1 - b2)              # EMA update (uses old m)
                D = _as3d(c).float()
                row = D.norm(dim=-1, keepdim=True).clamp_min(1e-12)  # per-row (over cols)
                col = D.norm(dim=-2, keepdim=True).clamp_min(1e-12)  # per-col (over rows)
                U = D / row + D / col                              # one-shot row+col normalize
                rms = U.pow(2).mean(dim=(-2, -1), keepdim=True).sqrt().clamp_min(1e-12)
                U = (U * (al / rms)).reshape_as(p).to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(U, alpha=-lr)


class Dion(Optimizer):
    def __init__(self, params, lr=1e-3, momentum=0.95, weight_decay=0.0, rank_frac=0.25):
        super().__init__(params, dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                                      rank_frac=rank_frac))

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mu, wd, rf = g["lr"], g["momentum"], g["weight_decay"], g["rank_frac"]
            for p in g["params"]:
                if p.grad is None:
                    continue
                st = self.state[p]
                E, m, n = _as3d(p).shape
                r = max(1, int(round(rf * min(m, n))))
                if "mom" not in st:                                    # error-feedback momentum + warm Q
                    st["mom"] = torch.zeros(E, m, n, device=p.device, dtype=torch.float32)
                    q = torch.randn(E, n, r, device=p.device, dtype=torch.float32)
                    st["Q"] = torch.linalg.qr(q).Q
                B, Q = st["mom"], st["Q"]
                B.add_(_as3d(p.grad).float())                          # accumulate grad
                P = torch.linalg.qr(torch.matmul(B, Q)).Q              # (E,m,r) orthonormal cols
                R = torch.matmul(B.transpose(-2, -1), P)               # (E,n,r) ~ V*Sigma
                B.sub_(torch.matmul(P, R.transpose(-2, -1)))           # error feedback: drop captured
                B.mul_(mu)                                             # decay residual
                Vr = R / R.norm(dim=-2, keepdim=True).clamp_min(1e-12)  # -> orthonormal right vecs
                st["Q"] = Vr                                           # warm start next step
                U = torch.matmul(P, Vr.transpose(-2, -1))              # (E,m,n) rank-r orthonormal upd
                U = _scale_rms(U).reshape_as(p).to(p.dtype)
                if wd:
                    p.mul_(1 - lr * wd)
                p.add_(U, alpha=-lr)
