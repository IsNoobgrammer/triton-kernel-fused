"""Regression: shipped ManasOptimizer (rank-8 default + comp) must match the validated
sandbox ExpManas bit-for-bit on the same seed/data. If parity holds, the wave-11 held-out
number (+0.0235) transfers to the shipped class verbatim.
"""
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))
import torch
import torch.nn.functional as F

from exp_manas import ExpManas
from kernels.sm75.manas import ManasOptimizer

DEV = "cuda"


def run(cls, **kw):
    torch.manual_seed(0)
    m = torch.nn.Sequential(torch.nn.Linear(64, 96, bias=False),
                            torch.nn.Linear(96, 48, bias=False)).to(DEV)
    opt = cls([p for p in m.parameters()], lr=2e-3, probe_gamma=0.08, probe_rho=0.98,
              probe_rank=8, probe_refresh=5, weight_decay=0.01, **kw)
    g = torch.Generator(device="cpu").manual_seed(7)
    for _ in range(40):
        x = torch.randn(32, 64, generator=g).to(DEV)
        with opt.probe():
            loss = (m(x) ** 2).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
        opt.step()
    return torch.cat([p.detach().reshape(-1) for p in m.parameters()])


for comp in (None, 1.0):
    a = run(ManasOptimizer, comp=comp)
    b = run(ExpManas, comp=comp)
    d = (a - b).abs().max().item()
    print(f"comp={comp}: shipped vs sandbox max|dw| = {d:.3e}  {'PARITY' if d < 1e-5 else 'MISMATCH'}")
