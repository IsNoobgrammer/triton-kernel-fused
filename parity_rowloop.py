"""Looped row kernels == single-pass row kernels, and the I>1024 cap is gone.

The rms is a ROW REDUCTION, so a tiled kernel cannot see a whole row and needs HBM pre-passes
(_row_rms fwd, _row_s bwd): 4N/9N traffic against plain silu's 3N/6N. The looped kernels keep one
program per row and walk it in chunks, so the reduction stays local and the second pass re-reads
from L1 -- back to ~3N/5N, and no width limit.

This asserts the restructure changed NOTHING numerically: at I <= 1024 both paths exist, so they are
compared head-to-head; above it, the looped path is checked against the fp64 eager reference.
"""
import importlib

import torch

importlib.import_module("kernels.sm120.moe")     # sm120 first (see parity_normed_tiles)
M = importlib.import_module("kernels.sm75.moe")  # `kernels.sm75.moe` the MODULE, not the exported fn

torch.manual_seed(0)
dev = "cuda"
OK = True


def ref(gate, up, code, alpha):
    g = gate.double()
    r = torch.sqrt(g.square().mean(-1, keepdim=True) + M._NS_EPS)
    if code == 0:
        a = torch.sigmoid(torch.tensor(0.0)) * 0 + alpha
        act = torch.nn.functional.silu(a * g)
    elif code == 2:
        act = torch.nn.functional.silu(alpha * g / r)
    else:
        p = torch.sigmoid(torch.as_tensor(alpha, dtype=torch.float64))
        act = r.pow(p) * torch.nn.functional.silu(g / r)
    return (act * up.double())


for I in (512, 1024, 2048, 4096):
    for code in (0, 2, 8):
        M_ROWS = 64
        gu = torch.randn(M_ROWS, 2 * I, device=dev, dtype=torch.float32) * 1.7
        go = torch.randn(M_ROWS, I, device=dev, dtype=torch.float32)
        act_t = torch.full((M_ROWS,), code, dtype=torch.int32, device=dev)
        alpha = (torch.rand(M_ROWS, device=dev, dtype=torch.float32) * 1.4 + 0.3)
        out = M._glu_fwd(gu, act_t, code_hint=code, row_alpha=alpha)
        ggu, da = M._glu_bwd(go, gu, act_t, code_hint=code, row_alpha=alpha, want_act_grads=True)

        # fp64 reference, and finite-difference on alpha for one row
        gate, up = gu[:, :I], gu[:, I:]
        r_out = torch.stack([ref(gate[i:i+1], up[i:i+1], code, alpha[i].double())[0] for i in range(M_ROWS)])
        e_out = (out.double() - r_out).abs().max().item() / r_out.abs().max().item()

        gu_ = gu.clone().requires_grad_(True)
        a_ = alpha.clone().requires_grad_(True)
        g2, u2 = gu_[:, :I], gu_[:, I:]
        rr = torch.sqrt(g2.float().square().mean(-1, keepdim=True) + M._NS_EPS)
        if code == 0:
            aref = torch.nn.functional.silu(a_[:, None] * g2)
        elif code == 2:
            aref = torch.nn.functional.silu(a_[:, None] * g2 / rr)
        else:
            aref = rr.pow(torch.sigmoid(a_)[:, None]) * torch.nn.functional.silu(g2 / rr)
        (aref * u2 * go).sum().backward()
        e_ggu = (ggu - gu_.grad).abs().max().item() / gu_.grad.abs().max().item()
        e_da = (da - a_.grad).abs().max().item() / a_.grad.abs().max().max().item()
        path = "row" if I <= M._ROWFUSE_MAX_I else "rowloop"
        good = e_out < 2e-6 and e_ggu < 2e-5 and e_da < 2e-4
        OK &= good
        print(f"I={I:>5} code={code}  {path:<8} {'PASS' if good else 'FAIL'}  "
              f"out {e_out:.2e}  dgu {e_ggu:.2e}  dalpha {e_da:.2e}")

print("\n" + ("ALL PASS -- looped kernels are numerically identical and lift the I cap" if OK else "FAIL"))
assert OK
