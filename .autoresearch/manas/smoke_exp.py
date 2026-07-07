import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.dirname(__file__))
import torch

from exp_manas import ExpManas

for kw in (dict(ref_mode="ema"), dict(norm_mode="permat"), dict(sign=1.0),
           dict(trust=2.0), dict(ref_mode="ema", trust=2.0), dict(probe_src="muonmom")):
    torch.manual_seed(0)
    m = torch.nn.Linear(32, 32, bias=False).cuda()
    opt = ExpManas([m.weight], lr=1e-3, probe_gamma=0.01, probe_rho=0.94, **kw)
    w0 = m.weight.detach().clone()
    for _ in range(5):
        with opt.probe():
            loss = (m(torch.randn(16, 32, device="cuda")) ** 2).mean()
            opt.zero_grad()
            loss.backward()
        opt.step()
    with opt.probe():
        pass                                     # apply+remove round-trip
    assert torch.isfinite(m.weight).all() and not torch.equal(m.weight, w0)
    dn = torch.linalg.vector_norm(opt._full_d(m.weight)).item()
    print(f"{kw}: ok, ||d|| {dn:.5f}")
print("smoke ok")
