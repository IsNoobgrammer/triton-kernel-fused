"""sm120 Manas: (1) gamma=0 is EXACTLY FusedMuon, (2) gamma>0 actually changes the outcome.

Part (2) is not optional ceremony. The per-expert-alpha bug shipped INERT with a green kernel
parity test because that test called the kernel directly and never asserted the feature moved an
end-to-end number ([[parity-vs-plumbing]]). Manas has three separate ways to be silently inert --
probe_min_votes gating to plain Muon, probe_gamma left at 0, vote() never called -- so the control
arm and the live arm are BOTH asserted here.

Run:  python parity_check/parity_manas_sm120.py            (CPU is fine; use_gram/symmul self-gate to cuBLAS)
"""
import copy

import _paths  # noqa: F401  -- repo root on sys.path
import torch

from kernels.sm120.muon import FusedMuon
from kernels.sm120.manas import ManasOptimizer, NS8_COEFFS

STEPS, ACCUM, SEED = 6, 4, 0
MUON_KW = dict(lr=3e-3, momentum=0.95, weight_decay=0.1, coeffs=NS8_COEFFS,
               ns_dtype=torch.float32, aurora_k=1, scale_mode="aurora",
               cautious_decay=False, use_gram=False, use_symmul=False)


def _model():
    torch.manual_seed(SEED)
    return torch.nn.Sequential(torch.nn.Linear(32, 64, bias=False),
                               torch.nn.Tanh(),
                               torch.nn.Linear(64, 32, bias=False))


def _run(opt_fn):
    torch.manual_seed(SEED)
    m = _model()
    opt = opt_fn(list(m.parameters()))
    probe = getattr(opt, "probe", None)
    vote = getattr(opt, "vote", lambda: None)
    torch.manual_seed(1234)                       # identical data for every arm
    batches = [[torch.randn(8, 32) for _ in range(ACCUM)] for _ in range(STEPS)]
    losses = []
    for step in batches:
        opt.zero_grad(set_to_none=True)
        tot = 0.0
        for x in step:
            if probe is not None:
                with probe():
                    (m(x).pow(2).mean() / ACCUM).backward()
                vote()
            else:
                (m(x).pow(2).mean() / ACCUM).backward()
            tot += float(m(x).pow(2).mean()) / ACCUM
        opt.step()
        losses.append(tot)
    return opt, m, losses


def main():
    global ACCUM
    muon_opt, muon_m, muon_loss = _run(lambda p: FusedMuon(p, **MUON_KW))

    manas_kw = dict(micro_vote=True, probe_rank=None, probe_rho=1.0, probe_rho_step=0.96, **MUON_KW)
    ctl_opt, ctl_m, ctl_loss = _run(lambda p: ManasOptimizer(p, probe_gamma=0.0, **manas_kw))
    live_opt, live_m, live_loss = _run(lambda p: ManasOptimizer(p, probe_gamma=0.04, **manas_kw))

    # (0) the shim really is riding the sm120 base, not sm75's
    assert ManasOptimizer._polar.__module__ == "kernels.sm120.muon", ManasOptimizer._polar.__module__

    # (1) CONTROL: gamma=0 -> probe never displaces theta -> bit-identical to FusedMuon
    ctl_err = max(float((a - b).abs().max()) for a, b in zip(muon_m.parameters(), ctl_m.parameters()))
    print(f"gamma=0 vs FusedMuon   max|dW| = {ctl_err:.3e}   (want exactly 0)")
    assert ctl_err == 0.0, "gamma=0 must be EXACTLY FusedMuon -- the arms are confounded otherwise"

    # (2) LIVE: the probe must (a) build a real consensus buffer, (b) move the weights,
    #     (c) leave theta CLEAN after step() -- no probe offset leaking into evals/checkpoints.
    d = [live_opt.state[p]["manas_d"] for p in live_opt._probe_params() if "manas_d" in live_opt.state[p]]
    dmax = max(float(x.abs().max()) for x in d)
    live_err = max(float((a - b).abs().max()) for a, b in zip(muon_m.parameters(), live_m.parameters()))
    print(f"gamma=0.04 |D|max      = {dmax:.4f}   over {len(d)} probed matrices")
    print(f"gamma=0.04 vs muon     max|dW| = {live_err:.3e}   (want >> 0)")
    print(f"votes last step        = {live_opt._votes_last} (accum {ACCUM})")
    print(f"loss@final  muon {muon_loss[-1]:.6f}  ctl {ctl_loss[-1]:.6f}  manas {live_loss[-1]:.6f}")
    assert len(d) == 2 and dmax > 1e-3, f"probe buffer inert: {len(d)} buffers, |D|max {dmax}"
    assert live_opt._votes_last == ACCUM, f"expected {ACCUM} votes/step, got {live_opt._votes_last}"
    assert live_err > 1e-6, "probe_gamma>0 did not change the weights -- Manas is plumbed but INERT"
    assert not live_opt._shift_on, "theta left shifted after step() -- evals would read theta+d"

    # (3) the ga1 self-gate: 1 vote/step must fall back to plain Muon, exactly
    ACCUM = 1
    ga1_opt, ga1_m, _ = _run(lambda p: ManasOptimizer(p, probe_gamma=0.04, **manas_kw))
    mu1_opt, mu1_m, _ = _run(lambda p: FusedMuon(p, **MUON_KW))
    ga1_err = max(float((a - b).abs().max()) for a, b in zip(mu1_m.parameters(), ga1_m.parameters()))
    print(f"ga1 self-gate vs muon  max|dW| = {ga1_err:.3e}   (want exactly 0)")
    assert ga1_err == 0.0, "probe_min_votes=2 must make ga1 exactly plain Muon"

    print("\nPASS: control inert, live arm moves weights, theta clean, ga1 gates to Muon.")


if __name__ == "__main__":
    main()
