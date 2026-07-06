# ablate_muon (retired)

The OLM (online LM-emulator) benchmark and its Muon-ablation driver **moved to the standalone
`olm-bench` repo**: https://github.com/IsNoobgrammer/olm-bench

That repo has the clean, reusable version — fixed synthetic task + metrics + harness, with the
model and optimizer as swap points (SwiGLU, aurora/normuon/aurora_ema scale modes, spectral_wd,
xorth / xorth_post are all there). The Muon optimizer it screens is a **vendored snapshot** of
this repo's `kernels/` (`kernels/sm75/muon.py` + `kernels/muon/muon_scaling.py`) — those remain
here as the product; re-vendor into olm-bench when they change.

The grok-MoE ablation driver that used to live here (`run_ablation.py`, `grok_moe.py`) is retired —
grokking was the wrong regime for LM optimizer screening (see `.autoresearch/`), which is why OLM
replaced it.
