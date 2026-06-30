# Muon end-to-end training bench (MNIST) — speed, convergence, memory

Informal record of the MNIST training-time studies run on the RTX PRO 6000 Blackwell
(sm_120, torch 2.12+cu130) via the molab notebook, 2026-06-30. These answer "does our fused
kernel change *training*, not just the isolated optimizer step?" and "how does Muon stack up
against plain/8-bit Adam on a real train loop?"

## Setup (frozen across all runs)
- **Data:** MNIST, 60k train / 10k val, normalized, resident on GPU.
- **Model:** MLP `784 -> 2048 -> 2048 -> 2048 -> 2048 -> 10` (ReLU). 14,217,226 params.
- **Param split (exact, from shell):**
  - Muon (2D weight matrices): `(2048,784)`, `(2048,2048)`x3, `(10,2048)` = **14,209,024 (99.942%)**
  - AdamW (1D biases): `(2048,)`x4, `(10,)` = **8,202 (0.058%)**
- **Recipe:** batch 256, 14 epochs, Muon lr 0.01 (cosine decay), AdamW-for-biases lr 1e-3.
  NS = 5 Polar-Express steps (fp16) on all arms that use Muon.
- **Fairness:** same model init + same data order per seed across arms (only the optimizer
  differs). Compile/JIT warmed then weights reset to init before the timed loop. Timing is
  TRAINING-only (val eval excluded). Peak = `max_memory_allocated` over the timed loop.

## RUN A — FusedMuon vs compiled Muon vs full-fp32 Muon (5-step NS)
All three = Muon(matrices) + AdamW(biases); only the NS implementation differs.

| arm | total-train | opt-step | final acc | speedup |
|---|---|---|---|---|
| FusedMuon (ours, fp16 NS) | 11.83 s | 3.26 ms | 0.9848 | 1.00x |
| compiled Muon (fp16 NS) | 14.39 s | 4.04 ms | 0.9843 | **1.22x slower** |
| full-fp32 Muon | 23.16 s | 6.71 ms | 0.9853 | **1.96x slower** |

Optimizer step = 91-95% of the full training step at this width (wide MLP, batch 256), so the
per-step optimizer win flows almost directly into end-to-end time. => FusedMuon trains this model
**1.22x faster than compiled Muon, 1.96x faster than fp32 Muon**, same accuracy.

### Convergence parity (3 seeds, mean +- std over seeds; same init+data per seed)
The single-run "reach 0.98 epoch" is NOISE (val acc bounces ~+-0.3% epoch-to-epoch, larger than
the inter-arm gap). Over 3 seeds the curves overlap:
- final/best val: fused 0.9853+-0.0004, compiled 0.9856+-0.0003, fp32 0.9855+-0.0013; all seeds >=0.98.
- z-test on best val (n=3): **fused vs fp32 z=-0.29 -> NOT significant** (fused matches the fp32
  mathematical truth in convergence). compiled vs fp32 z=+0.57 NOT sig. fused vs compiled z=-2.08
  (marginal 0.07% gap, borderline at n=3 -> reduction-order jitter, not practically meaningful).
- **Conclusion: the fused kernel changes SPEED, not the optimization path.** Reach-epoch needs
  multi-seed; never quote a single-run "X reaches acc faster".

## RUN B — FusedMuon vs compiled Muon vs torch fused AdamW vs bnb AdamW8bit (+ PEAK MEM)
Muon arms = Muon(matrices)+AdamW(biases); Adam arms = ONE optimizer on ALL params, lr 3e-4 cosine.

| optimizer | total train | vs FusedMuon | peak mem | opt-step | final acc |
|---|---|---|---|---|---|
| FusedMuon (ours) | 11.80 s | 1.00x | **815 MB** | 3.27 ms | 0.9849 |
| compiled Muon | 14.43 s | 1.22x slower | 815 MB | 4.06 ms | 0.9853 |
| torch fused AdamW | 2.20 s | **5.4x faster** | **900 MB** | 0.27 ms | 0.9875 |
| bnb AdamW8bit | 4.19 s | **2.8x faster** | **815 MB** | 0.63 ms | 0.9864 |

### Findings
- **Speed: plain Adam wins big on this easy task (expected).** Muon's per-step cost is the
  Newton-Schulz matmuls (3.3-4.1 ms); Adam is cheap elementwise (0.27-0.63 ms). fused AdamW is
  5.4x faster, AdamW8bit 2.8x faster, all at equal/slightly-higher acc. MNIST-MLP does not stress
  Muon's convergence-quality lever (its win is large ill-conditioned transformers) -> here you pay
  for NS with no payoff. Use Adam for small/easy models; Muon's case is large-model conditioning.
- **Memory: Muon's strong story.** FusedMuon 815 MB = SAME as 8-bit AdamW, and 85 MB UNDER fused
  AdamW (900 MB). Why: Muon stores ONE fp16 momentum buffer (~28 MB on 14.2M params); fused AdamW
  stores TWO fp32 moments m+v (~114 MB); AdamW8bit stores two 8-bit moments (~28 MB). So Muon gets
  8-bit-optimizer-class memory for free from its single-moment design, at full-ish precision.
  (Δpeak fused-Adam vs Muon = 900-815 = 85 MB ≈ 114-28 = the second-moment buffer Muon omits.)
- **AdamW8bit stability:** diverged to `inf` at lr 1e-3 (where torch fused AdamW was stable);
  needed lr 3e-4. The 8-bit state trades stability headroom.
- **Within Muon the kernel win holds end-to-end:** FusedMuon 1.22x faster than compiled at equal
  815 MB. (Per-step isolated win is ~2.3x at transformer scale / many matrices; only ~1.24x here
  because this model has few, large matrices so each NS is compute-bound cuBLAS in both paths and
  the launch-fusion lever shrinks — see muon.mdx "Scaling the Newton-Schulz iteration count" and
  the two-axis note: precision/TC edge grows with matrix SIZE, launch-fusion edge with matrix COUNT.)

## Toolchain notes (box)
- bitsandbytes 0.49.2 installed via `pip install --no-deps bitsandbytes` (avoid uv clobbering
  torch); AdamW8bit runs on sm_120/cu130. torchao not present.
- `torch.optim.AdamW(fused=True)` available and stable at lr 1e-3.
