# ablate_muon — Grok-MoE optimizer ablation (T4-ready)

Self-contained ablation of Muon (+ expert-repulsion / AdamW control) on a hard synthetic task:
**multi-op modular arithmetic** (add/sub/mul/div mod 97) with a **BiBo-style MoE** in each block,
trained under a **skewed op mix** (40/30/20/10). It measures not just held-out accuracy and
time-to-grok, but **functional specialization** — the mutual information (bits) between which
expert fires and which op the token is — per layer.

Question: does uniform-load balancing fight specialization under skew, and can optimizer-level
diversity pressure (expert weight repulsion) buy it back?

## Why T4
T4 **is** the `sm75` target the repo's `FusedMuon` is built for, so this uses the real optimizer
(fp16 Newton-Schulz on T4 tensor cores), not a stand-in. No custom compile needed — the Muon path
is pure-torch matmuls; nothing here imports Triton.

## Dependencies
**Only `torch`** — which is preinstalled on Kaggle and Colab, so there is nothing to `pip install`.
(`FusedMuon` imports torch and a pure-torch scaling module; no triton, no einops, no extras.)
On a bare box without torch: `pip install torch` (a CUDA build matching your driver).

## Run it — one command (auto-detects T4 x2)

New Notebook → Settings → Accelerator = **GPU T4 x2**. Then one cell:

```python
!git clone --depth 1 https://github.com/IsNoobgrammer/triton-kernel-fused
%cd triton-kernel-fused
!bash ablate_muon/run.sh
```

`run.sh` detects the GPU count. On **T4 x2** it forks two processes — shard 0 (even-indexed
arms) on GPU 0, shard 1 (odd) on GPU 1, in parallel — each streaming to the **console AND** its
own log (`ablate_muon/logs/g0.log`, `g1.log`, prefixed `[gpu0]`/`[gpu1]`), then auto-merges into
`results.jsonl` + the final table. On a single GPU it runs all arms sequentially. Force single
with `bash ablate_muon/run.sh 1`. Watch a GPU live: `!tail -f ablate_muon/logs/g0.log`.

Runtime: ~3000 steps/arm, a few minutes each on T4. x1 ~20-30 min total; x2 ~half that.

Manual equivalent (same one-process-per-GPU pattern as `bench/run.sh`):
```bash
export PYTHONPATH=.
CUDA_VISIBLE_DEVICES=0 python ablate_muon/run_ablation.py --shard 0 --nshards 2 &
CUDA_VISIBLE_DEVICES=1 python ablate_muon/run_ablation.py --shard 1 --nshards 2 &
wait
python ablate_muon/run_ablation.py --merge          # combined table
```

## What runs (edit `run_ablation.py` → `ARMS` / `COMMON`)
| arm | what |
|---|---|
| default s0, s1 | Muon (dsv4_10 coeffs, aurora_k1, wd 2.0) — the baseline |
| adamw s0, s1 | AdamW control (same wd) |
| default rep 1e-3, rep 1e-2 | Muon + expert weight repulsion `W_e += beta*(W_e - mean_E W)` after each step |

Config lives in `COMMON` (steps, batch, p, frac, wd, experts, top_k, bias_tokens, op_mix).
For a quick smoke: set `steps=400` in `COMMON`, or run one arm: `python ablate_muon/grok_moe.py`.

## Reading the output
Final table per arm: `acc` (held-out over ALL ops), `grok` (first step ≥90%), `MI(L)` = bits of
expert↔op MI per layer (higher = more specialized; ~1.85 is the ceiling under this op entropy),
`per-op` accuracy. The load balancer (DeepSeek-V3 selection-bias) fires every `bias_tokens`
(300k) tokens — a slow global fairness nudge that leaves per-batch routing free to specialize.

## Files
- `grok_moe.py` — model + data + `run(cfg)->dict` (importable; bootstraps `FusedMuon`).
- `run_ablation.py` — the driver + arm list + table.
- `results.jsonl` — one JSON per arm (written on run), includes full acc/MI curve.
