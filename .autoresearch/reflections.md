# Reflections — kernel optimization loop

## Round 0 (baseline, T4 compiled — 2026-06-28)
- Established frozen eval (`bench.py --compile --json moe ce` on T4). Baselines in state.json.
- **MoE per-expert = 2.85× fwd+bwd** — the durable win (compile can't fuse data-dependent routing).
- **CE = 0.61× time but 1.34× less memory** — a memory/OOM play, not speed.
- grouped MoE = 0.10× on Turing (tl.dot cliff) → auto-disabled sm_<80.
- Already landed (pre-loop): CE backward `float()` host-sync removed (was a graph break in the log);
  grouped gated off Turing.

## Open levers to try (ranked)
1. MoE per-expert: GPU-resident dispatch — kill `.tolist()` + Python schedule loop (host syncs).
2. MoE: `torch._grouped_mm` (torch 2.10 native cuBLAS grouped GEMM) → a Turing-fast grouped path,
   replacing the dead tl.dot one. Highest upside if the op exists on the T4 image.
3. CE backward: cut the 3-GEMM recompute (the 0.52× bwd is the bottleneck).

## Round 1 (candidates pushed, awaiting T4 sweep)
`python bench.py --compile` now sweeps, vs **compiled** eager, with backward derived as
(fwd+bwd − fwd) (the retain_graph re-backward was bogus under compile — gave CE bwd 24× at 0.99ms).
- **MoE candidates**: `per-expert` (baseline 2.85×), `grouped` (tl.dot control, ~0.10× on T4),
  **`grouped_cublas`** (`torch._grouped_mm`, GPU-resident, cuBLAS — the Turing candidate; guarded,
  reports FAILED if the op/signature differs on the T4 image).
- **CE candidates**: chunk-budget sweep `384MB / 1GB / 128MB` (launches vs peak memory).
- Default run = moe + ce only (swiglu/xsa/conv are fallbacks, named-only).
- Workflow: `git pull` + `python bench.py --compile` → paste @@RESULT lines. Nothing else.
WATCH on T4: does `grouped_cublas` exist + grad-pass + beat per-expert's 2.85×? Does any CE budget
push fwd+bwd toward ≥1.0×?

## Lessons (do not relearn)
- tl.dot GEMM never beats cuBLAS on Turing. Don't tune it — replace it.
- Measure only vs COMPILED eager on T4. Ampere/eager numbers mislead.
- A faster kernel that fails grad_rel < 1.5e-2 is a fail, not a win.
