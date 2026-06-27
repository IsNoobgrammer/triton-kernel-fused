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

## Round 1 RESULTS (T4, compiled) — convergence
- **MoE per-expert = 2.87× fwd+bwd, grad PASS. The SOLE durable win on T4+compile.** Keep.
- MoE grouped (tl.dot) = 0.10× — dead on Turing (confirmed). Disabled sm_<80.
- **MoE grouped_cublas FAILED**: `torch._grouped_mm` is **bf16/fp8-only + needs sm_80+** tensor
  cores → cannot run on T4 (sm_75). It's an **Ampere/Hopper-only** path. Made it bf16 + arch-gated
  (clean skip on Turing); UNTESTED (no sm_80+ box in the loop). This is THE lever if we leave T4.
- **CE = REDUNDANT under compile.** All budgets ~0.74× time AND the memory edge is gone:
  inductor's compiled `cross_entropy` already chunks, so 384MB/1GB use *more* mem than compiled
  eager; only 128MB squeaks 1.11× less. The old "1.34× less" was vs *un*-compiled eager. → CE is a
  no-compile fallback, not a win. **Close the CE optimization thread.**

## Verdict / where the loop stands
On **T4 + torch.compile**, the repo's one production-grade win is **MoE per-expert**. Everything
else is either redundant under compile (CE, SwiGLU, XSA, conv) or off-arch for Turing (both grouped
paths). MoE per-expert at 2.87× is likely near the T4 ceiling for the cuBLAS-loop approach — the only
clear path past it is a **bf16 grouped GEMM on Ampere/Hopper** (grouped_cublas), a different GPU.
Remaining T4 lever for per-expert (marginal): make dispatch fully GPU-resident (kill the one
`.tolist()` + Python schedule) — but it already wins *with* that sync, so expect small gains.

## Lessons (do not relearn)
- `torch._grouped_mm` = bf16/fp8 + sm_80+ ONLY. Useless on Turing. Right tool on Hopper/Ampere.
- Under torch.compile, inductor already chunks cross_entropy → hand-rolled chunked CE has no edge.
- The MoE win survives compile ONLY because compile can't fuse data-dependent routing.
- tl.dot GEMM never beats cuBLAS on Turing. Don't tune it — replace it.
- Measure only vs COMPILED eager on T4. Ampere/eager numbers mislead.
- A faster kernel that fails grad_rel < 1.5e-2 is a fail, not a win.
