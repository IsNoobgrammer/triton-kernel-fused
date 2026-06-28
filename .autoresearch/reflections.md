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

## Round 2 (outsourced kernels — do THEY beat compile?) — pushed, awaiting T4
Control test before concluding "hand-written kernels are pointless under compile": bench the famous
EXTERNAL kernels in the same --compile harness vs compiled eager.
- **Liger SwiGLU** + **Liger fused-linear CE** (`liger-kernel`, full fwd+bwd, canonical reference).
- **bassrehab/triton-kernels** `fused_moe_forward` (FORWARD-ONLY, standard SwiGLU, self-routing) vs
  compiled-eager MoE forward — forward speed + output-match only (no backward in their kernel).
Default run now = moe, ce, swiglu, liger_swiglu, liger_ce, bassrehab. Harness pip-installs liger;
bassrehab self-clones (blanks its __init__ to dodge the eager broken-import bug).
WATCH: if Liger/bassrehab ALSO lose to compiled eager on T4 → confirms "compile already does this"
is universal, not our incompetence. If they WIN → our kernels are subpar; adopt their approach.

## Round 2 partial (T4) — data + 2 bugs found
Data (compiled eager baseline): MoE per-expert **2.98×** (winner, stable), grouped 0.10× (dead),
grouped_cublas cleanly SKIPPED (sm_<80), CE ~0.75× + no mem edge (redundant confirmed), SwiGLU 0.96×.
- **BUG 1**: Liger SwiGLU CRASHED the whole run — torch.compile can't compile Liger's
  autograd.Function ("leaf Variable ... in-place op"), and the main loop only caught OOM → aborted
  everything after SwiGLU. FIX: main loop now catches ALL exceptions per-bench (one crash never
  aborts the sweep).
- **BUG 2 (methodology)**: I was compiling the KERNEL side too. Wrong — you don't wrap a hand-written
  autograd.Function in torch.compile (unrepresentative + triggers the Liger crash). FIX: compile
  ONLY the eager baseline; kernels run native Triton (eager). Liger's own benches do this.
Round 2b pushed with both fixes + added Cut-Cross-Entropy (Apple cut_cross_entropy) and bassrehab
SwiGLU (auto fwd-only detect). Re-run to get Liger/CCE/bassrehab numbers cleanly.

## Lessons (do not relearn)
- Do NOT torch.compile a custom autograd.Function — unrepresentative AND crashes some (Liger).
  Compile only the eager baseline; run the kernel as its native Triton.
- Every contender in its own try/except — one crash must not abort the sweep.
- `torch._grouped_mm` = bf16/fp8 + sm_80+ ONLY. Useless on Turing. Right tool on Hopper/Ampere.
- Under torch.compile, inductor already chunks cross_entropy → hand-rolled chunked CE has no edge.
- The MoE win survives compile ONLY because compile can't fuse data-dependent routing.
- tl.dot GEMM never beats cuBLAS on Turing. Don't tune it — replace it.
- Measure only vs COMPILED eager on T4. Ampere/eager numbers mislead.
- A faster kernel that fails grad_rel < 1.5e-2 is a fail, not a win.

## Round 3 (user pushback — investigated, not gated)
- **CCE on T4 = FIXABLE**: root cause was `CCE_AUTOTUNE=0` (default) → no early_config_prune → one
  fixed config needs 96KB shared mem > T4's 64KB. Set `CCE_AUTOTUNE=1` → prune drops over-budget
  configs + caps num_stages<=2 on Turing → runs. (Was wrong to gate it off; now enabled.)
- **Why Liger SwiGLU uses less mem than compile despite being SLOWER**: Liger recomputes silu in
  backward and writes grads IN-PLACE into the saved input buffers (its comment: "recomputation to
  save memory") → no grad allocation. Speed and memory are separate axes; Liger trades a hair of
  recompute for ~6% mem, still 3% slower than compiled eager.
- **Tried matching it (in-place backward in OUR swiglu) → NaN**. Our bwd kernel is @triton.autotune;
  autotune benchmarks it ~10x on the SAME buffer, so an in-place write corrupts trial 2+. Liger
  avoids this by NOT autotuning. In-place + autotune are incompatible. Reverted — marginal mem win
  not worth dropping autotune on a kernel that loses to compile on speed anyway.
- **Why ours can't beat compile on elementwise (1x)**: for memory-bound elementwise ops inductor is
  already at the HBM bandwidth ceiling — no fusion left to reclaim; a hand kernel just pays ~4%
  launch/codegen overhead. You don't out-code the compiler on trivial ops. Confirmed across ours +
  Liger + bassrehab (all ~0.95-0.97x).

## Round 2 (CE latency at fixed memory) — 2026-06-28
Re-scoped: CE is the memory/ENABLING kernel (only CE that runs in ce_oom). Goal = cut its latency
vs compiled WHILE keeping the memory saving. NEW: lifted inductor's CE to raw Triton
(`ce_compiled.py`, materialize-once + no-bwd-recompute) so the loop runs LOCALLY (3050 proxy
N=4096/V=32000) as the optimization set; T4 ce_fit = held-out.
- **iter1 fused forward = clear KEEP.** Removing the `.float()` (C,V) fp32 buffer and fusing
  logsumexp+gather into ONE online-softmax Triton kernel (read fp16, fp32-accum in registers) cut
  mem-parity latency 1.88x->1.44x AND memory ~40% (64MB budget now 1.31x LESS mem than compiled,
  was 1.01x). The `.float()` was doubling the chunk transient AND adding 2 extra passes. grad PASS.
- Lesson: our old forward did mm->.float()->logsumexp->gather = 1 GEMM + 3 reduction passes + an
  fp32 (C,V) alloc. The fp32 buffer is what made small-V proxy peak WORSE than compiled. Fuse first.
- Remaining gap (~1.44x) is the BACKWARD recompute GEMM = the deliberate memory price; can't remove
  without saving logits (= compiled's memory). So ~1.4x may be near the low-mem floor. Next levers:
  chunk-size tuning (fewer launches now fwd is light), grad-logit kernel tiling — expect small.

## Round 2 cont. — #3 refuted, #1 (int8) is the real win
- **#3 (one-shot budget) REFUTED by T4**: chunk count barely affects backward (recompute GEMM dominates, not chunk overhead). one-shot wastes memory for no speed. Kept budget as a pure MEMORY dial; corrected the docstring that oversold one-shot. The eval did its job — killed my wrong hypothesis.
- **#1 (int8 saved-logits) = big latency win.** Save logits as int8+per-row-scale in fwd, DEQUANTIZE in bwd -> skip the recompute GEMM (3->2 GEMM). Naive torch quantize LOST (1.44x: abs/amax/div/round/to-int8 = ~5 passes + fp16 temps cost more than the GEMM saved). FUSING the quantize (absmax folded into the reduce kernel + one int8 write kernel) -> **1.09x latency (nearly ties compiled)**, grad PASS 1.2e-2, NaN-free. Same lesson as iter1: fuse the elementwise, never torch-op it.
- int8 grad_rel 1.2e-2 is close to the 1.5e-2 gate (it IS ~the fp16 GEMM noise floor; compiled = 1.11e-2). If T4 V=81000 pushes it over, fallback = per-row-BLOCK scales (finer quant). Watch on T4.

## #1 (int8) REFUTED on T4 held-out — opt-set (proxy) overfit
- Proxy V=32000 said int8 = 1.09x (great). T4 V=81000 said 0.67x AND 2x worse memory (2633 vs recompute 1305). DISCARD.
- Why: per-row int8 needs abs-max -> a 2nd full (N,V) forward pass (+33ms) that ate most of the -69ms backward GEMM saving (net only -36ms / 10%). And it HOLDS (N,V) int8 = 1.3GB -> peak 2x worse than recompute, breaking the keep-memory goal. Disqualified by the objective's memory constraint, not just slow.
- **Lesson (GEPA opt-set-overfit): the proxy's small V cheapened both the extra forward pass AND the held int8 (131MB vs 1.3GB) — it flattered int8 on the exact axes T4 punished. A win on the opt-set is a hypothesis; the held-out (T4) is the verdict.** Always confirm latency+memory at true V before believing a quant win.
- CE loop CONVERGED: ship recompute + fused-forward (iter1) + gw.add_ (iter2). fwd tied compiled, bwd at recompute floor, budget = memory dial (2.35x @384MB .. 3.5x @128MB less mem). int8 dead, one-shot dead.

## Round 3 (NEW external baseline = Liger; "recompute floor" REFUTED) — 2026-06-28
- Benched ours vs **Liger fused-linear CE** (`liger_ce_sweep`, same mem-budget grid as `ce_sweep`).
  T4 verdict: **Liger BEAT our recompute path** at matched memory (Liger@2048 300ms/1083MB vs ours
  ~372ms). Our "recompute GEMM is the irreducible floor" diagnosis (round-2 state.json) was WRONG.
- **Why Liger wins / why our floor was self-inflicted**: CE's grad w.r.t. logits = (softmax-onehot)/n
  needs ONLY logits+labels (loss scalar → upstream grad is a scalar). Liger computes the FULL grad in
  the FORWARD chunk loop while logits are live → never stores (N,V), never recomputes. We had split
  fwd(lse)/bwd(recompute) — the 4th GEMM was our design choice, not a law. **Lesson: an external SOTA
  baseline is worth more than N self-comparisons — the local proxy only ever raced us against
  ourselves, so it never exposed the wrong split.**
- **FIX = `_CEFusedFwdBwd`**: grad-in-forward, no recompute (4→3 GEMM). T4: ours @192MB **260ms/904MB
  beats Liger@1024 321ms/836MB (19% faster) AND Liger@2048 (faster, 180MB less)**. Pareto-ahead.
- **Sub-trap (occupancy)**: first cut fused reduce+grad into ONE per-row kernel (V streamed twice
  serially, only `chunk` programs) — LOCALLY 1.41x faster, but T4 LOST to Liger (launch/occupancy-
  bound; latency *dropped* with bigger chunk = the tell). FIX: split into per-row `_fwd_reduce` (lse)
  + 2D-grid `_grad_logits_inplace` (~chunk/32×V/256 programs) → saturates SMs → 400→260ms @128MB,
  curve flattened (GEMM-bound). **Lesson: local (3050, small V) hides occupancy; a one-program-per-row
  kernel over a huge V is launch-bound on the wider T4 — prefer 2D grids. Local ≠ T4, AGAIN.**
- **Grads tighter than Liger**: ours grad_hidden rel 1.9e-3 vs Liger 1.1e-2 (vs fp32), grad_weight
  7e-4 both, loss bit-identical to Liger / eager 1e-6. We out-accuracy the SOTA baseline too.
- BMM/grouped-GEMM for the chunk GEMMs: REJECTED (analysis, not benched). Batching needs all chunk
  operands resident = the full (N,V) = the memory we chunk to avoid; the fully-batched limit IS the
  compiled baseline. And our chunk GEMMs are already large (M~1242), not launch-bound. No free lunch.
- CONVERGED (round 3): ship `_CEFusedFwdBwd` @192MB default. Beats Liger on speed AND grad accuracy;
  compiled std CE still fastest when logits fit (CE is the memory/OOM play). int8 + recompute REMOVED
  (dominated). Ported to BiBo `src/kernels/fused_ce.py`.

## Round 3b (XSA beats inductor) — 2026-06-28
- XSA was 0.86x fwd+bwd vs compiled (README "fallback only"). Diagnosed: the kernel launched
  **one program per (b,kv,s) row, BLOCK_D=128**, doing cross-lane D-reductions — the SAME
  fine-granularity occupancy trap as the CE one-pass kernel. Bandwidth-starved, not the op being hard.
- **FIX = re-tile**: XBLOCK rows/program + vectorized D-reduction (axis=1), keep the single fused
  kernel per phase. T4: **fwd 1.20x, fwd+bwd 1.15x** (was 0.36/0.86), grad PASS. BEATS inductor.
- **Why it WINS (a real structural edge, unlike pure elementwise)**: our ONE fused kernel reads V
  once + computes ‖v‖² inline; inductor emits TWO kernels and reads V twice (norm kernel, then main).
  Traffic ~10 vs ~12 units (Hq4/Hkv2) → ~15-20% less. A graph can't recover this: inductor won't fuse
  a reduction's consumer back into the reduction. Predicted ~15%, measured 20% on fwd.
- Backward = 0.98x (tie) as predicted: more terms (grad_V accumulates over the group), no traffic edge.
- **Lesson: "elementwise loses to compile" is NOT universal — it holds for ops with NO structural edge
  (SwiGLU, conv). XSA has one (the shared-V reduction across the GQA group → read V once in a single
  kernel). The re-tile is the same move that fixed CE: replace one-program-per-row with XBLOCK-rows +
  vectorized inner reduction. Fine-grained per-row launches are the recurring T4 anti-pattern.**
- **fwd+bwd-fusion line clarified**: ONE kernel for fwd+bwd is possible ONLY for terminal/scalar-loss
  ops (CE — grad needs no upstream tensor). Mid-network ops (XSA, SwiGLU, MoE, conv) MUST run backward
  as a separate phase (grad depends on the downstream grad_output, absent at forward) → max fusion =
  one fused kernel PER phase, which XSA now has.
