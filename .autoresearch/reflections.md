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

## Round 4 (the WHOLE conv router vs compile) — 2026-06-28
- Dump diagnosis: compiled router = transpose+pad(Triton) -> **cuDNN extern conv** -> sigmoid/bias ->
  native topk -> gather/sum/div. The compiler itself keeps the conv on cuDNN; only the cheap glue is
  Triton. Op is tiny (369 MFLOP, ~16MB x) => fwd is LAUNCH/overhead-bound (5 kernels), not compute.
- User chose "build reference, then attack". Built: `ref` (dump recipe hand-assembled, uncompiled,
  pure-autograd -> cuDNN convolution_backward) + `readonce` (tldot fwd with H-outer/K-inner loop
  reorder, hypothesis: K overlapping taps reread x from L1/L2 not HBM). Dropped `cublas` from sweep.
- All three CORRECT locally (idx 1.0, count==bincount, NaN-free, grad PASS; ref bit-exact).
- **Yellow flag (Ampere, weak proxy)**: readonce FWD = 1.394ms vs tldot 1.007ms — reorder HURT.
  Likely x (1MB/batch) already fits the 4MB L2, so loop order doesn't change HBM traffic, only tl.dot
  scheduling. If T4 confirms, the conv has NO compute edge over cuDNN -> pivot to ITER 2.
- **ITER 2 (gated on T4)**: fuse top-2-of-11 + gather + norm INTO the conv epilogue (in-register,
  scores never touch HBM) — the one thing the compiler CANNOT fuse (topk is a native op). This is the
  real candidate structural edge (cf. MoE/XSA wins), stronger than the conv-compute reorder. Build
  only if T4 shows the conv itself can ~tie cuDNN; else the honest answer is "use ref (cuDNN) and only
  the glue fusion is ours to win."
- AWAIT: `python bench.py --compile router` on T4.

## Round 4 iter1 T4 verdict + iter2 (the win) — 2026-06-28
- **readonce REFUTED on T4** (fwd 0.33x): the H-outer/K-inner reorder wrecked tl.dot pipelining on
  Turing; x (~1MB/batch) was already L2-resident so there was no HBM traffic to save. Ampere yellow
  flag was right. REMOVED. Lesson: don't reorder loops to "save" traffic that the cache already serves
  — you only perturb the MMA schedule. (Same class as prior local≠T4 traps.)
- **Profiler nailed the real structure** (the marksaroufim payoff): our tl.dot conv is ~2.5x slower
  than cuDNN (fwd kernel 862us vs cudnn_convolution 296us; bwd dx+dw 1610us vs cuDNN
  convolution_backward 643us). The dump calling cuDNN was correct — stop reimplementing the conv.
- **Two measured edges:** (1) ref's UNCOMPILED cuDNN backward BEATS compiled (1.20x) — inductor's
  compiled bwd adds a 137us constant_pad_nd_3 + scatter kernels. (2) aten::topk = ~295us + gatherTopK
  216us is a FLAT tax on every path INCLUDING compiled (a general bitonic kernel for top-2-of-11).
- **iter2 = cudnn backend** (data-justified, not guessed): F.conv1d (autograd gives cuDNN
  convolution_backward free = the 1.20x bwd) + `FusedTop2Epilogue` — ONE Triton kernel doing
  sigmoid+bias+top-k-argmax+unbiased-gather (BLOCK_N rows/program, vectorized argmax over E=11),
  killing the native topk+gather. This is THE structural edge: compile MUST call native topk; we fuse
  a bespoke top-2-of-11 (~free). Local: correct, grad_rel 3.4e-4. AWAIT T4 — expect first conv WIN.
- **General lesson reinforced**: the conv-router win (like MoE routing, XSA read-once) comes from
  fusing/replacing the op the compiler is FORCED to leave as a separate library call (topk), NOT from
  out-computing cuDNN. Find the native-op seam, not the GEMM.

## Round 4 CONVERGED — cudnn ties compile at the topk seam — 2026-06-28
- **cudnn T4 = 0.97x fwd+bwd** (best backend; tldot 0.73x, ref 0.85x), grad_rel 2.2e-7 (exact), mem
  parity. The FusedTop2Epilogue eliminated aten::topk (gone from the profile). A real 33% jump over
  tldot, now TYING compiled — but NOT a clear >1.0x speed win.
- **Why >1.0x isn't reachable (profiler-proven, matches the prior-round pattern):** our Triton conv
  kernel alone = 818us ~= compiled's WHOLE forward (821us). cuDNN owns the conv — inductor itself
  punts to cuDNN (extern_kernels.convolution), so there's no Triton conv that beats it on T4. Backward
  is the SAME convolution_backward (647us) on cudnn/ref/compiled -> tie by construction. The earlier
  ref-bwd "1.20x" was baseline throttling noise (eager fwd swung 0.44-1.0x run-to-run).
- **The win we DID get = the topk seam.** Like MoE (routing) and XSA (read-once), the edge is fusing
  the op the compiler is FORCED to leave as a separate library call (torch.topk). Killing the ~295us
  native topk + 216us gather is what closed 0.73 -> 0.97. There was exactly one such seam here.
- iter3 micro-opt: F.pad -> conv1d(padding=K-1)[...,:S] (causal-equivalent, kills the 16.8MB pad
  copy). Marginal. Set default backend = cudnn.
- **SHIP cudnn** as the conv-router default: ties compile on speed, wins on grad accuracy (2.2e-7 vs
  compiled's fp16) and on the tldot comparison. tldot kept for the mem-bound case (1.12x less mem).
- **Lesson (reinforced):** on T4+compile you win at native-op seams (topk, data-dependent routing,
  read-once reductions), NOT by out-computing cuDNN/cuBLAS. Profile FIRST to find the seam vs the
  immovable library call — the marksaroufim dump tactic paid off twice this round (found cuDNN-is-the-
  conv, found topk-is-the-tax). readonce (out-compute attempt) was the one refuted candidate.

## Round 4 REOPENED — fwd wins, merging the backward — 2026-06-28
- **I converged too early (corrected).** The iter3 micro-opt (F.pad -> conv1d(padding=K-1)) made the
  cudnn FORWARD WIN on T4: 1.15x (0.703 vs 0.810ms), topk gone from the profile. The op is bandwidth-
  bound (~52us fwd floor; compiled ~16x off) so there WAS headroom — anchoring on "can't beat cuDNN"
  was wrong; cuDNN is itself far off the floor on this degenerate 11-channel conv.
- **Backward 0.83x = the remaining gap.** Profiler: same convolution_backward (642us) as compiled, but
  wrapped in ~700us of UNFUSED glue (aten::copy_ + elementwise dominate: autograd copies x->contiguous
  twice, casts, transposes grad). compiled fuses it; our autograd path didn't.
- **iter4 = merged backward** (`FusedConvRouterCuDNN`, custom autograd): save contiguous xc ONCE in
  fwd (reused in bwd, not recopied) + call torch.ops.aten.convolution_backward DIRECTLY + fused
  `_router_epilogue_bwd_kernel` (sigmoid'+scatter in one kernel). Local (uncompiled) bwd 0.96->1.28x,
  fwd+bwd 1.17->1.41x. Correct (grad_rel 8.5e-5). AWAIT T4 — if bwd clears ~1.0x = first OVERALL win.
- Lesson: don't accept a "tie" while a phase still carries unfused glue — the profiler's copy_/
  elementwise rows ARE the to-do list. And re-question a convergence when one phase is bandwidth-bound
  and far off the HBM floor.

## Round 4 — FIRST WIN (cudnn 1.13x) + channels-last A/B — 2026-06-28
- **cudnn = first overall conv-router WIN on T4: fwd 1.15x, bwd 1.16x, fwd+bwd 1.13x**, grad 3.1e-7
  (exact), mem parity. The merged manual backward (convolution_backward direct + fused epilogue-bwd +
  save-contiguous-once) flipped bwd 0.83->1.16x. Banked.
- Profile (per-iter): 93% is cuDNN — conv_bwd 611us + conv_fwd 312us + **layout transposes 482us**
  (nchwToNhwc 315 + nhwcToNchw 167). Our fused epilogue+norm = 25us (already minimal). Backward ~1.5x
  forward is EXPECTED (2 grad GEMMs vs 1 fwd GEMM), not recompute waste — corrected the user's "save
  fwd values" intuition: there's no recompute to cut; we already save the input.
- **Next lever = the 482us transpose tax is a self-inflicted DOUBLE transpose**: x is natively
  (B,S,H)=NHWC; we .contiguous() to NCHW; cuDNN converts BACK to NHWC for its sm75_..._nhwc kernel.
  `cudnn_cl` feeds the channels-last strided view so cuDNN skips nchwToNhwc (+saves the 16.8MB copy).
  REGRESSED on Ampere local (1.39->1.19) — but Ampere doesn't emit those explicit transposes; T4 does.
  Classic local!=T4 -> A/B on T4, champion protected (don't overwrite the proven win with a gamble).
- Honest ceiling: beyond the transpose tax, the conv fwd/bwd GEMMs (923us) are the cuDNN floor; only a
  bandwidth-optimal transpose-free Triton conv (fwd+bwd, 52us/105us floors, 6-15x headroom but hard —
  tl.dot attempt was 818us) beats that. That's the next stepping stone if cudnn_cl lands.

## Round 4 — channels-last A/B REFUTED; cudnn 1.12x stands — 2026-06-28
- `cudnn_cl` (channels-last to skip cuDNN transposes) LOST on T4: fwd+bwd 0.95x vs cudnn 1.12x (bwd
  0.84x). Profile: nchwToNhwc STILL 6.31ms (identical to cudnn) — cuDNN copies to its own layout
  regardless; the strided input just ADDED copies (13.2->20.0ms) + slowed convolution_backward
  (613->948us). The double-transpose theory was wrong; the 482us transpose tax is NOT cuDNN-removable.
- **Process win: ZERO regression.** Champion was kept as a SEPARATE backend (not overwritten), so the
  failed gamble cost nothing. This is why you A/B on held-out instead of replacing the champion with an
  unconfirmed change — exactly the discipline that saved us here. Dropped cudnn_cl, restored clean cudnn.
- Confirmed ceiling for the cuDNN approach: conv fwd 312 + conv bwd 613 + transposes 482 are all cuDNN
  and immovable from outside. The ONLY remaining lever is a transpose-free bandwidth-optimal Triton conv
  (fwd+bwd; 52/105us floors vs cuDNN 312/613) — the hard rewrite. cudnn 1.12x is the shipped win.

## Round 5 OPENED — transpose-free Triton conv (beat cudnn 1.12x) — 2026-06-28
- Shipped cudnn (1.12x) as default; BiBo parity PASS @ E=11 (parity_bibo.py: idx 1.0, weights 4.5e-8,
  loss 8e-6, grads tight, bias update EXACT vs BiBoMoELayer.update_bias). Gate cleared before shipping.
- Round 5 goal: beat cudnn by going transpose-free (kill the 482us cuDNN layout-transpose tax; op is
  bandwidth-bound, 52us fwd / 105us bwd floors, cuDNN 6-16x off). Champion PROTECTED (separate backends,
  A/B on T4) — the discipline that saved us when channels-last regressed.
- **iter1 = `tlconv`**: merged-contraction forward (K taps folded into one K*H=2048 contraction, fatter
  tl.dots vs tldot's K skinny-512 dots). Forward logits bit-identical to tldot (correct). Whether it's
  FASTER than cuDNN's forward is a T4 question (tldot's fwd was 818us = 15x off floor; merged may or may
  not close that — Ampere local can't predict Turing tl.dot).
- Gotcha logged: triton @autotune can leave a stale output buffer during the FIRST correctness check
  (cold cache) -> false grad CHECK. Fixed: bench warms autotune before the check. Kernel proven correct
  in isolation regardless.

## Round 5 CLOSED — cudnn shipped, repo cleaned to cudnn-only — 2026-06-28
- tlconv (merged contraction) crashed on T4 (128x512 fp16 tile = 128KB > T4's 64KB SRAM); capped to
  BLOCK_C<=128 it collapses to ~tldot (loses). T4's 64KB SRAM is the wall for every transpose-free
  Triton conv attempt on Turing — the bandwidth headroom (fwd 52us / bwd 105us floors, cuDNN+compile
  8-16x off) is real but NOT reachable on sm_75 (needs Ampere/Hopper SRAM).
- **DECISION (user): keep cudnn only, remove all other backends, document the rest.** Done:
  `kernels/router.py` is now cudnn-only (removed tldot/cublas/tlconv/ref + their kernels + the dx/dw
  tl.dot imports). bench router = cudnn vs compiled. Full refuted-approaches ledger + the Ampere/Hopper
  revisit plan written to `.autoresearch/conv_router_findings.md`. BiBo integration unchanged (already
  cudnn-only). Parity PASS @ E=11 after cleanup.
- Round 4-5 final: conv router = cudnn 1.11-1.17x on T4, shipped + integrated into BiBo
  (router_type='conv' fast path). The transpose-free Triton conv round is documented for future GPUs.
