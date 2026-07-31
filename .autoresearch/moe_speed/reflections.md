## run 1 - diagnosis
Profile says the dominant MoE cost for radial is NOT arithmetic and NOT the RMS: it is
Memcpy DtoH at 11.2% (1426 calls / 8 steps = 178 per step), caused by the per-expert
dispatch path. _glu_bwd_row_kernel fires 1863x/step where a grouped path would fire 32x.
Radial is on that path ONLY because `ap32 is not None` disqualifies it from `uniform`.
I predicted load imbalance would dominate - WRONG. gate_up GEMM is 41.7% but that is work.
Lesson: count kernel launches before theorising about occupancy.

## run 2 - intervention DESIGNED, not yet applied (context budget)

THE FIX (atomic, one change):
  moe.py:866   uniform = all(c <= 2 or c >= 6 for c in codes) and ap32 is None
                                                              ^^^^^^^^^^^^^^^^ drop this rider
`uniform`'s REAL requirement is "no special experts" (codes 3/4 must skip the GLU, and a
whole-buffer call would run it on them) -- see the comment at moe.py:861-865. The ap32 clause is
a separate, over-conservative rider: _glu_fwd ALREADY takes a PER-ROW alpha with a stride
(_ap_stride returns 1 for per-row, 0 for broadcast), and rows are sorted by expert, so the
per-expert scalar expands to a per-row vector with a single on-device repeat_interleave.

FORWARD (easy):
    cnts   = tensor([bounds[e+1]-bounds[e] for e in range(E)], device=dev)   # bounds already host
    row_ap = torch.repeat_interleave(ap32[:, 0].contiguous(), cnts)
    it_all = _glu_fwd(gu_all, row_act, code_hint=hint, row_alpha=row_ap)
  ...and the same in the use_gmm branch.

BACKWARD (the part that must NOT be skipped -- this is why run 2 was not shipped):
  the per-expert path does `grad_act_params[e, 0] = da.sum()`. Batched, `da` comes back PER ROW,
  so it needs a segment-sum by expert:
    grad_act_params[:, 0] = torch.zeros(E, device=dev, dtype=torch.float32) \
                                 .index_add_(0, row_expert_idx, da)
  where row_expert_idx = repeat_interleave(arange(E), cnts). Shipping the forward alone would
  leave dalpha wrong AND SILENT -- radial would train with a corrupted theta gradient. Invariant 1
  says a speedup that changes the math is a FAIL, so this lands as one change or not at all.

TOGGLE TEST (what confirms the diagnosis, per Non-Negotiable 3):
  1. DtoH calls/step must drop from ~178 (re-run prof_moe.py, compare the Memcpy DtoH row)
  2. _glu_bwd_row_kernel launches/step must drop from 1863 toward ~32
  3. radial tps must rise from 158.2k
  4. ALL 5 parity suites green -- parity_radial in particular checks d_theta vs finite difference,
     which is exactly the gradient the segment-sum could break
  If 1+2 move but 3 does not, the launches were not the bottleneck and the diagnosis was wrong.

EXPECTED CEILING: DtoH is 11.2% of CUDA time, pure stall. Even partial recovery plausibly clears
the 165k target on its own (158.2 -> 165 is +4.3%). If it lands, silu is untouched by this change
(silu is already uniform), so the 170k silu target needs a DIFFERENT intervention -- likely the
gate_up GEMM at 41.7%.
## run 3 - DISCARD, diagnosis refuted (the useful kind)
Cut GLU launches 58x (14901->256) and tps got WORSE (158.2->157.2k). Memcpy DtoH did not move
one call (1426->1426). So launch count was never the bottleneck at this scale.
REAL cause of the 178 syncs/step: _sort_by_expert hands counts/bounds to the HOST, and BOTH the
per-expert and batched branches slice the GEMMs with those host bounds. Identical either way.
Only _grouped_mm dodges it (device-side offs).
LESSON: I confirmed a mechanism and assumed it was the cause. The toggle test is what caught it -
without it I would have shipped a regression and called it a win. Attack the GEMM side next.

## run 4 - overhead family CLOSED
Killed 248 of 1426 DtoH calls; total DtoH TIME did not change (1540->1557ms). 1.3ms per copy is
pipeline drain, not transfer - the sync waits on queued GPU work, so the wait is conserved.
Two refutations now (run 3 launches, run 4 syncs). STOP optimising overhead; we are GPU-bound.
The act_codes cache STAYS in tree - it is free, safe, and helps every arm - but it is not a win.
ESCALATE: gate_up GEMM 41.5%. Radial does 64 cuBLAS torch.mm/layer and never touches the fused
Triton GEMM (behind uniform+use_gmm). That is the only remaining item big enough to matter.

## runs 5-7 - GEMM refuted, small win banked
Run 5: radial on the fused/grouped GEMM = 156.7k, WORSE. Third refutation. The per-expert cuBLAS
loop is already good at this size; the recorded prior (grouped wins microbench, loses in training)
held. num_warps sweep: 8 is already optimal (12.0/14.1us; 16 is much worse).
Runs 6-7: act_codes cache + removing the counts DtoH->HtoD round trip -> 159.0k over 6 readings
vs 158.1k baseline. +0.9k, MARGINAL - baseline itself spans 157.5-158.5 between runs.
SCOREBOARD: 3 refutations (launches, syncs, GEMM path), 1 marginal win, 1 no-op (warps).
The path is GPU-bound with a tuned GEMM and a tuned GLU. Only unexamined bucket: 'other' 35.4%.

## run 8 + STOP (patience 3, and context exhausted)
BLOCK_I=256 exact tiling: microbench -6% on bwd, end-to-end NEUTRAL (158.6k). That is the FIFTH
time a microbench win failed to transfer. Standing rule for this codebase, now with 5 data points:
NEVER trust a microbench for MoE dispatch or tiling - only the 40-step end-to-end eval counts.
FINAL: radial 158.9k over 9 readings (best 159.7k) vs 158.1k baseline. +0.8k. TARGET 160k NOT MET.
Scoreboard: 4 refutations (launches, syncs, grouped GEMM, tiling), 1 marginal win (act_codes cache
+ counts round-trip), 1 no-op (num_warps already optimal).
CONCLUSION: this path is GPU-bound with an already-tuned GEMM and GLU. The remaining 1.1k is not
in any bucket I profiled. Only 'other' (35.4%) is unexamined.

## run 9 - 'other' bucket broken down. No single lever left.
  Memcpy DtoH        1583ms 14.6%   922 calls  (1.7ms each = DRAIN, not transfer; refuted twice)
  vectorized_elem     423ms  3.9%  4080
  indexFuncLargeIndex 305ms  2.8% 10255  (the grad_hidden scatter)
  bfloat16_copy       274ms  2.5%  7047  (dtype casts, ~1175/step)
  Memcpy HtoD         244ms  2.2%   162
  direct_copy         229ms  2.1%  3744
  elementwise         220ms  2.0%  3654
  Memcpy DtoD          74ms  0.7% 13898  (2316 launches/step, trivial time)
The remaining time is SPREAD across many small copies/casts inherent to per-expert dispatch.
No single item is worth the 1.1k gap; the largest (DtoH) is drain we already proved is conserved.
STOPPING at target-not-met. Getting 160k needs a structural change (fewer experts touched per
token, or a genuinely fused dispatch), not another micro-optimisation.

