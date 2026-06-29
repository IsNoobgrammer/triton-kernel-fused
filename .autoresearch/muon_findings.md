# Fused Muon — findings

## Round 1 (local RTX 3050 / Ampere, fp16 params, 48-tensor 75.5M synthetic BiBo set)
Levers: torch._foreach_* per-param sweeps + baddbmm NS epilogue folding + optional fp16-tensor-core NS.

| variant     | parity vs eager | speed   |
|-------------|-----------------|---------|
| fused-fp32  | 2.44e-04 (bit-tight, PASS) | 1.16x |
| fused-fp16  | 4.88e-04 (diff op) | **2.70x** |

fp16-NS stability gate PASS: SV mean 0.92-0.94 (slightly under-orth vs fp32's ~1, fp16 rounding),
|Δp|/lr attribution 0.175-0.179 flat across all shapes (vs eager ~0.19 — softer but in [0.15,0.25] gate).

The old AGENTS.md conclusion ("compile is the only lever") missed: baddbmm folds the NS axpy into the
GEMM (no pointwise kernels), foreach collapses the N-param launch tax, and fp16 NS engages tensor cores.
NEXT: verify on T4 (sm_75) — fp32 has NO tensor cores there, so the fp16 gap should be even larger.
Champion = fused-fp32 (bit-tight). fp16 = opt-in pending T4 stability + a real training-loss check.

## Round 2 — Polar-Express baseline + full pipeline (single + distributed)
Baseline switched to nprime06/parameter-golf Polar-Express Muon (5 per-iter NS coeffs, bf16, Jordan scale).
Pipeline now in `bench.py` (`python bench.py --compile muon`, or `torchrun --nproc_per_node=2 ... muon`):

PARITY GATE (correctness before speed), local RTX 3050:
- fused(quintic,fp32) vs **BiBo's trusted Muon** = 4.88e-4  PASS  <- the fusion is correct vs the in-repo anchor
- fused-bf16 vs PE reference (verbatim golf) = 5.86e-3  PASS
- fp16-NS stability (SV mean ~1, NaN-free) = PASS

SINGLE-GPU speed (uncompiled local): fused-bf16 1.11x, fused-fp16 1.15x. fp16 edge small on Ampere
(bf16 already on tensor cores) — expected to OPEN UP on T4 (bf16 has NO tensor cores on sm_75; fp16 does).

DISTRIBUTED option-B (DistributedMuon, exact whole-param round-robin, validated 2-rank gloo/CPU):
- B vs A replicated = 2.4e-7 (BIT-EXACT — same grads in, same weights out; NS work relocated not changed)
- B cross-rank weight agreement = 0.0 (all ranks stay in sync)
- comm = world_size packed-blob broadcasts (~one all-gather) on top of DDP's grad all-reduce;
  each rank does ~1/ws of the NS and stores momentum only for its owned params (less optim memory).

NEXT (T4): run single-GPU A/B (--compile) to size fp16 win + the optimizer's % of step; then 2x T4
torchrun to see if B's halved NS beats A's redundancy net of the broadcast. Ship winner to BiBo/bench/optim.py.

## Round 3 — CUDA-graph capture (attack the launch bound)
T4 profile verdict: fused-mixed is LAUNCH-BOUND, not compute-bound. `Command Buffer Full` = 60% of the
step at 48 tensors and **81% at 192 tensors** — the GPU stalls waiting on the CPU to submit ~1787 / ~7087
tiny kernels. GEMM (baddbmm+bmm) ~62% is the recipe floor (3 matmuls/iter x5 NS steps; tl.dot already
refuted). copy_+Memcpy DtoD ~24% is data movement. Fusion (foreach+baddbmm+batched-state) already halved
launches vs the compiled baseline, but the residual launch tax is exactly why fused-mixed is only 1.07x
(48t) -> 1.03x (192t): the tax GROWS with param count.

HYPOTHESIS: capture the whole momentum->NS->scatter as ONE CUDA graph, replay it each step. The grad
gather stays eager (reads current p.grad -> robust to zero_grad(set_to_none=True) rebinding); everything
downstream replays on persistent buffers. Collapses ~1787/7087 launches to (a few foreach gathers) + 1
replay -> kills Command Buffer Full. Biggest upside on muon_big (the real-training regime). Recipe math
untouched -> parity must stay == fused-mixed (~2.3e-5 vs fp32); a divergence flags a capture bug.
CAVEAT: graph bakes in static lr/wd/momentum; call set_graph(None) to recapture after an LR-sched change.
Candidate = `fused-graph` (use_graph=True), try/except falls back to eager champion on any capture error.
STATUS: dispatched, awaiting T4. Predicted 1.3-2x (GPU spends ~50%/81% of the step in launch stalls).

### Round 3 RESULT — graph REFUTED, but the frontier sweep handed us a real win
**CUDA-graph = DISCARD.** Launches collapsed 1768->830 and "Command Buffer Full" VANISHED from the
profile — but wall-clock was unchanged (48t: 72.2 vs 73.0ms tie; 192t: 329.7 vs 328.3 loss), total CUDA
time held (1.486->1.508s), and it cost +148/+607 MB. Parity was bit-correct (2.31e-5 == fused-mixed), so
the capture works perfectly — it just buys nothing. **The launch-bound diagnosis was WRONG: the step is
GPU-COMPUTE-bound** on the small fp16 GEMMs (turing_fp16 gemm ~839ms + Memcpy DtoD 166ms, IDENTICAL in
both profiles). "Command Buffer Full" was CPU submission OVERLAPPED behind a saturated GPU, not idle
stall — a profiler-accounting trap. Reinforces the standing lesson: GPU already saturated -> only a
STRUCTURAL edge wins, never launch reduction. use_graph kept as documented opt-in (would pay off on a
launch-bound host: slow CPU, many tiny params); off by default.

**ns_batch_elems 4M -> 64M = KEEP (the round's real win).** T4 frontier knee:
| cap | 48t x / MB | 192t x / MB |
|-----|------------|-------------|
| 4M (old default) | 1.09 / 862 | 1.05 / 3180 |
| **64M (new default)** | **1.16 / 1155** | **1.10 / 3830** |
| uncapped | 1.15 / 1155 | 1.09 / 4521 |
64M >= uncapped on speed at LESS memory (192t 3830 vs 4521). Mechanism: at 4M the (9,1536,512) expert
stacks row-chunk (row_cap = 4M/(1536*512) = 5 -> a 5+4 split); at 64M all 9 experts batch into ONE bmm
= bigger GEMM = better T4 SM utilization. The structural edge the compute-bound step actually responds to.
Consistent across both sizes and both runs, past noise. Champion is now fused-mixed @ 64M = 1.16x/1.10x.

NEXT lever: Memcpy DtoD = 11% of CUDA (166ms, 3760 calls) is the only non-GEMM cost left and is present
in BOTH profiles -> inherent to the NS math, likely a contiguity copy of a transposed bmm/baddbmm operand
inside newton_schulz. Bounded ~11% ceiling. GEMM 62% is the recipe floor (tl.dot refuted).
