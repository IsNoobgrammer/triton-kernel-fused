# Scope contract — Blackwell grouped MoE kernel

Frozen at 2026-06-30. Read this every iteration; the watchdog checks drift against it.

## Real goal
Ship a **grouped-GEMM MoE path that wins on Blackwell (sm_120)** so `moe()` can auto-dispatch to it
at scale instead of always falling back to per-expert. The win that matters: one block/grouped GEMM
over all routed GLU tokens uses the Blackwell tensor cores far better than E separate small cuBLAS
GEMMs (the per-expert loop), *and* drops the per-expert Python loop + host syncs.

## Artifact (what changes each iteration)
A single function `grouped_candidate(hidden, idx, wt, gate_up_proj, down_proj, act_codes) -> (N,H)`
with a working backward. Developed on the GPU box (notebook cells / scratch module); folded into
`kernels/sm120/moe.py` ONLY after it provably beats the baseline. Same signature as `moe_grouped`.

## Eval (frozen — never modify)
`bench.bench_moe(N, H=512, I=768, E_glu=9, n_special=2, top_k=2)` with `bench.DTYPE=bfloat16`,
`bench.COMPILE=True`. Drive the candidate by setting `bench.moe_grouped = grouped_candidate` then
calling `bench.bench_moe(...)` — this swaps only the "grouped" slot; run()/grad-check/timing/mem are
untouched. The eval reports, per variant: grad abs/rel vs compiled `moe_eager`, fwd ms, bwd ms,
fwd+bwd ms, peak mem. Baseline column = compiled `moe_eager`; the bar we care about = `per-expert`.

The BiBo stack the eval uses: `act_codes = [0,1,2,0,1,2,0,1,2,3,4]` — 9 GLU experts (PolyGLU triples
SiLU/ReLU2/Tanh) + Identity(3) + Zero(4). `gate_up_proj`/`down_proj` are sized **E_glu=9** (specials
carry NO weight slot here; GLU experts are ids 0..8, contiguous). top_k=2 -> N*top_k routed tokens.

## Objective (the numbers to push)
1. **Speed**: candidate fwd+bwd faster than `per-expert`. Target ~2.0x ("1x more faster"); any
   provable >1.0x at a defensible scale is progress.
2. **Memory**: candidate peak mem <= per-expert peak mem (lower or equal — hard constraint).
3. **Correctness (gate)**: grad rel parity vs eager comparable to per-expert (PASS, ~1e-2 bf16 floor);
   forward matches. A wrong-grad candidate is killed regardless of speed.
4. Holds on the **special-experts stack** (the eval's default) AND scales: confirm at N=16384 and
   scaled N (65536, 131072+). The user explicitly allows scaling inputs to show where grouped helps.

## Constraints / invariants
- **Must support PolyGLU** (heterogeneous act codes 0/1/2 per expert) — non-negotiable.
- **Must be correct with special experts present** (codes 3/4 in the stack). Clean design: the 9 GLU
  experts go through the grouped GEMM; Identity/Zero are handled by the cheap scatter/skip path
  exactly as per-expert does (they have no weight GEMM). A GLU-only grouped kernel FAILS the eval
  (grad rel ~1.6e3 on this stack) — that is the current `moe_grouped` and is the thing to fix.
- Never modify the eval. Optimize the artifact only.
- sm_120 only target here; leave sm75 reference and the proven per-expert path untouched.
- bf16 path (the eval runs --bf16). fp32 accumulate for the combine (MiMo), matches existing paths.
- No emoji anywhere.

## In-scope changes
- Grouped GEMM impl: `torch._grouped_mm` (cuBLAS, GPU-resident dispatch) vs Triton `tl.dot`
  `_grouped_mm` with Blackwell-tuned autotune configs.
- Dispatch construction (GPU-resident cumsum offsets vs host `.tolist()` + Python tile loop).
- Special-expert handling (route GLU subset through grouped GEMM, specials via `_combine_scatter`).
- Activation fusion (PolyGLU `_glu_fwd/_glu_bwd`), combine/scatter fusion, fp32 accum.
- Backward: autograd-native (if `torch._grouped_mm` is differentiable) vs manual grouped backward.

## Out of scope / off-limits
- The eval, the dataset, the per-expert champion, the router, sm75.
- Changing the BiBo stack definition or act-code semantics.
- Forward-only kernels (must have a correct backward — training library).

## Prior art (do not re-tread)
- `moe_per_expert` = Blackwell champion (~4x fwd+bwd vs compiled eager), correct on all codes. BASELINE.
- `moe_grouped` (tl.dot) = GLU-only, autotune configs tuned for T4; ~0.07x on Turing; on Blackwell
  faster than T4 but WRONG on the special-experts stack (no 3/4 handling). The starting point to fix.
- `moe_grouped_cublas` = uses `torch._grouped_mm`, GLU-only, casts to bf16, UNTESTED end-to-end,
  sm_80+ only. A scaffold for the cuBLAS-grouped candidate family.
- Recurring Blackwell lessons (.autoresearch): tl.dot defaults to TF32 on fp32 -> use ieee for fp32
  correctness gates; low-program serial reductions -> split-K + atomic fp32 accum.

## Definition of done
A grouped candidate that, on the special-experts stack at a defensible scale, beats per-expert on
fwd+bwd (ideally ~2x), uses <= per-expert memory, and PASSES grad parity. Then: fold into
kernels/sm120/moe.py, wire moe() to dispatch to it on sm_120, update docs + findings.

## Resources
Live RTX PRO 6000 Blackwell (sm_120) via marimo notebook; torch 2.12.0+cu130, triton 3.7.0.
Agent cannot run GPU locally (Windows) — all measurement runs on the box through visible cells.
