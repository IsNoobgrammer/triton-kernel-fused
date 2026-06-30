# Scope contract — Symmetric-matmul ("symmul") Newton-Schulz fused into our Muon

Status: SHIPPED (2026-06-30). sm120 FusedMuon now DEFAULTS to the symmetric-matmul ("symmul") NS;
`use_symmul=False` gives the pure-cuBLAS champion; `AmalgamatedMuon` is a back-compat alias.
Verified on RTX PRO 6000: parity vs cuBLAS champion 1.95e-3 (<2e-2), 1.31x faster at 0.61B, mem ==.
Full results (NS micro, 4-way, 1B-2.6B scaling, flash-muon head-to-head, B@X profiler/roofline/fp8)
in symmul_reflections.md. NO precision tradeoff -> symmul is the Blackwell default.
Date: 2026-06-30. GPU dev box: RTX PRO 6000 Blackwell sm_120 (no nvcc -> Triton-only).

## RESOLVED (user call, 2026-06-30)
- Regime = **(B)** large-matrix / "as we scale". The headline is the square sweep {1024..8192}.
- Baseline to beat = the OLD FusedMuon (our champion `newton_schulz`); target ~**1.8x**, and
  amalg peak mem **<= compiled** peak. The candidate must beat ALL THREE baselines as it scales.
- Frozen eval = `.autoresearch/bench_symmul.py` (4-way, algorithm fixed, only the symmetric GEMMs
  vary). Three baselines + the candidate:
    compiled (torch.compile cuBLAS NS) | triu (flash-muon single-matrix, UNbatched) |
    fused (champion eager cuBLAS NS) | amalg (our BATCHED symmul kernel = the candidate).
- Artifacts: candidate `kernels/sm120/newton_schulz_symmul.py` (batched symmul + PE-NS, additive,
  champion untouched); vendored baseline `.autoresearch/baselines/flash_muon_mmt.py`.
- First candidate (v1): replace BOTH symmetric GEMMs (X X^T and A A) with the batched symmul;
  polynomial `b*A + c*AA` as explicit axpy (loses the baddbmm fold on those two terms — the
  measured tradeoff); `B X` stays cuBLAS. Shape-dispatch: symmul -> cuBLAS below SYMMUL_MIN_DIM.

## The thesis (why this could be multiplicative)
Our FusedMuon already optimizes dimensions that are ORTHOGONAL to the Newton-Schulz GEMM
FLOP count: `_foreach_*` (launch collapse), `baddbmm` (axpy folded into the GEMM), batched
same-shape state (many matrices -> one big GEMM), fp16-TC NS. flash-muon / StarrickLiu /
Laker-Newhouse all exploit a DIFFERENT axis: the two NS GEMMs `A=X·Xᵀ` and `A·A` are
SYMMETRIC, so you compute only one triangle (~half the FLOPs) and mirror it. We do NOT do
this — we use full `torch.bmm`. Stacking the symmetric FLOP-cut ON TOP of our batching/launch
fusion is a new dimension -> potentially multiplicative return in the compute-bound regime.

## Real goal
A faster FusedMuon NS step by adding the symmetric-matmul FLOP cut (proven ~1.5-1.8x on the
symmetric matmul alone at dim>=2048, A100/H800/4090; flash-muon's Triton default) without
losing our existing launch/batching/epilogue wins or our hard memory gate. The honest win
region is the COMPUTE-BOUND / large-matrix regime — exactly where our fused-vs-compiled gap
shrank to ~1.24x (the wide-MLP / big-model case, documented in muon_training_bench.md).

## Artifact (what changes each iteration)
A NEW `kernels/sm120` Newton-Schulz variant (e.g. `newton_schulz_symmul`) that replaces the
two SYMMETRIC bmms with a BATCHED Triton symmetric-matmul (triangle + transpose-copy, adapted
from flash-muon's 2D `mmt_kernel` to a batch dim), behind a flag. The non-symmetric `B·X`
stays cuBLAS `baddbmm`. The current `FusedMuon` / `newton_schulz` champion is UNTOUCHED
(git-versioned, safe to branch). Knobs the loop may move: the batched-symmul kernel
(tiling/autotune), the shape-dispatch threshold (symmul vs cuBLAS-bmm), whether to fuse the
polynomial epilogue (`b·A + c·A·A`) into the Triton kernel, fp16 vs fp32 accumulate.

  current NS iter:  A=bmm(X,Xᵀ); B=baddbmm(A,A,A,β=b,α=c); X=baddbmm(X,B,X,β=a)
  symmul NS iter:   A=symmul(X); AA=symmul(A); B=b·A+c·AA; X=baddbmm(X,B,X,β=a)
  (symmul(M) := batched M·Mᵀ via triangle+mirror; symmul(A)=A·Aᵀ=A·A since A symmetric)

## Frozen eval (define now, then NEVER edit)
Reuse bench_muon's discipline (parity gate -> ms/peak/speedup, hard mem gate peak<=baseline),
EXTENDED to a shape grid that spans both regimes (the default muon_shapes are H=512 = small,
where symmul won't show; we MUST include large shapes to see the lever):
  - NS-step micro: single + batched matmul_transpose AND a full NS(5) step, for square-ish
    dims {512, 1024, 2048, 4096} and batched-small {B in [8..192] x 512²} (the BiBo-like regime).
  - Metrics per shape: (a) PARITY — symmul-NS output vs full-fp32 NS within NS tolerance
    (SV~1, NaN-free, |Δ| vs the cuBLAS-bmm NS) ; (b) NS-step ms ; (c) peak mem ; (d) full
    Muon-step ms via bench_muon on the muon_shapes (end-to-end, the real number).
Frozen once written. The artifact is judged ONLY by this.

## Objective
Minimize NS-step (and end-to-end Muon-step) time vs the CURRENT FusedMuon, at PARITY PASS and
peak<=baseline, PER SHAPE. Headline = large-matrix-regime speedup over current FusedMuon.
Stretch framing ("multiplicative"): our batching win (~2.3x vs compiled) composing with the
symmetric FLOP cut (~1.5x) in the compute-bound regime.

## Constraints / invariants
- Triton-only (no nvcc on box). No CUDA C++ / CUTLASS.
- Keep fp16 NS + fp32 normalization (unchanged math; only the GEMM impl changes).
- HARD MEM GATE: symmul variant peak <= current FusedMuon peak (the standing Muon rule).
- PARITY: symmul-NS must be numerically equivalent to the cuBLAS-bmm NS within the existing
  Muon parity tolerance (the transpose-copy epilogue is the correctness risk — Laker
  Newhouse's version had a transpose-store bug; flash-muon claims fixed -> we verify hard).
- DO NOT regress the small-matrix regime -> per-shape dispatch (symmul only where it wins).
- Champion FusedMuon stays intact; new variant is additive + flagged.

## In scope / out of scope
IN: batched symmetric-matmul Triton kernel; shape-dispatch threshold; optional fused
    polynomial epilogue; autotune configs; fp16/fp32 accumulate choice.
OUT: editing the frozen eval; changing the NS algorithm or PE coeffs; touching the
    non-symmetric B·X GEMM (stays cuBLAS); nvcc/CUDA C++; the AdamW-for-1D path.

## Prior art / known traps (do not relearn)
- `tl.dot` LOST to cuBLAS ~3x on T4 (Turing, no bf16 TC) — but that's Turing. flash-muon
  proves a Triton symmetric matmul BEATS torch/cuBLAS ~1.3-1.8x at dim>=2048 on A100/H800/4090;
  at dim<=1024 it's a WASH. => the win is large-dim only -> shape dispatch is mandatory.
- Losing the `baddbmm` epilogue fold on the 2 symmetric GEMMs adds elementwise traffic
  (b·A + c·AA as a separate op). May erode the FLOP win on smaller shapes -> consider fusing
  the polynomial into the Triton epilogue (StarrickLiu-style) as a later iteration.
- flash-muon's kernel is 2D single-matrix; batching it (add batch dim to grid/strides) is new
  code and the main implementation risk.
- A=X·Xᵀ iterates on the SMALLER Gram (we already transpose so rows<=cols); symmul(A) where A
  is n×n symmetric is the second target. B·X is full -> cuBLAS.
- Our shipped Muon regime (BiBo: H=512, many small matrices) is SMALL-matrix-batched, where
  symmul likely does NOT help -> see the key open question below.

## KEY OPEN QUESTION (confirm before Phase 1 — it decides if this is worth it)
Which regime is the target?
  (A) BiBo / transformer Muon as shipped: H=512, MANY small (512²-ish) matrices, batched.
      Here the NS GEMMs are launch/util-bound, NOT compute-bound -> the symmetric FLOP cut
      likely buys ~nothing, and small-dim Triton tl.dot may even lose to cuBLAS. If THIS is
      the target, the symmul lever is probably the wrong bet.
  (B) Large-matrix / big-model Muon (1024²-4096², fewer matrices): the compute-bound regime
      where flash-muon's ~1.5-1.8x lives and our fused-vs-compiled gap shrank to 1.24x. The
      symmul lever pays here.
=> If the goal is "faster BiBo Muon", (A) means low expected return; if "a better general /
   large-model Muon (and a real lever for the regime where we're currently weakest)", (B) is
   the bet and the effort is justified. NEED USER CALL.

## Definition of done
A flagged sm120 symmul-NS Muon variant that, on Blackwell:
- beats current FusedMuon on the large-matrix NS-step / Muon-step (target: capture most of
  flash-muon's ~1.5x symmetric-matmul win on dim>=2048), PARITY PASS, peak<=baseline; AND
- does NOT regress small-matrix shapes (dispatch falls back to cuBLAS bmm there).
Stop on: target met, user stop, budget, or patience/plateau per autoresearcher rules.

## Resources
Blackwell box (Triton 3.x, no nvcc), bench_muon as the harness, flash-muon's mmt_kernel as the
2D reference to batch. ~minutes per NS-micro iteration; end-to-end Muon step ~seconds. Watchdog
interval (when Phase 1 starts): ~20-30 min (heavy-ish iterations with parity+timing+mem).
