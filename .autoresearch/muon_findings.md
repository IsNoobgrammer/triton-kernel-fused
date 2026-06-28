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
