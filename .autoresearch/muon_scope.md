# Scope — Fused Muon optimizer step

**Artifact:** `kernels/muon.py` (`FusedMuon` + `newton_schulz`) — develop/bench in triton-kernel-fused,
ship the winner into `BiBo/bench/optim.py` (`Muon`).

**Eval (frozen):** `.autoresearch/bench_muon.py` — parity (max|Δp| per step vs the exact BiBo eager
recipe) + speed (do_bench, fp16 params) for: eager-BiBo vs fused-fp32 (champion) vs fused-fp16. Run on
T4 (sm_75). Champion gate: parity bit-tight to eager (fp32 path) and faster.

**Objective:** wall-clock of one `step()` over the real BiBo param set, lower is better. fp16-NS only
kept if it ALSO passes: SV mean ~1, |Δp|/lr attribution flat ~0.2, NaN-free over real training steps.

**Real goal:** cut optimizer-step overhead in BiBo training without changing the Moonlight recipe's
math (momentum→orthogonalize order, 0.2·√max(A,B) scale, decoupled WD, per-expert 3D batching).

**Constraints / off-limits:**
- Do NOT change the recipe math. Bit-parity to the current eager Muon is the gate for the fp32 champion.
- No Triton `tl.dot` GEMM (proven 3× loss vs cuBLAS; 512² won't fit T4 64KB SRAM) — settled prior round.
- No cross-param shape-bucketing into one big bmm (reverted: 2× transient memory thrashed 4GB GPU).
- Per-param NS stays (3D experts batch over expert dim internally).

**In-scope levers:** `torch._foreach_*` for the per-param momentum/scale/update sweeps; `baddbmm`
epilogue folding inside NS (`b·A+c·A@A`, `a·X+B@X`); fp16 NS GEMMs (T4 fp16 tensor cores vs fp32
CUDA cores) — the only lever on the 71%-GEMM-bound dominant cost.

**Prior art (AGENTS.md, Jun 27):** Muon step is GEMM-bound ~68-71%, NS=3 matmuls/iter floor, cuBLAS.
compile_ns is "the only lever" they found — but they MISSED baddbmm folding + foreach sweeps + fp16
tensor-core NS. Those are this round's hypotheses.

**Done:** champion (foreach+baddbmm, fp32 NS) bit-parity + faster on T4, shipped to BiBo. fp16-NS
kept as opt-in only if stability gates pass on T4.
