# Reflections — Blackwell grouped MoE

## What worked
- `torch._grouped_mm` (cu130, sm_120) is **autograd-native** (grad_x + grad_w flow), bf16 AND fp16,
  16-byte-aligned strides required (real shapes H=512/2I=1536/I=768 all satisfy). All four MoE GEMMs
  map onto it without transpose tricks: fwd gate_up = gm(x, gup^T, offs); eo = gm(inter, dwn^T, offs);
  bwd grad_inter = gm(grad_eo, dwn, offs); grad_x = gm(grad_gate_up, gup, offs).
- **Correct grouped on the special-experts stack** = route GLU experts (codes 0/1/2, weight slots
  0..E_glu-1, contiguous prefix after sort) through ONE grouped GEMM; handle Identity(3)/Zero(4) on the
  sorted tail with a cheap weighted scatter / skip (no weight GEMM). grad rel ~7e-3 PASS. This is the
  fix the old GLU-only `moe_grouped` (grad rel ~5.7 WRONG) lacked.
- **bf16-accumulate scatter (v2)** beat fp32-accumulate (v1): top_k small -> each out row sums only k
  bf16 terms, so bf16 accumulate stays within grad tol AND drops the (n_glu x H) fp32 temp + halves the
  fp32 out buffer. Strictly faster AND leaner than v1.

## The governing variable: tokens-per-expert (= routed / E)
Grouped's win over per-expert is a function of tokens-per-expert, NOT N or E alone:
  ~496 -> 3.0x, ~655 -> 2.5x, ~963 -> 2.1x, ~1820 -> 1.6x, ~3277 (BiBo E=9) -> 1.17x, ~14564 -> 0.73x.
Many small per-expert GEMMs underutilize the Blackwell tensor cores; ONE grouped cuBLAS GEMM does not.
Scaling N *up* at fixed E HURTS grouped (more tokens/expert -> per-expert GEMMs get efficient). The win
regime is MANY experts / FEW tokens each (the real-MoE shape), not big N. top_k also raises tokens/expert.

## The speed/memory Pareto frontier (the key structural finding)
You CANNOT get max-speed AND min-memory from a grouped approach — they are opposite ends of one frontier:
- **v4 (manual grouped backward)**: FASTEST (2.55x @ E=32/k=2, beats v2) but FATTEST (+18..35% mem).
  The grouped backward materializes ALL token-gradients + full weight-grad buffers at once.
- **v3 (gradient-checkpoint the GEMM core)**: LEANEST (<= per-expert mem everywhere) but slowest
  (1.26..1.85x; recompute tax).
- **v2 (autograd-native, bf16 scatter)**: BALANCED — 2-3x AND ~tie-to-under memory at top_k=2 (the
  BiBo regime). The recommended default.
Root cause: grouped = fewer/bigger ops (fast, more live memory); per-expert = many small ops looped
expert-by-expert freeing as it goes (slow, less memory). The trade is structural, not an impl wart.

## De-risk (all PASS)
- empty-GLU-group (zero-token expert): PASS first/middle/last -> torch._grouped_mm tolerates zero-size.
- fp16 path: 2.27x AND lower mem @ E=32, grad rel 5e-4 PASS.
- top_k 2/4/8: win shrinks as top_k raises tokens/expert; high top_k pushes v2 mem a few % over.

## Shipping decision
- File: kernels/sm120/moe_grouped.py. Default = v2 (balanced, memory-safe at k=2). Document v3/v4 as
  the lean/fast alternates on the frontier.
- moe() on sm_120 dispatches to grouped when tokens/expert is low (speed crossover ~3000-4000; mem
  ~tie below ~2000). per-expert otherwise + as fallback (torch._grouped_mm missing / unaligned / sm_<80).
- Grouped now handles specials -> drop the old `glu_only` dispatch guard for the grouped path.
- BiBo's own stack is E=9 (high tokens/expert at training N) -> per-expert still wins there; grouped is
  the win for higher-expert-count MoEs on Blackwell.

## Open / not pursued
- Last % memory of v4 via progressive `del` of saved activations — blocked: ctx holds saved_tensors
  refs through backward, so del of locals does not free. Would need recompute (=v3) or per-chunk bwd.
