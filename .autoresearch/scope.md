# Scope contract — kernel optimization loop (MoE + CE)

Semi-manual autoresearch loop. The eval runs on a **Tesla T4** (Kaggle); the agent cannot run it
locally (local = RTX 3050, torch.compile broken). Each round: agent commits a candidate → user runs
`.autoresearch/kaggle_eval.py` on the T4 notebook → pastes the `@@RESULT` lines → agent keeps/kills.

## Real goal
Make BiBo's training step faster / fit bigger on the **actual training GPU (T4, sm_75) under
`torch.compile`** — not on paper, not on Ampere, not in eager. A kernel only earns its place if it
beats what `torch.compile` already gives for free on T4.

## Frozen eval (the only ground truth)
`python bench.py --compile --json moe ce` on a **Tesla T4**, fp16. Both kernel and eager are
torch.compile'd; compile + autotune happen in warmup (excluded from the timed step). Reports
fwd / bwd / fwd+bwd speedup vs **compiled eager**, peak memory, and grad-rel vs eager.
**Never edit the eval to make numbers look better.** Shapes are pinned (see state.json baseline).

## Targets (artifacts that may change)
- `kernels/moe.py` — MoE per-expert (the one speed win) + grouped (Ampere+ only).
- `kernels/cross_entropy.py` — fused-linear CE (the memory win).
- `kernels/swiglu.py` is in-scope ONLY as the MoE per-expert activation dependency.

## Objectives
- **MoE per-expert**: push fwd+bwd beyond the 2.85× baseline AND make dispatch fully GPU-resident
  (no `.tolist()` / Python schedule host syncs). Stretch: a Turing-fast grouped path via
  `torch._grouped_mm` (torch 2.10 native cuBLAS grouped GEMM) to replace the dead tl.dot path.
- **CE**: lift fwd+bwd from 0.61× toward ≥1.0× (tie compiled eager) WHILE keeping the ≥1.3× memory
  win. Lever: cut the 3-GEMM backward recompute cost.

## Constraints / invariants (hard)
- **Grad-equivalence**: every candidate must keep `grad_rel < 1.5e-2` vs eager (the bench gate). A
  faster-but-wrong kernel is a fail.
- **GPU-resident**: minimize host syncs (`.item()/.tolist()/float()/bool()`); they show up as
  torch.compile graph breaks. This is an explicit goal, not just a nicety.
- Don't touch the eval (`bench.py` measurement logic) to flatter results. Adding `--json` parsing is fine.
- PolyGLU stays (per-expert act_codes: SiLU/ReLU²/Tanh).

## Out of scope (decided, do not revisit)
- SwiGLU / XSA / causal-conv1d router as **speed** plays — they lose to compiled inductor on T4
  (0.96× / 0.89× / 0.75×). Kept only as no-compile fallbacks; do NOT spend loop budget on them.
- grouped MoE on Turing — tl.dot cliff (0.10×); auto-disabled on sm_<80. Only revisit via a
  cuBLAS-backed grouped GEMM (`torch._grouped_mm`), never more tl.dot tuning.

## Prior art (this session)
- Triton `tl.dot` GEMMs are ~2% SoL on Turing vs cuBLAS ~50% (proven 3× in BiBo). Never bet on tl.dot
  GEMM beating cuBLAS on T4.
- torch.compile/inductor fuses elementwise ops as well as hand-written Triton; an `autograd.Function`
  is an opaque graph break that can BLOCK inductor fusion. So fused-elementwise kernels rarely win
  under compile.
- The MoE per-expert win is real because compile graph-breaks on data-dependent routing.

## Definition of done
- MoE per-expert: GPU-resident (no host sync in the hot path) AND fwd+bwd ≥ current 2.85× on T4.
- CE: fwd+bwd ≥ 0.9× compiled eager with memory still ≥ 1.3× less, grad-rel < 1.5e-2.
- Each kept change verified on T4 via the frozen eval, grad gate green.
