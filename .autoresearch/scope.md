# Scope contract — MoE MLP-block megakernel

Frozen 2026-08-08. Re-read at the start of every iteration. The watchdog checks drift against this.

## Real goal

Make the **whole MLP block** faster in real training, not a faster microbenchmark of one phase.
User's framing: *"it's the whole block you have to consider — like F1, in some sectors you may lose
time but overall the lap should be season best."* A candidate that speeds up norm+router while
slowing the expert GEMMs is a LOSS if the block total does not improve.

Secondary but real: be closer to fp64 than today's stack. Today the model routes ~1.7% of tokens to
different experts than exact arithmetic would choose (1,133 of 65,536 per micro-batch).

## Artifact (what changes)

`kernels/sm120/megakernel/moe/**` — the fused block and its custom `autograd.Function`, including a
hand-written fused backward. Nothing else.

## Frozen eval

`bench/eval_mlp_block.py`. Scores a block implementation and returns, per shape:

    fwd_ms, fwdbwd_ms, peak_gb, out_maxerr_vs_fp64, flips_vs_fp64

Baseline = liger RMSNorm + eager router + `kernels.sm120.moe.moe`, i.e. the stack the model runs
today (patch list is `{liger_norm, liger_rope, moe, xsa}` — there is no router patch).

**The eval is READ-ONLY once frozen.** Changing it to make a number look better is the cardinal sin.

## Objectives

1. **PRIMARY** — block `fwd+bwd_ms`, lower is better. This is the lap time.
2. **PRIMARY** — block `fwd_ms`, lower is better.
3. **SECOND OBJECTIVE (not a gate)** — `out_maxerr_vs_fp64` and `flips_vs_fp64`, lower is better.
   Keep a Pareto archive over (speed, accuracy); do not collapse to one.
4. Tiebreak — `peak_gb`, lower is better. The custom backward should free ~17 GB by recomputing
   `gate_up` instead of saving it; that memory is only worth something if it converts to speed.

## Splits

- Optimization set: `(B,S) = (32,1024), (64,1024)` — the shapes training actually uses.
- Held-out: `(B,S) = (32,2048), (16,4096)` — never tuned against; promotion is judged here.
- Standing slices (regression trip-wires): `flips_vs_fp64` must not exceed today's stack; per-expert
  counts must equal `bincount(indices)` and sum to `T*K`.

## Constraints / invariants

- **One activation per build**, chosen at compile time: radial NormSiLU OR SiLU, never both, no
  polyglu. The backward hardcodes one derivative rather than branching.
- **The router's load-balancing bias update is a side effect that must fire exactly once per step.**
  The backward MUST reuse the saved top-k indices and never re-run the router — recomputing would
  apply the balancing twice and silently double the balancing rate.
- Model dtype bf16, sm120 (RTX PRO 6000 Blackwell, 97 GB). fp32 master weights.
- Router semantics are fixed by `src/modeling/ffn/router.py`: sigmoid on fp32 logits, bias steers
  SELECTION ONLY, weights gathered from UNBIASED scores, sum-norm with `+1e-20`.

## Out of scope

- Changing model architecture, expert count, top-k, or the activation choice.
- Multi-GPU / expert parallelism. Cursor's Mixture-of-Kittens solves an all-to-all communication
  problem across 72 GPUs; we have 64 experts on ONE device and none of that applies.
- Touching the eval, the attention path, or the training loop.

## Prior art — measured this session, do NOT re-derive

| Finding | Number |
|---|---|
| Block forward baseline @T=65536 | liger 0.050 + router 0.235 + moe() 4.255 = **4.541 ms** |
| `moe()` share of the block | **94%** — and it is already the output of a dedicated speed round |
| Fusing norm+router only | 1.21x on that sub-block = **1.01x on the block** |
| Ceiling if everything except the expert GEMMs were free | **1.07x** |
| `h_norm` round-trip cost | 0.002 ms (1%), NOT the 134 MB claimed |
| Megakernel routing accuracy | **0 flips** vs 1,133 for today's stack; weights 3,400x closer |
| Attention (unrelated, closed) | 3.7% of a step — a custom flash kernel buys 1.04x, not worth it |
| FA3/FA4 wheels | sm90a / sm100 only — cannot target sm120 |

**Three times this session a benchmark measured MY OWN reference instead of the model's real path**
(a dense attention mask, a bf16 rstd, an eager norm standing in for Liger). Every eval number must
come from the code the model actually runs. Check the patch list, not the assumption.

## Definition of done

Stop when the held-out shapes show **>= 1.10x on block fwd+bwd with accuracy no worse than today's
stack**, or on any external stopping condition (patience 3, max 30 iterations, budget, user stop,
Goodhart trip-wire). A tie inside the noise floor is reported as a tie, not a win.

## Resources

Single RTX PRO 6000 Blackwell on the marimo box, currently idle. The box has died mid-session
before; state lives on disk here, not in conversation.
