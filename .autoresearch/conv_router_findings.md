# Conv MoE router — optimization ledger (Round 4-5) + what to revisit on better hardware

**Shipped (T4 / sm_75):** the `cudnn` backend — cuDNN conv (padding=K-1, no F.pad copy) + a fused
Triton epilogue (sigmoid+bias+top-k argmax+unbiased gather, in-register) + a merged manual backward
(`torch.ops.aten.convolution_backward` direct on a saved-once contiguous input + fused epilogue-bwd).
**T4: 1.11-1.17x fwd+bwd vs torch.compile, exact grads (rel ~3e-7), mem parity.** BiBo-parity PASS @
E=11. This is the ONLY backend kept in code (`kernels/router.py`, `BiBo/src/kernels/fused_conv_router.py`).

Shape that all numbers refer to: B=16, S=1024, H=512, E=11, K=4, top_k=2, fp16, Tesla T4.

## Why the win is what it is (the two real edges)
1. **Fuse the native op compile can't.** `torch.topk` is a library call inductor must keep separate
   (~295us + 216us gather on T4 — ~36% of the compiled forward). A bespoke top-2-of-11 argmax in the
   epilogue is ~free. This is the same class of win as MoE data-dependent routing and XSA read-once.
2. **Own the backward glue.** cuDNN `convolution_backward` (~613us) is identical on every path; the
   gap was ~700us of UNFUSED glue around it (autograd copies x->contiguous twice, casts, transposes).
   A custom backward (save contiguous input once, call convolution_backward directly, fuse sigmoid'+
   scatter) removed it: bwd 0.83x -> 1.16x.

## Refuted on T4 (REMOVED) — and WHY (so we don't re-try them on Turing)
| approach | T4 result | root cause |
|---|---|---|
| `tldot` — transpose-free fused Triton conv (K skinny dots, contraction 512) | 0.73x | conv kernel 818us = ~15x off the 52us bandwidth floor; skinny E=11->16 tl.dot wastes the MMA; bwd dx/dw 1502us >> cuDNN 613us |
| `cublas` — K cuBLAS GEMMs on native layout | 0.35x | K+2K Python-loop GEMMs RMW-ing fp32 buffers; launch + fp32 traffic dominate |
| `readonce` — tldot fwd with H-outer/K-inner loop reorder | 0.33x fwd | x (~1MB/batch) already L2-resident, so reorder cut no HBM traffic — only wrecked tl.dot pipelining on Turing |
| `cudnn_cl` — feed cuDNN channels-last to skip its nchwToNhwc transposes | 0.95x | cuDNN copies to its own layout regardless (nchwToNhwc still 6.3ms); strided input ADDED copies + slowed convolution_backward 613->948us |
| `tlconv` — merged (k,h)=K*H=2048 contraction (fatter dots) | CRASH / ~tldot | T4 64KB SRAM caps the contraction tile (128x512 fp16 = 128KB > 64KB); capped to BLOCK_C<=128 it collapses back to tldot |

## The unrealized headroom (revisit on Ampere/Hopper — NOT T4)
The conv is **bandwidth-bound and tiny**: x = 16.8MB -> forward floor ~52us, backward floor ~105us
(read x + write grad_x). The compiled path runs **8-16x off** those floors; cuDNN's general conv
(312us fwd / 613us bwd) is itself ~6x off (degenerate 11-output-channel conv it was never tuned for).
So a **bandwidth-optimal, transpose-free Triton conv (fwd+bwd)** that reads x once and fuses the
epilogue could in principle hit 2-4x overall — but on T4 it never landed because:
- T4 SRAM is only 64KB (Ampere 100-164KB, Hopper 228KB) — the read-once sliding-window tile + fat
  contraction need more SRAM than Turing has. **On Ampere/Hopper this constraint largely lifts.**
- Turing tl.dot is weak for skinny-N (E=11) and the codegen overhead is high; Ampere/Hopper tensor
  cores + larger register/SRAM budgets change the calculus (tl.dot is NOT dead there — prior CE/MoE
  rounds saw tl.dot competitive on Ampere local).

**If you move off T4, re-open these (highest ceiling first):**
1. Transpose-free fused conv FORWARD: read x[s-K+1:s+BLOCK_S,:] once into SRAM, slide across K taps,
   fuse sigmoid+top-k+gather in-register. Kills cuDNN's ~482us layout-transpose tax + all topk/glue.
   Needs > 64KB SRAM to tile well -> Ampere/Hopper only.
2. Transpose-free custom BACKWARD (dx/dw) to beat cuDNN convolution_backward (613us, ~6x off the 105us
   floor). The bigger absolute prize; same SRAM story.
3. `grouped_mm`/bf16 tensor-core paths (sm_80+) — see the MoE round's `grouped_cublas` finding (bf16 +
   sm_80+ only), the analogous "right tool on Hopper, useless on Turing" lever.

## Recurring T4 lessons (don't relearn)
- On T4 + torch.compile you win at **native-op seams** (topk, data-dependent routing, read-once
  reductions), NOT by out-computing cuDNN/cuBLAS. Profile to find the seam vs the immovable library call.
- Local (RTX 3050, Ampere) does NOT predict T4: tl.dot scheduling, occupancy, and layout-transpose
  behavior all differ. Local is for CORRECTNESS; T4 is the perf verdict.
- Protect the champion: every candidate is a SEPARATE backend, A/B on T4 — never overwrite a proven
  win with an unconfirmed change (this saved us when channels-last regressed).
