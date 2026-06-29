# Scope contract — optimize the conv router for Blackwell (sm_120)

## Real goal
The conv MoE router (`fused_router` [cudnn]) REGRESSED on Blackwell: **0.82× uncompiled / 0.91× compiled**
fwd+bwd (T4 was 1.11–1.17×). Get it back to ≥ 1.0× vs compiled eager on sm_120 without losing grad
correctness or memory parity. It's the only Blackwell loss; bring it to at least a tie, ideally a win.

## Frozen eval (ground truth — HUMAN-GATED)
`python bench.py --compile router` on the RTX PRO 6000 Blackwell box (arch=sm120, bf16). Reports
fwd / bwd / fwd+bwd × vs compiled eager, peak mem, and **grad parity (idx-agree 1.0, grad_rel < 1.5e-2,
count==bincount)**. The agent CANNOT run this — the user runs it and pastes results. Each iteration =
one user round-trip. So: high-confidence, correctness-de-risked candidates only; no blind scattershot.
Local = syntax/import only (and the local 3050 resolves to sm75, not sm120, so it can't even exercise
this backend) — the user's run is also the correctness gate.

## Diagnosis (from the Blackwell profile, compiled fwd+bwd: ours 5.58 ms vs eager 4.86 ms)
- **`aten::copy_` = 48% of our CUDA (2.68 ms, 100 calls)** vs eager 27.5% (1.34 ms, 60 calls). The killer.
- ROOT CAUSE (hypothesis): our forward does `xc = x.transpose(1,2).contiguous()` → forces the input to
  channels-FIRST (B,H,S) contiguous. But x (B,S,H) with H innermost is ALREADY channels-LAST (NLC) for a
  conv over S with H in-channels. cuDNN wants NHWC, so it then transposes our channels-first copy BACK to
  NHWC (`nchwToNhwc` 410 µs + `nhwcToNchw` 226 µs + `tensorTransformGeneric` 234 µs ≈ 870 µs). We pay a
  self-inflicted double layout change + a 16.8 MB contiguous copy.
- Evidence inductor avoids it: its compiled backward returns grad_x as `reinterpret_tensor(... stride
  (524288,1,1024))` — a channels-last VIEW, no transpose copy.
- We DO fuse away eager's native top-k (~1.26 ms) — the T4 edge — but spend ~1.3 ms MORE in copies. Net loss.

## In-scope changes (champion-protected)
Fork `kernels/sm120/router.py` (today a 1-line re-export of sm75). sm75 cudnn champion stays UNTOUCHED;
every candidate is a new sm120 backend, A/B'd on Blackwell. Allowed levers, cheap→expensive:
1. **channels-last conv** — feed cuDNN x in NHWC (no explicit contiguous, conv1d-as-conv2d channels_last);
   return grad_x as a view. Targets the copy_ 48% + nchwToNhwc. T4-REFUTED (cuDNN copied regardless) but
   Blackwell + cu130 + bf16 UNTESTED. Cheapest, highest-ceiling first shot.
2. **transpose-free fused Triton conv (fwd+bwd)** — read x (B,S,H) once, conv over S in-SRAM, fuse the
   sigmoid+top-k+gather epilogue, no cuDNN, zero layout copies. The real ceiling (T4 shelved it: 64 KB
   SRAM; Blackwell ~228 KB + bf16 TC is the "revisit on better hardware" case). Bigger, escalation.

## Constraints / invariants (hard)
- NEVER touch the eval or the sm75 champion. Candidates are separate sm120 backends.
- grad parity must hold (idx 1.0, grad_rel < 1.5e-2, count==bincount, NaN-free) BEFORE any perf number is
  trusted — the first run on each candidate is the correctness gate.
- mem ≤ ~1.0× compiled (router memory parity is the bar; small headroom OK if it buys speed).

## Refuted (T4) — re-open ONLY because Blackwell differs, and say why
channels-last (cuDNN copied to its layout regardless, nchwToNhwc still present, slowed conv_bwd 613→948µs);
tl.dot conv 0.73×; cuBLAS K-GEMM 0.35×; readonce loop-reorder 0.33×; merged-contraction SRAM-crash. All on
T4. The arch + cuDNN-version change is the only justification to retry channels-last / Triton conv here.

## Definition of done
Router ≥ 1.0× fwd+bwd vs compiled eager on Blackwell, grad PASS, mem parity. Stop target: tie-or-win.
If after a few candidates nothing beats compiled, SHIP the sm75 cudnn reuse (accept the ~0.9× on Blackwell)
and record honestly — the router may just be inductor's to own on this arch too.
