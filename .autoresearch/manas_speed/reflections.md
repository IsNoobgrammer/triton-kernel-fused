# Reflections — manas speed
- Baseline: launch-bound, 28.6k CUDA launches/manas-step (K=4). vote()=44ms (dominant), apply=12ms. Both per-param Python loops over 70 mats x K.
- Boundary (_finish_micro_step) already shape-grouped + foreach (a2e9260) - mirror that pattern.
- iter1 KEEP: shape-grouped batching. overhead_k4 70->28, kinv 84->22, parity max_err 6e-8. Still K-linear residual (~+22ms k2->k8); per-vote cost not yet negligible. Next: profile remaining launches.
- iter2: profile of batched=12k launches (from 28.6k). Residual = DtoD copies(1.1ms)+small gemms+~900 manas launches CPU dispatch; +harness randn artifact 1.2ms (not manas). CUDA-graph is the path to fully match muon but blocked by eval grad-realloc; finalized iter1 as champion (correct, 2.5-3.8x win).
- REAL-VERIFY (ungameable, in-process interleaved fair bench): old-manas +10.1% real tps cost -> batched +5.8% (1.75x less, -20ms/step). bit-exact losses. Isolation eval overhead (+28ms k4) matched real (+26ms) -> eval was faithful, not gamed.
- Remaining +5.8% = per-group loop x K CPU dispatch (12k launches). Only lever to fully match muon = CUDA-graph capture of the per-micro step. VIABLE in real training (in-place grad accum = static storage), BUT: risky (stateful optimizer), and MoE dynamic routing may not be graph-capturable. Not attempted autonomously - needs explicit go + capturability check.
