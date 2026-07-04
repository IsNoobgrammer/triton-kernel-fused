#!/bin/bash
# ablate_muon — Grok-MoE optimizer ablation launcher.
#
# Usage:
#   bash ablate_muon/run.sh              # grok ablation: auto T4 x2 -> 2 GPUs; else 1 GPU
#   bash ablate_muon/run.sh olm          # ONLINE LM-emulator ablation (run_olm.py) instead
#   bash ablate_muon/run.sh [olm] 1      # force single-GPU (all arms sequential on cuda:0)
#
# Dual-GPU: shard 0 (even-indexed arms) on GPU 0, shard 1 (odd) on GPU 1, simultaneously —
# same one-process-per-GPU pattern as bench/run.sh. Each GPU streams to the console AND to its
# own log file (logs/g0.log, logs/g1.log); results merged into results.jsonl + final table.
#
# No `set -e`: if one GPU hiccups the other still finishes and --merge prints whatever completed.

# Determinism / alloc env (mirrors bench/run.sh)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

# torch is the only dep and is preinstalled on Kaggle/Colab — nothing to pip install.
DRIVER="ablate_muon/run_ablation.py"; PFX=""
if [ "$1" = "olm" ]; then
    DRIVER="ablate_muon/run_olm.py"; PFX="olm_"; shift
fi
GPU_COUNT=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
FORCE_SINGLE="${1:-}"
echo "[run.sh] driver $DRIVER; detected $GPU_COUNT GPU(s); logs -> $LOGDIR"

if [ "$GPU_COUNT" -ge 2 ] && [ "$FORCE_SINGLE" != "1" ]; then
    echo "[run.sh] DUAL-GPU: shard 0 on GPU 0, shard 1 on GPU 1 (parallel)"
    CUDA_VISIBLE_DEVICES=0 python -u "$DRIVER" --shard 0 --nshards 2 \
        2>&1 | sed -u 's/^/[gpu0] /' | tee "$LOGDIR/${PFX}g0.log" &
    P0=$!
    CUDA_VISIBLE_DEVICES=1 python -u "$DRIVER" --shard 1 --nshards 2 \
        2>&1 | sed -u 's/^/[gpu1] /' | tee "$LOGDIR/${PFX}g1.log" &
    P1=$!
    wait $P0 $P1
    echo "[run.sh] both GPUs done — merging"
    python -u "$DRIVER" --merge 2>&1 | tee "$LOGDIR/${PFX}merge.log"
else
    echo "[run.sh] SINGLE-GPU: all arms sequential on cuda:0"
    python -u "$DRIVER" 2>&1 | tee "$LOGDIR/${PFX}single.log"
fi

echo "[run.sh] done. table + per-arm results in ablate_muon/results.jsonl"
