#!/bin/bash
# ablate_muon — single-GPU FAN-OUT launcher for a big-memory card (e.g. RTX 6000 Ada, 48GB).
#
# Instead of one-process-per-GPU (run.sh), this runs 4 shards CONCURRENTLY on the SAME GPU -
# the model is ~3.6M params so 4 arms fit easily in VRAM and the card is under-utilized by one.
#
# Usage:
#   bash ablate_muon/run_6000.sh              # grok ablation, 4 concurrent shards on cuda:0
#   bash ablate_muon/run_6000.sh olm          # ONLINE LM-emulator ablation (run_olm.py)
#   bash ablate_muon/run_6000.sh olm 8        # override: 8 concurrent shards
#
# Each shard streams to console ([t0]..[tN]) and its own log; results merged into results.jsonl.
# No `set -e`: one shard hiccuping must not kill the others; --merge prints whatever finished.

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT"
LOGDIR="$HERE/logs"
mkdir -p "$LOGDIR"

DRIVER="ablate_muon/run_ablation.py"; PFX=""
if [ "$1" = "olm" ]; then
    DRIVER="ablate_muon/run_olm.py"; PFX="olm_"; shift
fi
NSHARDS="${1:-4}"                                    # concurrent shards on the one GPU
echo "[run_6000] driver $DRIVER; $NSHARDS concurrent shards on cuda:0; logs -> $LOGDIR"

PIDS=()
for s in $(seq 0 $((NSHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=0 python -u "$DRIVER" --shard "$s" --nshards "$NSHARDS" \
        2>&1 | sed -u "s/^/[t$s] /" | tee "$LOGDIR/${PFX}t$s.log" &
    PIDS+=($!)
done
wait "${PIDS[@]}"

echo "[run_6000] all $NSHARDS shards done — merging"
python -u "$DRIVER" --merge 2>&1 | tee "$LOGDIR/${PFX}merge.log"
echo "[run_6000] done. table + per-arm results in ablate_muon/results${PFX:+_olm}.jsonl"
