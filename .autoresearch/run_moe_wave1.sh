#!/bin/bash
PY="/c/Users/shaur/OneDrive/Documents/BiBo/.venv/Scripts/python.exe"
export PYTHONPATH="/c/Users/shaur/OneDrive/Documents/triton-kernel-fused"
cd "/c/Users/shaur/OneDrive/Documents/triton-kernel-fused/.autoresearch"
B="--steps 3000 --batch 768"
"$PY" -u train_grok_moe.py --arm default --seed 0 $B > moe_default_s0.log 2>&1 &
"$PY" -u train_grok_moe.py --arm default --seed 1 $B > moe_default_s1.log 2>&1 &
"$PY" -u train_grok_moe.py --arm adamw   --seed 0 $B > moe_adamw_s0.log 2>&1 &
"$PY" -u train_grok_moe.py --arm adamw   --seed 1 $B > moe_adamw_s1.log 2>&1 &
"$PY" -u train_grok_moe.py --arm default --seed 0 --repulse 0.001 $B > moe_rep001_s0.log 2>&1 &
"$PY" -u train_grok_moe.py --arm default --seed 0 --repulse 0.01  $B > moe_rep01_s0.log 2>&1 &
wait
echo MOE_WAVE1_DONE
