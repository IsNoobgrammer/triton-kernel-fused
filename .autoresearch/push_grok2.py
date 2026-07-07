
arms = ["default", "ns6", "ns8", "jns6", "k2", "normuon"]
lines = ["#!/bin/bash", "export PYTHONPATH=/home/marimo/triton-kernel-fused", "cd /home/marimo", "pids=()"]
for arm in arms:
    for seed in (0, 1, 2):
        lines.append(f"python -u train_grok.py --arm {arm} --seed {seed} --frac 0.45 --wd 1.0 > grok_g2_{arm}_s{seed}.log 2>&1 & pids+=($!)")
for wd in ("0.6", "2.0"):
    lines.append(f"python -u train_grok.py --arm default --seed 0 --frac 0.45 --wd {wd} > grok_g2_wd{wd.replace('.','')}_s0.log 2>&1 & pids+=($!)")
lines += ['for p in "${pids[@]}"; do wait $p; done', "echo GRID2_DONE"]
open("/home/marimo/run_grok_grid2.sh", "w").write("\n".join(lines) + "\n")
import subprocess
subprocess.Popen(["bash", "-c", "nohup bash /home/marimo/run_grok_grid2.sh > /home/marimo/grok_grid2_pipeline.log 2>&1 &"])
import time; time.sleep(10)
print(subprocess.run(["bash", "-c", "pgrep -fc train_grok.py"], capture_output=True, text=True).stdout.strip(), "jobs (expect 20)")
