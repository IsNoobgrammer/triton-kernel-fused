
lines = ["#!/bin/bash", "export PYTHONPATH=/home/marimo/triton-kernel-fused", "cd /home/marimo", "pids=()"]
for arm in ("default", "ns6"):
    for d in (128, 256, 512):
        for frac in ("0.35", "0.55"):
            for seed in (0, 1):
                tag = f"{arm}_d{d}_f{frac.replace('0.','')}_s{seed}"
                lines.append(f"python -u train_grok.py --arm {arm} --seed {seed} --frac {frac} --d {d} --wd 2.0 > grok_g3_{tag}.log 2>&1 & pids+=($!)")
for wd in ("3.0", "4.0"):
    lines.append(f"python -u train_grok.py --arm default --seed 0 --frac 0.45 --wd {wd} > grok_g3_wd{wd.replace('.','')}_s0.log 2>&1 & pids+=($!)")
lines += ['for p in "${pids[@]}"; do wait $p; done', "echo GRID3_DONE"]
open("/home/marimo/run_grok_grid3.sh", "w").write("\n".join(lines) + "\n")
import subprocess
subprocess.Popen(["bash", "-c", "nohup bash /home/marimo/run_grok_grid3.sh > /home/marimo/grok_grid3_pipeline.log 2>&1 &"])
import time; time.sleep(10)
print(subprocess.run(["bash", "-c", "pgrep -fc train_grok.py"], capture_output=True, text=True).stdout.strip(), "jobs (expect 26)")
