"""C14 diagnostic — user's rho-batch-coupling hypothesis: rho-window x batch ~ const.

Prediction if true: at BS 32 (noisier grads) long memory (rho .995) beats short (.9);
at BS 512 (cleaner grads) short memory matches or beats long. Paired WITHIN each BS
(base at that BS is the reference); protocol otherwise identical to the frozen eval
(imported, BS overridden here — eval_manas.py untouched).

Run: ../../BiBo/.venv/Scripts/python.exe bs_rho.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
import numpy as np

import eval_manas as E
from run_wave import Trainer

GAMMA, SEEDS = 0.08, (0, 1, 2)
ARMS = {"base": 0.0, "rho90": 0.90, "rho98": 0.98, "rho995": 0.995}

for bs in (32, 512):
    E.BS = bs
    logs = {}
    for name, rho in ARMS.items():
        logs[name] = {sd: E.run_seed(
            lambda model, sd, r=rho: Trainer(model, gamma=GAMMA if r else 0.0, rho=r or 0.85),
            sd) for sd in SEEDS}
    print(f"\n== BS {bs} (paired vs base@BS{bs}, seeds {SEEDS}) ==", flush=True)
    for name in ("rho90", "rho98", "rho995"):
        s = E.score(logs[name], logs["base"], SEEDS)
        print(f"  {name}: frontier {s['delta_frontier_mean']:+.4f} (sem {s['noise_sem']:.4f})  "
              f"best_acc {s['delta_best_acc_mean']:+.4f}  train {s['delta_final_train_mean']:+.4f}",
              flush=True)
