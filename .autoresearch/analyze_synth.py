"""Analyze synth_results.jsonl: steps-to-threshold + loss curves per arm/ratio, paired by seed."""
import json
from collections import defaultdict

rows = [json.loads(l) for l in open("synth_results.jsonl", encoding="utf-8-sig") if l.strip()]
curves = defaultdict(dict)     # (arm, ratio, seed) -> {step: (loss, acc)}
for r in rows:
    if "step" in r:
        curves[(r["arm"], r["ratio"], r["seed"])][r["step"]] = (r["val_loss"], r["val_acc"])

ARMS = ["adamw", "default", "b12", "champ"]
TH_ACC = [0.90, 0.99, 0.999]
TH_LOSS = [0.5, 0.1, 0.01]


def first_step(c, key, th, above):
    for s in sorted(c):
        v = c[s][0] if key == "loss" else c[s][1]
        if (v >= th if above else v <= th):
            return s
    return None


for ratio in (1, 4):
    print(f"\n===== MLP ratio {ratio} " + ("(ALL Muon mats square)" if ratio == 1 else "(rect control)") + " =====")
    print(f"{'arm':>8} | " + " | ".join(f"acc>={t:<5}" for t in TH_ACC) + " | " +
          " | ".join(f"loss<={t:<4}" for t in TH_LOSS) + " | loss@500    | loss@1500")
    for arm in ARMS:
        cells = []
        for th in TH_ACC:
            ss = [first_step(curves[(arm, ratio, s)], "acc", th, True) for s in (0, 1)]
            cells.append("/".join(str(x) if x else "--" for x in ss))
        for th in TH_LOSS:
            ss = [first_step(curves[(arm, ratio, s)], "loss", th, False) for s in (0, 1)]
            cells.append("/".join(str(x) if x else "--" for x in ss))
        l5 = [curves[(arm, ratio, s)].get(500, (float("nan"),))[0] for s in (0, 1)]
        l15 = [curves[(arm, ratio, s)].get(1500, (float("nan"),))[0] for s in (0, 1)]
        cells.append(" ".join(f"{v:.4f}" for v in l5))
        cells.append(" ".join(f"{v:.5f}" for v in l15))
        print(f"{arm:>8} | " + " | ".join(f"{c:^9}" for c in cells[:6]) + f" | {cells[6]} | {cells[7]}")

# early-phase paired curves (loss at each checkpoint, seed-averaged)
print("\n===== seed-mean val_loss trajectory (steps 100..700) =====")
for ratio in (1, 4):
    print(f"-- ratio {ratio} --")
    hdr = "step:  " + "  ".join(f"{s:>7}" for s in range(100, 701, 100))
    print(hdr)
    for arm in ARMS:
        vals = []
        for s in range(100, 701, 100):
            vs = [curves[(arm, ratio, sd)].get(s, (float("nan"),))[0] for sd in (0, 1)]
            vals.append(sum(vs) / 2)
        print(f"{arm:>6}: " + "  ".join(f"{v:7.4f}" for v in vals))
