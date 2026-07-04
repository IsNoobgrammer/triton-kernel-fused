"""Emergence-curve plots + noise-robust curve metrics for the olm ablation.

WHY: frac@final has huge seed variance because the task learns by a sharp phase transition
whose TIMING is seed-sensitive. The trajectory (AUC, step-to-threshold) is far less noisy
than the endpoint. This renders labeled figures + prints the curve metrics.

Data: reads .autoresearch/results_olm.jsonl (curve field) if present; else uses the v7
trajectories embedded below (parsed from the v7 console output, 2026-07-04). Saves PNGs to
.autoresearch/plots/emergence/.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "emergence")
os.makedirs(OUT, exist_ok=True)
LNP = np.log(97)
FLOOR = 0.0924                                                        # 5% noise floor (frac)

STEPS = [1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000]
# label -> (config, seed, frac trajectory over STEPS). v8 NS-config comparison (fixed bias).
V7 = {
    "8-iter (ns8, default)":  ("8it", 0, [0.997, 0.796, 0.722, 0.584, 0.573, 0.566, 0.550, 0.535]),
    "8-iter (ns8) s1":        ("8it", 1, [0.997, 0.872, 0.684, 0.560, 0.517, 0.489, 0.464, 0.451]),
    "10-iter (dsv4_10)":      ("10it", 0, [0.997, 0.804, 0.709, 0.610, 0.603, 0.602, 0.595, 0.592]),
    "10-iter (dsv4_10) s1":   ("10it", 1, [0.996, 0.858, 0.732, 0.717, 0.706, 0.634, 0.612, 0.609]),
    "k2 (aurora_k2)":         ("k2", 0, [0.997, 0.835, 0.723, 0.611, 0.592, 0.585, 0.568, 0.558]),
    "k2 (aurora_k2) s1":      ("k2", 1, [0.996, 0.871, 0.713, 0.603, 0.583, 0.555, 0.509, 0.495]),
}


def load_runs():
    """Prefer real results_olm.jsonl; fall back to embedded v7."""
    p = os.path.join(HERE, "results_olm.jsonl")
    if os.path.exists(p):
        runs = {}
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            steps = [c[0] for c in r["curve"]]
            frac = [c[1] for c in r["curve"]]
            runs[f"{r['arm']}_s{r['seed']}"] = (r.get("arm"), r["seed"], steps, frac)
        return runs, True
    return {k: (v[0], v[1], STEPS, v[2]) for k, v in V7.items()}, False


def auc(steps, frac):
    """Mean frac over training (trapezoidal / range). Lower = compressed faster + deeper."""
    s, f = np.array(steps, float), np.array(frac, float)
    trap = getattr(np, "trapezoid", getattr(np, "trapz", None))       # np2 renamed trapz
    return trap(f, s) / (s[-1] - s[0])


def step_to(steps, frac, thr):
    """First step where frac crosses <= thr (linear interp). None if never."""
    s, f = np.array(steps, float), np.array(frac, float)
    for i in range(1, len(f)):
        if f[i] <= thr:
            if f[i - 1] <= thr:
                return s[i - 1]
            t = (f[i - 1] - thr) / (f[i - 1] - f[i])
            return s[i - 1] + t * (s[i] - s[i - 1])
    return None


def main():
    runs, real = load_runs()
    src = "results_olm.jsonl" if real else "embedded v7 console data"

    # ---- metrics table ----
    print(f"[plot_emergence] source: {src}\n")
    print(f"{'run':22s} {'final':>7s} {'AUC':>7s} {'->0.65':>8s} {'->0.60':>8s}")
    rows = {}
    for name, (arm, seed, steps, frac) in runs.items():
        a = auc(steps, frac)
        t65, t60 = step_to(steps, frac, 0.65), step_to(steps, frac, 0.60)
        rows[name] = (frac[-1], a, t65, t60)
        print(f"{name:22s} {frac[-1]:7.3f} {a:7.3f} "
              f"{('%.0f' % t65) if t65 else '  --':>8s} {('%.0f' % t60) if t60 else '  --':>8s}")

    # ---- Fig 1: all emergence curves ----
    plt.figure(figsize=(9, 6))
    for name, (arm, seed, steps, frac) in runs.items():
        ls = "--" if seed == 1 else "-"
        plt.plot(steps, frac, ls, marker="o", ms=3, label=name)
    plt.axhline(FLOOR, color="k", ls=":", lw=1, label=f"noise floor {FLOOR:.3f}")
    plt.xlabel("training step"); plt.ylabel("frac  (val CE / ln 97;  lower = better)")
    plt.title("olm v8 NS-config emergence curves — compression vs step\n8-iter (ns8) vs 10-iter (dsv4_10) vs k2, 2 seeds each (one epoch, 5% noise)")
    plt.legend(fontsize=8, ncol=2); plt.grid(alpha=0.3); plt.ylim(0.4, 1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "emergence_curves.png"), dpi=130)
    plt.close()

    # ---- Fig 2: seed-noise story (paired configs with 2 seeds) ----
    plt.figure(figsize=(9, 6))
    pairs = {"8-iter": "tab:blue", "10-iter": "tab:red", "k2": "tab:green"}
    for base, col in pairs.items():
        for name, (arm, seed, steps, frac) in runs.items():
            if name.startswith(base):
                ls = "--" if seed == 1 else "-"
                plt.plot(steps, frac, ls, color=col, marker="o", ms=3,
                         label=f"{name} (final {frac[-1]:.3f}, AUC {auc(steps, frac):.3f})")
    plt.axhline(FLOOR, color="k", ls=":", lw=1)
    plt.xlabel("training step"); plt.ylabel("frac (lower = better)")
    plt.title("v8 NS-config, 2 seeds each: 8-iter (blue) < k2 (green) < 10-iter (red)\nsame-seed ranking is consistent despite the timing noise in the tail")
    plt.legend(fontsize=8); plt.grid(alpha=0.3); plt.ylim(0.4, 1.02)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "seed_noise.png"), dpi=130)
    plt.close()

    # ---- Fig 3: AUC vs final, grouped by config (noise-robust ranking) ----
    cfgs = {}
    for name, (arm, seed, steps, frac) in runs.items():
        cfgs.setdefault(arm, []).append((frac[-1], auc(steps, frac)))
    names = list(cfgs)
    x = np.arange(len(names))
    fin_m = [np.mean([v[0] for v in cfgs[n]]) for n in names]
    fin_s = [np.std([v[0] for v in cfgs[n]]) for n in names]
    auc_m = [np.mean([v[1] for v in cfgs[n]]) for n in names]
    auc_s = [np.std([v[1] for v in cfgs[n]]) for n in names]
    plt.figure(figsize=(9, 6))
    plt.bar(x - 0.2, fin_m, 0.4, yerr=fin_s, capsize=4, label="final frac (noisy)")
    plt.bar(x + 0.2, auc_m, 0.4, yerr=auc_s, capsize=4, label="AUC / mean frac (robust)")
    plt.xticks(x, names, rotation=20, ha="right")
    plt.ylabel("frac (lower = better)")
    plt.title("Per-config: final-frac vs curve-AUC  (error bar = seed spread)\nAUC error bars are tighter -> less seed noise")
    plt.legend(); plt.grid(alpha=0.3, axis="y")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "auc_vs_final.png"), dpi=130)
    plt.close()

    print(f"\n[plot_emergence] wrote 3 PNGs -> {OUT}")


if __name__ == "__main__":
    main()
