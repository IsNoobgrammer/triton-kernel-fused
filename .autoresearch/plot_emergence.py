"""OLM ablation dashboard — lots of plots. Compression (frac), compositional learning from
SPARSE signal (depth-2/3, the real-LM axis), utilization (eff), specialization (spec), and
noise-robust curve metrics (AUC). Saves individual PNGs + one dashboard.png to plots/emergence/.

Reads .autoresearch/results_olm.jsonl if present (per_depth in curve); else uses embedded v8.
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
FLOOR = 0.0924
STEPS = [1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000]

# v9 coefficient/iteration comparison. Per arm: config color-key, seed, frac/d2/d3
# trajectories, FINAL per-depth acc (d1..d6), per-layer spec, per-layer eff.
# 8-iter KJ = the champ (floor); 6-iter (ns_4) = cheaper KJ; PE-8 = Polar-Express minimax
# (rejected: s0 NaN'd - not plottable - and s1 emerged late & worst).
RUNS = {
 "8-iter KJ (champ)": dict(c="8it", s=0,
    frac=[.997,.796,.722,.584,.573,.566,.550,.535], d2=[.016,.036,.113,.135,.144,.156,.194,.225],
    d3=[.015,.022,.026,.032,.032,.038,.047,.056], dfin=[.947,.225,.056,.023,.016,.019],
    spec=[.13,.26,.20], eff=[7.5,7.4,7.5]),
 "8-iter KJ (champ) s1": dict(c="8it", s=1,
    frac=[.997,.872,.684,.560,.517,.489,.464,.451], d2=[.016,.019,.022,.094,.179,.232,.309,.421],
    d3=[.018,.017,.022,.031,.050,.080,.120,.177], dfin=[.948,.421,.177,.068,.031,.018],
    spec=[.17,.19,.25], eff=[7.6,7.5,7.3]),
 "6-iter (ns_4)": dict(c="6it", s=0,
    frac=[.997,.794,.668,.583,.573,.567,.552,.542], d2=[.018,.032,.126,.144,.144,.156,.176,.205],
    d3=[.016,.019,.028,.037,.038,.042,.049,.060], dfin=[.947,.205,.060,.026,.017,.019],
    spec=[.23,.28,.13], eff=[7.6,7.1,7.5]),
 "6-iter (ns_4) s1": dict(c="6it", s=1,
    frac=[.996,.865,.733,.616,.581,.539,.500,.484], d2=[.014,.021,.035,.094,.112,.141,.186,.240],
    d3=[.015,.021,.019,.026,.032,.045,.076,.098], dfin=[.940,.240,.098,.050,.025,.019],
    spec=[.07,.19,.20], eff=[7.4,7.5,7.3]),
 "PE-8 (s0 NaN'd) s1": dict(c="pe8", s=1,
    frac=[.996,.875,.720,.696,.695,.687,.603,.580], d2=[.017,.023,.048,.080,.078,.088,.107,.138],
    d3=[.016,.019,.018,.027,.031,.030,.036,.045], dfin=[.560,.138,.045,.020,.019,.021],
    spec=[.19,.36,.28], eff=[7.5,6.9,7.1]),
}
COL = {"8it": "tab:blue", "6it": "tab:orange", "pe8": "tab:red"}
CFGS = ["8it", "6it", "pe8"]
CLABEL = {"8it": "8-iter KJ", "6it": "6-iter (ns_4)", "pe8": "PE-8"}


def load():
    p = os.path.join(HERE, "results_olm.jsonl")
    if os.path.exists(p):
        runs = {}
        ok = True
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            if not r["curve"] or len(r["curve"][0]) < 4:
                ok = False; break
            cu = r["curve"]
            runs[r.get("tag", r.get("arm"))] = dict(
                c=r.get("arm", "?"), s=r["seed"], steps=[c[0] for c in cu],
                frac=[c[1] for c in cu], d2=[c[3][1] for c in cu], d3=[c[3][2] for c in cu],
                dfin=r["per_depth"], spec=r.get("spec_frac", []), eff=r.get("eff_experts", []))
        if ok and runs:
            return runs, True
    return {k: dict(steps=STEPS, **v) for k, v in RUNS.items()}, False


def auc(steps, y):
    s, f = np.array(steps, float), np.array(y, float)
    if len(s) < 2:
        return float(f[-1])
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trap(f, s) / (s[-1] - s[0])


def by_cfg(runs, fn):
    """fn(run)->scalar; returns {cfg: [values over its seeds]}."""
    out = {c: [] for c in CFGS}
    for r in runs.values():
        if r["c"] in out:
            out[r["c"]].append(fn(r))
    return out


def _line(ax, runs, key, ylab, title, floor=None, ylim=None):
    for name, r in runs.items():
        ax.plot(r["steps"], r[key], "--" if r["s"] == 1 else "-", color=COL.get(r["c"]),
                marker="o", ms=3, label=name)
    if floor is not None:
        ax.axhline(floor, color="k", ls=":", lw=1, label=f"floor {floor:.3f}")
    ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    if ylim:
        ax.set_ylim(*ylim)


def _bar(ax, runs, fn, ylab, title, better="low"):
    m = by_cfg(runs, fn)
    x = np.arange(len(CFGS))
    means = [np.mean(m[c]) for c in CFGS]
    ax.bar(x, means, color=[COL[c] for c in CFGS], alpha=0.8)
    for i, c in enumerate(CFGS):                                    # seed dots
        ax.scatter([i] * len(m[c]), m[c], color="k", s=18, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([CLABEL[c] for c in CFGS], fontsize=8)
    ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=0.3, axis="y")


def dashboard(runs):
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    _line(ax[0, 0], runs, "frac", "frac (lower=better)", "Compression (frac) vs step",
          floor=FLOOR, ylim=(0.4, 1.02))
    _line(ax[0, 1], runs, "d2", "depth-2 acc", "Depth-2 (sparse signal) vs step", ylim=(0, 0.5))
    _line(ax[0, 2], runs, "d3", "depth-3 acc", "Depth-3 (sparser) vs step", ylim=(0, 0.2))
    # per-depth final bars (grouped by depth, config colors, mean over seeds)
    m = {c: np.mean([r["dfin"] for r in runs.values() if r["c"] == c], axis=0) for c in CFGS}
    nd = len(next(iter(m.values()))); xx = np.arange(nd)
    for i, c in enumerate(CFGS):
        ax[1, 0].bar(xx + (i - 1) * 0.27, m[c], 0.27, color=COL[c], label=CLABEL[c])
    ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels([f"d{d+1}" for d in range(nd)])
    ax[1, 0].set_ylabel("final acc"); ax[1, 0].set_title("Final accuracy by depth (learning hierarchy)")
    ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=0.3, axis="y")
    _bar(ax[1, 1], runs, lambda r: r["frac"][-1], "frac", "Final frac (dots=seeds)")
    _bar(ax[1, 2], runs, lambda r: auc(r["steps"], r["frac"]), "AUC frac", "AUC frac (noise-robust)")
    _bar(ax[2, 0], runs, lambda r: r["d2"][-1], "depth-2 acc", "Final depth-2 (sparse signal)")
    _bar(ax[2, 1], runs, lambda r: np.mean(r["eff"]) if r["eff"] else 0, "eff experts",
         "Effective experts (util; higher=better)")
    _bar(ax[2, 2], runs, lambda r: np.mean(r["spec"]) if r["spec"] else 0, "spec frac",
         "Specialization (mean over layers)")
    fig.suptitle("OLM v9 KPI dashboard — 8-iter KJ champ (blue) vs 6-iter ns_4 (orange) vs "
                 "PE-8 (red, s0 NaN'd)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, "dashboard.png"), dpi=120); plt.close(fig)


def main():
    runs, real = load()
    print(f"[plot] source: {'results_olm.jsonl' if real else 'embedded v9'}\n")
    print(f"{'run':24s} {'frac':>6s} {'d2':>6s} {'d3':>6s} {'AUC':>6s} {'eff':>5s} {'spec':>5s}")
    for name, r in runs.items():
        print(f"{name:24s} {r['frac'][-1]:6.3f} {r['d2'][-1]:6.3f} {r['d3'][-1]:6.3f} "
              f"{auc(r['steps'], r['frac']):6.3f} {np.mean(r['eff']) if r['eff'] else 0:5.1f} "
              f"{np.mean(r['spec']) if r['spec'] else 0:5.2f}")

    # individual plots
    for key, ylab, title, fn, floor, ylim in [
        ("frac", "frac (val CE / ln 97; lower=better)", "OLM v9 — compression (frac)",
         "emergence_curves.png", FLOOR, (0.4, 1.02)),
        ("d2", "depth-2 accuracy (higher=better)",
         "OLM v9 — DEPTH-2 (learning from SPARSE compositional signal)", "depth2_accuracy.png",
         None, (0, 0.5)),
        ("d3", "depth-3 accuracy (higher=better)", "OLM v9 — DEPTH-3 (even sparser signal)",
         "depth3_accuracy.png", None, (0, 0.2))]:
        fig, a = plt.subplots(figsize=(9, 6))
        _line(a, runs, key, ylab, title, floor=floor, ylim=ylim)
        fig.tight_layout(); fig.savefig(os.path.join(OUT, fn), dpi=130); plt.close(fig)

    dashboard(runs)
    print(f"\n[plot] wrote dashboard.png + emergence/depth2/depth3 -> {OUT}")


if __name__ == "__main__":
    main()
