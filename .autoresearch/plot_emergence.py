"""OLM ablation KPI dashboards — ACCUMULATING store, one dashboard per ablation axis.

Design: every benched run is embedded ONCE below, grouped by axis (coeff / wd / scale).
New waves ADD entries — we never reset. Each axis renders its own dashboard_<axis>.png +
frac/depth2/depth3 curves, so new data is always plotted against the old data on the same
instrument. The champ (8-iter KJ aurora_k1, wd 0.1) appears in every axis as the anchor.

Metrics: frac (compression, lower=better), depth-2/3 (emergence from SPARSE signal = the
real-LM axis), eff (utilization), spec (specialization), AUC (noise-robust curve metric).
  python plot_emergence.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "plots", "emergence")
os.makedirs(OUT, exist_ok=True)
FLOOR = 0.0924
STEPS = [1000, 2000, 3000, 4000, 4500, 5000, 5500, 6000]

# ---------------------------------------------------------------------------------------
# THE STORE. Per run: c=config color-key, s=seed, frac/d2/d3 trajectories (over STEPS),
# FINAL per-depth acc (d1..d6), per-layer spec, per-layer eff. ADD new waves here.
# ---------------------------------------------------------------------------------------

# --- AXIS: coefficient / iteration (all wd 0.1, aurora_k1 base) -------------------------
# v8 (8it/10it/k2) + v9 (6it/pe8). 8-iter KJ = champ.
RUNS_COEFF = {
 "8-iter KJ (champ)": dict(c="8it", s=0,
    frac=[.997,.796,.722,.584,.573,.566,.550,.535], d2=[.016,.036,.113,.135,.144,.156,.194,.225],
    d3=[.015,.022,.026,.032,.032,.038,.047,.056], dfin=[.947,.225,.056,.023,.016,.019],
    spec=[.13,.26,.20], eff=[7.5,7.4,7.5]),
 "8-iter KJ (champ) s1": dict(c="8it", s=1,
    frac=[.997,.872,.684,.560,.517,.489,.464,.451], d2=[.016,.019,.022,.094,.179,.232,.309,.421],
    d3=[.018,.017,.022,.031,.050,.080,.120,.177], dfin=[.948,.421,.177,.068,.031,.018],
    spec=[.17,.19,.25], eff=[7.6,7.5,7.3]),
 "10-iter (dsv4_10)": dict(c="10it", s=0,
    frac=[.997,.804,.709,.610,.603,.602,.595,.592], d2=[.015,.021,.024,.021,.020,.018,.021,.022],
    d3=[.016,.021,.021,.022,.021,.022,.019,.021], dfin=[.947,.022,.021,.019,.015,.015],
    spec=[.08,.49,.54], eff=[7.6,7.7,7.7]),
 "10-iter (dsv4_10) s1": dict(c="10it", s=1,
    frac=[.996,.858,.732,.717,.706,.634,.612,.609], d2=[.016,.020,.026,.040,.059,.073,.093,.090],
    d3=[.016,.017,.020,.021,.022,.025,.030,.032], dfin=[.584,.090,.032,.023,.019,.021],
    spec=[.13,.30,.36], eff=[7.4,7.1,6.9]),
 "k2 (aurora_k2)": dict(c="k2", s=0,
    frac=[.997,.835,.723,.611,.592,.585,.568,.558], d2=[.015,.039,.067,.093,.098,.103,.131,.156],
    d3=[.016,.021,.023,.025,.025,.029,.034,.034], dfin=[.947,.156,.034,.021,.015,.017],
    spec=[.09,.27,.16], eff=[7.5,7.1,7.6]),
 "k2 (aurora_k2) s1": dict(c="k2", s=1,
    frac=[.996,.871,.713,.603,.583,.555,.509,.495], d2=[.016,.022,.039,.132,.143,.171,.313,.386],
    d3=[.016,.017,.023,.024,.020,.021,.024,.027], dfin=[.946,.386,.027,.022,.019,.018],
    spec=[.04,.38,.32], eff=[7.8,7.5,7.5]),
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
COL_COEFF = {"8it": "tab:blue", "10it": "tab:red", "k2": "tab:green", "6it": "tab:orange",
             "pe8": "tab:purple"}
CFGS_COEFF = ["8it", "6it", "10it", "k2", "pe8"]
CLABEL_COEFF = {"8it": "8-iter KJ", "6it": "6-iter", "10it": "10-iter", "k2": "k2", "pe8": "PE-8"}

# --- AXIS: weight decay (all 8-iter KJ aurora_k1) - RTX 6000, DEVICE-MATCHED --------------
# v13 (0.1/0.2) + v12 (0.3-1.0), all on RTX 6000. T4 numbers dropped (device shift ~+0.03-0.06
# frac; keeping them here would put a device seam at 0.1). Peak = wd 0.2. Low side (0.007-0.05)
# lands with v14 - ADD here then. (T4 low-side archived in git history if ever needed.)
RUNS_WD = {
 "wd 0.1": dict(c="w0.1", s=0,
    frac=[.997,.808,.734,.599,.584,.574,.558,.549], d2=[.016,.043,.103,.125,.125,.136,.160,.183],
    d3=[.016,.017,.026,.028,.031,.035,.040,.047], dfin=[.946,.183,.047,.024,.019,.020],
    spec=[.07,.31,.19], eff=[7.8,7.3,7.7]),
 "wd 0.1 s1": dict(c="w0.1", s=1,
    frac=[.996,.860,.729,.671,.587,.570,.527,.503], d2=[.015,.021,.030,.067,.104,.146,.193,.268],
    d3=[.016,.020,.020,.021,.027,.035,.049,.077], dfin=[.945,.268,.077,.026,.020,.021],
    spec=[.13,.24,.23], eff=[7.6,7.3,7.0]),
 "wd 0.2 (champ)": dict(c="w0.2", s=0,
    frac=[.997,.831,.692,.602,.592,.589,.572,.559], d2=[.015,.036,.072,.087,.091,.096,.115,.150],
    d3=[.018,.017,.021,.024,.023,.026,.030,.035], dfin=[.944,.150,.035,.021,.017,.018],
    spec=[.09,.27,.17], eff=[7.5,7.5,7.7]),
 "wd 0.2 (champ) s1": dict(c="w0.2", s=1,
    frac=[.996,.889,.728,.628,.577,.519,.479,.454], d2=[.014,.020,.022,.057,.142,.371,.473,.492],
    d3=[.016,.017,.023,.024,.029,.099,.207,.262], dfin=[.857,.492,.262,.073,.022,.019],
    spec=[.02,.23,.22], eff=[7.8,7.4,7.2]),
 "wd 0.3": dict(c="w0.3", s=0,
    frac=[.997,.836,.674,.600,.590,.588,.570,.561], d2=[.016,.034,.058,.075,.081,.077,.097,.122],
    d3=[.016,.019,.023,.022,.026,.027,.033,.034], dfin=[.945,.122,.034,.024,.017,.019],
    spec=[.14,.20,.25], eff=[7.6,7.7,7.6]),
 "wd 0.3 s1": dict(c="w0.3", s=1,
    frac=[.997,.831,.705,.689,.688,.676,.613,.585], d2=[.019,.026,.068,.085,.076,.081,.093,.127],
    d3=[.018,.019,.023,.027,.030,.027,.035,.039], dfin=[.605,.127,.039,.025,.017,.017],
    spec=[.02,.25,.31], eff=[7.8,7.5,7.6]),
 "wd 0.5": dict(c="w0.5", s=0,
    frac=[.997,.838,.734,.611,.595,.590,.579,.563], d2=[.016,.032,.100,.121,.127,.134,.145,.156],
    d3=[.015,.018,.025,.026,.029,.028,.032,.038], dfin=[.935,.156,.038,.020,.016,.016],
    spec=[.10,.07,.20], eff=[7.8,7.8,7.9]),
 "wd 0.5 s1": dict(c="w0.5", s=1,
    frac=[.996,.918,.713,.688,.650,.577,.546,.524], d2=[.015,.020,.056,.085,.078,.139,.203,.260],
    d3=[.017,.019,.019,.028,.027,.028,.032,.036], dfin=[.846,.260,.036,.021,.020,.019],
    spec=[.04,.30,.22], eff=[7.9,7.7,7.7]),
 "wd 0.7": dict(c="w0.7", s=0,
    frac=[.997,.856,.715,.644,.624,.647,.582,.567], d2=[.017,.024,.049,.063,.073,.065,.107,.141],
    d3=[.015,.018,.018,.020,.023,.022,.026,.027], dfin=[.945,.141,.027,.021,.019,.019],
    spec=[.04,.25,.32], eff=[7.7,7.8,7.8]),
 "wd 0.7 s1 (no emerge)": dict(c="w0.7", s=1,
    frac=[.996,.968,.748,.722,.724,.721,.705,.687], d2=[.015,.016,.020,.024,.022,.024,.029,.034],
    d3=[.015,.014,.020,.020,.022,.023,.021,.022], dfin=[.277,.034,.022,.018,.016,.015],
    spec=[.06,.20,.11], eff=[7.9,7.8,7.9]),
 "wd 1.0": dict(c="w1.0", s=0,
    frac=[.996,.868,.791,.658,.639,.627,.598,.574], d2=[.016,.021,.045,.055,.057,.060,.079,.123],
    d3=[.018,.015,.021,.022,.023,.024,.021,.028], dfin=[.940,.123,.028,.017,.018,.018],
    spec=[.08,.19,.18], eff=[7.9,7.8,8.0]),
 "wd 1.0 s1": dict(c="w1.0", s=1,
    frac=[.996,.998,.744,.666,.618,.591,.574,.561], d2=[.013,.017,.020,.039,.077,.120,.147,.157],
    d3=[.015,.017,.018,.021,.025,.027,.034,.045], dfin=[.945,.157,.045,.023,.019,.020],
    spec=[.07,.06,.28], eff=[7.6,7.8,7.7]),
}
_WD_ORDER = ["w0.1", "w0.2", "w0.3", "w0.5", "w0.7", "w1.0"]        # add 0.007-0.05 when v14 lands
CFGS_WD = _WD_ORDER
CLABEL_WD = {k: f"wd {k[1:]}" for k in _WD_ORDER}
COL_WD = {k: plt.cm.viridis(i / (len(_WD_ORDER) - 1)) for i, k in enumerate(_WD_ORDER)}

# --- AXIS: scale mode (all wd 0.1, 8-iter KJ) ------------------------------------------
# aurora_k1 (= coeff champ) + aurora_k2 (= k2) HAVE curves. normuon + polar land with v11 -
# ADD them here when the run finishes (same schema).
RUNS_SCALE = {
 "aurora_k1 (champ)": dict(c="ak1", s=0,
    frac=[.997,.796,.722,.584,.573,.566,.550,.535], d2=[.016,.036,.113,.135,.144,.156,.194,.225],
    d3=[.015,.022,.026,.032,.032,.038,.047,.056], dfin=[.947,.225,.056,.023,.016,.019],
    spec=[.13,.26,.20], eff=[7.5,7.4,7.5]),
 "aurora_k1 (champ) s1": dict(c="ak1", s=1,
    frac=[.997,.872,.684,.560,.517,.489,.464,.451], d2=[.016,.019,.022,.094,.179,.232,.309,.421],
    d3=[.018,.017,.022,.031,.050,.080,.120,.177], dfin=[.948,.421,.177,.068,.031,.018],
    spec=[.17,.19,.25], eff=[7.6,7.5,7.3]),
 "aurora_k2": dict(c="ak2", s=0,
    frac=[.997,.835,.723,.611,.592,.585,.568,.558], d2=[.015,.039,.067,.093,.098,.103,.131,.156],
    d3=[.016,.021,.023,.025,.025,.029,.034,.034], dfin=[.947,.156,.034,.021,.015,.017],
    spec=[.09,.27,.16], eff=[7.5,7.1,7.6]),
 "aurora_k2 s1": dict(c="ak2", s=1,
    frac=[.996,.871,.713,.603,.583,.555,.509,.495], d2=[.016,.022,.039,.132,.143,.171,.313,.386],
    d3=[.016,.017,.023,.024,.020,.021,.024,.027], dfin=[.946,.386,.027,.022,.019,.018],
    spec=[.04,.38,.32], eff=[7.8,7.5,7.5]),
 "normuon": dict(c="normuon", s=0,
    frac=[.997,.808,.743,.599,.582,.574,.559,.549], d2=[.017,.019,.097,.122,.129,.132,.154,.180],
    d3=[.015,.021,.025,.027,.033,.033,.042,.042], dfin=[.946,.180,.042,.022,.019,.020],
    spec=[.08,.19,.22], eff=[7.9,7.4,7.5]),
 "normuon s1": dict(c="normuon", s=1,
    frac=[.996,.832,.664,.586,.579,.571,.565,.560], d2=[.016,.020,.090,.129,.130,.144,.152,.158],
    d3=[.015,.019,.021,.021,.019,.024,.024,.023], dfin=[.947,.158,.023,.019,.019,.018],
    spec=[.05,.37,.40], eff=[7.8,7.7,7.3]),
 "polar (base muon)": dict(c="polar", s=0,
    frac=[.996,.842,.751,.628,.603,.595,.575,.566], d2=[.017,.025,.054,.073,.079,.081,.092,.117],
    d3=[.016,.019,.024,.023,.025,.027,.028,.028], dfin=[.945,.117,.028,.021,.016,.020],
    spec=[.04,.30,.17], eff=[7.8,7.5,7.1]),
 "polar (base muon) s1": dict(c="polar", s=1,
    frac=[.996,.887,.702,.586,.578,.570,.562,.556], d2=[.014,.017,.061,.133,.145,.151,.158,.157],
    d3=[.016,.020,.023,.030,.031,.036,.043,.054], dfin=[.948,.157,.054,.025,.021,.022],
    spec=[.14,.36,.24], eff=[7.3,7.2,7.5]),
}
COL_SCALE = {"ak1": "tab:blue", "ak2": "tab:green", "normuon": "tab:orange", "polar": "tab:red"}
CFGS_SCALE = ["ak1", "ak2", "normuon", "polar"]
CLABEL_SCALE = {"ak1": "aurora_k1", "ak2": "aurora_k2", "normuon": "normuon", "polar": "polar"}

AXES = [
    ("coeff", "coefficient / iteration axis (wd 0.1, aurora_k1)", RUNS_COEFF, COL_COEFF,
     CFGS_COEFF, CLABEL_COEFF),
    ("wd", "weight-decay axis (RTX 6000, 8-iter KJ aurora_k1) - peak at wd 0.2", RUNS_WD,
     COL_WD, CFGS_WD, CLABEL_WD),
    ("scale", "scale-mode axis (wd 0.1, 8-iter KJ)", RUNS_SCALE, COL_SCALE, CFGS_SCALE,
     CLABEL_SCALE),
]


def auc(y):
    s, f = np.array(STEPS, float), np.array(y, float)
    trap = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
    return trap(f, s) / (s[-1] - s[0])


def wsd(grid, total=6000, warmup=500, decay_frac=0.2, min_lr=0.1):
    """WSD LR multiplier (olm.py): linear warmup -> stable 1.0 -> cosine decay to min_lr.
    Identical for every run at 6000 steps, so it overlays as a shared reference line."""
    g = np.asarray(grid, float); out = np.ones_like(g)
    wu = g <= warmup; out[wu] = g[wu] / warmup
    t0 = total * (1 - decay_frac); dec = g > t0
    prog = (g[dec] - t0) / max(total - t0, 1)
    out[dec] = min_lr + (1 - min_lr) * 0.5 * (1 + np.cos(np.pi * prog))
    return out


def _line(ax, runs, col, key, ylab, title, floor=None, ylim=None):
    for name, r in runs.items():
        if r.get(key) is None:
            continue
        ax.plot(STEPS, r[key], "--" if r["s"] == 1 else "-", color=col.get(r["c"]),
                marker="o", ms=3, label=f"{name} ({r[key][-1]:.3f})")
    if floor is not None:
        ax.axhline(floor, color="k", ls=":", lw=1, label=f"floor {floor:.3f}")
    ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2, loc="upper right")
    if ylim:
        ax.set_ylim(*ylim)
    # WSD LR-schedule reference (shared by all runs): dark line on a right-hand axis so its
    # true 0-1 shape (warmup / stable / cosine-decay) shows without distorting the metric.
    ax2 = ax.twinx()
    g = np.linspace(0, STEPS[-1], 400)
    ax2.plot(g, wsd(g), color="k", lw=2.2, ls="-.", alpha=0.85, zorder=0, label="WSD LR mult")
    ax2.set_ylim(0, 1.08); ax2.set_ylabel("LR mult (WSD)", fontsize=8)
    ax2.legend(loc="lower left", fontsize=6)


def _bar(ax, runs, col, cfgs, clabel, fn, ylab, title):
    m = {c: [fn(r) for r in runs.values() if r["c"] == c] for c in cfgs}
    present = [c for c in cfgs if m[c]]
    x = np.arange(len(present))
    ax.bar(x, [np.mean(m[c]) for c in present], color=[col[c] for c in present], alpha=0.8)
    for i, c in enumerate(present):                                 # seed dots
        ax.scatter([i] * len(m[c]), m[c], color="k", s=18, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels([clabel[c] for c in present], fontsize=8, rotation=15)
    ax.set_ylabel(ylab); ax.set_title(title); ax.grid(alpha=0.3, axis="y")


def dashboard(axname, title, runs, col, cfgs, clabel):
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    _line(ax[0, 0], runs, col, "frac", "frac (lower=better)", "Compression (frac) vs step",
          floor=FLOOR, ylim=(0.4, 1.02))
    _line(ax[0, 1], runs, col, "d2", "depth-2 acc", "Depth-2 (sparse signal) vs step", ylim=(0, 0.6))
    _line(ax[0, 2], runs, col, "d3", "depth-3 acc", "Depth-3 (sparser) vs step", ylim=(0, 0.32))
    # per-depth final bars (grouped by depth, config colors, mean over seeds)
    present = [c for c in cfgs if any(r["c"] == c for r in runs.values())]
    m = {c: np.mean([r["dfin"] for r in runs.values() if r["c"] == c], axis=0) for c in present}
    nd = len(next(iter(m.values()))); xx = np.arange(nd); w = 0.8 / len(present)
    for i, c in enumerate(present):
        ax[1, 0].bar(xx + (i - len(present) / 2) * w + w / 2, m[c], w, color=col[c], label=clabel[c])
    ax[1, 0].set_xticks(xx); ax[1, 0].set_xticklabels([f"d{d+1}" for d in range(nd)])
    ax[1, 0].set_ylabel("final acc"); ax[1, 0].set_title("Final accuracy by depth (hierarchy)")
    ax[1, 0].legend(fontsize=7); ax[1, 0].grid(alpha=0.3, axis="y")
    _bar(ax[1, 1], runs, col, cfgs, clabel, lambda r: r["frac"][-1], "frac", "Final frac (dots=seeds)")
    _bar(ax[1, 2], runs, col, cfgs, clabel, lambda r: auc(r["frac"]), "AUC frac", "AUC frac (noise-robust)")
    _bar(ax[2, 0], runs, col, cfgs, clabel, lambda r: r["d2"][-1], "depth-2 acc",
         "Final depth-2 (sparse signal)")
    _bar(ax[2, 1], runs, col, cfgs, clabel, lambda r: np.mean(r["eff"]) if r["eff"] else 0,
         "eff experts", "Effective experts (util; higher=better)")
    _bar(ax[2, 2], runs, col, cfgs, clabel, lambda r: np.mean(r["spec"]) if r["spec"] else 0,
         "spec frac", "Specialization (mean over layers)")
    fig.suptitle(f"OLM KPI dashboard - {title}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(os.path.join(OUT, f"dashboard_{axname}.png"), dpi=120); plt.close(fig)


def main():
    for axname, title, runs, col, cfgs, clabel in AXES:
        if not runs:
            continue
        print(f"\n=== {axname}: {title} ===")
        print(f"{'run':24s} {'frac':>6s} {'d2':>6s} {'d3':>6s} {'AUC':>6s} {'eff':>5s} {'spec':>5s}")
        for name, r in runs.items():
            print(f"{name:24s} {r['frac'][-1]:6.3f} {r['d2'][-1]:6.3f} {r['d3'][-1]:6.3f} "
                  f"{auc(r['frac']):6.3f} {np.mean(r['eff']) if r['eff'] else 0:5.1f} "
                  f"{np.mean(r['spec']) if r['spec'] else 0:5.2f}")
        dashboard(axname, title, runs, col, cfgs, clabel)
        # individual curves per axis (the user's priority: compression + emergence depth>1)
        for key, ylab, ttl, fn, floor, ylim in [
            ("frac", "frac (lower=better)", "compression / phase transition", f"frac_{axname}.png",
             FLOOR, (0.4, 1.02)),
            ("d2", "depth-2 acc", "DEPTH-2 emergence (sparse signal)", f"depth2_{axname}.png",
             None, (0, 0.6)),
            ("d3", "depth-3 acc", "DEPTH-3 emergence (sparser)", f"depth3_{axname}.png",
             None, (0, 0.32))]:
            fig, a = plt.subplots(figsize=(9, 6))
            _line(a, runs, col, key, ylab, f"OLM {axname} - {ttl}", floor=floor, ylim=ylim)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, fn), dpi=130); plt.close(fig)
    print(f"\n[plot] wrote dashboard_/frac_/depth2_/depth3_ per axis -> {OUT}")


if __name__ == "__main__":
    main()
