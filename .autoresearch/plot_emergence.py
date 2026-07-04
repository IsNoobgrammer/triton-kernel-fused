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

# --- AXIS: weight decay (all 8-iter KJ aurora_k1; wd 0.1 = the champ, same run) ---------
# v10 sweep. HIGHER wd better; wd 0.2 = new champ. wd 0.1 reuses the coeff champ curves.
RUNS_WD = {
 "wd 0.01": dict(c="wd001", s=0,
    frac=[.997,.823,.714,.594,.581,.572,.562,.556], d2=[.018,.022,.063,.104,.129,.138,.151,.157],
    d3=[.017,.018,.022,.028,.030,.034,.038,.048], dfin=[.946,.157,.048,.023,.020,.019],
    spec=[.02,.23,.22], eff=[7.7,7.4,7.5]),
 "wd 0.01 s1": dict(c="wd001", s=1,
    frac=[.996,.892,.678,.595,.540,.508,.480,.466], d2=[.016,.019,.023,.061,.116,.180,.263,.328],
    d3=[.015,.022,.023,.024,.041,.066,.099,.141], dfin=[.947,.328,.141,.056,.024,.016],
    spec=[.19,.27,.22], eff=[7.7,7.0,7.4]),
 "wd 0.03": dict(c="wd003", s=0,
    frac=[.997,.822,.696,.597,.584,.580,.564,.554], d2=[.014,.047,.079,.104,.112,.117,.140,.162],
    d3=[.015,.018,.024,.027,.032,.034,.033,.040], dfin=[.946,.162,.040,.021,.019,.016],
    spec=[.12,.26,.15], eff=[7.4,7.5,7.6]),
 "wd 0.03 s1": dict(c="wd003", s=1,
    frac=[.996,.892,.722,.614,.605,.599,.587,.566], d2=[.016,.018,.022,.022,.020,.024,.051,.080],
    d3=[.015,.019,.021,.022,.019,.025,.024,.027], dfin=[.946,.080,.027,.021,.020,.023],
    spec=[.14,.27,.27], eff=[7.4,7.1,7.0]),
 "wd 0.05": dict(c="wd005", s=0,
    frac=[.997,.810,.697,.587,.577,.571,.561,.557], d2=[.017,.023,.088,.121,.135,.136,.152,.157],
    d3=[.016,.015,.024,.027,.028,.032,.038,.041], dfin=[.946,.157,.041,.025,.021,.021],
    spec=[.07,.28,.25], eff=[7.6,7.4,7.2]),
 "wd 0.05 s1": dict(c="wd005", s=1,
    frac=[.996,.846,.714,.702,.679,.612,.563,.548], d2=[.014,.023,.050,.084,.081,.096,.148,.176],
    d3=[.015,.023,.021,.029,.031,.031,.047,.063], dfin=[.584,.176,.063,.030,.022,.019],
    spec=[.11,.21,.23], eff=[7.7,7.5,7.2]),
 "wd 0.1 (champ)": dict(c="wd01", s=0,
    frac=[.997,.796,.722,.584,.573,.566,.550,.535], d2=[.016,.036,.113,.135,.144,.156,.194,.225],
    d3=[.015,.022,.026,.032,.032,.038,.047,.056], dfin=[.947,.225,.056,.023,.016,.019],
    spec=[.13,.26,.20], eff=[7.5,7.4,7.5]),
 "wd 0.1 (champ) s1": dict(c="wd01", s=1,
    frac=[.997,.872,.684,.560,.517,.489,.464,.451], d2=[.016,.019,.022,.094,.179,.232,.309,.421],
    d3=[.018,.017,.022,.031,.050,.080,.120,.177], dfin=[.948,.421,.177,.068,.031,.018],
    spec=[.17,.19,.25], eff=[7.6,7.5,7.3]),
 "wd 0.2 (NEW champ)": dict(c="wd02", s=0,
    frac=[.997,.821,.652,.525,.490,.471,.444,.427], d2=[.016,.026,.085,.165,.259,.319,.440,.541],
    d3=[.019,.020,.026,.060,.089,.131,.188,.272], dfin=[.946,.541,.272,.130,.057,.018],
    spec=[.07,.22,.21], eff=[7.9,7.6,7.5]),
 "wd 0.2 (NEW champ) s1": dict(c="wd02", s=1,
    frac=[.996,.851,.708,.692,.636,.540,.479,.463], d2=[.013,.022,.056,.085,.096,.264,.473,.499],
    d3=[.015,.016,.024,.026,.028,.074,.220,.297], dfin=[.637,.499,.297,.109,.030,.022],
    spec=[.32,.27,.19], eff=[7.5,7.4,7.3]),
}
COL_WD = {"wd001": "#c6dbef", "wd003": "#9ecae1", "wd005": "#6baed6", "wd01": "#3182bd",
          "wd02": "#08306b"}                                        # blues: light=low wd, dark=high
CFGS_WD = ["wd001", "wd003", "wd005", "wd01", "wd02"]
CLABEL_WD = {"wd001": "wd 0.01", "wd003": "wd 0.03", "wd005": "wd 0.05", "wd01": "wd 0.1",
             "wd02": "wd 0.2"}

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
    ("wd", "weight-decay axis (8-iter KJ aurora_k1) - HIGHER wd better", RUNS_WD, COL_WD,
     CFGS_WD, CLABEL_WD),
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
