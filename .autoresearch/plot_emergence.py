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
STEPS_10K = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 9500, 10000]  # v20 eval grid
STEPS_BY_AXIS = {"v20": STEPS_10K}                                   # axes on a non-default budget

# ---------------------------------------------------------------------------------------
# THE STORE. Per run: c=config color-key, s=seed, frac/d2/d3 trajectories (over STEPS),
# FINAL per-depth acc (d1..d6), per-layer spec, per-layer eff. ADD new waves here.
# ---------------------------------------------------------------------------------------

# --- AXIS: coefficient / iteration (RTX 6000, DEVICE-MATCHED; all wd 0.1, aurora_k1) -----
# v15 re-bench: ns_8 = v13 wd0.1 (anchor), ns_6/ns_10/PE-8 new on RTX 6000. T4 dropped
# (device shift). k2 lives in the scale axis. NOTE: 8/6/10-iter now TIE within seed noise
# (T4's clean "8>6>10" did NOT reproduce - seed noise dominates, as on the wd axis).
# PE-8 s0 NaN'd on BOTH devices (seed-deterministic overshoot); s1 trains but 50% NaN = dead.
RUNS_COEFF = {
 "8-iter KJ": dict(c="8it", s=0,
    frac=[.997,.808,.734,.599,.584,.574,.558,.549], d2=[.016,.043,.103,.125,.125,.136,.160,.183],
    d3=[.016,.017,.026,.028,.031,.035,.040,.047], dfin=[.946,.183,.047,.024,.019,.020],
    spec=[.07,.31,.19], eff=[7.8,7.3,7.7]),
 "8-iter KJ s1": dict(c="8it", s=1,
    frac=[.996,.860,.729,.671,.587,.570,.527,.503], d2=[.015,.021,.030,.067,.104,.146,.193,.268],
    d3=[.016,.020,.020,.021,.027,.035,.049,.077], dfin=[.945,.268,.077,.026,.020,.021],
    spec=[.13,.24,.23], eff=[7.6,7.3,7.0]),
 "6-iter (ns_4)": dict(c="6it", s=0,
    frac=[.997,.836,.763,.664,.628,.597,.578,.567], d2=[.014,.032,.052,.074,.077,.077,.095,.120],
    d3=[.017,.018,.020,.019,.026,.027,.026,.031], dfin=[.945,.120,.031,.022,.021,.018],
    spec=[.10,.31,.16], eff=[7.6,7.4,7.4]),
 "6-iter (ns_4) s1": dict(c="6it", s=1,
    frac=[.996,.870,.724,.660,.583,.546,.508,.483], d2=[.016,.021,.039,.084,.127,.161,.212,.296],
    d3=[.016,.019,.023,.025,.038,.057,.087,.129], dfin=[.904,.296,.129,.048,.022,.018],
    spec=[.12,.20,.25], eff=[7.6,7.4,7.2]),
 "10-iter (dsv4_10)": dict(c="10it", s=0,
    frac=[.997,.841,.707,.604,.593,.592,.572,.564], d2=[.016,.028,.048,.071,.073,.083,.100,.129],
    d3=[.016,.017,.022,.025,.028,.023,.028,.032], dfin=[.945,.129,.032,.020,.020,.018],
    spec=[.07,.32,.14], eff=[7.7,7.4,7.7]),
 "10-iter (dsv4_10) s1": dict(c="10it", s=1,
    frac=[.997,.839,.696,.582,.573,.568,.554,.534], d2=[.017,.020,.045,.122,.134,.144,.160,.178],
    d3=[.015,.019,.023,.029,.034,.040,.048,.056], dfin=[.945,.178,.056,.025,.025,.021],
    spec=[.02,.29,.24], eff=[7.8,7.1,7.0]),
 "PE-8 (s0 NaN'd) s1": dict(c="pe8", s=1,
    frac=[.996,.849,.727,.588,.552,.513,.483,.467], d2=[.015,.020,.029,.102,.148,.190,.246,.353],
    d3=[.016,.021,.020,.035,.060,.077,.103,.157], dfin=[.937,.353,.157,.059,.033,.021],
    spec=[.16,.22,.22], eff=[7.5,7.2,7.2]),
}
COL_COEFF = {"8it": "tab:blue", "10it": "tab:red", "6it": "tab:orange", "pe8": "tab:purple"}
CFGS_COEFF = ["8it", "6it", "10it", "pe8"]
CLABEL_COEFF = {"8it": "8-iter KJ", "6it": "6-iter", "10it": "10-iter", "pe8": "PE-8"}

# --- AXIS: weight decay (all 8-iter KJ aurora_k1) - RTX 6000, DEVICE-MATCHED --------------
# v13 (0.1/0.2) + v12 (0.3-1.0), all on RTX 6000. T4 numbers dropped (device shift ~+0.03-0.06
# frac; keeping them here would put a device seam at 0.1). Peak = wd 0.2. Low side (0.007-0.05)
# lands with v14 - ADD here then. (T4 low-side archived in git history if ever needed.)
RUNS_WD = {
 "wd 0.007": dict(c="w0.007", s=0,
    frac=[.997,.829,.664,.582,.574,.569,.560,.555], d2=[.017,.035,.107,.137,.140,.145,.154,.160],
    d3=[.015,.020,.026,.029,.030,.037,.043,.048], dfin=[.945,.160,.048,.025,.022,.017],
    spec=[.13,.30,.14], eff=[7.7,7.2,7.3]),
 "wd 0.007 s1": dict(c="w0.007", s=1,
    frac=[.996,.897,.702,.565,.522,.491,.468,.454], d2=[.017,.017,.033,.098,.176,.221,.301,.386],
    d3=[.015,.019,.021,.036,.053,.081,.118,.183], dfin=[.947,.386,.183,.083,.048,.016],
    spec=[.27,.22,.24], eff=[7.6,7.8,7.4]),
 "wd 0.01": dict(c="w0.01", s=0,
    frac=[.997,.787,.609,.531,.508,.498,.473,.462], d2=[.015,.049,.114,.172,.203,.228,.281,.338],
    d3=[.017,.022,.027,.046,.060,.073,.108,.139], dfin=[.944,.338,.139,.060,.032,.019],
    spec=[.10,.24,.20], eff=[7.8,7.5,7.3]),
 "wd 0.01 s1": dict(c="w0.01", s=1,
    frac=[.996,.883,.725,.590,.502,.454,.426,.415], d2=[.015,.021,.024,.077,.328,.484,.548,.559],
    d3=[.015,.016,.022,.027,.060,.164,.262,.310], dfin=[.947,.559,.310,.119,.022,.020],
    spec=[.10,.26,.22], eff=[7.4,7.2,7.3]),
 "wd 0.02": dict(c="w0.02", s=0,
    frac=[.997,.791,.679,.587,.577,.572,.562,.556], d2=[.016,.040,.108,.137,.137,.141,.154,.157],
    d3=[.014,.021,.025,.028,.035,.039,.040,.045], dfin=[.946,.157,.045,.022,.016,.015],
    spec=[.06,.29,.23], eff=[7.9,7.3,7.5]),
 "wd 0.02 s1": dict(c="w0.02", s=1,
    frac=[.997,.850,.743,.733,.669,.618,.578,.568], d2=[.014,.016,.020,.027,.040,.077,.127,.161],
    d3=[.014,.021,.021,.021,.023,.028,.035,.042], dfin=[.594,.161,.042,.023,.020,.017],
    spec=[.14,.20,.31], eff=[7.6,7.3,6.6]),
 "wd 0.05": dict(c="w0.05", s=0,
    frac=[.997,.803,.659,.588,.573,.570,.561,.556], d2=[.015,.030,.068,.129,.140,.141,.153,.156],
    d3=[.016,.020,.023,.029,.030,.035,.036,.045], dfin=[.947,.156,.045,.022,.019,.021],
    spec=[.11,.30,.30], eff=[7.3,7.2,7.5]),
 "wd 0.05 s1": dict(c="w0.05", s=1,
    frac=[.996,.927,.748,.744,.694,.617,.588,.577], d2=[.013,.020,.020,.038,.054,.079,.095,.109],
    d3=[.014,.016,.017,.021,.021,.024,.027,.026], dfin=[.923,.109,.026,.021,.021,.018],
    spec=[.07,.36,.34], eff=[7.5,7.5,6.9]),
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
_WD_ORDER = ["w0.007", "w0.01", "w0.02", "w0.05", "w0.1", "w0.2", "w0.3", "w0.5", "w0.7", "w1.0"]
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

# --- AXIS: muon LR (RTX 6000, bf16-amp, wd 0.1, aurora_k1, 8-iter) -----------------------
# v18 sweep. FIRST knob with a clear signal through seed noise (coarse). U-curve, optimum 1e-3.
RUNS_LR = {
 "lr 3e-4": dict(c="lr0.0003", s=0,
    frac=[.996,.995,.813,.745,.716,.670,.622,.608], d2=[.016,.020,.027,.032,.040,.058,.085,.099],
    d3=[.016,.014,.018,.023,.025,.024,.026,.027], dfin=[.859,.099,.027,.017,.018,.019],
    spec=[.02,.29,.26], eff=[7.8,7.7,7.8]),
 "lr 3e-4 s1": dict(c="lr0.0003", s=1,
    frac=[.996,.998,.955,.758,.720,.684,.654,.642], d2=[.015,.015,.018,.029,.035,.046,.057,.068],
    d3=[.016,.015,.021,.020,.019,.019,.022,.026], dfin=[.551,.068,.026,.018,.019,.019],
    spec=[.10,.29,.22], eff=[7.7,7.7,7.7]),
 "lr 1e-3 (best)": dict(c="lr0.001", s=0,
    frac=[.997,.809,.651,.580,.573,.568,.558,.553], d2=[.017,.034,.099,.128,.142,.149,.158,.162],
    d3=[.017,.019,.025,.029,.034,.036,.045,.054], dfin=[.947,.162,.054,.020,.019,.017],
    spec=[.06,.25,.29], eff=[7.8,7.1,7.5]),
 "lr 1e-3 (best) s1": dict(c="lr0.001", s=1,
    frac=[.996,.862,.726,.672,.593,.533,.489,.472], d2=[.018,.022,.032,.079,.117,.323,.464,.497],
    d3=[.015,.018,.018,.023,.031,.082,.195,.264], dfin=[.697,.497,.264,.070,.021,.019],
    spec=[.26,.19,.22], eff=[7.5,7.2,7.5]),
 "lr 2e-3": dict(c="lr0.002", s=0,
    frac=[.994,.863,.718,.700,.694,.695,.685,.682], d2=[.015,.026,.042,.037,.039,.039,.053,.060],
    d3=[.016,.021,.019,.019,.024,.025,.024,.026], dfin=[.158,.060,.026,.016,.018,.013],
    spec=[.04,.31,.17], eff=[7.9,7.1,7.8]),
 "lr 2e-3 s1 (best run)": dict(c="lr0.002", s=1,
    frac=[.996,.806,.615,.590,.491,.433,.409,.400], d2=[.015,.018,.018,.049,.304,.513,.544,.550],
    d3=[.015,.015,.020,.027,.087,.234,.311,.324], dfin=[.948,.550,.324,.194,.120,.064],
    spec=[.32,.16,.15], eff=[7.6,7.5,7.4]),
 "lr 4e-3": dict(c="lr0.004", s=0,
    frac=[.995,.946,.726,.715,.708,.708,.699,.688], d2=[.015,.022,.024,.026,.024,.020,.030,.038],
    d3=[.017,.020,.020,.021,.021,.020,.019,.019], dfin=[.155,.038,.019,.019,.014,.018],
    spec=[.10,.30,.23], eff=[7.7,7.1,7.2]),
 "lr 4e-3 s1": dict(c="lr0.004", s=1,
    frac=[.996,.754,.717,.712,.714,.711,.708,.712], d2=[.019,.020,.020,.019,.018,.022,.024,.025],
    d3=[.016,.019,.021,.017,.024,.018,.021,.018], dfin=[.176,.025,.018,.017,.017,.014],
    spec=[.39,.15,.25], eff=[7.3,6.6,6.0]),
 "lr 8e-3 (dead)": dict(c="lr0.008", s=0,
    frac=[.996,.992,.897,.889,.889,.888,.884,.884], d2=[.016,.015,.021,.026,.021,.026,.030,.030],
    d3=[.018,.013,.016,.018,.020,.018,.016,.020], dfin=[.030,.030,.020,.017,.015,.015],
    spec=[.01,.30,.17], eff=[7.4,7.3,7.3]),
 "lr 8e-3 s1": dict(c="lr0.008", s=1,
    frac=[.996,.916,.802,.799,.684,.622,.598,.566], d2=[.015,.017,.017,.019,.024,.028,.032,.121],
    d3=[.016,.015,.017,.018,.024,.020,.022,.022], dfin=[.947,.121,.022,.018,.018,.014],
    spec=[.14,.11,.22], eff=[7.6,7.5,7.4]),
}
_LR_ORDER = ["lr0.0003", "lr0.001", "lr0.002", "lr0.004", "lr0.008"]
CFGS_LR = _LR_ORDER
CLABEL_LR = {"lr0.0003": "3e-4", "lr0.001": "1e-3", "lr0.002": "2e-3", "lr0.004": "4e-3", "lr0.008": "8e-3"}
COL_LR = {k: plt.cm.viridis(i / (len(_LR_ORDER) - 1)) for i, k in enumerate(_LR_ORDER)}

# --- AXIS: scale mode @ 10k STEPS (RTX 6000, bf16-amp) - the RANKING REVERSES vs 6k ----------
# v20. At 6k aurora won; at 10k NORMUON wins (later but deeper - d2 0.72). ns_8 == ns_10.
RUNS_V20 = {
 "aurora ns8": dict(c="au8", s=0,
    frac=[.996,.823,.691,.598,.582,.577,.562,.551,.530,.519,.515], d2=[.014,.041,.067,.086,.097,.115,.136,.172,.229,.269,.303],
    d3=[.018,.020,.021,.023,.030,.031,.033,.032,.040,.047,.049], dfin=[.946,.303,.049,.023,.021,.020],
    spec=[.12,.23,.07], eff=[7.4,7.2,7.5]),
 "aurora ns8 s1": dict(c="au8", s=1,
    frac=[.996,.855,.727,.675,.529,.447,.418,.408,.399,.394,.391], d2=[.014,.020,.028,.074,.144,.484,.537,.546,.558,.561,.563],
    d3=[.016,.017,.020,.027,.044,.210,.284,.312,.333,.349,.353], dfin=[.948,.563,.353,.203,.124,.079],
    spec=[.11,.28,.18], eff=[7.6,7.6,7.4]),
 "aurora ns10": dict(c="au10", s=0,
    frac=[.997,.819,.702,.603,.587,.577,.566,.553,.533,.521,.516], d2=[.017,.042,.067,.093,.099,.112,.129,.158,.218,.261,.293],
    d3=[.017,.017,.024,.028,.026,.032,.031,.032,.039,.044,.049], dfin=[.947,.293,.049,.021,.019,.019],
    spec=[.11,.20,.14], eff=[7.7,7.6,7.9]),
 "aurora ns10 s1": dict(c="au10", s=1,
    frac=[.996,.833,.716,.637,.514,.459,.434,.413,.398,.389,.383], d2=[.015,.019,.019,.045,.323,.441,.476,.502,.521,.537,.542],
    d3=[.016,.016,.022,.027,.072,.236,.285,.311,.324,.339,.342], dfin=[.948,.542,.342,.202,.116,.018],
    spec=[.07,.17,.16], eff=[7.7,7.6,7.6]),
 "normuon (WINS)": dict(c="nor", s=0,
    frac=[.997,.846,.680,.585,.569,.509,.479,.462,.413,.391,.381], d2=[.017,.029,.062,.124,.144,.189,.243,.290,.473,.628,.721],
    d3=[.014,.019,.022,.028,.032,.062,.091,.094,.171,.250,.303], dfin=[.946,.721,.303,.151,.076,.043],
    spec=[.12,.20,.16], eff=[7.7,7.8,7.4]),
 "normuon (WINS) s1": dict(c="nor", s=1,
    frac=[.996,.856,.704,.605,.581,.573,.532,.493,.464,.454,.445], d2=[.014,.022,.047,.079,.121,.138,.181,.246,.354,.427,.487],
    d3=[.016,.017,.021,.023,.027,.033,.039,.074,.114,.161,.193], dfin=[.947,.487,.193,.026,.023,.024],
    spec=[.17,.24,.27], eff=[7.7,7.4,7.3]),
 "polar": dict(c="pol", s=0,
    frac=[.996,.846,.737,.610,.585,.576,.569,.562,.542,.529,.522], d2=[.015,.030,.066,.085,.103,.107,.124,.140,.177,.226,.249],
    d3=[.016,.018,.020,.024,.030,.031,.031,.033,.039,.043,.045], dfin=[.948,.249,.045,.023,.018,.022],
    spec=[.24,.25,.12], eff=[7.4,7.2,7.6]),
 "polar s1": dict(c="pol", s=1,
    frac=[.996,.893,.674,.582,.571,.564,.560,.551,.530,.517,.511], d2=[.013,.019,.058,.119,.135,.142,.158,.180,.236,.282,.302],
    d3=[.016,.017,.023,.031,.040,.040,.046,.050,.056,.068,.070], dfin=[.947,.302,.070,.025,.024,.021],
    spec=[.11,.25,.24], eff=[7.2,7.3,7.3]),
}
COL_V20 = {"au8": "tab:blue", "au10": "tab:cyan", "nor": "tab:orange", "pol": "tab:red"}
CFGS_V20 = ["nor", "au8", "au10", "pol"]
CLABEL_V20 = {"au8": "aurora ns8", "au10": "aurora ns10", "nor": "normuon", "pol": "polar"}

AXES = [
    ("coeff", "coeff/iter axis (RTX 6000, wd 0.1) - 8/6/10-iter TIE in noise; PE-8 s0 NaN",
     RUNS_COEFF, COL_COEFF, CFGS_COEFF, CLABEL_COEFF),
    ("lr", "muon LR axis (RTX 6000, bf16-amp) - U-curve, optimum 1e-3; 2e-3 = edge of stability",
     RUNS_LR, COL_LR, CFGS_LR, CLABEL_LR),
    ("v20", "scale mode @ 10k steps - RANKING REVERSES: normuon WINS (later but deeper); ns8==ns10",
     RUNS_V20, COL_V20, CFGS_V20, CLABEL_V20),
    ("wd", "weight-decay axis (RTX 6000, 8-iter KJ) - broad NOISY basin 0.01-0.2 "
     "(0.01 best on frac+AUC; non-monotone = seed noise)", RUNS_WD, COL_WD, CFGS_WD, CLABEL_WD),
    ("scale", "scale-mode axis (wd 0.1, 8-iter KJ)", RUNS_SCALE, COL_SCALE, CFGS_SCALE,
     CLABEL_SCALE),
]


def auc(y, steps=STEPS):
    s, f = np.array(steps, float), np.array(y, float)
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


def _line(ax, runs, col, key, ylab, title, floor=None, ylim=None, steps=STEPS):
    for name, r in runs.items():
        if r.get(key) is None:
            continue
        ax.plot(steps, r[key], "--" if r["s"] == 1 else "-", color=col.get(r["c"]),
                marker="o", ms=3, label=f"{name} ({r[key][-1]:.3f})")
    if floor is not None:
        ax.axhline(floor, color="k", ls=":", lw=1, label=f"floor {floor:.3f}")
    ax.set_xlabel("step"); ax.set_ylabel(ylab); ax.set_title(title)
    ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2, loc="upper right")
    if ylim:
        ax.set_ylim(*ylim)
    # WSD phase boundaries: warmup ends at 500, cosine decay starts at steps*(1-decay_frac).
    total = steps[-1]
    ax.axvline(500, color="red", ls="--", lw=1.2, alpha=0.7, label="warmup end (500)")
    ax.axvline(total * 0.8, color="red", ls="--", lw=1.2, alpha=0.7,
               label=f"decay start ({int(total * 0.8)})")
    # WSD LR-schedule reference (shared by all runs): dark line on a right-hand axis so its
    # true 0-1 shape (warmup / stable / cosine-decay) shows without distorting the metric.
    ax2 = ax.twinx()
    g = np.linspace(0, total, 400)
    ax2.plot(g, wsd(g, total=total), color="k", lw=2.2, ls="-.", alpha=0.85, zorder=0, label="WSD LR mult")
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


def dashboard(axname, title, runs, col, cfgs, clabel, steps=STEPS):
    fig, ax = plt.subplots(3, 3, figsize=(17, 13))
    _line(ax[0, 0], runs, col, "frac", "frac (lower=better)", "Compression (frac) vs step",
          floor=FLOOR, ylim=(0.35, 1.02), steps=steps)
    _line(ax[0, 1], runs, col, "d2", "depth-2 acc", "Depth-2 (sparse signal) vs step", ylim=(0, 0.75), steps=steps)
    _line(ax[0, 2], runs, col, "d3", "depth-3 acc", "Depth-3 (sparser) vs step", ylim=(0, 0.4), steps=steps)
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
    _bar(ax[1, 2], runs, col, cfgs, clabel, lambda r: auc(r["frac"], steps), "AUC frac", "AUC frac (noise-robust)")
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
        steps = STEPS_BY_AXIS.get(axname, STEPS)
        print(f"\n=== {axname}: {title} ===")
        print(f"{'run':24s} {'frac':>6s} {'d2':>6s} {'d3':>6s} {'AUC':>6s} {'eff':>5s} {'spec':>5s}")
        for name, r in runs.items():
            print(f"{name:24s} {r['frac'][-1]:6.3f} {r['d2'][-1]:6.3f} {r['d3'][-1]:6.3f} "
                  f"{auc(r['frac'], steps):6.3f} {np.mean(r['eff']) if r['eff'] else 0:5.1f} "
                  f"{np.mean(r['spec']) if r['spec'] else 0:5.2f}")
        dashboard(axname, title, runs, col, cfgs, clabel, steps)
        # individual curves per axis (the user's priority: compression + emergence depth>1)
        for key, ylab, ttl, fn, floor, ylim in [
            ("frac", "frac (lower=better)", "compression / phase transition", f"frac_{axname}.png",
             FLOOR, (0.35, 1.02)),
            ("d2", "depth-2 acc", "DEPTH-2 emergence (sparse signal)", f"depth2_{axname}.png",
             None, (0, 0.75)),
            ("d3", "depth-3 acc", "DEPTH-3 emergence (sparser)", f"depth3_{axname}.png",
             None, (0, 0.4))]:
            fig, a = plt.subplots(figsize=(9, 6))
            _line(a, runs, col, key, ylab, f"OLM {axname} - {ttl}", floor=floor, ylim=ylim, steps=steps)
            fig.tight_layout(); fig.savefig(os.path.join(OUT, fn), dpi=130); plt.close(fig)
    print(f"\n[plot] wrote dashboard_/frac_/depth2_/depth3_ per axis -> {OUT}")


if __name__ == "__main__":
    main()
