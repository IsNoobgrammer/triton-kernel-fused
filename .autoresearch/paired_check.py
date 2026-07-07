"""Paired per-seed comparison over 10 fresh seeds (20..29): rownorm vs rownorm+sperm:0.05
at B=10 and B=12. Uses the frozen eval's pipeline/metric unmodified."""
import math

import eval_kappa as ev

KJ = [3.4445, -4.775, 2.0315]; PIN = [2.0, -1.5, 0.5]
CANDS = {
    "B10_rown":  {"prescale": "rownorm", "stages": [[KJ] * 8 + [PIN] * 2]},
    "B10_sperm": {"prescale": "rownorm+sperm:0.05", "stages": [[KJ] * 8 + [PIN] * 2]},
    "B12_rown":  {"prescale": "rownorm", "stages": [[KJ] * 10 + [PIN] * 2]},
    "B12_sperm": {"prescale": "rownorm+sperm:0.05", "stages": [[KJ] * 10 + [PIN] * 2]},
}
SEEDS = list(range(20, 30))
res = {n: [] for n in CANDS}
for s in SEEDS:
    M = ev.gen(ev.N, ev.N, 2.0, s)
    for n, c in CANDS.items():
        res[n].append(ev.kappa_of(ev.run_pipeline(M, c)))
for B in ("B10", "B12"):
    a, b = res[f"{B}_rown"], res[f"{B}_sperm"]
    wins = sum(1 for x, y in zip(a, b) if y < x)
    print(f"{B}: per-seed rown vs sperm")
    print("  rown : " + " ".join(f"{k:8.3g}" for k in a))
    print("  sperm: " + " ".join(f"{k:8.3g}" for k in b))
    gm = lambda v: math.exp(sum(math.log(x) for x in v) / len(v))
    print(f"  geomean {gm(a):.3f} -> {gm(b):.3f} | max {max(a):.3g} -> {max(b):.3g} | sperm wins {wins}/10")
