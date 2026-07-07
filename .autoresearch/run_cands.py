"""Runner: evaluate a list of candidate dicts through the FROZEN eval (imports eval_kappa, no edits).
Usage: python run_cands.py cands.json [--holdout]
cands.json = {"name": candidate, ...}
"""
import json
import math
import sys

import eval_kappa as ev


def score(cand, seeds):
    cost = sum(len(s) for s in cand["stages"])
    out = {"cost": cost}
    for decay in ev.DECAYS:
        ks = [ev.kappa_of(ev.run_pipeline(ev.gen(ev.N, ev.N, decay, s), cand)) for s in seeds]
        out[f"d{decay}"] = math.exp(sum(math.log(k) for k in ks) / len(ks))
    O = ev.run_pipeline(ev.gen(2 * ev.N, ev.N, 2.0, 0), cand)
    k2 = ev.kappa_of(O)
    rn = O.float().norm(dim=-1)
    dead = (rn < 0.1 * rn.mean()).float().mean().item() * 100
    out["slice_ok"] = bool(k2 <= 1.05 and dead == 0.0)
    out["k_r2"] = k2
    return out


if __name__ == "__main__":
    cands = json.load(open(sys.argv[1], encoding="utf-8-sig"))
    seeds = ev.HOLD_SEEDS if "--holdout" in sys.argv else ev.OPT_SEEDS
    print(f"seeds={seeds}")
    for name, cand in cands.items():
        r = score(cand, seeds)
        print(f"{name:>28} | cost {r['cost']:>2} | kappa d2 {r['d2.0']:>10.4g} | d0 {r['d0.0']:>10.4g} "
              f"| r2 {r['k_r2']:.3f} {'OK' if r['slice_ok'] else 'REGRESSION'}", flush=True)
