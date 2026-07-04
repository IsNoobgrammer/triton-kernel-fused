"""Online LM-emulator ablation driver (one-epoch regime; see olm.py for the task).

Same shard/merge pattern as run_ablation.py:
    bash ablate_muon/run.sh olm            # auto dual-GPU
    PYTHONPATH=. python ablate_muon/run_olm.py [--shard N --nshards K | --merge]
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

COMMON = dict(steps=6000, batch=768)

# OLM wave 1: calibration + baselines. Default arch = dense first layer, MoE layers 2-4
# (user call); depth 1-6 Zipf mix; 5% label noise -> CE floor 0.42 nats = frac 0.092,
# matching LM's residual ~0.09 (the race is toward the floor, never to zero, like text).
# Regime predictions to test:
#  - wd optimum flips small online (grok-optimal 2.0 should HURT -> regime check)
#  - Muon vs AdamW gap in the compute-bound regime (the gap that matters for BiBo)
#  - dense-first-1 (default) vs all-MoE contrast
# v5 wave: RE-BENCH the mechanism backlog on the VALIDATED proxy (default is now ns8).
# grok (wrong regime) called these harmful/null; olm is the LM-correct screen. Noise
# floor from 2 default seeds (v4: default seeds spanned frac 0.566-0.592). A mechanism
# must clear that spread to count. Survivors -> 2-seed confirm (v6) -> real LM.
#   cautious = LM-predicted-good (compression without signal tax); scap = wd substitute
#   (smax logged now); repulse/grad_rep/xorth/grokfast = grok-null, regime re-test.
# v8 wave: DEFAULT NS-CONFIG comparison, MULTI-SEED (v7 killed normuon/xorth as broken-bias
# artifacts; plain Muon+ns8 is the frontier). Now tune its NS budget on the validated proxy.
# 3 configs x 2 seeds (compute-limited); compare MEANS vs the ~0.084 seed spread. Does more
# NS fidelity (10 iter) or more aurora passes (k2) beat the cheap ns8 (8 iter), or is the
# iter/coeff axis dead here too (-> ns8 confirmed cheapest-tied = perf-per-flop win)?
#   aurora_k1 8 iter (ns_kj=6) = current default | aurora_k1 10 iter (ns_kj=8) = dsv4_10 |
#   aurora_k2 8 iter (ns_kj=6, k2)
# v9 wave: coefficient-family / cheaper-iters exploration (default ns8=8-iter floor known:
# s0 0.535 / s1 0.451). NEW configs only, 2 seeds each -> compare MEANS to that floor.
#   ns_4+2  = KJ*4 + 2 pin = 6 iters (even cheaper than the 8-iter default - does it hold?)
#   PE-8    = Polar-Express minimax 8-iter schedule (different coeff FAMILY) + aurora
# v10 wave: WEIGHT-DECAY sweep - the dominant knob from the whole program (7x generalization
# speed on grok) that we NEVER swept on olm (only spot-checked: 0.1 works, 2.0 dead=regime
# check). Fix the established winners (aurora_k1, 8-iter KJ) and vary ONLY muon wd. 2 seeds
# each, rank on AUC (noise-robust). wd=0.1 NOT re-run (known: s0 0.535 / s1 0.451 = anchor).
# Optimum likely BELOW 0.1 since 2.0 is already dead -> probe 0.01/0.03/0.05 + one above (0.2).
# v14 wave: LOW-SIDE wd sweep on the RTX 6000 - map the LEFT of the peak device-matched.
# v12 (0.3-1.0) turns over above 0.2; probe below with 0.007/0.01/0.02/0.05 (0.02 & 0.007 are
# NEW; 0.01/0.05 device-matched vs their T4 numbers). 2 seeds each, aurora_k1 8-iter KJ.
# Full device-consistent curve = v14 (0.007-0.05) + v13 (0.1/0.2) + v12 (0.3-1.0).
ARMS = [
    dict(arm="default", seed=s, wd=w)
    for w in (0.007, 0.01, 0.02, 0.05)
    for s in (0, 1)
]


def _tag(r):
    t = f"olm_{r['arm']}_s{r['seed']}_wd{r['wd'] if r['arm'] != 'adamw' else r['adamw_wd']}"
    if r.get("dense_first"):
        t += f"_df{r['dense_first']}"
    if r.get("warmup", 500) != 500:
        t += f"_wu{r['warmup']}"
    if r["arm"] == "default":
        if r.get("scale_mode", "aurora") != "aurora":
            t += f"_{r['scale_mode']}"
        if r.get("aurora_k", 1) != 1:
            t += f"_k{r['aurora_k']}"
        if r.get("coeffs", "kj") == "pe":
            t += "_pe8"
        elif r.get("ns_kj", 6) != 6:
            t += f"_it{r['ns_kj'] + 2}"                           # total NS iters
    for key, pre in (("repulse", "rep"), ("decor", "dec"), ("grad_rep", "gr"),
                     ("niche", "ni"), ("scap", "sc"), ("cautious", "cw"),
                     ("grokfast", "gf"), ("lookahead", "la")):
        if r.get(key):
            t += f"_{pre}{r[key]}"
    if r.get("xorth"):
        t += "_xo"
    if r.get("mult", 4) != 4:
        t += f"_m{r['mult']}"
    if r.get("steps", 6000) != 6000:
        t += f"_{r['steps']}st"
    return t


def _table(results):
    print("\n" + "=" * 108)
    print("ONLINE LM-EMULATOR  (one epoch, fresh data, 5% noise; gap = CE above the 0.42-nat floor)")
    print("=" * 108)
    for r in results:
        spec = "/".join(f"{s:.2f}" for s in r.get("spec_frac", r.get("mi_final", [])))
        eff = "/".join(f"{e:.1f}" for e in r.get("eff_experts", []))
        print(f"{_tag(r):30s} frac {r['frac']:.3f}  acc {r['acc']:.4f}  "
              f"spec {spec:14s}  eff/{r.get('experts', 8)} {eff}")
    print("=" * 108)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()

    if args.merge:
        results = []
        for f in sorted(glob.glob(os.path.join(HERE, "results_olm_shard*.jsonl"))):
            results += [json.loads(l) for l in open(f) if l.strip()]
        with open(os.path.join(HERE, "results_olm.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        _table(results)
        return

    from olm import run
    mine = ARMS[args.shard::args.nshards]
    out = os.path.join(HERE, f"results_olm_shard{args.shard}.jsonl")
    open(out, "w").close()
    results = []
    for i, a in enumerate(mine, 1):
        print(f"\n===== shard {args.shard} arm {i}/{len(mine)}: {a} =====", flush=True)
        try:
            r = run({**COMMON, **a})
        except Exception as e:                                     # one bad arm must not kill the shard
            import traceback
            print(f"[shard {args.shard}] arm {a} FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        results.append(r)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
    if args.nshards == 1:
        _table(results)
    else:
        print(f"\nshard {args.shard} done ({len(results)} arms) -> {out}; run --merge for the table",
              flush=True)


if __name__ == "__main__":
    main()
