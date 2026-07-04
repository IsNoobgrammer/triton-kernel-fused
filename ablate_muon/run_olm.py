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
ARMS = [
    dict(arm="default", seed=0),
    dict(arm="default", seed=1),
    dict(arm="default", seed=0, wd=2.0),
    dict(arm="adamw",   seed=0),
    dict(arm="adamw",   seed=1),
    dict(arm="default", seed=0, dense_first=0),
]


def _tag(r):
    t = f"olm_{r['arm']}_s{r['seed']}_wd{r['wd'] if r['arm'] != 'adamw' else r['adamw_wd']}"
    if r.get("dense_first"):
        t += f"_df{r['dense_first']}"
    if r.get("steps", 6000) != 6000:
        t += f"_{r['steps']}st"
    return t


def _table(results):
    print("\n" + "=" * 108)
    print("ONLINE LM-EMULATOR  (one epoch, fresh data, 5% noise; gap = CE above the 0.42-nat floor)")
    print("=" * 108)
    for r in results:
        mi = "/".join(f"{m:.2f}" for m in r["mi_final"])
        pd = " ".join(f"{a:.2f}" for a in r["per_depth"])
        print(f"{_tag(r):26s} CE {r['loss']:.4f}  gap {r.get('gap', -1):+.4f}  frac {r['frac']:.3f}  "
              f"acc {r['acc']:.4f}  d1-6 {pd}  MI(L) {mi}")
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
        r = run({**COMMON, **a})
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
