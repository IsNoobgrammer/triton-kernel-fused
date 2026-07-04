"""Grok-MoE ablation driver. Runs arms in ONE process per GPU (one CUDA context each, no
multi-proc CPU thrash), journals to results, prints the acc + grok + MI table.

Single GPU (T4 x1):
    PYTHONPATH=. python ablate_muon/run_ablation.py

Two GPUs (Kaggle T4 x2) — one process pinned per GPU, then merge:
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python ablate_muon/run_ablation.py --shard 0 --nshards 2 &
    PYTHONPATH=. CUDA_VISIBLE_DEVICES=1 python ablate_muon/run_ablation.py --shard 1 --nshards 2 &
    wait
    PYTHONPATH=. python ablate_muon/run_ablation.py --merge

Runtime: ~3000 steps/arm, a few min each on T4. x2 halves wall-clock. Edit ARMS / COMMON below.
"""
import argparse
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

# Shared config for every arm (override per-arm in ARMS). Trim steps/batch for a faster smoke.
COMMON = dict(steps=3000, batch=768, p=97, frac=0.45, wd=2.0, experts=8, top_k=2,
              bias_tokens=300_000, op_mix=(0.4, 0.3, 0.2, 0.1))

# Wave 3 (wave 2 done: AdamW fair control groks @5000 = Muon 2.4x faster; expert_wd 0.5
# REJECTED-low: slows grok AND lowers MI, only keeps dead experts alive; post-grok state
# is STABLE at MI 1.00 / 2-way split, not ongoing decay):
#  - expert_wd HIGHER direction (3.0, 4.0): experts want MORE compression than attn?
#  - cross-expert update decorrelation (idea #3): grad_e -= decor*mean_E(grad) on expert
#    stacks before the step — acts on the UPDATE (repulsion on state is rejected).
ARMS = [
    dict(arm="default", seed=0, decor=0.5),
    dict(arm="default", seed=0, decor=1.0),
    dict(arm="default", seed=1, decor=1.0),
    dict(arm="default", seed=0, expert_wd=3.0),
    dict(arm="default", seed=1, expert_wd=3.0),
    dict(arm="default", seed=0, expert_wd=4.0),
]


def _tag(r):
    t = f"{r['arm']}_s{r['seed']}" + (f"_rep{r['repulse']}" if r.get("repulse") else "")
    if r["arm"] == "adamw" and "adamw_wd" in r:
        t += f"_awd{r['adamw_wd']}"
    if r.get("expert_wd") is not None:
        t += f"_ewd{r['expert_wd']}"
    if r.get("decor"):
        t += f"_dec{r['decor']}"
    if r.get("steps", 3000) != 3000:
        t += f"_{r['steps']}st"
    return t


def _table(results):
    print("\n" + "=" * 92)
    print("GROK-MoE ABLATION  (skewed op mix, held-out acc; MI = bits of expert<->op per layer)")
    print("=" * 92)
    for r in results:
        mi = "/".join(f"{m:.2f}" for m in r["mi_final"])
        po = " ".join(f"{a:.2f}" for a in r["per_op"])
        gs = str(r["grok_step"]) if r["grok_step"] is not None else "----"
        print(f"{_tag(r):22s} acc {r['acc']:.4f}  grok {gs:>5s}  MI(L) {mi:20s}  per-op {po}")
    print("=" * 92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--merge", action="store_true", help="read all result shards -> combined table")
    args = ap.parse_args()

    if args.merge:
        results = []
        for f in sorted(glob.glob(os.path.join(HERE, "results_shard*.jsonl"))):
            results += [json.loads(l) for l in open(f) if l.strip()]
        with open(os.path.join(HERE, "results.jsonl"), "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        _table(results)
        return

    from grok_moe import run                                       # import after CUDA_VISIBLE_DEVICES
    mine = ARMS[args.shard::args.nshards]                          # this GPU's slice of the arms
    out = os.path.join(HERE, f"results_shard{args.shard}.jsonl")
    open(out, "w").close()
    results = []
    for i, a in enumerate(mine, 1):
        print(f"\n===== shard {args.shard} arm {i}/{len(mine)}: {a} =====", flush=True)
        r = run({**COMMON, **a})
        results.append(r)
        with open(out, "a") as f:
            f.write(json.dumps(r) + "\n")
    if args.nshards == 1:                                          # single-GPU: print table directly
        _table(results)
    else:
        print(f"\nshard {args.shard} done ({len(results)} arms) -> {out}; run --merge for the table",
              flush=True)


if __name__ == "__main__":
    main()
