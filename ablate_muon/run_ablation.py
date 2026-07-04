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

# Wave 5 — ledger backlog, 6 new mechanisms + user-requested combos (baseline band
# 1800-2200, bar: grok >=400 earlier at acc parity, or clear MI/structure win):
#  - micro-repulsion 1e-4/1e-5 (user: aux-loss regime; compounding drift only ~1.35x)
#  - grad-space repulsion (amplify each expert's deviation, inverse of decor)
#  - xorth: cross-expert grad whitening along E (E x E gram inv-sqrt) — Muon-native, ours
#  - niche: fitness-sharing lr (expert grads scaled by inverse recent load)
#  - scap: sigma-cap as wd SUBSTITUTE (wd 0.1 + clip top singular value @2.0 post-step)
#  - cautious: sign-masked wd 2.0 (decay only where the step already shrinks |w|)
#  - combos: rep+gf, rep+niche+gf (user ask)
ARMS = [
    dict(arm="default", seed=0, repulse=1e-4),
    dict(arm="default", seed=0, repulse=1e-5),
    dict(arm="default", seed=0, grad_rep=0.5),
    dict(arm="default", seed=0, xorth=1),
    dict(arm="default", seed=1, xorth=1),
    dict(arm="default", seed=0, niche=0.5),
    dict(arm="default", seed=0, scap=2.0, wd=0.1),
    dict(arm="default", seed=0, cautious=2.0),
    dict(arm="default", seed=0, repulse=1e-4, grokfast=2.0),
    dict(arm="default", seed=0, repulse=1e-4, niche=0.5, grokfast=2.0),
]


def _tag(r):
    t = f"{r['arm']}_s{r['seed']}" + (f"_rep{r['repulse']}" if r.get("repulse") else "")
    if r["arm"] == "adamw" and "adamw_wd" in r:
        t += f"_awd{r['adamw_wd']}"
    if r.get("expert_wd") is not None:
        t += f"_ewd{r['expert_wd']}"
    if r.get("decor"):
        t += f"_dec{r['decor']}"
    if r.get("grokfast"):
        t += f"_gf{r['grokfast']}"
    if r.get("lookahead"):
        t += f"_la{r['lookahead']}"
    if r.get("grad_rep"):
        t += f"_gr{r['grad_rep']}"
    if r.get("xorth"):
        t += "_xo"
    if r.get("niche"):
        t += f"_ni{r['niche']}"
    if r.get("scap"):
        t += f"_sc{r['scap']}"
    if r.get("cautious"):
        t += f"_cw{r['cautious']}"
    if r.get("wd", 2.0) != 2.0:
        t += f"_wd{r['wd']}"
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
