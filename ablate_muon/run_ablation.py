"""Grok-MoE ablation driver — runs all arms SEQUENTIALLY in one process (one CUDA context,
no multi-proc CPU thrash), journals to results.jsonl, prints the final acc + grok + MI table.

T4 (or any CUDA GPU):
    PYTHONPATH=. python ablate_muon/run_ablation.py
Runtime: ~6 arms x 3000 steps, a few minutes each on T4. Live per-eval prints; safe to leave.

Edit ARMS / COMMON below to change the wave.
"""
import json
import os

from grok_moe import run

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results.jsonl")

# Shared config for every arm (override per-arm in ARMS). Trim steps/batch for a faster smoke.
COMMON = dict(steps=3000, batch=768, p=97, frac=0.45, wd=2.0, experts=8, top_k=2,
              bias_tokens=300_000, op_mix=(0.4, 0.3, 0.2, 0.1))

# Wave 1: baseline (Muon vs AdamW, 2 seeds each) + expert-repulsion sweep.
ARMS = [
    dict(arm="default", seed=0),
    dict(arm="default", seed=1),
    dict(arm="adamw",   seed=0),
    dict(arm="adamw",   seed=1),
    dict(arm="default", seed=0, repulse=1e-3),
    dict(arm="default", seed=0, repulse=1e-2),
]


def main():
    open(OUT, "w").close()                                         # fresh journal
    results = []
    for i, a in enumerate(ARMS, 1):
        cfg = {**COMMON, **a}
        print(f"\n===== arm {i}/{len(ARMS)}: {a} =====", flush=True)
        r = run(cfg)
        results.append(r)
        with open(OUT, "a") as f:
            f.write(json.dumps(r) + "\n")

    # ── final table ──
    def row(r):
        tag = f"{r['arm']}_s{r['seed']}" + (f"_rep{r['repulse']}" if r["repulse"] else "")
        mi = "/".join(f"{m:.2f}" for m in r["mi_final"])
        po = " ".join(f"{a:.2f}" for a in r["per_op"])
        gs = str(r["grok_step"]) if r["grok_step"] is not None else "----"
        return f"{tag:22s} acc {r['acc']:.4f}  grok {gs:>5s}  MI(L) {mi:20s}  per-op {po}"

    print("\n" + "=" * 92)
    print("GROK-MoE ABLATION  (skewed op mix, held-out acc; MI = bits of expert<->op per layer)")
    print("=" * 92)
    for r in results:
        print(row(r))
    print("=" * 92)
    print(f"journal: {OUT}")


if __name__ == "__main__":
    main()
