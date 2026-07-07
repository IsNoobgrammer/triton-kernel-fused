"""FROZEN EVAL — kappa at r=1 per NS-iteration budget. Do not modify to improve numbers.

Candidate = JSON file: {
  "prescale": "none" | "rownorm" | "sink2",         # O(n^2) pre-op before the FIRST polar (free)
  "interstage": "rownorm" | "none",                 # O(n^2) op between polars (aurora-style; free)
  "stages": [[[a,b,c], ...], ...]                   # one coeff list PER POLAR; cost = total tuples
}
Pipeline (fp16 NS, matching prod): M -> prescale -> polar(stage1) -> {interstage -> polar(stageK)}...
Usage: python eval_kappa.py cand.json [--holdout]
Prints per-case kappa, aggregate (geomean), cost, and the r=2 standing slice.
"""
import json
import math
import sys

import torch

sys.path.insert(0, r"C:\Users\shaur\OneDrive\Documents\triton-kernel-fused")

dev = "cuda"
N = 2048
OPT_SEEDS = [0, 1, 2]
HOLD_SEEDS = [10, 11, 12]
DECAYS = [0.0, 2.0]


def gen(m, n, decay, seed):
    g = torch.Generator(device=dev).manual_seed(seed)
    rs = torch.logspace(0, -decay, m, device=dev, dtype=torch.float32).view(m, 1)
    return (rs * torch.randn(m, n, device=dev, dtype=torch.float32, generator=g)).half()


def polar(M, coeffs):
    X = M.unsqueeze(0)
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(1e-7).view(-1, 1, 1)
    X = X.half() / nrm.half()
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    return X[0]


def _one_prescale(X, kind):
    if kind == "rownorm":
        return X / X.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    if kind == "sink2":
        m, n = X.shape
        for _ in range(2):
            X = X * (math.sqrt(n) / X.norm(dim=-1, keepdim=True).clamp_min(1e-8))
            X = X * (math.sqrt(m) / X.norm(dim=-2, keepdim=True).clamp_min(1e-8))
        return X
    if kind.startswith("dither:"):        # + eps * Gaussian (deterministic dither, seed 999)
        eps = float(kind.split(":")[1])
        g = torch.Generator(device=dev).manual_seed(999)
        E = torch.randn(*X.shape, device=dev, generator=g)
        m, n = X.shape
        smax = 2.0 * X.norm() / math.sqrt(min(m, n))     # ~spectral norm of a Gaussian-ish matrix
        return X + (eps * smax / (2.0 * E.norm() / math.sqrt(min(m, n)))) * E
    if kind.startswith("sperm:"):          # + eps * signed permutation (exactly orthogonal, O(n))
        eps = float(kind.split(":")[1])
        m, n = X.shape
        k = min(m, n)
        g = torch.Generator(device=dev).manual_seed(999)
        r_idx = torch.randperm(m, device=dev, generator=g)[:k]
        c_idx = torch.randperm(n, device=dev, generator=g)[:k]
        sgn = (torch.randint(0, 2, (k,), device=dev, generator=g) * 2 - 1).float()
        smax = 2.0 * X.norm() / math.sqrt(k)
        E = torch.zeros_like(X)
        E[r_idx, c_idx] = sgn                              # exactly orthogonal on its k x k support
        return X + eps * smax * E
    if kind == "none":
        return X
    raise ValueError(kind)


def prescale(M, kind):
    X = M.float()
    for k in kind.split("+"):
        X = _one_prescale(X, k)
    return X.half()


def run_pipeline(M, cand):
    X = prescale(M, cand.get("prescale", "none"))
    stages = cand["stages"]
    for i, coeffs in enumerate(stages):
        if i > 0 and cand.get("interstage", "rownorm") == "rownorm":
            Xf = X.float()
            X = (Xf / Xf.norm(dim=-1, keepdim=True).clamp_min(1e-8)).half()
        X = polar(X, coeffs)
    return X


def kappa_of(O):
    A = O if O.size(0) >= O.size(1) else O.T
    G = A.double().T @ A.double()
    w = torch.linalg.eigvalsh(G).clamp_min(0).sqrt().clamp_min(1e-30)
    return (w[-1] / w[0]).item()


def main():
    cand = json.load(open(sys.argv[1]))
    seeds = HOLD_SEEDS if "--holdout" in sys.argv else OPT_SEEDS
    cost = sum(len(s) for s in cand["stages"])
    print(f"cost = {cost} NS iters | prescale={cand.get('prescale','none')} "
          f"interstage={cand.get('interstage','rownorm')} stages={[len(s) for s in cand['stages']]} "
          f"| seeds={seeds}")
    logs = []
    for decay in DECAYS:
        ks = []
        for s in seeds:
            k = kappa_of(run_pipeline(gen(N, N, decay, s), cand))
            ks.append(k)
        gm = math.exp(sum(math.log(k) for k in ks) / len(ks))
        logs.append((decay, gm))
        print(f"  r=1 decay={decay}: kappa {' '.join(f'{k:.3g}' for k in ks)}  geomean {gm:.4g}")
    # standing slice: r=2 decay=2 seed 0 through the same pipeline
    O = run_pipeline(gen(2 * N, N, 2.0, 0), cand)
    k2 = kappa_of(O)
    rn = O.float().norm(dim=-1)
    dead = (rn < 0.1 * rn.mean()).float().mean().item() * 100
    ok = k2 <= 1.05 and dead == 0.0
    print(f"  slice r=2: kappa {k2:.4g} dead {dead:.1f}%  {'OK' if ok else 'REGRESSION'}")
    score = logs[1][1]  # primary: decay=2 geomean
    print(f"SCORE kappa_r1_decay2_geomean = {score:.4g}  (cost {cost})")


if __name__ == "__main__":
    main()
