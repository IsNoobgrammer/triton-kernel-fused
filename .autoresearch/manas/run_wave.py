"""Wave runner — trains arms under the FROZEN eval protocol and scores them vs base.

Usage: ../../BiBo/.venv/Scripts/python.exe run_wave.py <arm> [<arm> ...] [--heldout]
Arm spec: name=key:val,key:val  e.g.  manas_default=gamma:1.5e-3,rho:0.85
Keys: gamma, rho, rank, refresh, sign (default -1), outer (muon|adamw).
Logs cached per (arm, seed) in logs/; base arm auto-included. Scores print as JSON.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))
sys.path.insert(0, HERE)
import torch
import torch.nn.functional as F

import eval_manas as E
from exp_manas import ExpManas
from kernels.sm75.manas import ManasOptimizer

LOGS = os.path.join(HERE, "logs")
os.makedirs(LOGS, exist_ok=True)


class AccumAdamWTrainer:
    """AdamW with K-step gradient accumulation — the paper's baseline arm [C7]."""

    def __init__(self, model, K=4):
        self.opt = torch.optim.AdamW(model.parameters(), lr=E.LR, weight_decay=E.WD)
        self.K, self.t = int(K), 0
        self.opt.zero_grad(set_to_none=True)

    def train_step(self, model, x, y):
        loss = F.cross_entropy(model(x), y) / self.K
        loss.backward()
        self.t += 1
        if self.t % self.K == 0:
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
        return float(loss) * self.K


class NexusCloneTrainer:
    """Faithful Nexus (arXiv:2604.09258 Alg. 3): inner clone runs momentum-free normalized
    SGD per microbatch; at accumulation boundaries the displacement (theta - theta_inner)
    is fed to AdamW as the pseudo-gradient; inner re-syncs. Positive control [C7/A7]."""

    def __init__(self, model, K=4, gamma_in=E.LR):
        import copy
        self.inner = copy.deepcopy(model)
        self.opt = torch.optim.AdamW(model.parameters(), lr=E.LR, weight_decay=E.WD)
        self.K, self.gamma_in, self.t = int(K), float(gamma_in), 0

    def train_step(self, model, x, y):
        loss = F.cross_entropy(self.inner(x), y)
        self.inner.zero_grad(set_to_none=True)
        loss.backward()
        with torch.no_grad():
            gs = [p.grad for p in self.inner.parameters() if p.grad is not None]
            gn = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(g) for g in gs])).clamp_min(1e-12)
            for p in self.inner.parameters():
                if p.grad is not None:
                    p.sub_(p.grad, alpha=float(self.gamma_in / gn))
        self.t += 1
        if self.t % self.K == 0:
            with torch.no_grad():
                for pm, pi in zip(model.parameters(), self.inner.parameters()):
                    pm.grad = (pm - pi) / self.gamma_in    # uphill pseudo-grad ~ sum of ghat
            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            self.inner.load_state_dict(model.state_dict())
        return float(loss)


class Trainer:
    """Muon(+probe) on 2D mats, AdamW on the rest. gamma=0 => base Muon (probe inert)."""

    def __init__(self, model, gamma=0.0, rho=0.85, rank=None, refresh=200, sign=-1.0,
                 outer="muon", ref=None, norm=None, trust=None, src=None, comp=None,
                 alr=None, awd=None):
        mats, rest = E.split_params(model)
        if outer == "adamw":
            self.opt = torch.optim.AdamW(mats + rest, lr=alr or E.LR,
                                         weight_decay=E.WD if awd is None else awd)
            self.aux = None
            self.probe = None
        elif ref or norm or src or trust is not None or comp is not None or sign != -1.0:
            self.opt = ExpManas(mats, lr=E.LR, probe_gamma=gamma, probe_rho=rho,
                                probe_rank=rank, probe_refresh=refresh,
                                weight_decay=E.WD, ref_mode=ref or "theta",
                                norm_mode=norm or "global", sign=sign, trust=trust,
                                probe_src=src or "buffer", comp=comp)
            self.aux = torch.optim.AdamW(rest, lr=E.LR, weight_decay=E.WD)
            self.probe = self.opt.probe if gamma != 0 else None
        else:
            self.opt = ManasOptimizer(mats, lr=E.LR, probe_gamma=gamma, probe_rho=rho,
                                      probe_rank=rank, probe_refresh=refresh,
                                      weight_decay=E.WD)
            self.aux = torch.optim.AdamW(rest, lr=E.LR, weight_decay=E.WD)
            self.probe = self.opt.probe if gamma != 0 else None

    def train_step(self, model, x, y):
        def fwd_bwd():
            loss = F.cross_entropy(model(x), y)
            self.opt.zero_grad(set_to_none=True)
            if self.aux:
                self.aux.zero_grad(set_to_none=True)
            loss.backward()
            return loss
        if self.probe is not None:
            with self.probe():
                loss = fwd_bwd()
        else:
            loss = fwd_bwd()
        self.opt.step()
        if self.aux:
            self.aux.step()
        return float(loss)


def parse_arm(spec):
    name, _, kvs = spec.partition("=")
    cfg = {}
    for kv in filter(None, kvs.split(",")):
        k, _, v = kv.partition(":")
        cfg[k] = v if k in ("outer", "ref", "norm", "src") else (None if v == "None" else float(v))
    if "rank" in cfg and cfg["rank"] and cfg["rank"] > 1:
        cfg["rank"] = int(cfg["rank"])
    if "refresh" in cfg:
        cfg["refresh"] = int(cfg["refresh"])
    return name, cfg


def build(model, cfg):
    outer = cfg.get("outer", "muon")
    if outer == "adamw_accum":
        return AccumAdamWTrainer(model, K=cfg.get("K", 4))
    if outer == "nexus":
        return NexusCloneTrainer(model, K=cfg.get("K", 4),
                                 gamma_in=cfg.get("gamma_in", E.LR))
    return Trainer(model, **cfg)


def get_log(name, cfg, seed):
    path = os.path.join(LOGS, f"{name}_s{seed}.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    log = E.run_seed(lambda model, sd: build(model, cfg), seed)
    with open(path, "w") as f:
        json.dump(log, f)
    print(f"  ran {name} seed {seed}: best_ood_acc {max(log['ood_acc']):.4f} "
          f"min_train {min(log['train']):.4f}", flush=True)
    return log


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    seeds = E.HELDOUT_SEEDS if "--heldout" in sys.argv else E.OPT_SEEDS
    base_name, base_cfg = ("base", {})
    if "--base" in sys.argv:
        base_name, base_cfg = parse_arm(sys.argv[sys.argv.index("--base") + 1])
    arms = [parse_arm(a) for a in args]
    base_logs = {sd: get_log(base_name, base_cfg, sd) for sd in seeds}
    out = {}
    for name, cfg in arms:
        logs = {sd: get_log(name, cfg, sd) for sd in seeds}
        out[name] = E.score(logs, base_logs, seeds)
        s = out[name]
        print(f"{name}: delta_frontier {s['delta_frontier_mean']:+.4f} "
              f"(sem {s['noise_sem']:.4f})  best_acc {s['delta_best_acc_mean']:+.4f}  "
              f"train_parity {s['delta_final_train_mean']:+.4f}", flush=True)
    with open(os.path.join(HERE, "last_wave.json"), "w") as f:
        json.dump(out, f, indent=1)
