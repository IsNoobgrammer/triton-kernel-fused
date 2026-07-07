"""ManasOptimizer toy validation — heterogeneous MNIST, conv stem + GLU trunk.

Question set (from the rolling-probe brainstorm):
  * does the alignment signal DO anything (loss/acc curves, cross-source grad cosine)?
  * claim 1 STALENESS: rho=0.85 window ~7 is right; long memory (0.99) is extra error not signal.
  * claim 2 PURITY: momentum-contaminated probe (violates Nexus's momentum-free-inner) is worse.
  * claim 3 OUTER: Muon outer digests the probe gradient at least as well as AdamW outer.
  * low-rank richness: how much of the probe signal does a rank-8 sketch keep on REAL gradients?

Setup: two "sources" = clean vs inverted (1-x) MNIST, batches alternate (pretraining-mixture toy).
Model: Conv2d(1,8,3,s2) stem [AdamW — muon is never for convs] -> proj -> 2x SwiGLU block -> head;
all 2D matrices on ManasOptimizer (aurora-K1, NS-8). Eval: loss/acc per source at theta (probe off),
cross-source grad cosine at theta, ||d||; the manas run also carries a SHADOW rank-8 sketch updated
with the same gradients to measure cos(Q C, d_full) and per-step capture ||Q^T g||/||g||.

Run: ../BiBo/.venv/Scripts/python.exe .autoresearch/manas_mnist.py
Outputs: manas_mnist_results.json + manas_mnist.png (this dir).
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

from kernels.sm75.manas import ManasOptimizer

DEV = "cuda"
STEPS, BS, EVAL_EVERY = 500, 128, 25
HERE = os.path.dirname(os.path.abspath(__file__))


# ---------------- data: two heterogeneous sources ----------------
def load_data():
    root = os.path.join(HERE, "data")
    tf = transforms.ToTensor()
    tr = datasets.MNIST(root, train=True, download=True, transform=tf)
    x = torch.stack([tr[i][0] for i in range(14000)]).to(DEV)
    y = torch.tensor([tr[i][1] for i in range(14000)], device=DEV)
    xt, yt = x[:12000], y[:12000]
    xv, yv = x[12000:], y[12000:]
    return xt, yt, xv, yv


def src(x, s):                       # source 0 = clean, source 1 = inverted
    return x if s == 0 else 1.0 - x


# ---------------- model: conv stem + GLU trunk ----------------
class GLUBlock(nn.Module):
    def __init__(self, h, i):
        super().__init__()
        self.gate = nn.Linear(h, i, bias=False)
        self.up = nn.Linear(h, i, bias=False)
        self.down = nn.Linear(i, h, bias=False)

    def forward(self, x):
        return x + self.down(F.silu(self.gate(x)) * self.up(x))


class ToyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Conv2d(1, 8, 3, stride=2)             # 28 -> 13; AdamW (4D, not for muon)
        self.proj = nn.Linear(8 * 13 * 13, 256, bias=False)
        self.blocks = nn.Sequential(GLUBlock(256, 512), GLUBlock(256, 512))
        self.head = nn.Linear(256, 10, bias=False)

    def forward(self, x):
        z = F.silu(self.stem(x)).flatten(1)
        return self.head(self.blocks(self.proj(z)))


def split_params(m):
    mats = [p for p in m.parameters() if p.ndim == 2]
    rest = [p for p in m.parameters() if p.ndim != 2]
    return mats, rest


# ---------------- claim-2 variant: momentum-contaminated probe ----------------
class ImpureManas(ManasOptimizer):
    """Deliberately VIOLATES probe purity: d accumulates a momentumized (beta 0.9) direction."""

    @torch.no_grad()
    def _update_probe(self):
        ps = [p for p in self._probe_params() if p.grad is not None]
        if not ps or self.probe_gamma == 0.0:
            return
        ms = []
        for p in ps:
            st = self.state[p]
            if "impure_m" not in st:
                st["impure_m"] = torch.zeros_like(p, dtype=torch.float32)
            st["impure_m"].mul_(0.9).add_(torch.nan_to_num(p.grad).to(torch.float32))
            ms.append(st["impure_m"])
        gn = torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(m) for m in ms]))
        inv = self.probe_gamma / gn
        inv = torch.where(torch.isfinite(inv) & (gn > 0), inv, torch.zeros_like(inv))
        for p, m in zip(ps, ms):
            d = self._full_d(p)
            d.mul_(self.probe_rho)
            d.addcmul_(m, inv, value=-1.0)


# ---------------- one training run ----------------
def run(name, seed=0, gamma=0.0, rho=0.85, rank=None, outer="muon", impure=False,
        shadow_rank=None):
    torch.manual_seed(seed)
    xt, yt, xv, yv = DATA
    model = ToyNet().to(DEV)
    mats, rest = split_params(model)
    cls = ImpureManas if impure else ManasOptimizer
    if outer == "muon":
        opt = cls(mats, lr=1e-3, probe_gamma=gamma, probe_rho=rho, probe_rank=rank,
                  weight_decay=0.01)
        aux = torch.optim.AdamW(rest, lr=1e-3, weight_decay=0.01)
        opts = [opt, aux]
    else:                                                     # AdamW outer + probe-only Manas (lr=0)
        opt = cls(mats, lr=0.0, probe_gamma=gamma, probe_rho=rho, probe_rank=rank,
                  weight_decay=0.0)
        aux = torch.optim.AdamW(mats + rest, lr=1e-3, weight_decay=0.01)
        opts = [aux, opt]                                     # AdamW moves params; Manas only moves d

    # fixed per-source eval batches (cosine + curves measured at theta, probe OFF)
    ev = {s: (src(xv[:512], s), yv[:512]) for s in (0, 1)}

    # shadow rank sketch riding the FULL-d run (signal-richness measurement)
    shadow = None
    if shadow_rank:
        shadow = {id(p): [None, None] for p in mats}          # per param: [Q, C]

    log = {"step": [], "loss": [], "acc0": [], "acc1": [], "cos": [], "dnorm": [],
           "cap": [], "dcos": []}
    g = torch.Generator(device="cpu").manual_seed(seed)
    for t in range(STEPS):
        s = t % 2
        idx = torch.randint(0, xt.shape[0], (BS,), generator=g).to(DEV)
        xb, yb = src(xt[idx], s), yt[idx]
        with opt.probe():
            loss = F.cross_entropy(model(xb), yb)
            for o in opts:
                o.zero_grad(set_to_none=True)
            loss.backward()
        if shadow is not None:                                # shadow sketch fed the same grads
            cap_num = cap_den = 0.0
            for p in mats:
                q, c = shadow[id(p)]
                gf = p.grad.to(torch.float32)
                if q is None or t % 100 == 0:
                    om = torch.randn(p.shape[1], shadow_rank, device=DEV)
                    qn = torch.linalg.qr(gf @ om)[0]
                    c = (qn.mT @ q) @ c if q is not None else torch.zeros(
                        shadow_rank, p.shape[1], device=DEV)
                    q = qn
                gn_ = torch.linalg.vector_norm(gf).clamp_min(1e-12)
                cap_num += torch.linalg.vector_norm(q.mT @ gf).square().item()
                cap_den += gn_.square().item()
                c.mul_(rho)
                c.add_((q.mT @ gf) / gn_, alpha=-gamma)       # same update rule as low-rank mode
                shadow[id(p)] = [q, c]
            log.setdefault("_cap_t", []).append(cap_num / max(cap_den, 1e-12))
        for o in opts:
            o.step()

        if t % EVAL_EVERY == 0 or t == STEPS - 1:
            with torch.no_grad():
                accs, losses = [], []
                for si in (0, 1):
                    logits = model(ev[si][0])
                    losses.append(F.cross_entropy(logits, ev[si][1]).item())
                    accs.append((logits.argmax(-1) == ev[si][1]).float().mean().item())
            gs = []
            for si in (0, 1):                                 # cross-source grad cosine at theta
                for o in opts:
                    o.zero_grad(set_to_none=True)
                F.cross_entropy(model(ev[si][0][:256]), ev[si][1][:256]).backward()
                gs.append(torch.cat([p.grad.reshape(-1) for p in mats]))
            cs = F.cosine_similarity(gs[0], gs[1], dim=0).item()
            for o in opts:
                o.zero_grad(set_to_none=True)
            dn = torch.cat([opt._d_of(p).reshape(-1) for p in mats]).norm().item()
            dcos = float("nan")
            if shadow is not None:
                df = torch.cat([opt._d_of(p).reshape(-1) for p in mats])
                dl = torch.cat([(shadow[id(p)][0] @ shadow[id(p)][1]).reshape(-1) for p in mats])
                dcos = F.cosine_similarity(df, dl, dim=0).item()
            log["step"].append(t); log["loss"].append(sum(losses) / 2)
            log["acc0"].append(accs[0]); log["acc1"].append(accs[1])
            log["cos"].append(cs); log["dnorm"].append(dn); log["dcos"].append(dcos)
            log["cap"].append(sum(log.get("_cap_t", [0.0])[-EVAL_EVERY:]) /
                              max(len(log.get("_cap_t", [0.0])[-EVAL_EVERY:]), 1))
    log.pop("_cap_t", None)
    fin = {"acc": (log["acc0"][-1] + log["acc1"][-1]) / 2, "loss": log["loss"][-1],
           "cos": log["cos"][-1]}
    print(f"  {name:<22} acc {fin['acc']:.4f}  loss {fin['loss']:.4f}  xsrc-cos {fin['cos']:+.3f}")
    return {"name": name, "log": log, "final": fin}


if __name__ == "__main__":
    DATA = load_data()
    print(f"arms x {STEPS} steps (bs {BS}), sources = clean vs inverted MNIST")
    R = []
    for seed in (0, 1):
        R.append(run(f"muon_base_s{seed}", seed))
        R.append(run(f"manas_g1e-3_s{seed}", seed, gamma=1e-3, shadow_rank=8 if seed == 0 else None))
    R.append(run("manas_g3e-3", 0, gamma=3e-3))
    R.append(run("stale_rho.99", 0, gamma=1e-3, rho=0.99))     # claim 1: long memory hurts
    R.append(run("short_rho.5", 0, gamma=1e-3, rho=0.5))       # claim 1: window too short = weaker
    R.append(run("impure_probe", 0, gamma=1e-3, impure=True))  # claim 2: purity violation
    R.append(run("adamw_base", 0, outer="adamw"))              # claim 3: outer comparison
    R.append(run("adamw_probe", 0, gamma=1e-3, outer="adamw"))
    R.append(run("lowrank_r8", 0, gamma=1e-3, rank=8))         # low-rank end-to-end

    with open(os.path.join(HERE, "manas_mnist_results.json"), "w") as f:
        json.dump(R, f)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    keyarms = ["muon_base_s0", "manas_g1e-3_s0", "manas_g3e-3", "stale_rho.99",
               "impure_probe", "lowrank_r8", "adamw_base", "adamw_probe"]
    for r in R:
        if r["name"] not in keyarms:
            continue
        lg = r["log"]
        acc = [(a + b) / 2 for a, b in zip(lg["acc0"], lg["acc1"])]
        ax[0][0].plot(lg["step"], acc, label=r["name"])
        ax[0][1].plot(lg["step"], lg["loss"], label=r["name"])
        ax[1][0].plot(lg["step"], lg["cos"], label=r["name"])
        ax[1][1].plot(lg["step"], lg["dnorm"], label=r["name"])
    for a, t in zip(ax.flat, ["val acc (mean of sources)", "val loss", "cross-source grad cosine",
                              "||d||"]):
        a.set_title(t); a.legend(fontsize=7); a.set_xlabel("step")
    sh = next(r for r in R if r["name"] == "manas_g1e-3_s0")["log"]
    print(f"\nlow-rank richness (rank-8 shadow on real grads): per-step grad capture "
          f"mean {sum(sh['cap'])/len(sh['cap']):.3f}, cos(d_r8, d_full) "
          f"mean {sum(c for c in sh['dcos'] if c == c)/max(sum(1 for c in sh['dcos'] if c == c),1):.3f}")
    fig.tight_layout()
    fig.savefig(os.path.join(HERE, "manas_mnist.png"), dpi=120)
    print("saved manas_mnist_results.json + manas_mnist.png")
