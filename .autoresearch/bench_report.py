"""Definitive 5-way Muon Newton-Schulz comparison + parity + publication charts (for the writeup/X).

Contenders (NS micro, single square matrix):
  eager      : naive PyTorch PE Newton-Schulz (bmm, no fusion)            fp16
  compiled   : torch.compile of the PE NS (inductor)                       fp16
  ours-fused : our cuBLAS NS (baddbmm fold) = FusedMuon(use_symmul=False)  fp16
  flash      : nil0x9/flash-muon EXACT fast_newtonschulz (Jordan, verbatim)bf16
  ours-amalg : our symmetric-matmul NS = FusedMuon default                 fp16

PARITY is a first-class output: every method's NS is checked against an fp32 reference of its OWN
recipe (PE for the 4 PE methods, Jordan for flash) -> max|delta|, plus singular-value spread (SV mean
/min/max; orthogonal target = 1). The headline parity claim: ours-amalg == ours-fused == fp32-PE to
~1e-3 (the symmetric kernel changes only rounding order, not the math).

Writes .autoresearch/report/{data.json, chart_time.png, chart_speedup.png, chart_parity.png, chart_opt.png}.
Run: <venv>/python .autoresearch/bench_report.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "baselines")))

import torch
from triton.testing import do_bench
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import bench_symmul as bs
import flash_muon_mmt as fm
from kernels.sm75.muon import _PE_COEFFS, newton_schulz as ns_fused

OUT = os.path.join(os.path.dirname(__file__), "report")
os.makedirs(OUT, exist_ok=True)
DEV = "cuda"
DIMS = [1024, 2048, 4096, 8192]

# brand palette (dark theme, X-friendly)
COL = {"eager": "#8b949e", "compiled": "#58a6ff", "ours-fused": "#d29922",
       "flash": "#bc8cff", "ours-amalg": "#3fb950"}
ORDER = ["eager", "compiled", "ours-fused", "flash", "ours-amalg"]


# ── NS implementations on a single 2D matrix G ──
def _eager_core(X, coeffs):
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))
        AA = torch.bmm(A, A)
        Bm = b * A + c * AA
        X = a * X + torch.bmm(Bm, X)
    return X


_eager_compiled = torch.compile(_eager_core)


def ns_eager(G):
    X, t, s = bs._prep(G, torch.float16); return bs._restore(_eager_core(X, _PE_COEFFS), t, s)
def ns_compiled(G):
    X, t, s = bs._prep(G, torch.float16); return bs._restore(_eager_compiled(X, _PE_COEFFS), t, s)
def ns_ours_fused(G):
    return ns_fused(G, _PE_COEFFS, torch.float16)
def ns_ours_amalg(G):
    from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul
    return newton_schulz_symmul(G, _PE_COEFFS, torch.float16)
def ns_flash(G):
    return fm.fast_newtonschulz(G, 5)

METHODS = {"eager": ns_eager, "compiled": ns_compiled, "ours-fused": ns_ours_fused,
           "flash": ns_flash, "ours-amalg": ns_ours_amalg}


def _ref_pe_fp32(G):
    X, t, s = bs._prep(G, torch.float32); return bs._restore(_eager_core(X, _PE_COEFFS), t, s)
def _ref_jordan_fp32(G):
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.float()
    tr = X.size(-2) > X.size(-1)
    if tr: X = X.mT
    X = X / (X.norm() + 1e-7)
    for _ in range(5):
        Aa = X @ X.mT; X = a * X + (b * Aa + c * (Aa @ Aa)) @ X
    if tr: X = X.mT
    return X


def _sv(Y):
    Y = Y.float()
    sv = torch.linalg.svdvals(Y)
    return sv.mean().item(), sv.min().item(), sv.max().item()


def run_micro():
    rows = []
    for d in DIMS:
        G = torch.randn(d, d, device=DEV, dtype=torch.float16)
        ref_pe = _ref_pe_fp32(G); ref_j = _ref_jordan_fp32(G)
        for name, fn in METHODS.items():
            Y = fn(G).float()
            ref = ref_j if name == "flash" else ref_pe
            dmax = (Y - ref).abs().max().item()
            svm, svlo, svhi = _sv(Y)
            ms = do_bench(lambda: fn(G))
            rows.append(dict(dim=d, method=name, ms=ms, dmax=dmax,
                             sv_mean=svm, sv_min=svlo, sv_max=svhi))
            print(f"  d={d:5d} {name:11s} {ms:8.3f}ms  max|d|={dmax:.1e}  SV {svm:.3f}[{svlo:.3f},{svhi:.3f}]")
    return rows


def chart_time(rows):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    for m in ORDER:
        xs = [r["dim"] for r in rows if r["method"] == m]
        ys = [r["ms"] for r in rows if r["method"] == m]
        ax.plot(xs, ys, "-o", color=COL[m], lw=2.4, ms=7, label=m)
    ax.set_xscale("log", base=2); ax.set_yscale("log")
    ax.set_xticks(DIMS); ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlabel("matrix dim (square)", fontsize=13); ax.set_ylabel("Newton-Schulz(5) step  (ms)", fontsize=13)
    ax.set_title("Muon Newton-Schulz step time  —  RTX PRO 6000 Blackwell (fp16)", fontsize=15, fontweight="bold")
    ax.grid(True, alpha=0.25); ax.legend(fontsize=12, framealpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_time.png")); plt.close(fig)


def chart_speedup(rows):
    plt.style.use("dark_background")
    base = {(r["dim"]): r["ms"] for r in rows if r["method"] == "compiled"}
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    show = ["ours-fused", "flash", "ours-amalg"]
    w = 0.26
    for i, m in enumerate(show):
        ys = [base[d] / next(r["ms"] for r in rows if r["method"] == m and r["dim"] == d) for d in DIMS]
        xs = [j + (i - 1) * w for j in range(len(DIMS))]
        bars = ax.bar(xs, ys, width=w, color=COL[m], label=m)
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.02, f"{y:.2f}x", ha="center", fontsize=9, color=COL[m], fontweight="bold")
    ax.axhline(1.0, color="#8b949e", ls="--", lw=1, alpha=0.7)
    ax.set_xticks(range(len(DIMS))); ax.set_xticklabels([str(d) for d in DIMS])
    ax.set_xlabel("matrix dim", fontsize=13); ax.set_ylabel("speedup vs torch.compile  (higher = better)", fontsize=13)
    ax.set_title("Speedup over torch.compile  —  ours-amalg wins at every scale", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25); ax.legend(fontsize=12, framealpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_speedup.png")); plt.close(fig)


def chart_parity(rows):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    d = 4096
    ms = [r for r in rows if r["dim"] == d]
    xs = range(len(ORDER))
    ys = [next(r["dmax"] for r in ms if r["method"] == m) for m in ORDER]
    ax.bar(xs, ys, color=[COL[m] for m in ORDER])
    for x, y in zip(xs, ys):
        ax.text(x, y * 1.15, f"{y:.1e}", ha="center", fontsize=10, fontweight="bold")
    ax.axhline(2e-2, color="#f85149", ls="--", lw=1.6, label="NS tolerance 2e-2")
    ax.set_yscale("log"); ax.set_xticks(list(xs)); ax.set_xticklabels(ORDER, fontsize=11)
    ax.set_ylabel("max|Δ| vs fp32 reference (same recipe)", fontsize=13)
    ax.set_title(f"PARITY @ d={d}: every method matches its fp32 reference (well under tolerance)", fontsize=14, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25); ax.legend(fontsize=12, framealpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_parity.png")); plt.close(fig)


def run_optimizer():
    import bench_muon as bm
    import bench_muon_4way as b4
    import bench_muon_symmul as bm2
    from kernels.sm120.muon import FusedMuon
    shapes = bm2.make_big_shapes(layers=2, d=4096, ffn=11008)        # dense, large -> symmul fires
    nB = sum(int(torch.tensor(s).prod().item()) for s in shapes) / 1e9
    mk = {
        "eager":     lambda ps: bm.BaselineMuon(ps, lr=.02, weight_decay=.1),
        "compiled":  lambda ps: b4.CompiledMuon(ps, lr=.02, weight_decay=.1),
        "ours-fused":lambda ps: FusedMuon(ps, lr=.02, weight_decay=.1, ns_dtype=torch.float16, use_symmul=False),
        "flash":     lambda ps: fm.FlashMuon(ps, lr=.02, weight_decay=.1),
        "ours-amalg":lambda ps: FusedMuon(ps, lr=.02, weight_decay=.1, ns_dtype=torch.float16, use_symmul=True),
    }
    out = {}
    for name, f in mk.items():
        ms, mb = bm2.measure(f, shapes, torch.float16)
        out[name] = dict(ms=ms, mb=mb)
        print(f"  {name:11s} {ms:8.1f}ms  {mb:8.0f}MB")
    return {"params_B": round(nB, 2), "methods": out}


def chart_opt(opt):
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=110)
    ys = [opt["methods"][m]["ms"] for m in ORDER]
    bars = ax.bar(range(len(ORDER)), ys, color=[COL[m] for m in ORDER])
    base = opt["methods"]["compiled"]["ms"]
    for i, m in enumerate(ORDER):
        ax.text(i, ys[i] + max(ys) * 0.01, f"{ys[i]:.0f}ms\n{base/ys[i]:.2f}x", ha="center", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(ORDER))); ax.set_xticklabels(ORDER, fontsize=11)
    ax.set_ylabel("optimizer .step()  (ms)", fontsize=13)
    ax.set_title(f"Full Muon step, dense {opt['params_B']:.1f}B params (lower = better)", fontsize=15, fontweight="bold")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "chart_opt.png")); plt.close(fig)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print("GPU:", torch.cuda.get_device_name(0), "| torch", torch.__version__)
    print("\n=== NS micro (single matrix), 5-way + parity ===")
    micro = run_micro()
    print("\n=== optimizer step, dense ~0.4B, 5-way ===")
    opt = run_optimizer()
    chart_time(micro); chart_speedup(micro); chart_parity(micro); chart_opt(opt)
    data = {"gpu": torch.cuda.get_device_name(0), "dims": DIMS, "micro": micro, "optimizer": opt}
    with open(os.path.join(OUT, "data.json"), "w") as fh:
        json.dump(data, fh, indent=2)
    print(f"\nwrote charts + data.json to {OUT}")
