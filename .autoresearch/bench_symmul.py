"""FROZEN EVAL — symmul Newton-Schulz, 4-way, AS WE SCALE (regime B). Do NOT edit to move numbers.

Holds the Polar-Express NS ALGORITHM fixed (same _PE_COEFFS, same normalize/orient, same non-
symmetric `B X` via cuBLAS) across four contenders and varies ONLY the two symmetric GEMMs'
implementation. That isolates the symmetric FLOP cut (the lever) and the batching dimension:

  compiled : PE-NS, cuBLAS bmm/baddbmm, torch.compile          (baseline 1 — inductor)
  triu     : PE-NS, flash-muon single-matrix matmul_transpose  (baseline 2 — symmetric, UNbatched)
  fused    : PE-NS, our champion newton_schulz (eager cuBLAS)   (baseline 3 — THE 1.8x target)
  amalg    : PE-NS, our BATCHED symmul kernel                   (candidate — symmetric + batched)

Bars (from scope): amalg must beat `fused` ~1.8x in the scaling regime AND peak mem <= `compiled`,
and beat all three baselines as the matrix grows. PARITY is a hard gate (the transpose-copy is the
correctness risk) — a contender that fails parity cannot be promoted regardless of speed.

Metrics per (shape, dtype): parity max|Δ| vs an fp32 cuBLAS PE-NS reference + SV-mean band,
NS(5)-step ms (do_bench), peak MB (max_memory_allocated). Square sweep {1024..8192} is the
headline; batched-small {B x 512^2} is the BiBo regression guard (amalg must NOT regress -> the
symmul dispatch falls back to cuBLAS below SYMMUL_MIN_DIM there).

Run on the box:  <venv>/python .autoresearch/bench_symmul.py
                 (optional: --dtypes fp16,bf16  --json out.json)
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "baselines")))

import torch
from triton.testing import do_bench

from kernels.sm75.muon import _PE_COEFFS, newton_schulz as fused_ns
from kernels.sm120.newton_schulz_symmul import newton_schulz_symmul as amalg_ns
import flash_muon_mmt as fm

DEV = "cuda"
_DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}
# Per-dtype parity tolerance on max|Δp| vs the fp32 cuBLAS PE-NS reference (NS is an approximate
# orthogonalization; fp16 has more mantissa than bf16 -> a tighter gate). SV-mean band is shared.
_TOL = {"fp16": 1.0e-2, "bf16": 2.0e-2}
_SV_BAND = (0.80, 1.20)


# ── shared prep/restore so every contender runs the IDENTICAL algorithm around the GEMMs ──
def _prep(G, dtype, eps=1e-7):
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(dtype) / nrm.to(dtype)
    return X, transposed, squeeze


def _restore(X, transposed, squeeze):
    if transposed:
        X = X.transpose(1, 2)
    return X.squeeze(0) if squeeze else X


def _core_cublas(X, coeffs):
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    return X


_compiled_core = torch.compile(_core_cublas)


def compiled_ns(G, coeffs=_PE_COEFFS, ns_dtype=torch.float16):
    X, t, s = _prep(G, ns_dtype)
    return _restore(_compiled_core(X, coeffs), t, s).to(G.dtype)


def triu_ns(G, coeffs=_PE_COEFFS, ns_dtype=torch.float16):
    """PE-NS using flash-muon's single-matrix kernel; loops over the batch (UNbatched baseline)."""
    X, t, s = _prep(G, ns_dtype)
    Bsz, M, _ = X.shape
    A = torch.empty((Bsz, M, M), device=X.device, dtype=ns_dtype)
    AA = torch.empty_like(A)
    for a, b, c in coeffs:
        for i in range(Bsz):
            fm.matmul_transpose_assign(X[i], A[i])
            fm.matmul_transpose_assign(A[i], AA[i])
        B = b * A + c * AA
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    return _restore(X, t, s).to(G.dtype)


def reference_fp32(G, coeffs=_PE_COEFFS):
    X, t, s = _prep(G, torch.float32)
    return _restore(_core_cublas(X, coeffs), t, s).float()


CONTENDERS = {"compiled": compiled_ns, "triu": triu_ns, "fused": fused_ns, "amalg": amalg_ns}
BASELINES = ("compiled", "triu", "fused")


def make_input(shape, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    return torch.randn(*shape, generator=g, device=DEV, dtype=torch.float16)


def parity(fn, G, dtype, ref):
    Y = fn(G, ns_dtype=dtype).float()
    nan = bool(Y.isnan().any() or Y.isinf().any())
    dmax = (Y - ref).abs().max().item()
    Y3 = Y if Y.ndim == 3 else Y.unsqueeze(0)
    sv = torch.linalg.svdvals(Y3)
    return dmax, sv.mean().item(), nan


def measure(fn, G, dtype):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    fn(G, ns_dtype=dtype)                                   # warm + autotune
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() / 1e6
    ms = do_bench(lambda: fn(G, ns_dtype=dtype))
    return ms, peak


def run(shapes, dtypes):
    print(f"GPU: {torch.cuda.get_device_name(0)}  | torch {torch.__version__}")
    cap = torch.cuda.get_device_capability(0)
    print(f"compute capability: sm_{cap[0]}{cap[1]}\n")
    out = {"gpu": torch.cuda.get_device_name(0), "sm": f"{cap[0]}{cap[1]}", "results": []}

    for dname in dtypes:
        dt = _DTYPES[dname]
        tol = _TOL[dname]
        print(f"================  dtype = {dname}  (parity tol max|Δ| < {tol:.0e})  ================")
        hdr = f"{'shape':>18} | {'contender':>9} | {'parity':>8} {'svμ':>6} | {'ms':>8} | {'MB':>8} | {'vs fused':>9}"
        print(hdr); print("-" * len(hdr))
        for shape in shapes:
            G = make_input(shape)
            ref = reference_fp32(G)
            row = {"shape": list(shape), "dtype": dname, "contenders": {}}
            t_fused = None
            lines = []
            for name, fn in CONTENDERS.items():
                dmax, svm, nan = parity(fn, G, dt, ref)
                ms, peak = measure(fn, G, dt)
                ok = (not nan) and dmax < tol and _SV_BAND[0] <= svm <= _SV_BAND[1]
                row["contenders"][name] = {"dmax": dmax, "svmean": svm, "nan": nan,
                                           "parity_ok": ok, "ms": ms, "peak_mb": peak}
                if name == "fused":
                    t_fused = ms
                lines.append((name, dmax, svm, ok, ms, peak))
            for name, dmax, svm, ok, ms, peak in lines:
                spd = f"{t_fused/ms:.2f}x" if t_fused else "-"
                tag = "PASS" if ok else "FAIL"
                print(f"{str(shape):>18} | {name:>9} | {tag:>4} {dmax:>7.1e} {svm:>6.2f} | {ms:>8.3f} | {peak:>8.1f} | {spd:>9}")
            # headline verdict for the candidate on this shape
            a = row["contenders"]["amalg"]; f = row["contenders"]["fused"]; c = row["contenders"]["compiled"]
            beats = {b: f"{row['contenders'][b]['ms']/a['ms']:.2f}x" for b in BASELINES}
            mem_ok = a["peak_mb"] <= c["peak_mb"] * 1.001
            verdict = ("PASS" if a["parity_ok"] else "PARITY-FAIL")
            print(f"{'':>18} -> amalg vs {beats}  mem {a['peak_mb']:.0f}/{c['peak_mb']:.0f}MB "
                  f"({'<=compiled OK' if mem_ok else 'OVER compiled'})  [{verdict}]")
            print()
            out["results"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtypes", default="fp16,bf16")
    ap.add_argument("--json", default="")
    ap.add_argument("--quick", action="store_true", help="skip 8192 (faster smoke run)")
    args = ap.parse_args()
    assert torch.cuda.is_available()

    square = [(1024, 1024), (2048, 2048), (4096, 4096)] + ([] if args.quick else [(8192, 8192)])
    batched_small = [(32, 512, 512), (128, 512, 512)]      # BiBo regression guard (must not regress)
    shapes = square + batched_small
    dtypes = [d for d in args.dtypes.split(",") if d in _DTYPES]

    print("SQUARE SWEEP = headline (regime B, as we scale). BATCHED-SMALL = BiBo regression guard.\n")
    out = run(shapes, dtypes)
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.json}")


if __name__ == "__main__":
    main()
