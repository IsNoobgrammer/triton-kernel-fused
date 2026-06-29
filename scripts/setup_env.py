#!/usr/bin/env python3
"""Prepare a bench/test environment for triton-kernel-fused — REUSE if present, install only if missing.

This is the fix for `uv sync` clobbering a container's torch. The policy:

  1. If this Python already has a CUDA-enabled torch + triton (per scripts/env_check.py), do NOTHING.
     This is the container / pre-provisioned-GPU case — e.g. an RTX PRO 6000 Blackwell image that ships
     torch 2.12+cu130. A working install is never reinstalled.
  2. Only when something is missing, install it — into THIS interpreter, preferring `uv pip` if uv is on
     PATH (`uv pip install --python <this python>`), else plain `pip`. The CUDA wheel channel is
     selectable so you are never forced onto a build that lacks your GPU.

    python scripts/setup_env.py                  # reuse if ready, else install (default cu124 channel)
    python scripts/setup_env.py --cuda cu128     # pick a CUDA channel for a fresh torch install
    python scripts/setup_env.py --torch-index https://download.pytorch.org/whl/cu130
    python scripts/setup_env.py --force          # install even if a usable env is detected
    python scripts/setup_env.py --cpu            # CPU-only torch (no GPU; for import/CI smoke tests)

Run inside whatever environment you want populated: an activated venv, a `uv venv` (ideally created with
`--system-site-packages` so it inherits a container's torch), or the container's raw Python directly.
"""
import argparse
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import env_check

DEFAULT_CUDA = "cu124"


def _installer():
    """Prefer uv (targeting THIS interpreter) so it installs into the same env; fall back to pip."""
    if shutil.which("uv"):
        return ["uv", "pip", "install", "--python", sys.executable]
    return [sys.executable, "-m", "pip", "install"]


def _run(cmd):
    print(">", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    ap = argparse.ArgumentParser(
        description="Reuse-or-install a CUDA torch + triton for triton-kernel-fused (never clobbers a working one).")
    ap.add_argument("--cuda", default=os.environ.get("TKF_CUDA", DEFAULT_CUDA),
                    help="CUDA wheel channel for a FRESH torch install (cu121, cu124, cu128, ...). "
                         "Ignored if a usable torch is already present.")
    ap.add_argument("--torch-index", default=os.environ.get("TKF_TORCH_INDEX"),
                    help="Full PyTorch wheel index URL (overrides --cuda).")
    ap.add_argument("--cpu", action="store_true", help="Install CPU-only torch (import/CI smoke tests, no GPU).")
    ap.add_argument("--force", action="store_true", help="Install even if a usable env is already detected.")
    args = ap.parse_args()

    info = env_check.probe()
    if info["ok"] and not args.force:
        print("Environment already has a CUDA torch + triton — reusing it, installing nothing.")
        print(f"  torch {info['torch']} (CUDA {info['cuda_version']}) | triton {info['triton']} | "
              f"{info['gpu']} -> kernels.{info['arch']}")
        print("Run:  python bench.py")
        return 0

    base = _installer()
    index = None if args.cpu else (args.torch_index or f"https://download.pytorch.org/whl/{args.cuda}")
    print(f"Setting up via: {' '.join(base)}" + (f"   (torch index: {index})" if index else "   (CPU torch)"))

    # torch — only if missing or below the minimum; never reinstall a satisfactory one.
    if info["torch"] is None or any(p.startswith("torch ") for p in info["problems"]):
        cmd = base + ["torch>=2.4.0"] + (["--index-url", index] if index else [])
        _run(cmd)
    else:
        print(f"torch {info['torch']} already present and adequate — keeping it (no reinstall).")

    # triton — Linux needs the separate wheel; on Windows it ships inside the torch wheel.
    if info["triton"] is None or any(p.startswith("triton ") for p in info["problems"]):
        if sys.platform.startswith("linux") and not args.cpu:
            _run(base + ["triton>=3.0.0"])
        elif sys.platform.startswith("linux"):
            _run(base + ["triton>=3.0.0"])
        else:
            print("triton missing, but on this platform it ships inside the torch wheel — the torch step "
                  "above should provide it. If not, install a CUDA torch build.")
    else:
        print(f"triton {info['triton']} already present and adequate — keeping it (no reinstall).")

    # The kernels package itself — no deps, so this never drags torch/triton along.
    _run(base + ["-e", ".", "--no-deps"])

    print("\nRe-checking environment:")
    return env_check.main()


if __name__ == "__main__":
    raise SystemExit(main())
