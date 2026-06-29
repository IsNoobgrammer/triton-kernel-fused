#!/usr/bin/env python3
"""Environment doctor for triton-kernel-fused.

Checks whether the *current* Python already has a CUDA-enabled PyTorch + Triton good enough to run the
kernels and `bench.py`. Probes only — installs nothing. Exit 0 = ready, 1 = something missing/too old.

    python scripts/env_check.py

Run it before benching, or let `scripts/setup_env.py` call it (that script installs only when this fails).
The whole point: a container or machine that already ships a working CUDA torch (e.g. an RTX PRO 6000
Blackwell image with torch cu130) is "ready" as-is — nothing should reinstall over it.
"""
import os
import sys

MIN_TORCH = (2, 4, 0)
MIN_TRITON = (3, 0, 0)


def _ver_tuple(v):
    """'2.12.0+cu130' -> (2, 12, 0); tolerant of odd suffixes."""
    nums = []
    for part in v.split("+")[0].split(".")[:3]:
        digits = "".join(c for c in part if c.isdigit())
        nums.append(int(digits) if digits else 0)
    while len(nums) < 3:
        nums.append(0)
    return tuple(nums)


def probe():
    """Return a dict describing the env; `ok` is True when torch(+CUDA) and triton meet the minimums."""
    info = {"torch": None, "cuda_version": None, "cuda_available": False, "gpu": None,
            "capability": None, "arch": None, "triton": None, "ok": False, "problems": []}

    try:
        import torch
        info["torch"] = torch.__version__
        if _ver_tuple(torch.__version__) < MIN_TORCH:
            info["problems"].append(f"torch {torch.__version__} < {'.'.join(map(str, MIN_TORCH))} required")
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["gpu"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            info["capability"] = f"{cap[0]}.{cap[1]}"
            try:                                            # map cc -> shipped kernel package
                sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                from kernels import arch_for_capability
                info["arch"] = arch_for_capability(*cap)
            except Exception:
                info["arch"] = None
        else:
            info["problems"].append("torch is installed but torch.cuda.is_available() is False (no CUDA)")
    except ImportError:
        info["problems"].append("torch not installed")

    try:                                                    # Linux: separate wheel; Windows: bundled in torch
        import triton
        info["triton"] = triton.__version__
        if _ver_tuple(triton.__version__) < MIN_TRITON:
            info["problems"].append(f"triton {triton.__version__} < {'.'.join(map(str, MIN_TRITON))} required")
    except ImportError:
        info["problems"].append("triton not installed")

    info["ok"] = (
        info["torch"] is not None
        and info["cuda_available"]
        and info["triton"] is not None
        and not any(p.startswith(("torch ", "triton ")) for p in info["problems"])
    )
    return info


def main():
    info = probe()
    print("triton-kernel-fused — environment check")
    print("-" * 44)
    print(f"  torch        : {info['torch'] or 'NOT INSTALLED'}")
    print(f"  CUDA (torch) : {info['cuda_version'] or 'n/a'}   available={info['cuda_available']}")
    if info["gpu"]:
        print(f"  GPU          : {info['gpu']}  (cc {info['capability']} -> kernels.{info['arch']})")
    print(f"  triton       : {info['triton'] or 'NOT INSTALLED'}")
    print("-" * 44)
    if info["ok"]:
        print(f"READY. Run:  python bench.py        (kernels import from kernels.{info['arch']})")
        return 0
    print("NOT READY:")
    for p in info["problems"]:
        print(f"  - {p}")
    print("\nFix:  python scripts/setup_env.py     (reuses what's there, installs only what's missing)")
    print("or provide a CUDA PyTorch + Triton yourself (a GPU container usually already ships them).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
