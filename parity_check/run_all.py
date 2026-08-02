"""Run every parity gate and print one line per gate. Exit 1 if any failed.

    python parity_check/run_all.py          # all gates that run in this venv
    python parity_check/run_all.py --all    # also parity_bibo (needs the BiBo venv, see its docstring)

Each gate already exits non-zero on failure, so this only has to collect return codes.
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
NEEDS_BIBO_VENV = {"parity_bibo.py", "parity_xsa_alpha.py", "parity_swa_flex.py"}


def main():
    include_all = "--all" in sys.argv
    gates = sorted(p for p in HERE.glob("parity_*.py")
                   if include_all or p.name not in NEEDS_BIBO_VENV)
    failed = []
    for g in gates:
        r = subprocess.run([sys.executable, str(g)], capture_output=True, text=True)
        tail = (r.stdout.strip().splitlines() or [r.stderr.strip().splitlines()[-1:] or ""])[-1]
        print(f"{'PASS' if r.returncode == 0 else 'FAIL'}  {g.name:<26} {tail[:90]}")
        if r.returncode:
            failed.append((g.name, r.stdout, r.stderr))
    for name, out, err in failed:
        print(f"\n{'=' * 20} {name} {'=' * 20}\n{out}\n{err}")
    print(f"\n{len(gates) - len(failed)}/{len(gates)} gates passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
