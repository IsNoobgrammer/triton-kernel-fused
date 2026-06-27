"""Frozen eval — RUN THIS CELL ON THE KAGGLE T4 NOTEBOOK each loop round.

It pulls the candidate branch, runs the compiled bench for MoE + CE on the T4, and prints the
`@@RESULT` lines. Copy the WHOLE output back to the agent.

Usage in a Kaggle cell:
    import os; os.environ["ARC_BRANCH"] = "arc/round-1"   # the branch the agent pushed (default: master)
    exec(open("/kaggle/working/triton-kernel-fused/.autoresearch/kaggle_eval.py").read())
  (or just paste this file's contents into a cell)
"""
import os
import sys
import subprocess

REPO = "/kaggle/working/triton-kernel-fused"
URL = "https://github.com/IsNoobgrammer/triton-kernel-fused"
BRANCH = os.environ.get("ARC_BRANCH", "master")


def _run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


if not os.path.isdir(REPO):
    _run(["git", "clone", "-q", URL, REPO])
_run(["git", "-C", REPO, "fetch", "-q", "--all"])
_run(["git", "-C", REPO, "checkout", "-q", BRANCH])
_run(["git", "-C", REPO, "reset", "--hard", "-q", f"origin/{BRANCH}"])
head = _run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"]).stdout.strip()

print(f"=== eval branch={BRANCH} @ {head} ===")
r = _run([sys.executable, "bench.py", "--compile", "--json", "moe", "ce"], cwd=REPO)
print(r.stdout[-6000:])
tail = r.stderr.strip().splitlines()[-8:]
if tail:
    print("--- stderr tail ---")
    print("\n".join(tail))
print("=== paste everything above (esp. the @@RESULT lines) back to the agent ===")
