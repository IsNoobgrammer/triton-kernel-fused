"""expert_counts (block histogram) must equal torch.bincount exactly, with no host sync; then timing
vs the old scatter_add_ at the board shape (65536 tokens x top-6, 64 experts).
    python parity_check/parity_expert_counts.py"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from kernels.sm75.moe import expert_counts

ok = True
for E in (8, 64, 65):
    for n in (0, 1, 4095, 4096, 4097, 393216):
        for srt in (False, True):
            e = torch.randint(0, E, (n,), device="cuda")
            if srt:
                e = e.sort().values
            ok &= torch.equal(expert_counts(e, E), torch.bincount(e, minlength=E))
e = torch.randint(0, 64, (393216,), device="cuda")
n = [0]
warnings.showwarning = lambda msg, *a, **k: n.__setitem__(0, n[0] + ("called a synchronizing" in str(msg)))  # real syncs, not the mode notice
warnings.simplefilter("always")
torch.cuda.synchronize(); torch.cuda.set_sync_debug_mode("warn")
expert_counts(e, 64)
torch.cuda.set_sync_debug_mode(0)
ok &= n[0] == 0
for name, fn in (("scatter_add_", lambda: torch.zeros(64, dtype=torch.long, device="cuda").scatter_add_(0, e, torch.ones_like(e))),
                 ("expert_counts", lambda: expert_counts(e, 64))):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(50):
        fn()
    b.record(); torch.cuda.synchronize()
    print(f"   {name:14s} {a.elapsed_time(b) / 50 * 1e3:8.1f} us  (393216 ids, 64 bins)")
print(f"host syncs {n[0]}")
# the MoE sort switched to int16 keys + stable=True: the permutation must be IDENTICAL to the old one
for E_, n_ in ((64, 393216), (8, 524288), (64, 1000)):
    ids = torch.randint(0, E_, (n_,), device="cuda")
    v0, o0 = ids.sort()
    v1, o1 = ids.to(torch.int16).sort(stable=True)
    same = torch.equal(o0, o1) and torch.equal(v0, v1.long())
    ok &= same
    print(f"   sort permutation int64 default vs int16 stable, E={E_} n={n_}: {'identical' if same else 'DIFFERS'}")
for name, fn in (("sort int64", lambda: ids64.sort()), ("sort int16 stable", lambda: ids64.to(torch.int16).sort(stable=True))):
    ids64 = torch.randint(0, 64, (393216,), device="cuda")
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    a_, b_ = torch.cuda.Event(True), torch.cuda.Event(True)
    a_.record()
    for _ in range(20):
        fn()
    b_.record(); torch.cuda.synchronize()
    print(f"   {name:18s} {a_.elapsed_time(b_) / 20 * 1e3:8.1f} us")
print("COUNTS PASS" if ok else "COUNTS FAIL")
