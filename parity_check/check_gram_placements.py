"""gram restart placements follow the schedule length (ns_router.gram_placements).

    python -m parity_check.check_gram_placements
"""
from parity_check import _paths  # noqa: F401
from kernels.sm120.ns_router import GRAM_PLACEMENTS, gram_placements

assert gram_placements(8) == GRAM_PLACEMENTS, "ns8 must keep its exact candidate list (logged runs reproduce)"
for n in (5, 6, 8, 10, 12):
    ps = gram_placements(n)
    assert ps[0] == () and len(set(ps)) == len(ps), n
    assert all(1 <= r < n for p in ps for r in p), f"placement at/after the last iteration for n={n}"
    print(f"n={n:2d}: {len(ps)} placements, latest restart {max(max(p) for p in ps if p)}")
assert max(max(p) for p in gram_placements(10) if p) == 8, "ns10 must try restarts near its tail"
print("GRAM_PLACEMENTS PASS")
