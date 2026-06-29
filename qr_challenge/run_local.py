"""Faithful local driver for the qr_v2 frozen eval.

This is NOT the frozen eval -- reference.py / task.py / utils.py are. This only
*reproduces* the GPU MODE scoring locally so we can iterate:

  - test gate : every one of the 22 task.yml `tests` specs through
                check_implementation (the hard correctness gate).
  - benchmark : every one of the 12 task.yml `benchmarks` specs through the same
                timing logic as eval.py::_run_single_benchmark (l2 clear, the
                256MB input-batch sizing, cuda events, early-stop), then the
                geometric mean of the per-case means -- the leaderboard score.

Swap only submission.py between runs. Never edit reference/task/utils.
"""
import math
import time

import torch

from reference import generate_input, check_implementation
from utils import clear_l2_cache, set_seed
from submission import custom_kernel

MAX_ITERATIONS_PER_BENCHMARK = 50
BENCHMARK_INPUT_BYTES_TARGET = 256 * 1024 * 1024

# task.yml `tests` (22) -- the correctness gate.
TESTS = [
    {"batch": 20, "n": 32, "cond": 1, "seed": 53124},
    {"batch": 40, "n": 176, "cond": 1, "seed": 3321},
    {"batch": 40, "n": 352, "cond": 1, "seed": 1200},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32523},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4327},
    {"batch": 1, "n": 4096, "cond": 1, "seed": 75342},
    {"batch": 16, "n": 512, "cond": 4, "seed": 32524, "case": "dense"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32525, "case": "rankdef"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32526, "case": "clustered"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32527, "case": "band"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32528, "case": "rowscale"},
    {"batch": 16, "n": 512, "cond": 0, "seed": 32529, "case": "nearcollinear"},
    {"batch": 4, "n": 1024, "cond": 4, "seed": 4328, "case": "dense"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4329, "case": "rankdef"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4330, "case": "nearrank"},
    {"batch": 4, "n": 1024, "cond": 0, "seed": 4331, "case": "clustered"},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 224466, "case": "dense"},
    {"batch": 2, "n": 2048, "cond": 0, "seed": 224467, "case": "rankdef"},
    {"batch": 1, "n": 4096, "cond": 0, "seed": 75343, "case": "upper"},
    {"batch": 16, "n": 512, "cond": 2, "seed": 32530, "case": "mixed"},
    {"batch": 4, "n": 1024, "cond": 2, "seed": 4332, "case": "mixed"},
    {"batch": 2, "n": 2048, "cond": 2, "seed": 224468, "case": "mixed"},
]

# task.yml `benchmarks` (12) -- the ranked geomean.
BENCHMARKS = [
    {"batch": 20, "n": 32, "cond": 1, "seed": 43214},
    {"batch": 40, "n": 176, "cond": 1, "seed": 423011},
    {"batch": 40, "n": 352, "cond": 1, "seed": 123456},
    {"batch": 640, "n": 512, "cond": 2, "seed": 1029},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 75342},
    {"batch": 8, "n": 2048, "cond": 1, "seed": 224466},
    {"batch": 2, "n": 4096, "cond": 1, "seed": 32412},
    {"batch": 640, "n": 512, "cond": 2, "seed": 770001, "case": "mixed"},
    {"batch": 60, "n": 1024, "cond": 2, "seed": 770002, "case": "mixed"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770003, "case": "rankdef"},
    {"batch": 640, "n": 512, "cond": 0, "seed": 770004, "case": "clustered"},
    {"batch": 60, "n": 1024, "cond": 0, "seed": 770005, "case": "nearrank"},
]


def _clone(d):
    return d.clone() if isinstance(d, torch.Tensor) else d


def _spec(s):
    return ", ".join(f"{k}={v}" for k, v in s.items() if k != "seed")


def run_tests(verbose=True):
    ok = True
    for i, spec in enumerate(TESTS):
        data = generate_input(**spec)
        torch.cuda.synchronize()
        out = custom_kernel(_clone(data))
        torch.cuda.synchronize()
        good, msg = check_implementation(data, out)
        ok = ok and good
        if verbose:
            tag = "pass" if good else "FAIL"
            if good:
                # pull the worst scaled factor residual out of the message
                sf = [p for p in msg.split(";") if "scaled_factor" in p]
                print(f"  [{tag}] t{i:02d} {_spec(spec):42s} {sf[0].strip() if sf else ''}")
            else:
                print(f"  [{tag}] t{i:02d} {_spec(spec):42s} {msg}")
    print(f"TEST GATE: {'PASS' if ok else 'FAIL'} ({sum(1 for _ in TESTS)} specs)")
    return ok


def _bench_count(spec):
    b, n = int(spec.get("batch", 1)), int(spec.get("n", 1))
    bpi = b * n * n * 4
    if bpi <= 0:
        return 1
    return max(1, min(MAX_ITERATIONS_PER_BENCHMARK, BENCHMARK_INPUT_BYTES_TARGET // bpi))


def _make_batch(spec, count):
    args = dict(spec)
    out = []
    for _ in range(count):
        if "seed" in args:
            args["seed"] += 42
        out.append(generate_input(**args))
    return out


def bench_one(spec, max_repeats=200, max_time_ns=10e9):
    count = _bench_count(spec)
    data_list = _make_batch(spec, count)
    # warmup + correctness on the actual timed inputs
    outs = [custom_kernel(_clone(d)) for d in data_list]
    for d, o in zip(data_list, outs):
        good, msg = check_implementation(d, o)
        if not good:
            return None, msg
    durations = []
    t0 = time.perf_counter_ns()
    for i in range(max_repeats):
        torch.cuda.synchronize()
        clear_l2_cache()
        se, ee = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        se.record()
        outs = [custom_kernel(d) for d in data_list]
        ee.record()
        torch.cuda.synchronize()
        durations.append(se.elapsed_time(ee) * 1e6 / len(data_list))  # ns/call
        tot = time.perf_counter_ns() - t0
        if i > 1 and tot > 1e8:
            mean = sum(durations) / len(durations)
            var = sum((x - mean) ** 2 for x in durations) / (len(durations) - 1)
            err = math.sqrt(var) / math.sqrt(len(durations))
            if err / mean < 0.001 or mean * len(durations) > max_time_ns or tot > 120e9:
                break
    mean = sum(durations) / len(durations)
    return mean, f"runs={len(durations)} count={count}"


def run_benchmarks():
    # warmup pass on case 0 (mirrors eval.py)
    bench_one(BENCHMARKS[0], max_repeats=50, max_time_ns=10e7)
    means_us = []
    for i, spec in enumerate(BENCHMARKS):
        mean_ns, info = bench_one(spec)
        if mean_ns is None:
            print(f"  b{i:02d} {_spec(spec):42s} FAIL: {info}")
            return None
        us = mean_ns / 1e3
        means_us.append(us)
        print(f"  b{i:02d} {_spec(spec):42s} {us:10.2f} us   ({info})")
    geo = math.exp(sum(math.log(x) for x in means_us) / len(means_us))
    print(f"GEOMEAN: {geo:.3f} us  over {len(means_us)} benchmark cases")
    return geo


if __name__ == "__main__":
    set_seed(42)
    print(f"device: {torch.cuda.get_device_name(0)}  cc={torch.cuda.get_device_capability(0)}  torch={torch.__version__}")
    print("=== CORRECTNESS GATE (22 test specs) ===")
    gate = run_tests()
    print("\n=== BENCHMARK (12 specs, geomean = leaderboard score) ===")
    if gate:
        run_benchmarks()
    else:
        print("skipping benchmark -- gate failed")
