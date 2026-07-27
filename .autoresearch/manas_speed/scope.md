# Scope — manas probe-path speedup
Real goal: manas usable in real training w/o meaningful tps cost (currently -7% seq1024, -24% seq512).
Artifact: kernels/sm75/manas.py — vote() + apply_probe()/_restore_theta() (per-micro loops). Boundary already batched.
Frozen eval: /home/marimo/eval_manas.py on box (correctness checksum vs frozen ref + timing K=2/4/8 vs muon). NEVER edit.
Objective: overhead=manas-muon -> ~0 (match muon 15.2ms); kinvariance=manas_k8-manas_k2 -> ~0.
Invariants: DO NOT change algorithm (numerical parity gate max_err<1e-6); fp32 params; never touch eval.
In-scope: batch per-param loops (shape-group bmm + foreach), torch.compile/CUDA-graph, fuse ops.
Out-of-scope: changing the probe math, rank, gamma, sketch semantics.
Baseline: muon 15.2 | manas k2 56.4 k4 84.8 k8 140.4 | overhead +41/+70/+125 | kinv 84.
Done: overhead_k4 < ~10ms AND kinvariance < ~15ms, correctness pass.
