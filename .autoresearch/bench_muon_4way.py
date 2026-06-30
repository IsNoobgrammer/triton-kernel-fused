"""4-way Muon bench on bench_muon.py's EXACT methodology, swept across model width H.

Reuses bench_muon's make_shapes / make_params / prime_grads (the real BiBo Muon inventory:
attention + dense MLP + 3D stacked experts) and its do_bench speed harness. Adds a 'compiled'
contender (the reference Polar-Express recipe torch.compile'd) and 'amalg' (AmalgamatedMuon =
FusedMuon + symmul), so we measure compiled / fused / amalg under the SAME shapes the champion
was tuned on -- and sweep H to expose the regime transition:

  small H (512, BiBo): matrices are tiny -> NS is launch-bound -> fused's foreach+batching beats
      compiled big; symmul is INERT (gram < 2048, gated to champion) so amalg == fused.
  large H (>=2048): matrices are compute-bound -> batching fades -> symmul fires so amalg > fused.

per-contender peak mem is measured with empty_cache + reset_peak so it is the contender's own
working set (not floor-contaminated). Timing via do_bench.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import torch
from triton.testing import do_bench

import bench_muon as bm
from bench_muon import baseline_ns, BaselineMuon, make_shapes, make_params, prime_grads
from kernels.sm120.muon import FusedMuon
from kernels.sm120.newton_schulz_symmul import AmalgamatedMuon

DEV = "cuda"
_compiled_ns = torch.compile(baseline_ns)


class CompiledMuon(BaselineMuon):
    """The reference Polar-Express Muon (bench_muon.BaselineMuon) with the NS iteration
    torch.compile'd -- i.e. exactly 'the original, compiled', per-param (no cross-param batching)."""

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr, momentum, wd = group["lr"], group["momentum"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None or p.ndim not in (2, 3):
                    continue
                g = p.grad.bfloat16()
                st = self.state[p]
                if "momentum_buffer" not in st:
                    st["momentum_buffer"] = torch.zeros_like(g)
                buf = st["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                update = g.add(buf, alpha=momentum) if group["nesterov"] else buf
                update = _compiled_ns(update)                       # <-- compiled NS (only change)
                scale = max(1, p.shape[-2] / p.shape[-1]) ** 0.5
                if wd > 0.0:
                    p.data.mul_(1.0 - lr * wd)
                p.add_(update.to(p.dtype), alpha=-lr * scale)


def measure(make_opt, shapes):
    """do_bench step ms + clean per-contender peak MB (empty_cache + reset_peak isolates it)."""
    params = make_params(shapes, 0)
    opt = make_opt(params)
    prime_grads(params)
    opt.step()                                                      # warm (compile/autotune)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    ms = do_bench(lambda: (prime_grads(params), opt.step()))
    peak = torch.cuda.max_memory_allocated() / 1e6
    del opt, params
    torch.cuda.empty_cache()
    return ms, peak


def run_sweep(h_list=(512, 1024, 2048, 4096), layers=2, E=4):
    rows = []
    for H in h_list:
        I = 768 if H == 512 else int(1.5 * H)
        shapes = make_shapes(layers=layers, H=H, I=I, E=E)
        nparam = sum(int(torch.tensor(s).prod().item()) for s in shapes)
        mk = {
            "compiled": lambda ps: CompiledMuon(ps, lr=0.02, weight_decay=0.1),
            "fused": lambda ps: FusedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16),
            "amalg": lambda ps: AmalgamatedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16),
        }
        dpar = bm.parity(lambda ps: AmalgamatedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16), shapes)
        res = {name: measure(fn, shapes) for name, fn in mk.items()}
        tc = res["compiled"][0]; tf = res["fused"][0]
        for name, (ms, mb) in res.items():
            rows.append(dict(H=H, params_M=round(nparam / 1e6, 1), gram=min(H, H // 2),
                             kernel=name, ms=round(ms, 3), MB=round(mb, 1),
                             x_vs_compiled=round(tc / ms, 2), x_vs_fused=round(tf / ms, 2),
                             amalg_parity=("%.1e" % dpar) if name == "amalg" else ""))
        print(f"H={H:5d} ({nparam/1e6:5.0f}M): "
              f"compiled {res['compiled'][0]:7.2f}ms  fused {res['fused'][0]:7.2f}ms "
              f"({tc/res['fused'][0]:.2f}x)  amalg {res['amalg'][0]:7.2f}ms "
              f"({tc/res['amalg'][0]:.2f}x vs compiled, {tf/res['amalg'][0]:.2f}x vs fused)  parity {dpar:.1e}")
    return rows


def run_configs(configs):
    """configs = [(H, layers, E), ...] -- scale BOTH width and depth. Same contenders/metrics."""
    rows = []
    for H, layers, E in configs:
        I = 768 if H == 512 else int(1.5 * H)
        shapes = make_shapes(layers=layers, H=H, I=I, E=E)
        nparam = sum(int(torch.tensor(s).prod().item()) for s in shapes)
        mkf = lambda ps: FusedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16)
        mk = {
            "compiled": lambda ps: CompiledMuon(ps, lr=0.02, weight_decay=0.1),
            "fused": mkf,
            "amalg": lambda ps: AmalgamatedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16),
        }
        dpar = bm.parity(lambda ps: AmalgamatedMuon(ps, lr=0.02, weight_decay=0.1, ns_dtype=torch.float16), shapes)
        res = {name: measure(fn, shapes) for name, fn in mk.items()}
        tc = res["compiled"][0]; tf = res["fused"][0]
        for name, (ms, mb) in res.items():
            rows.append(dict(H=H, layers=layers, params_B=round(nparam / 1e9, 2), kernel=name,
                             ms=round(ms, 2), MB=round(mb, 0),
                             x_vs_compiled=round(tc / ms, 2), x_vs_fused=round(tf / ms, 2),
                             mem_vs_compiled=round(mb / res["compiled"][1], 2),
                             amalg_parity=("%.1e" % dpar) if name == "amalg" else ""))
        print(f"H={H} L={layers} ({nparam/1e9:.2f}B): compiled {tc:8.1f}ms  "
              f"fused {tf:8.1f}ms ({tc/tf:.2f}x)  amalg {res['amalg'][0]:8.1f}ms "
              f"({tc/res['amalg'][0]:.2f}x cmp, {tf/res['amalg'][0]:.2f}x fused)  "
              f"mem a/c {res['amalg'][1]/res['compiled'][1]:.2f}x  parity {dpar:.1e}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--H", default="512,1024,2048,4096")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--E", type=int, default=4)
    args = ap.parse_args()
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)} | torch {torch.__version__}\n")
    run_sweep([int(x) for x in args.H.split(",")], args.layers, args.E)
