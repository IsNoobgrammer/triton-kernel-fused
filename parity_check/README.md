# Parity gates

Numerical-equivalence gates for the fused kernels. Each one compares a kernel against an
independent eager/autograd reference and exits non-zero on failure. Run them after any kernel
change — a green gate is the only evidence the refactor was numerics-neutral.

```
python parity_check/run_all.py         # everything that runs in this venv
python parity_check/parity_radial.py   # one gate
```

Most need CUDA (Triton). `parity_manas_sm120.py` runs on CPU.

| Gate | Pins |
|---|---|
| `parity_radial.py` | act code 8, radial NormSiLU `r^p * SiLU(g/r)` — fwd, d_gate_up, d_theta, and end-to-end through `moe_per_expert` |
| `parity_rowloop.py` | the looped row kernels are bit-equal to the single-pass ones, and the `I <= 1024` cap is gone |
| `parity_normed_tiles.py` | the RMS-normed codes still take the Triton grouped GEMMs instead of silently falling back |
| `parity_expert_alpha.py` | per-expert input scale alpha: fwd, d_gate_up, d_alpha, and `alpha == 1` is bit-identical to passing nothing |
| `parity_specials.py` | ±Identity experts (codes 3/4) across `moe_eager`, `moe_per_expert` and the sm120 grouped path |
| `parity_cautious_wd.py` | cautious weight decay in `FusedMuon` |
| `parity_manas_sm120.py` | sm120 Manas: `gamma=0` is exactly `FusedMuon`, and `gamma>0` actually moves the weights |
| `parity_bibo.py` | `fused_router` vs the real `BiBoMoERouter` — needs the BiBo venv, excluded from `run_all` unless `--all` |

`parity_manas_sm120.py` and `parity_radial.py` both check that the feature **changes an end-to-end
result**, not just that the kernel matches a reference. Per-expert alpha once shipped completely
inert behind a green kernel-level gate; kernel parity cannot see a dispatcher that drops the
parameter.
