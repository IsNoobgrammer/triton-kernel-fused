"""FusedMuon `muon_rc`: a parameter stored in another shape but orthogonalised as (r, c) matrices must step EXACTLY
like a real parameter of that matrix shape (same grads, same steps).

Cases (the ASR model's): pointwise Conv1d (out, in, 1) -> (out, in); pointwise Conv2d (out, in, 1, 1) -> (out, in);
gate-stacked LSTM (4H, in) -> 4 x (H, in) (compared with a real (4, H, in) parameter); attention q/k/v per head
(512, 512) -> 8 x (64, 512). Variants aurora (default) and
muown, fused tail on / off, nesterov, weight decay. Gate: bitwise equal weights after 5 steps.

    python parity_check/parity_muon_rc.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch

from kernels.sm120.muon import FusedMuon

ok = True


def run(make, shape_ref, variant, fused_tail, steps=5, seed=0):
    """make() -> (param stored shape, muon_rc or None). Returns the weights after `steps`, viewed as shape_ref."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    stored, rc = make()
    w0 = torch.randn(shape_ref, device="cuda", generator=g) * 0.05
    p = torch.nn.Parameter(w0.reshape(stored).clone())
    if rc is not None:
        p.muon_rc = rc
    opt = FusedMuon([p], lr=1e-2, momentum=0.95, weight_decay=0.1, variant=variant, fused_tail=fused_tail,
                    ns_backend="cublas")
    for _ in range(steps):
        p.grad = (torch.randn(shape_ref, device="cuda", generator=g) * 0.01).reshape(stored)
        opt.step()
    return p.detach().reshape(shape_ref).clone()


def main():
    global ok
    cases = [("pointwise Conv1d (1024,512,1) as (1024,512)", (1024, 512, 1), (1024, 512), (1024, 512)),
             ("pointwise Conv2d (256,256,1,1) as (256,256)", (256, 256, 1, 1), (256, 256), (256, 256)),
             ("LSTM (2560,640) as 4 x (640,640)", (2560, 640), (640, 640), (4, 640, 640)),
             ("attention per head (512,512) as 8 x (64,512)", (512, 512), (64, 512), (8, 64, 512))]
    for variant in ("aurora", "muown"):
        for ft in (True, False):
            for name, stored, rc, ref in cases:
                a = run(lambda: (stored, rc), ref, variant, ft)
                b = run(lambda: (ref, None), ref, variant, ft)          # a real parameter of the matrix shape
                same = torch.equal(a, b)
                ok &= same
                print(f"  [{'PASS' if same else 'FAIL'}] {variant:6s} fused_tail={ft!s:5s} {name}: "
                      f"{'bitwise equal' if same else f'max diff {(a - b).abs().max().item():.3e}'}", flush=True)
    # without muon_rc a (out, in, 1) conv is the WRONG thing (out separate (in, 1) slices): it must differ
    a = run(lambda: ((1024, 512, 1), None), (1024, 512), "aurora", True)
    b = run(lambda: ((1024, 512), None), (1024, 512), "aurora", True)
    differs = not torch.equal(a, b)
    ok &= differs
    print(f"  [{'PASS' if differs else 'FAIL'}] control: an UNMARKED (1024,512,1) conv steps differently from the "
          f"(1024,512) matrix (max diff {(a - b).abs().max().item():.3e}) -- i.e. the attribute is what fixes it", flush=True)
    print("\nALL PASS" if ok else "\nFAILED", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
