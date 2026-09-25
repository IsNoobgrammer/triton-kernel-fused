"""Where does the fused CE's extra gradient error (vs plain PyTorch bf16) come from? Stage by stage,
from the SAME bf16 logits, against fp64:
    g   = d loss / d logits          ours: bf16, in place    eager: fp32 (autocast CE), cast to bf16
    dh  = g @ W                      both bf16 GEMMs
    dW  = g^T @ h                    ours: bf16 addmm_ accumulated over chunks    eager: one GEMM

    python parity_check/diag_ce_grad.py
"""
import importlib
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F

CE = importlib.import_module("kernels.sm75.cross_entropy")
dev, bf = "cuda", torch.bfloat16
N, H, V = 16384, 512, 81920
g0 = torch.Generator(device=dev).manual_seed(0)
x = torch.randn(N, H, device=dev, generator=g0).to(bf)
w = (torch.randn(V, H, device=dev, generator=g0) * H ** -0.5 * 2).to(bf)
lab = torch.randint(0, V, (N,), device=dev, generator=g0)
lab[::7] = -100
L = torch.mm(x, w.t())                                     # bf16 logits, shared by every arm


def rel(a, b):
    return ((a.double() - b).norm() / b.norm()).item()


# fp64 truth of every stage, from the shared bf16 logits / operands
Ld = L.double().requires_grad_()
F.cross_entropy(Ld, lab, ignore_index=-100).backward()
G64 = Ld.grad
dh64 = G64 @ w.double()
dW64 = G64.t() @ x.double()

# eager: autocast CE runs in fp32 on the bf16 logits; its grad is cast back to bf16 for the mm
Lf = L.float().requires_grad_()
F.cross_entropy(Lf, lab, ignore_index=-100).backward()
Ge32 = Lf.grad
Ge = Ge32.to(bf)
dhe = torch.mm(Ge, w)
dWe = torch.mm(Ge.t(), x)

# ours: lse from the kernel, grad written in place (bf16), dW accumulated over chunks like the loop
valid = lab != -100
nv = valid.sum().clamp(min=1).to(torch.float32).reshape(1)
lse = torch.logsumexp(L.float(), -1)
Go = L.clone()
CE._grad_logits_inplace(Go, lse, lab, nv, -100)
dho = torch.mm(Go, w)
C = CE._chunk_rows(N, V, 256 * 1024 * 1024)
dWo = torch.zeros_like(w)
for i in range(0, N, C):
    dWo.addmm_(Go[i:i + C].t(), x[i:i + C])
dWo32 = torch.zeros(V, H, device=dev, dtype=torch.float32)
for i in range(0, N, C):
    dWo32 += torch.mm(Go[i:i + C].t(), x[i:i + C]).float()

print(f"stage g  (d loss / d logits):  eager fp32 {rel(Ge32, G64):.3e}  eager->bf16 {rel(Ge, G64):.3e}  ours bf16 {rel(Go, G64):.3e}")
print(f"stage dh (g @ W):              eager {rel(dhe, dh64):.3e}  ours {rel(dho, dh64):.3e}")
print(f"stage dW (g^T @ h):            eager one GEMM {rel(dWe, dW64):.3e}  ours bf16 addmm_ over {-(-N // C)} chunks "
      f"{rel(dWo, dW64):.3e}  ours fp32 chunk-sum {rel(dWo32, dW64):.3e}")
d = (Go.float() - Ge.float())
nz = (d != 0)
print(f"g elements differing ours vs eager-bf16: {nz.float().mean().item():.4%}; "
      f"where they differ, mean |ours-truth| {((Go.double() - G64)[nz]).abs().mean().item():.3e} "
      f"vs eager {((Ge.double() - G64)[nz]).abs().mean().item():.3e}")
