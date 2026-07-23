"""Fused PolyGLU MoE experts — the hard one. Per-expert + grouped paths + naive-eager reference.

PolyGLU: experts are GLU MLPs with *heterogeneous* activations — each expert carries an
activation code (0=SiLU, 1=ReLU², 2=NormSiLU, 5=SiTU), e.g. groups of three [SiLU, ReLU², NormSiLU].
Pass an `act_codes` (E,) int32 tensor alongside the expert weights.

SiTU (code 5, Jul 22 2026): tanh(gate) * sigmoid(gate) — SiLU with the linear factor replaced by
tanh; bounded, fully elementwise, and PARAMETER-FREE by default (the menu design: activations carry
no learnable params or norms — scaling lives in the gate/up projections; NormSiLU is likewise
gain-free). Optional learnable variant: pass `act_params` (E, 2) fp32 [alpha, gamma] to compute
gamma * tanh(alpha*gate) * sigmoid(gate) with per-expert scalars (DyT-style; kept for A/Bs — the
toy MLP round measured alpha as load-bearing there). Unlike NormSiLU there is NO row reduction in
fwd; the learnable variant adds one in-kernel dalpha/dgamma pass in bwd, gated on requires_grad.
Per-expert path only when act_params is used (grouped paths reject code 5, like the specials).

NormSiLU (code 2, replaced Tanh Jul 7 2026): SiLU(gate / rms(gate)) — per-row RMS over the
intermediate dim, gain-free, eps 1e-6 (DECO intra-expert stage adapted to GLU; matches BiBo's
eager _POLYGLU_ACTIVATIONS). The RMS is a row reduction, so the elementwise GLU kernels get a
one-program-per-row pre-pass each way: _row_rms_kernel (fwd: r per row) and _row_s_kernel
(bwd: S = Σ_j go·up·silu'(ĝ)·ĝ, the RMS-coupling term of the gradient
grad_gate = (go·up·silu'(ĝ) − (S/I)·ĝ)/r, where ĝ = gate/r).

Why MoE can't be ONE fused kernel (unlike SwiGLU / XSA / CE)
------------------------------------------------------------
SwiGLU/XSA/CE are *dense*: every row does identical work, so one kernel with a fixed grid
covers them. An MoE is *data-dependent* — the router sends each token to a runtime-chosen
subset of experts, so the work is a ragged collection of per-expert GEMMs whose sizes aren't
known until the router fires. That breaks single-kernel fusion at three points, "from weights,
to dispatch, to the final summed tensor":

  1. DISPATCH (gather): tokens for expert e are scattered across the batch; you must gather them
     into a contiguous block before a GEMM can touch them.
  2. RAGGED GEMM: expert e gets `count[e]` tokens — runtime-shaped, different every step. A plain
     batched GEMM needs equal sizes; here each "batch" is a different M. You loop (one GEMM per
     expert) or block-schedule a grouped GEMM over sorted tokens.
  3. COMBINE (scatter): each token went to top-k experts, so the output is a weighted sum of k
     expert outputs scattered back to its row — an index-add reduction, not a plain write.

So a real MoE is a *pipeline* of fused stages wired by a sort, not one kernel. The router itself
stays in your model; pass its top-k indices/weights in. Two expert-pipeline drop-ins:

  moe_per_expert(...) — sort by expert, then per expert: cuBLAS gate_up GEMM -> fused PolyGLU
        activation (Triton) -> cuBLAS down GEMM -> weighted scatter. MANUAL backward (no
        autograd-composition glue: no grad-accum add_, no per-op fill_). Best at LOW token counts.
  moe_grouped(...)    — ONE block-scheduled grouped-GEMM over all sorted tokens (Triton tl.dot)
        + matched grouped-GEMM backward. Best at HIGH token counts. ⚠ tl.dot: re-bench per arch.
  moe(...)            — auto: grouped at >= GROUPED_MIN_TOKENS rows, else per-expert.

Why naive eager is so slow
---------------------------
`moe_eager` is the hand-written version: loop experts, boolean-mask each (`idx == e`), gather,
two `F.linear`s, activation, scatter. Slow because (a) the per-expert boolean-mask/index forces
a GPU→CPU **sync every iteration** (the launch queue drains E times per layer), (b) the GLU
activation is unfused elementwise kernels + an intermediate write, (c) zero GEMM batching — E
tiny GEMMs each under-utilizing the device. The fused paths kill all three: one sort instead of
E masks, a fused-activation Triton kernel, and (grouped) a single batched GEMM.

Weights: gate_up_proj (E, 2*I, H), down_proj (E, H, I), act_codes (E,) int32.
"""
import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["moe", "moe_per_expert", "moe_grouped", "moe_grouped_cublas", "moe_eager",
           "BatchedGLU", "GROUPED_MIN_TOKENS"]

GROUPED_MIN_TOKENS = 4096
SCHED_BLOCK_M = 64
_NS_EPS = 1e-6         # NormSiLU rms eps — must match BiBo eager (_NORMSILU_EPS in ffn/moe.py)
_NS_BLOCK_I = 256      # row-reduction chunk (I=768 -> 3 iters); fixed block, no autotune (MoE rule)


def _code_max(act_codes):
    """Max act code with a ONE-TIME host sync per tensor: act_codes is static model config, so the
    result is cached as a python attribute on the tensor (dispatchers call this per layer per step)."""
    m = getattr(act_codes, "_code_max_cache", None)
    if m is None:
        m = int(act_codes.max())
        try:
            act_codes._code_max_cache = m
        except Exception:
            pass                      # exotic tensor subclass: fall back to syncing each call
    return m


def _amp_cast(*ts):
    """Under autocast, cast float tensors to the ACTIVE autocast dtype (fp16/bf16) so the custom
    Functions see one consistent dtype end-to-end; no-op outside autocast. Grads returned for the
    cast tensors are dtype-converted back by the autograd engine at the Function boundary."""
    if torch.is_autocast_enabled("cuda"):
        dt = torch.get_autocast_dtype("cuda")
        return tuple(t.to(dt) if t.is_floating_point() else t for t in ts)
    return ts


# ───────────────────────── PolyGLU activation (per-row act code) ─────────────────────────
# NormSiLU (code 2) needs a per-row RMS over the gate half — a row reduction the elementwise
# tile kernels can't see. One-program-per-row pre-pass kernels compute the row scalars; the rms
# is recomputed in backward from the saved gate_up (one cheap read; keeps _glu_fwd/_glu_bwd
# signatures and every autograd path's ctx unchanged).
@triton.jit
def _row_rms_kernel(GateUp_ptr, Act_ptr, Rms_ptr, I, s_gu_m, s_gu_i,
                    EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    if (at == 2) or (at == 6):                           # NormSiLU (2) and radial NormSiLU (6) both need rms
        acc = tl.zeros([BLOCK_I], dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            offs = i0 + tl.arange(0, BLOCK_I)
            g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=offs < I, other=0.0).to(tl.float32)
            acc += g * g
        tl.store(Rms_ptr + row, tl.sqrt(tl.sum(acc) / I + EPS))
    else:
        tl.store(Rms_ptr + row, 1.0)


@triton.jit
def _row_s_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Rms_ptr, S_ptr, T_ptr, I,
                  s_go_m, s_go_i, s_gu_m, s_gu_i, BLOCK_I: tl.constexpr):
    # RMS-coupling prepass (codes 2 & 6). S = sum_j gu*silu'(g/r)*(g/r) ; T = sum_j gu*silu(g/r)
    # (T is the radial d(r^0.5)/dg coupling, consumed only for code 6; harmless/ignored for code 2).
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    if (at == 2) or (at == 6):
        r = tl.load(Rms_ptr + row)
        accS = tl.zeros([BLOCK_I], dtype=tl.float32)
        accT = tl.zeros([BLOCK_I], dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            offs = i0 + tl.arange(0, BLOCK_I)
            m = offs < I
            go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=m, other=0.0).to(tl.float32)
            gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
            up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
            gn = gate / r
            sig = 1.0 / (1.0 + tl.exp(-gn))
            gu = go * up
            accS += gu * (sig * (1.0 + gn * (1.0 - sig))) * gn
            accT += gu * (gn * sig)
        tl.store(S_ptr + row, tl.sum(accS))
        tl.store(T_ptr + row, tl.sum(accT))
    else:
        tl.store(S_ptr + row, 0.0)
        tl.store(T_ptr + row, 0.0)


# ── row-fused (v2) GLU kernels: one program per row spans the FULL intermediate dim, so the
# NormSiLU rms, the backward's S-coupling term, and SiTU's dalpha/dgamma row sums all happen
# IN-REGISTER in the same pass. 1 launch fwd + 1 launch bwd for every code — no pre-pass kernels,
# no extra HBM reads (fwd ~4N->3N, bwd ~9N->5N for NormSiLU). Used when I <= _ROWFUSE_MAX_I;
# larger I falls back to the tiled kernels + pre-pass path below (kept unchanged).
_ROWFUSE_MAX_I = 1024


@triton.jit
def _glu_fwd_row_kernel(GateUp_ptr, Act_ptr, Alpha_ptr, Gamma_ptr, Out_ptr, I,
                        s_gu_m, s_gu_i, s_o_m, s_o_i, s_ap,
                        EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I)
    msk = offs < I
    gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    gg = tl.load(Gamma_ptr + row * s_ap).to(tl.float32)
    r = tl.sqrt(tl.sum(gate * gate) / I + EPS)           # consumed where at==2 (NormSiLU) / at==6 (radial)
    is_norm = (at == 2) | (at == 6)
    gn = tl.where(is_norm, gate / r, gate)
    sig = 1.0 / (1.0 + tl.exp(-gn))                       # sigma(gate) for codes 0/1/5, sigma(gate/r) for 2/6
    f = gn * sig                                          # silu: code 0 (gn=gate) AND codes 2/6 (gn=gate/r)
    relu = tl.maximum(gate, 0.0)
    t5 = 2.0 / (1.0 + tl.exp(-2.0 * aa * gate)) - 1.0    # tanh(alpha*g) = 2*sigmoid(2*alpha*g)-1 (exact)
    act = tl.where(at == 1, relu * relu,
                   tl.where(at == 5, gg * t5 * sig,
                            tl.where(at == 6, tl.sqrt(r) * f, f)))   # code 6: sqrt(r)*SiLU(g/r) (radial NormSiLU)
    tl.store(Out_ptr + row * s_o_m + offs * s_o_i, (act * up).to(Out_ptr.dtype.element_ty), mask=msk)


@triton.jit
def _glu_bwd_row_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Alpha_ptr, Gamma_ptr,
                        GradGateUp_ptr, DA_ptr, DG_ptr, I,
                        s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i, s_ap,
                        EPS: tl.constexpr, WANT_AP: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I)
    msk = offs < I
    go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=msk, other=0.0).to(tl.float32)
    gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    gg = tl.load(Gamma_ptr + row * s_ap).to(tl.float32)
    r = tl.sqrt(tl.sum(gate * gate) / I + EPS)
    is_norm = (at == 2) | (at == 6)
    gn = tl.where(is_norm, gate / r, gate)
    sig = 1.0 / (1.0 + tl.exp(-gn))
    f = gn * sig
    df = sig * (1.0 + gn * (1.0 - sig))
    relu = tl.maximum(gate, 0.0)
    t5 = 2.0 / (1.0 + tl.exp(-2.0 * aa * gate)) - 1.0
    situ = gg * t5 * sig
    dsitu = gg * (aa * (1.0 - t5 * t5) * sig + t5 * sig * (1.0 - sig))
    act = tl.where(at == 1, relu * relu,
                   tl.where(at == 5, situ, tl.where(at == 6, tl.sqrt(r) * f, f)))
    gu_ = go * up
    # RMS-coupling. S = sum(gu*silu'(gn)*gn) (codes 2 & 6); T = sum(gu*silu(gn)) (radial code 6 only,
    # from d(r^0.5)/dg). code 2 (p=0): (gu*silu'(gn) - (S/I)*gn)/r ;
    # code 6 (p=0.5): (gu*silu'(gn) - (gn/I)*(S - 0.5*T))/sqrt(r).
    S = tl.sum(tl.where(is_norm, gu_ * df * gn, 0.0))
    T = tl.sum(tl.where(at == 6, gu_ * f, 0.0))
    grad_norm = tl.where(at == 6, (gu_ * df - (gn / I) * (S - 0.5 * T)) / tl.sqrt(r),
                         (gu_ * df - (S / I) * gn) / r)
    grad_gate = tl.where(is_norm, grad_norm,
                         gu_ * tl.where(at == 0, df, tl.where(at == 5, dsitu, 2.0 * relu)))
    tl.store(GradGateUp_ptr + row * s_ggu_m + offs * s_ggu_i, grad_gate, mask=msk)
    tl.store(GradGateUp_ptr + row * s_ggu_m + (I + offs) * s_ggu_i, go * act, mask=msk)
    if WANT_AP:
        if at == 5:
            tl.store(DA_ptr + row, tl.sum(gu_ * gg * gate * (1.0 - t5 * t5) * sig))
            tl.store(DG_ptr + row, tl.sum(gu_ * t5 * sig))
        else:
            tl.store(DA_ptr + row, 0.0)
            tl.store(DG_ptr + row, 0.0)


@triton.jit
def _glu_fwd_kernel(GateUp_ptr, Act_ptr, Rms_ptr, Alpha_ptr, Gamma_ptr, Out_ptr, M, I,
                    s_gu_m, s_gu_i, s_o_m, s_o_i, s_ap,
                    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0); pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M; mask = mask_m[:, None] & (offs_i < I)[None, :]
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + offs_m, mask=mask_m, other=0)[:, None]
    r = tl.load(Rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    aa = tl.load(Alpha_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]   # 1.0 where at!=5
    gg = tl.load(Gamma_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    sig = 1.0 / (1.0 + tl.exp(-gate))
    silu = gate * sig
    relu = tl.maximum(gate, 0.0)
    gn = gate / r                                    # r==1.0 for non-norm rows (rms prepass sets 1.0)
    nsilu = gn * (1.0 / (1.0 + tl.exp(-gn)))
    t5 = 2.0 / (1.0 + tl.exp(-2.0 * aa * gate)) - 1.0   # tanh(alpha*g) = 2*sigmoid(2*alpha*g)-1 (exact)
    situ = gg * t5 * sig                                 # SiTU: gamma * tanh(alpha*g) * sigmoid(g)
    act = tl.where(at == 0, silu,
                   tl.where(at == 1, relu * relu,
                            tl.where(at == 5, situ,
                                     tl.where(at == 6, tl.sqrt(r) * nsilu, nsilu))))   # code 6: radial NormSiLU
    tl.store(Out_ptr + offs_m[:, None] * s_o_m + offs_i[None, :] * s_o_i,
             (act * up).to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _glu_bwd_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Rms_ptr, S_ptr, T_ptr, Alpha_ptr, Gamma_ptr,
                    GradGateUp_ptr, M, I,
                    s_go_m, s_go_i, s_gu_m, s_gu_i, s_ggu_m, s_ggu_i, s_ap,
                    BLOCK_M: tl.constexpr, BLOCK_I: tl.constexpr):
    pid_m = tl.program_id(0); pid_i = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_i = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
    mask_m = offs_m < M; mask = mask_m[:, None] & (offs_i < I)[None, :]
    go = tl.load(GradOut_ptr + offs_m[:, None] * s_go_m + offs_i[None, :] * s_go_i, mask=mask, other=0.0).to(tl.float32)
    gate = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + offs_i[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + offs_m[:, None] * s_gu_m + (I + offs_i)[None, :] * s_gu_i, mask=mask, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + offs_m, mask=mask_m, other=0)[:, None]
    r = tl.load(Rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    sv = tl.load(S_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    tv = tl.load(T_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    aa = tl.load(Alpha_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]   # 1.0 where at!=5
    gg = tl.load(Gamma_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    sig = 1.0 / (1.0 + tl.exp(-gate)); silu = gate * sig; dsilu = sig * (1.0 + gate * (1.0 - sig))
    relu = tl.maximum(gate, 0.0); relu2 = relu * relu; drelu2 = 2.0 * relu
    gn = gate / r
    sig_n = 1.0 / (1.0 + tl.exp(-gn)); nsilu = gn * sig_n
    dnsilu = sig_n * (1.0 + gn * (1.0 - sig_n))
    t5 = 2.0 / (1.0 + tl.exp(-2.0 * aa * gate)) - 1.0   # tanh(alpha*g), exact sigmoid identity
    situ = gg * t5 * sig
    dsitu = gg * (aa * (1.0 - t5 * t5) * sig + t5 * sig * (1.0 - sig))
    act = tl.where(at == 0, silu,
                   tl.where(at == 1, relu2,
                            tl.where(at == 5, situ,
                                     tl.where(at == 6, tl.sqrt(r) * nsilu, nsilu))))
    # RMS-coupling grad. code 2 (p=0): (gu·silu'(ĝ) − (S/I)·ĝ)/r ;
    # code 6 (p=0.5): (gu·silu'(ĝ) − (ĝ/I)·(S − 0.5·T))/sqrt(r).
    gu_ = go * up
    grad_norm = tl.where(at == 6, (gu_ * dnsilu - (gn / I) * (sv - 0.5 * tv)) / tl.sqrt(r),
                         (gu_ * dnsilu - (sv / I) * gn) / r)
    grad_gate = tl.where((at == 2) | (at == 6), grad_norm,
                         gu_ * tl.where(at == 0, dsilu, tl.where(at == 5, dsitu, drelu2)))
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + offs_i[None, :] * s_ggu_i, grad_gate, mask=mask)
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + (I + offs_i)[None, :] * s_ggu_i, go * act, mask=mask)


@triton.jit
def _row_situ_bwd_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Alpha_ptr, Gamma_ptr, DA_ptr, DG_ptr, I,
                         s_go_m, s_go_i, s_gu_m, s_gu_i, s_ap, BLOCK_I: tl.constexpr):
    # per-row SiTU param-grad sums (only at==5 rows):
    #   dgamma_row = sum_j go*up*tanh(a*g)*sig(g) ; dalpha_row = sum_j go*up*gamma*g*(1-t^2)*sig(g)
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    if at == 5:
        a = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
        gm = tl.load(Gamma_ptr + row * s_ap).to(tl.float32)
        acc_a = tl.zeros([BLOCK_I], dtype=tl.float32)
        acc_g = tl.zeros([BLOCK_I], dtype=tl.float32)
        for i0 in range(0, I, BLOCK_I):
            offs = i0 + tl.arange(0, BLOCK_I)
            m = offs < I
            go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=m, other=0.0).to(tl.float32)
            gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
            up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
            t = 2.0 / (1.0 + tl.exp(-2.0 * a * gate)) - 1.0   # tanh(a*g), exact sigmoid identity
            sig = 1.0 / (1.0 + tl.exp(-gate))
            gu = go * up
            acc_g += gu * t * sig
            acc_a += gu * gm * gate * (1.0 - t * t) * sig
        tl.store(DA_ptr + row, tl.sum(acc_a))
        tl.store(DG_ptr + row, tl.sum(acc_g))
    else:
        tl.store(DA_ptr + row, 0.0)
        tl.store(DG_ptr + row, 0.0)


# reusable fp32 ones — dummy Rms/S input for slices KNOWN to have no NormSiLU rows (r=1, S=0
# semantics: the kernels only consume them where at==2). Slicing a cached buffer costs zero
# kernel launches, vs one fill launch per _glu call.
_ONES_CACHE = {}


def _ones(M, device):
    buf = _ONES_CACHE.get(device)
    if buf is None or buf.numel() < M:
        buf = torch.ones(max(M, 8192), device=device, dtype=torch.float32)
        _ONES_CACHE[device] = buf
    return buf[:M]


def _row_rms(gate_up, row_act, I):
    M = gate_up.shape[0]
    rms = torch.empty(M, device=gate_up.device, dtype=torch.float32)
    if M > 0:
        _row_rms_kernel[(M,)](gate_up, row_act, rms, I, gate_up.stride(0), gate_up.stride(1),
                              EPS=_NS_EPS, BLOCK_I=_NS_BLOCK_I)
    return rms


def _ap_stride(row_alpha):
    """0 = broadcast one scalar to every row (uniform per-expert slice); 1 = per-row values."""
    return 0 if (row_alpha is not None and row_alpha.numel() == 1) else 1


def _ts_norm_eager(gate_up):
    """ts_norm (act code 7) applied EAGERLY to a (M,2I) gate_up slice: tanh(sigmoid(g/rms(g))) * up.
    Pure autograd-native torch — lets the FUSED per-expert MoE (fast dispatch + cuBLAS GEMMs) use code 7
    with NO Triton activation kernel; the elementwise activation is the only eager part."""
    I = gate_up.shape[1] // 2
    g = gate_up[:, :I].float(); u = gate_up[:, I:]
    z = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
    return torch.tanh(torch.sigmoid(z)).to(gate_up.dtype) * u


def _glu_fwd(gate_up, row_act, code_hint=None, row_alpha=None, row_gamma=None):
    """code_hint: host-side int when EVERY row shares one act code (the per-expert path) — lets
    non-NormSiLU slices skip the _row_rms launch on the tiled fallback path (row-fused path needs
    no pre-pass at all). row_alpha/row_gamma: per-row fp32 SiTU scalars (code 5); None -> 1.0.
    code_hint==7 (ts_norm): eager torch activation, no Triton kernel (GEMMs upstream stay fused)."""
    if code_hint == 7:
        return _ts_norm_eager(gate_up)
    M, twoI = gate_up.shape; I = twoI // 2
    ra = _ones(M, gate_up.device) if row_alpha is None else row_alpha
    rg = _ones(M, gate_up.device) if row_gamma is None else row_gamma
    out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
    if I <= _ROWFUSE_MAX_I:
        if M > 0:
            BLOCK_I = max(16, triton.next_power_of_2(I))
            _glu_fwd_row_kernel[(M,)](gate_up, row_act, ra, rg, out, I,
                                      gate_up.stride(0), gate_up.stride(1), out.stride(0), out.stride(1),
                                      _ap_stride(row_alpha),
                                      EPS=_NS_EPS, BLOCK_I=BLOCK_I, num_warps=(8 if BLOCK_I >= 1024 else 4))
        return out
    skip_ns = code_hint is not None and code_hint not in (2, 6)   # codes 2 & 6 both need rms
    rms = _ones(M, gate_up.device) if skip_ns else _row_rms(gate_up, row_act, I)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_fwd_kernel[grid](gate_up, row_act, rms, ra, rg, out, M, I, gate_up.stride(0), gate_up.stride(1),
                          out.stride(0), out.stride(1), _ap_stride(row_alpha),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    return out


def _glu_bwd(grad_out, gate_up, row_act, code_hint=None, row_alpha=None, row_gamma=None,
             want_situ_grads=False):
    """Returns ggu, or (ggu, da_rows, dg_rows) when want_situ_grads (requires row_alpha/row_gamma).
    Row-fused path (I <= _ROWFUSE_MAX_I): ONE kernel computes grad_gate_up + the SiTU per-row
    param grads in the same pass; tiled fallback uses the pre-pass kernels + _row_situ_bwd.
    code_hint==7 (ts_norm): eager-activation backward via autograd (no Triton, no hand-derived math)."""
    if code_hint == 7:
        gu = gate_up.detach().requires_grad_(True)
        with torch.enable_grad():
            inter = _ts_norm_eager(gu)
        ggu_c7, = torch.autograd.grad(inter, gu, grad_out.to(inter.dtype))
        return (ggu_c7, None, None) if want_situ_grads else ggu_c7
    M, twoI = gate_up.shape; I = twoI // 2
    ra = _ones(M, gate_up.device) if row_alpha is None else row_alpha
    rg = _ones(M, gate_up.device) if row_gamma is None else row_gamma
    ggu = torch.empty_like(gate_up)
    if I <= _ROWFUSE_MAX_I:
        if want_situ_grads:
            da = torch.empty(M, device=gate_up.device, dtype=torch.float32)
            dg = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        else:
            da = dg = gate_up                 # dead pointers: WANT_AP=0 compiles the stores out
        if M > 0:
            BLOCK_I = max(16, triton.next_power_of_2(I))
            _glu_bwd_row_kernel[(M,)](grad_out, gate_up, row_act, ra, rg, ggu, da, dg, I,
                                      grad_out.stride(0), grad_out.stride(1),
                                      gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                                      _ap_stride(row_alpha),
                                      EPS=_NS_EPS, WANT_AP=want_situ_grads, BLOCK_I=BLOCK_I,
                                      num_warps=(8 if BLOCK_I >= 1024 else 4))
        return (ggu, da, dg) if want_situ_grads else ggu
    skip_ns = code_hint is not None and code_hint not in (2, 6)   # codes 2 & 6 both need rms + S/T coupling
    if skip_ns:
        rms = _ones(M, gate_up.device)      # r=1 / S=0 / T=0 semantics; values unread where at not in {2,6}
        sbuf = tbuf = rms
    else:
        rms = _row_rms(gate_up, row_act, I)  # recompute (one gate-half read) — keeps ctx/signatures unchanged
        sbuf = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        tbuf = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        if M > 0:
            _row_s_kernel[(M,)](grad_out, gate_up, row_act, rms, sbuf, tbuf, I,
                                grad_out.stride(0), grad_out.stride(1),
                                gate_up.stride(0), gate_up.stride(1), BLOCK_I=_NS_BLOCK_I)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_bwd_kernel[grid](grad_out, gate_up, row_act, rms, sbuf, tbuf, ra, rg, ggu, M, I,
                          grad_out.stride(0), grad_out.stride(1),
                          gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                          _ap_stride(row_alpha),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    if want_situ_grads:
        da, dg = _row_situ_bwd(grad_out, gate_up, row_act, ra, rg)
        return ggu, da, dg
    return ggu


def _row_situ_bwd(grad_out, gate_up, row_act, row_alpha, row_gamma):
    """Per-row SiTU dalpha/dgamma sums (fp32, (M,) each; zero for at!=5 rows)."""
    M, twoI = gate_up.shape; I = twoI // 2
    da = torch.empty(M, device=gate_up.device, dtype=torch.float32)
    dg = torch.empty(M, device=gate_up.device, dtype=torch.float32)
    if M > 0:
        _row_situ_bwd_kernel[(M,)](grad_out, gate_up, row_act, row_alpha, row_gamma, da, dg, I,
                                   grad_out.stride(0), grad_out.stride(1),
                                   gate_up.stride(0), gate_up.stride(1), _ap_stride(row_alpha),
                                   BLOCK_I=_NS_BLOCK_I)
    return da, dg


class BatchedGLU(torch.autograd.Function):
    """PolyGLU activation: out = act_{row}(gate) * up, with a per-row activation code.
    Optional per-row SiTU scalars (code 5): grads returned PER ROW (fp32) — build row_alpha/row_gamma
    differentiably (e.g. repeat_interleave of an (E,) param) and autograd sums them per expert."""
    @staticmethod
    def forward(ctx, gate_up, row_act, row_alpha=None, row_gamma=None):
        if (row_alpha is None) != (row_gamma is None):
            raise ValueError("row_alpha and row_gamma must be passed together (both or neither)")
        ctx.save_for_backward(gate_up, row_act,
                              row_alpha if row_alpha is not None else torch.empty(0),
                              row_gamma if row_gamma is not None else torch.empty(0))
        ctx.has_situ = row_alpha is not None
        return _glu_fwd(gate_up, row_act, row_alpha=row_alpha, row_gamma=row_gamma)

    @staticmethod
    def backward(ctx, grad_out):
        gate_up, row_act, row_alpha, row_gamma = ctx.saved_tensors
        ra = row_alpha if ctx.has_situ else None
        rg = row_gamma if ctx.has_situ else None
        go = grad_out   # kernels take explicit strides: broadcast/expanded grads work un-materialized
        if ctx.has_situ and (ctx.needs_input_grad[2] or ctx.needs_input_grad[3]):
            ggu, da, dg = _glu_bwd(go, gate_up, row_act, row_alpha=ra, row_gamma=rg,
                                   want_situ_grads=True)
            return ggu, None, da, dg
        return _glu_bwd(go, gate_up, row_act, row_alpha=ra, row_gamma=rg), None, None, None


# ───────────────────────── grouped GEMM kernels ─────────────────────────
@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 128, "BLOCK_K": 64}, num_warps=8, num_stages=2),
    ], key=["K", "N"])
@triton.jit
def _grouped_mm_kernel(X_ptr, W_ptr, Out_ptr, TileExpert_ptr, TileStart_ptr, ExpertEnd_ptr,
                       K, N, stride_xm, stride_xk, stride_we, stride_wn, stride_wk,
                       stride_om, stride_on,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_t = tl.program_id(0); pid_n = tl.program_id(1)
    e = tl.load(TileExpert_ptr + pid_t); m0 = tl.load(TileStart_ptr + pid_t); m_end = tl.load(ExpertEnd_ptr + e)
    offs_m = m0 + tl.arange(0, BLOCK_M); offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < m_end; mask_n = offs_n < N
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    W_e = W_ptr + e * stride_we
    for k0 in range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K); mask_k = offs_k < K
        x = tl.load(X_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        w = tl.load(W_e + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
                    mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    tl.store(Out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
             acc.to(Out_ptr.dtype.element_ty), mask=mask_m[:, None] & mask_n[None, :])


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64, "BLOCK_M": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128, "BLOCK_M": 64}, num_warps=4, num_stages=2),
    ], key=["N", "K"])
@triton.jit
def _grouped_wgrad_kernel(A_ptr, B_ptr, GW_ptr, ExpertStart_ptr, ExpertEnd_ptr, N, K,
                          stride_am, stride_an, stride_bm, stride_bk, stride_ge, stride_gn, stride_gk,
                          BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, BLOCK_M: tl.constexpr):
    pid_e = tl.program_id(0); pid_n = tl.program_id(1); pid_k = tl.program_id(2)
    m_start = tl.load(ExpertStart_ptr + pid_e); m_end = tl.load(ExpertEnd_ptr + pid_e)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N); offs_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_n = offs_n < N; mask_k = offs_k < K
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    m = m_start
    while m < m_end:
        offs_m = m + tl.arange(0, BLOCK_M); mask_m = offs_m < m_end
        a = tl.load(A_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an,
                    mask=mask_m[:, None] & mask_n[None, :], other=0.0)
        b = tl.load(B_ptr + offs_m[:, None] * stride_bm + offs_k[None, :] * stride_bk,
                    mask=mask_m[:, None] & mask_k[None, :], other=0.0)
        acc += tl.dot(tl.trans(a), b)
        m += BLOCK_M
    tl.store(GW_ptr + pid_e * stride_ge + offs_n[:, None] * stride_gn + offs_k[None, :] * stride_gk,
             acc.to(GW_ptr.dtype.element_ty), mask=mask_n[:, None] & mask_k[None, :])


def _build_schedule(counts, bounds, E, device, block_m=SCHED_BLOCK_M):
    tile_expert, tile_start = [], []
    for e in range(E):
        for ti in range((counts[e] + block_m - 1) // block_m):
            tile_expert.append(e); tile_start.append(bounds[e] + ti * block_m)
    if not tile_expert:
        return None, None
    return (torch.tensor(tile_expert, dtype=torch.int32, device=device),
            torch.tensor(tile_start, dtype=torch.int32, device=device))


def _grouped_mm(x_sorted, W, te, ts, e_end, N, trans_w=False):
    M, K = x_sorted.shape
    out = torch.empty(M, N, device=x_sorted.device, dtype=x_sorted.dtype)
    s_we, s_wn, s_wk = (W.stride(0), W.stride(2), W.stride(1)) if trans_w else (W.stride(0), W.stride(1), W.stride(2))
    grid = lambda meta: (te.numel(), triton.cdiv(N, meta["BLOCK_N"]))
    _grouped_mm_kernel[grid](x_sorted, W, out, te, ts, e_end, K, N, x_sorted.stride(0), x_sorted.stride(1),
                             s_we, s_wn, s_wk, out.stride(0), out.stride(1))
    return out


def _grouped_wgrad(A, B, e_start, e_end, E, N, K):
    gW = torch.zeros(E, N, K, device=A.device, dtype=A.dtype)
    grid = lambda meta: (E, triton.cdiv(N, meta["BLOCK_N"]), triton.cdiv(K, meta["BLOCK_K"]))
    _grouped_wgrad_kernel[grid](A, B, gW, e_start, e_end, N, K, A.stride(0), A.stride(1),
                                B.stride(0), B.stride(1), gW.stride(0), gW.stride(1), gW.stride(2))
    return gW


def _sort_by_expert(idx, wt, E):
    ntok, top_k = idx.shape
    flat_t = torch.arange(ntok, device=idx.device).unsqueeze(1).expand_as(idx).flatten()
    sorted_e, order = idx.flatten().sort()
    counts = torch.bincount(sorted_e, minlength=E).tolist()
    bounds = [0]
    for c in counts:
        bounds.append(bounds[-1] + c)
    return flat_t[order], wt.flatten()[order], order, counts, bounds


# ───────────────────────── grouped path ─────────────────────────
class _GroupedMoE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, idx, wt, gate_up_proj, down_proj, act_codes):
        x, wt, gate_up_proj, down_proj = _amp_cast(x, wt, gate_up_proj, down_proj)
        ntok, H = x.shape
        top_k = idx.shape[1]; E = gate_up_proj.shape[0]; I = gate_up_proj.shape[1] // 2
        dev = x.device
        st, sw, order, counts, bounds = _sort_by_expert(idx, wt, E)
        e_start = torch.tensor(bounds[:E], dtype=torch.int32, device=dev)
        e_end = torch.tensor(bounds[1:], dtype=torch.int32, device=dev)
        te, ts = _build_schedule(counts, bounds, E, dev)
        counts_t = torch.tensor(counts, device=dev)
        row_act = torch.repeat_interleave(act_codes, counts_t).to(torch.int32)
        x_s = x[st].contiguous()
        gate_up = _grouped_mm(x_s, gate_up_proj, te, ts, e_end, 2 * I)
        inter = _glu_fwd(gate_up, row_act)
        eo = _grouped_mm(inter, down_proj, te, ts, e_end, H)
        out = torch.zeros(ntok, H, device=dev, dtype=torch.float32)   # fp32 accumulate (MiMo)
        out.index_add_(0, st, (eo * sw.unsqueeze(-1)).float())
        out = out.to(x.dtype)
        ctx.save_for_backward(x_s, gate_up, inter, eo, st, sw, order, te, ts, e_start, e_end,
                              row_act, gate_up_proj, down_proj)
        ctx.shapes = (ntok, H, I, top_k, E)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        (x_s, gate_up, inter, eo, st, sw, order, te, ts, e_start, e_end, row_act,
         gate_up_proj, down_proj) = ctx.saved_tensors
        ntok, H, I, top_k, E = ctx.shapes
        go_s = grad_out[st].contiguous()
        grad_w_s = (go_s.float() * eo.float()).sum(-1).to(grad_out.dtype)
        grad_eo = go_s * sw.unsqueeze(-1)
        grad_inter = _grouped_mm(grad_eo, down_proj, te, ts, e_end, I, trans_w=True)
        grad_down_proj = _grouped_wgrad(grad_eo, inter, e_start, e_end, E, H, I)
        grad_gate_up = _glu_bwd(grad_inter, gate_up, row_act)
        grad_x_s = _grouped_mm(grad_gate_up, gate_up_proj, te, ts, e_end, H, trans_w=True)
        grad_gate_up_proj = _grouped_wgrad(grad_gate_up, x_s, e_start, e_end, E, 2 * I, H)
        grad_x = torch.zeros(ntok, H, device=grad_out.device, dtype=grad_out.dtype)
        grad_x.index_add_(0, st, grad_x_s)
        grad_wt = torch.zeros(ntok * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = grad_w_s
        return grad_x, None, grad_wt.view(ntok, top_k), grad_gate_up_proj, grad_down_proj, None


def moe_grouped(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    """Block-scheduled grouped-GEMM PolyGLU MoE. hidden (N,H), indices/weights (N,k),
    gate_up_proj (E,2I,H), down_proj (E,H,I), act_codes (E,) int32 -> (N,H).
    Code 5 (SiTU) is rejected — it needs act_params, which only the per-expert path carries. Codes
    3/4 are ACCEPTED for legacy diagnostics (grouped_parity.py / bench.py) but produce the documented
    wrong grads on specials stacks; moe() never routes them here."""
    if _code_max(act_codes) > 4:
        raise ValueError("codes >4 (SiTU code 5 / radial NormSiLU code 6) unsupported on the grouped path; "
                         "use moe_per_expert")
    return _GroupedMoE.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)


# ── candidate: grouped GEMM via torch._grouped_mm (cuBLAS, GPU-resident, no host sync) ──
# This is the Turing candidate: a cuBLAS grouped GEMM instead of the slow tl.dot one, with the
# dispatch built entirely on-GPU (cumsum offsets, no .tolist()/Python schedule loop). Composition
# of autograd-native ops, so no custom backward — IF torch._grouped_mm is differentiable. Requires
# torch with _grouped_mm (>= ~2.5/2.8); raises otherwise (the bench catches it and reports FAILED).
def moe_grouped_cublas(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes):
    # NOTE (measured T4, round 1): torch._grouped_mm is **bf16/fp8-only** and needs bf16 tensor
    # cores → **sm_80+ (Ampere/Hopper) only**. On Turing (T4, sm_75) it cannot run; we hard-skip.
    # Where it DOES run, the grouped GEMM is cuBLAS + the whole dispatch is GPU-resident (cumsum
    # offsets, no .tolist()/Python loop). GEMMs go through bf16 (cast in/out); grad flows through
    # _grouped_mm (autograd-native). UNTESTED end-to-end — no sm_80+ box in the loop yet.
    if not hasattr(torch, "_grouped_mm"):
        raise RuntimeError("torch._grouped_mm unavailable in this torch build")
    if torch.cuda.get_device_capability(hidden.device)[0] < 8:
        raise RuntimeError("torch._grouped_mm needs bf16 tensor cores (sm_80+); skipped on this GPU")
    if _code_max(act_codes) > 4:
        raise ValueError("codes >4 (SiTU code 5 / radial NormSiLU code 6) unsupported on the grouped-cublas path; "
                         "use moe_per_expert")
    N, H = hidden.shape
    E = gate_up_proj.shape[0]
    flat_t = torch.arange(N, device=hidden.device).unsqueeze(1).expand_as(top_k_indices).flatten()
    sorted_e, order = top_k_indices.flatten().sort()
    st = flat_t[order]
    sw = top_k_weights.flatten()[order]
    counts = torch.bincount(sorted_e, minlength=E)                        # GPU
    offs = counts.cumsum(0).to(torch.int32)                               # GPU end-exclusive offsets
    row_act = torch.repeat_interleave(act_codes, counts).to(torch.int32)  # GPU
    x_s = hidden[st].contiguous()                                         # (M,H)
    bf = torch.bfloat16
    gate_up = torch._grouped_mm(x_s.to(bf), gate_up_proj.transpose(-2, -1).to(bf), offs=offs).to(hidden.dtype)
    inter = BatchedGLU.apply(gate_up, row_act)                            # (M,I) in model dtype
    eo = torch._grouped_mm(inter.to(bf), down_proj.transpose(-2, -1).to(bf), offs=offs).to(hidden.dtype)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)   # fp32 accumulate (MiMo)
    out.index_add_(0, st, (eo * sw.unsqueeze(-1)).float())
    return out.to(hidden.dtype)


# ───────────────────────── fused weighted scatter / gather (combine tail) ─────────────────────────
# Forward combine per expert was: (eo * w).float() then index_add_ = 3 kernels (mul, cast, scatter)
# + 2 transient (m,H) tensors. Fuse into ONE scatter kernel: read eo(fp16)+w, scale in fp32,
# atomic-add into the fp32 out. Within an expert st[s:en] is UNIQUE (top-k = distinct experts),
# so the fp32 atomics never contend -> bit-deterministic, equals index_add.
@triton.jit
def _combine_scatter_kernel(EO_ptr, W_ptr, Tok_ptr, Out_ptr, m, H, s_eo_m, s_eo_h, s_out_n, s_out_h,
                            BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_m = tl.program_id(0); pid_h = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_h = pid_h * BLOCK_H + tl.arange(0, BLOCK_H)
    mask_m = offs_m < m; mask = mask_m[:, None] & (offs_h < H)[None, :]
    eo = tl.load(EO_ptr + offs_m[:, None] * s_eo_m + offs_h[None, :] * s_eo_h, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    tok = tl.load(Tok_ptr + offs_m, mask=mask_m, other=0)
    tl.atomic_add(Out_ptr + tok[:, None] * s_out_n + offs_h[None, :] * s_out_h, eo * w, mask=mask)


# Backward combine per expert: grad_eo = grad_out[tok] * w ; grad_w = sum_h(grad_out[tok] * eo).
# One kernel: gather grad_out[tok], emit grad_eo (m,H) + grad_w (m,). Replaces gather+mul+mul+reduce.
# BLOCK_H spans the full H (one block) so the grad_w row-reduction is complete (H<=~1024 fits).
@triton.jit
def _combine_bwd_kernel(GO_ptr, EO_ptr, W_ptr, Tok_ptr, GradEO_ptr, GradW_ptr, m, H,
                        s_go_n, s_go_h, s_eo_m, s_eo_h, s_geo_m, s_geo_h,
                        BLOCK_M: tl.constexpr, BLOCK_H: tl.constexpr):
    pid_m = tl.program_id(0)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); offs_h = tl.arange(0, BLOCK_H)
    mask_m = offs_m < m; mask = mask_m[:, None] & (offs_h < H)[None, :]
    tok = tl.load(Tok_ptr + offs_m, mask=mask_m, other=0)
    go = tl.load(GO_ptr + tok[:, None] * s_go_n + offs_h[None, :] * s_go_h, mask=mask, other=0.0).to(tl.float32)
    eo = tl.load(EO_ptr + offs_m[:, None] * s_eo_m + offs_h[None, :] * s_eo_h, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(W_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    tl.store(GradEO_ptr + offs_m[:, None] * s_geo_m + offs_h[None, :] * s_geo_h,
             (go * w).to(GradEO_ptr.dtype.element_ty), mask=mask)
    tl.store(GradW_ptr + offs_m, tl.sum(go * eo, axis=1), mask=mask_m)


def _combine_scatter(eo, w, tok, out):
    m, H = eo.shape
    BLOCK_M = max(16, min(64, triton.next_power_of_2(m))); BLOCK_H = max(16, min(128, triton.next_power_of_2(H)))
    grid = (triton.cdiv(m, BLOCK_M), triton.cdiv(H, BLOCK_H))
    _combine_scatter_kernel[grid](eo, w, tok, out, m, H, eo.stride(0), eo.stride(1),
                                  out.stride(0), out.stride(1), BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H)


def _combine_bwd(grad_out, eo, w, tok):
    m, H = eo.shape
    grad_eo = torch.empty_like(eo)
    grad_w = torch.empty(m, device=eo.device, dtype=torch.float32)   # fp32 reduction (MiMo)
    BLOCK_M = 16; BLOCK_H = triton.next_power_of_2(H)
    _combine_bwd_kernel[(triton.cdiv(m, BLOCK_M),)](grad_out, eo, w, tok, grad_eo, grad_w, m, H,
                        grad_out.stride(0), grad_out.stride(1), eo.stride(0), eo.stride(1),
                        grad_eo.stride(0), grad_eo.stride(1), BLOCK_M=BLOCK_M, BLOCK_H=BLOCK_H)
    return grad_eo, grad_w


# ───────────────────────── per-expert path (custom backward, cuBLAS GEMMs) ─────────────────────────
class _PerExpertMoE(torch.autograd.Function):
    """Sorted dispatch + per-expert cuBLAS GEMMs + fused PolyGLU + weighted fp32 scatter, with a
    MANUAL backward. The autograd-native composition (the old body) auto-generated backward and let
    autograd insert grad-accumulation add_ (~32/iter) and buffer fill_ (~40/iter) glue = ~21% of
    fwd+bwd time on T4. The manual backward does only the essential kernels: per-expert dX/dW GEMMs +
    the existing _glu_bwd + ONE index_add_ for grad_hidden (not 9 scatter-adds). Backward still has
    2× the GEMMs of forward (dX AND dW per fwd GEMM — irreducible matmul autodiff), but no glue."""

    @staticmethod
    def forward(ctx, hidden, idx, wt, gate_up_proj, down_proj, act_codes, act_params=None):
        # AMP-safe, dtype-agnostic: cast float args to the ACTIVE autocast dtype (fp16 or bf16) so
        # forward GEMMs, saved tensors, and the manual backward stay dtype-consistent. Without this,
        # autocast rewrote the fwd GEMMs while saving fp32 weights -> mixed-dtype backward.
        # act_params (E,2) fp32 [alpha,gamma] stays fp32 (scalar params, precision-sensitive).
        hidden, wt, gate_up_proj, down_proj = _amp_cast(hidden, wt, gate_up_proj, down_proj)
        N, H = hidden.shape
        E = act_codes.shape[0]                  # total routed experts (GLU + specials)
        codes = act_codes.tolist()              # 0/1/2/5 = GLU (weight slot e), 3 = Identity, 4 = Zero
        top_k = idx.shape[1]; dev = hidden.device
        st, sw, order, counts, bounds = _sort_by_expert(idx, wt, E)
        x_s = hidden.index_select(0, st)                                  # (M,H) contiguous gather
        counts_t = torch.tensor(counts, device=dev)
        M_rows = idx.numel()   # output_size: statically known -> repeat_interleave skips its host sync
        row_act = torch.repeat_interleave(act_codes, counts_t, output_size=M_rows).to(torch.int32)
        # SiTU (code 5) is tanh(g)*sigmoid(g) — parameter-free like the rest of the menu (act_params
        # None => alpha=gamma=1, the DEFAULT). Passing act_params (E,2) [alpha,gamma] enables the
        # learnable variant; uniform slices broadcast the expert's scalars (stride-0, no row tensors).
        ap32 = act_params.float().contiguous() if act_params is not None else None
        # per-expert activations kept as LISTS — the GEMM outputs ARE the storage; no contiguous
        # buffer + slice-copy (that was a pure DtoD-memcpy + memory tax).
        gate_up_l = [None] * E; inter_l = [None] * E; eo_l = [None] * E
        out = torch.zeros(N, H, device=dev, dtype=torch.float32)          # fp32 accumulate (MiMo)
        for e in range(E):
            s, en = bounds[e], bounds[e + 1]
            if en == s:
                continue
            if codes[e] == 4:                                            # Zero expert: contributes nothing
                continue
            if codes[e] == 3:                                            # Identity: weighted passthrough of input
                _combine_scatter(x_s[s:en], sw[s:en], st[s:en], out)
                continue
            gu = x_s[s:en] @ gate_up_proj[e].t()                         # GLU expert; weight slot = e (GLU first)
            _has_ap = codes[e] == 5 and ap32 is not None
            it = _glu_fwd(gu, row_act[s:en], code_hint=codes[e],   # uniform slice: non-NS skips rms launch
                          row_alpha=(ap32[e, 0:1] if _has_ap else None),
                          row_gamma=(ap32[e, 1:2] if _has_ap else None))
            eo = it @ down_proj[e].t()
            gate_up_l[e] = gu; inter_l[e] = it; eo_l[e] = eo
            _combine_scatter(eo, sw[s:en], st[s:en], out)   # fused (eo*w)->fp32 scatter (1 kernel, was mul+cast+index_add)
        ctx.save_for_backward(x_s, st, sw, order, row_act, gate_up_proj, down_proj,
                              ap32 if ap32 is not None else torch.empty(0))
        ctx.lists = (gate_up_l, inter_l, eo_l); ctx.bounds = bounds; ctx.shapes = (N, H, top_k, E)
        ctx.codes = codes; ctx.has_situ = ap32 is not None
        return out.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x_s, st, sw, order, row_act, gate_up_proj, down_proj, ap32) = ctx.saved_tensors
        gate_up_l, inter_l, eo_l = ctx.lists
        N, H, top_k, E = ctx.shapes; bounds = ctx.bounds; codes = ctx.codes
        M = st.numel()
        grad_w_s = torch.zeros(M, device=grad_out.device, dtype=grad_out.dtype)
        grad_gate_up_proj = torch.zeros_like(gate_up_proj)
        grad_down_proj = torch.zeros_like(down_proj)
        grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
        want_ap = ctx.has_situ and ctx.needs_input_grad[6]   # skip param-grad work when alpha/gamma frozen
        grad_act_params = (torch.zeros(E, 2, device=grad_out.device, dtype=torch.float32)
                           if want_ap else None)
        for e in range(E):
            s, en = bounds[e], bounds[e + 1]
            if en == s:
                continue
            if codes[e] == 4:                                          # Zero: no grad
                continue
            if codes[e] == 3:                                          # Identity: out=w*input -> dx=w*go, dw=sum(go*input)
                ge, gw = _combine_bwd(grad_out, x_s[s:en], sw[s:en], st[s:en])
                grad_w_s[s:en] = gw.to(grad_out.dtype)
                grad_hidden.index_add_(0, st[s:en], ge)
                continue
            it = inter_l[e]                                            # (m,I)
            # fused combine bwd: gather grad_out[tok], emit grad_eo=go*w and grad_w=sum_h(go*eo)
            ge, gw = _combine_bwd(grad_out, eo_l[e], sw[s:en], st[s:en])   # (m,H), (m,) fp32
            grad_w_s[s:en] = gw.to(grad_out.dtype)
            grad_inter = ge @ down_proj[e]                              # (m,H)@(H,I) -> (m,I)
            grad_down_proj[e] = ge.t() @ it                            # (H,m)@(m,I) -> (H,I)
            is_situ = codes[e] == 5
            if is_situ and want_ap:
                grad_gate_up, da, dg = _glu_bwd(grad_inter, gate_up_l[e], row_act[s:en],
                                                code_hint=5, row_alpha=ap32[e, 0:1],
                                                row_gamma=ap32[e, 1:2], want_situ_grads=True)
                grad_act_params[e, 0] = da.sum()
                grad_act_params[e, 1] = dg.sum()
            else:
                _has_ap = is_situ and ctx.has_situ
                grad_gate_up = _glu_bwd(grad_inter, gate_up_l[e], row_act[s:en], code_hint=codes[e],
                                        row_alpha=(ap32[e, 0:1] if _has_ap else None),
                                        row_gamma=(ap32[e, 1:2] if _has_ap else None))   # (m,2I)
            grad_gate_up_proj[e] = grad_gate_up.t() @ x_s[s:en]        # (2I,m)@(m,H) -> (2I,H)
            grad_hidden.index_add_(0, st[s:en], grad_gate_up @ gate_up_proj[e])  # scatter dX straight in
        grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
        grad_wt[order] = grad_w_s
        return (grad_hidden, None, grad_wt.view(N, top_k), grad_gate_up_proj, grad_down_proj, None,
                grad_act_params)


def moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
                   act_params=None):
    """Sorted dispatch + cuBLAS GEMMs + fused PolyGLU activation + weighted scatter, MANUAL backward
    (no autograd-composition glue). Wins at low token counts.
    act_params: (E,2) fp32 [alpha,gamma] per expert — required iff any act code is 5 (SiTU)."""
    return _PerExpertMoE.apply(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj,
                               act_codes, act_params)


def moe(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes, act_params=None):
    """Auto: grouped at >= GROUPED_MIN_TOKENS rows (N*top_k) on Ampere+ (sm_80+) AND only when every
    expert is a GLU (act_codes in {0,1,2}); else per-expert (which alone supports codes 3/4/5).

    The grouped path's tl.dot GEMMs are catastrophic on Turing (T4, sm_75) — measured ~0.1x vs compiled
    eager — so it is NEVER chosen on sm_<80; per-expert (cuBLAS) wins there. The grouped path also does
    NOT implement the Identity (code 3) / Zero (code 4) special experts — it runs GLU over every expert
    uniformly — so it is correct ONLY for pure-GLU stacks. With a special expert present it produces
    wrong output and gradients (measured: grad rel ~1.6e+03 on the 9-GLU+Identity+Zero stack), so we
    fall back to the per-expert path (which handles codes 3/4 in fwd and bwd) whenever a special expert
    is in the stack. per-expert is correct on every arch and is itself a large win (T4 ~2.9x; Blackwell
    ~4x fwd+bwd). To use grouped on a mixed stack, fix _GroupedMoE to special-case codes 3/4 first."""
    cap_major = torch.cuda.get_device_capability(hidden.device)[0]
    glu_only = _code_max(act_codes) <= 2       # codes 3/4 (specials) AND 5 (SiTU) -> per-expert; cached, no per-call sync
    if top_k_indices.numel() >= GROUPED_MIN_TOKENS and cap_major >= 8 and glu_only:
        return moe_grouped(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes)
    return moe_per_expert(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
                          act_params)


# ───────────────────────── naive eager reference (the slow baseline) ─────────────────────────
def _act_eager(gate, code, alpha=None, gamma=None):
    if code == 0:
        return F.silu(gate)
    if code == 1:
        return F.relu(gate) ** 2
    if code == 5:
        # SiTU: gamma * tanh(alpha*g) * sigmoid(g), fp32 math, learnable per-expert scalars
        g = gate.float()
        return (gamma * torch.tanh(alpha * g) * torch.sigmoid(g)).to(gate.dtype)
    if code == 6:
        # radial NormSiLU: sqrt(r) * SiLU(g/r), r = rms(gate) — restores half the gate radius (p=0.5)
        g = gate.float()
        r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
        return (torch.sqrt(r) * F.silu(g / r)).to(gate.dtype)
    if code == 7:
        # ts_norm: tanh(sigmoid(g/rms(g))), bounded normalized composition. EAGER-ONLY (no fused kernel
        # yet) — the moe_per_expert Triton path does NOT handle code 7; route code-7 stacks via moe_eager.
        g = gate.float()
        z = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
        return torch.tanh(torch.sigmoid(z)).to(gate.dtype)
    # NormSiLU: SiLU(rms-normed gate), gain-free — matches BiBo eager (_NORMSILU_EPS)
    g = gate.float()
    g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
    return F.silu(g).to(gate.dtype)


def moe_eager(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
              act_params=None):
    """Hand-written MoE: per-expert boolean mask (GPU sync each iter), unfused activation,
    E tiny GEMMs. Correct, and deliberately the slow baseline the fused paths beat."""
    N, H = hidden.shape
    twoI = gate_up_proj.shape[1]
    I = twoI // 2
    codes = act_codes.tolist()          # 0/1/2/5 = GLU (weight slot e), 3 = Identity, 4 = Zero
    E = len(codes)                       # total routed experts (GLU + specials)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)   # fp32 accumulate (MiMo)
    for e in range(E):
        rows = (top_k_indices == e).any(-1)
        if not bool(rows.any()):
            continue
        w = (top_k_weights * (top_k_indices == e)).sum(-1)[rows]
        if codes[e] == 4:                                                # Zero: contributes nothing
            continue
        if codes[e] == 3:                                                # Identity: weighted passthrough
            out[rows] += (hidden[rows] * w.unsqueeze(-1)).float()
            continue
        gate_up = hidden[rows] @ gate_up_proj[e].t()
        a, g = ((act_params[e, 0], act_params[e, 1]) if codes[e] == 5 and act_params is not None
                else (1.0, 1.0))
        inter = _act_eager(gate_up[:, :I], codes[e], a, g) * gate_up[:, I:]
        out[rows] += ((inter @ down_proj[e].t()) * w.unsqueeze(-1)).float()
    return out.to(hidden.dtype)
