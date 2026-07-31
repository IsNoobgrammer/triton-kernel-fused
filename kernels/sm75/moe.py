"""Fused PolyGLU MoE experts — the hard one. Per-expert + grouped paths + naive-eager reference.

PolyGLU: experts are GLU MLPs with *heterogeneous* activations — each expert carries an
activation code: 0=SiLU, 2=NormSiLU, 8=radial NormSiLU (+ 3/4 = the ±Identity specials).
Pass an `act_codes` (E,) int32 tensor alongside the expert weights.

SPECIAL (param-free, no GEMM) codes 3 and 4 are NOT activations — they mark an expert whose whole
body is a signed passthrough of the input: code 3 = +Identity (out += w*x), code 4 = -Identity
(out += -w*x). They own no slot in gate_up_proj/down_proj, so the GLU block must come FIRST in the
stack (weight slot == expert index) and the specials must be the trailing indices.
Code 4 was the ZERO expert (emit nothing) until Jul 26 2026. It was replaced because a Zero expert's
output is identically 0, so d(loss)/d(its router weight) = <grad_out, 0> = 0 — the router gets no
gradient telling it that picking Zero was right, and every Zero assignment is therefore driven by
the load balancer alone. The ± pair still spans "skip this layer" (equal weights cancel) but reaches
it with live gradient on both branches. LongCat-Flash (arXiv:2509.01322) likewise uses identity,
never zero, for its zero-COMPUTATION experts.

ACT MENU, Jul 31 2026 -- CUT TO THREE. Codes 1 (ReLU^2), 5 (SiTU), 6 (NormReLU^2), 7 (NormSiTU)
and 9 (decoupled gamma*SiLU(alpha*g)) are DELETED, and with them the per-expert `gamma` parameter,
which was live for code 9 alone. What remains: 0 = SiLU, 2 = NormSiLU, 8 = radial NormSiLU, plus the
parameter-free +/-Identity specials (3, 4), which are kept for the higher-scale special-expert work.
Numeric codes are deliberately UNCHANGED so existing checkpoints and run names still resolve.
Decided by the 1B-token four-way (bpb): radial 0.64313 < silu-a 0.64429 < normsilu 0.64646 <
silu 0.64768, against a 0.00037 same-seed floor. `act_params` is now (E,) or (E, 2) with only
column 0 (alpha) read.

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


# Cast cache for the big 3D EXPERT STACKS only, keyed on (storage, version, dtype).
# The expert weights are fp32 masters that change ONCE per optimizer step, but _amp_cast ran on
# every micro-batch: with grad_accum=4 that is 3 redundant casts of ~300 MB per MoE layer.
# Profiling put bfloat16_copy_kernel at 7.87 ms/micro-batch. Tensor._version bumps on any in-place
# write, so an optimizer step invalidates the entry automatically -- a stale cast is not possible.
_CAST_CACHE = {}
try:                                            # Blackwell-only fused gate_up GEMM+GLU epilogue
    from kernels.sm120 import moe_fused_glu as _FUSED_GLU
except Exception:
    _FUSED_GLU = None


def _cached_cast(t, dt):
    key = t.untyped_storage().data_ptr()
    hit = _CAST_CACHE.get(key)
    if hit is not None and hit[0] == t._version and hit[1] is dt and hit[2].shape == t.shape:
        return hit[2]
    c = t.to(dt)
    _CAST_CACHE[key] = (t._version, dt, c)
    if len(_CAST_CACHE) > 256:                     # bounded: ~2 entries per MoE layer
        for k in list(_CAST_CACHE)[:128]:
            _CAST_CACHE.pop(k, None)
    return c


def _amp_cast(*ts):
    """Under autocast, cast float tensors to the ACTIVE autocast dtype (fp16/bf16) so the custom
    Functions see one consistent dtype end-to-end; no-op outside autocast. Grads returned for the
    cast tensors are dtype-converted back by the autograd engine at the Function boundary.

    3D expert stacks go through _cached_cast (see above); activations are cast fresh every call."""
    if torch.is_autocast_enabled("cuda"):
        dt = torch.get_autocast_dtype("cuda")
        return tuple((_cached_cast(t, dt) if (t.ndim == 3 and t.dtype != dt) else t.to(dt))
                     if t.is_floating_point() else t for t in ts)
    return ts


# ───────────────────────── PolyGLU activation (per-row act code) ─────────────────────────
# NormSiLU (code 2) needs a per-row RMS over the gate half — a row reduction the elementwise
# tile kernels can't see. One-program-per-row pre-pass kernels compute the row scalars; the rms
# is recomputed in backward from the saved gate_up (one cheap read; keeps _glu_fwd/_glu_bwd
# signatures and every autograd path's ctx unchanged).
@triton.jit
def _row_rms_kernel(GateUp_ptr, Act_ptr, Rms_ptr, I, s_gu_m, s_gu_i,
                    EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    # per-row RMS of the gate half; consumed where at in {2 (NormSiLU), 8 (radial)}, else r=1.
    row = tl.program_id(0)
    at = tl.load(Act_ptr + row)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        g = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=offs < I, other=0.0).to(tl.float32)
        acc += g * g
    rms = tl.sqrt(tl.sum(acc) / I + EPS)
    tl.store(Rms_ptr + row, tl.where((at == 2) | (at == 8), rms, 1.0))


@triton.jit
def _row_s_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Rms_ptr, S_ptr, I,
                  s_go_m, s_go_i, s_gu_m, s_gu_i, BLOCK_I: tl.constexpr):
    # RMS-coupling term S = sum_j go*up*silu'(ĝ)*ĝ, ĝ=gate/r. Both surviving normed codes (2, 8)
    # apply SiLU to ĝ, so there is a single f' -- the old per-code branch is gone with 6/7.
    # r=1 for non-normalized rows (from _row_rms) and S is READ only where at in {2,8}, so the
    # value computed for other rows is harmless.
    row = tl.program_id(0)
    r = tl.load(Rms_ptr + row)
    acc = tl.zeros([BLOCK_I], dtype=tl.float32)
    for i0 in range(0, I, BLOCK_I):
        offs = i0 + tl.arange(0, BLOCK_I)
        m = offs < I
        go = tl.load(GradOut_ptr + row * s_go_m + offs * s_go_i, mask=m, other=0.0).to(tl.float32)
        gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=m, other=0.0).to(tl.float32)
        up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=m, other=0.0).to(tl.float32)
        gn = gate / r
        sig = 1.0 / (1.0 + tl.exp(-gn))
        acc += go * up * (sig * (1.0 + gn * (1.0 - sig))) * gn
    tl.store(S_ptr + row, tl.sum(acc))


# ── row-fused (v2) GLU kernels: one program per row spans the FULL intermediate dim, so the
# NormSiLU rms, the backward's S-coupling term, and the per-expert dalpha row sum all happen
# IN-REGISTER in the same pass. 1 launch fwd + 1 launch bwd for every code — no pre-pass kernels,
# no extra HBM reads (fwd ~4N->3N, bwd ~9N->5N for NormSiLU). Used when I <= _ROWFUSE_MAX_I;
# larger I falls back to the tiled kernels + pre-pass path below (kept unchanged).
_ROWFUSE_MAX_I = 1024


@triton.jit
def _glu_fwd_row_kernel(GateUp_ptr, Act_ptr, Alpha_ptr, Out_ptr, I,
                        s_gu_m, s_gu_i, s_o_m, s_o_i, s_ap,
                        EPS: tl.constexpr, BLOCK_I: tl.constexpr):
    row = tl.program_id(0)
    offs = tl.arange(0, BLOCK_I)
    msk = offs < I
    gate = tl.load(GateUp_ptr + row * s_gu_m + offs * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    up = tl.load(GateUp_ptr + row * s_gu_m + (I + offs) * s_gu_i, mask=msk, other=0.0).to(tl.float32)
    at = tl.load(Act_ptr + row)
    aa = tl.load(Alpha_ptr + row * s_ap).to(tl.float32)
    r = tl.sqrt(tl.sum(gate * gate) / I + EPS)           # consumed where at in {2, 8}
    gn = tl.where((at == 2) | (at == 8), gate / r, gate)
    # code 8 = RADIAL NormSiLU: r^p * SiLU(g/r), p = sigmoid(alpha) in (0,1). NormSiLU discards the
    # per-token gate radius r entirely; this puts a BOUNDED fraction of it back. p must be bounded --
    # the toy round measured full p=1 (raw magnitude passthrough) as harmful and unbounded-learnable
    # as worse than fixed p=0.5, while bounded-learnable was the best arm. Confirmed at 1B tokens:
    # radial 0.64313 bpb vs normsilu 0.64646 vs silu 0.64768, and p learns a DEPTH RAMP 0.11 -> 0.93
    # so the layer behaves as normsilu early and as full magnitude late.
    # alpha is REUSED as the exponent logit here, so code 8 takes no input scale (z = gn below).
    p8 = 1.0 / (1.0 + tl.exp(-aa))
    rp = tl.exp(p8 * tl.log(r))                          # r^p
    # PER-EXPERT INPUT SCALE alpha: z = alpha * x, x = gate (code 0) or gate/r (code 2). For the
    # NORMED code alpha MUST sit AFTER the rms -- rms is positively homogeneous, so alpha*g/rms(alpha*g)
    # == g/rms(g) and scaling before the norm is exactly inert.
    # alpha == 1 gives z == gn bit-exactly, so both codes are byte-identical when the feature is off.
    z = tl.where(at == 8, gn, aa * gn)                    # 8 spends alpha on the exponent instead
    sig = 1.0 / (1.0 + tl.exp(-z))
    f = z * sig                                           # silu(z) -- the ONLY nonlinearity now
    act = tl.where(at == 8, rp * f, f)
    tl.store(Out_ptr + row * s_o_m + offs * s_o_i, (act * up).to(Out_ptr.dtype.element_ty), mask=msk)


@triton.jit
def _glu_bwd_row_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Alpha_ptr,
                        GradGateUp_ptr, DA_ptr, I,
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
    r = tl.sqrt(tl.sum(gate * gate) / I + EPS)
    is_norm = (at == 2) | (at == 8)
    gn = tl.where(is_norm, gate / r, gate)
    p8 = 1.0 / (1.0 + tl.exp(-aa))                        # code 8 radial exponent, bounded (0,1)
    lr8 = tl.log(r)
    rp = tl.exp(p8 * lr8)                                 # r^p
    rpm1 = tl.exp((p8 - 1.0) * lr8)                       # r^(p-1)
    z = tl.where(at == 8, gn, aa * gn)                    # see the forward kernel for why alpha
    sig = 1.0 / (1.0 + tl.exp(-z))                        # sits AFTER the rms for code 2
    f = z * sig
    df = sig * (1.0 + z * (1.0 - sig))                    # silu'(z)
    act = tl.where(at == 8, rp * f, f)
    gu_ = go * up
    # RMS coupling (code 2): d/dg = alpha*(gu*silu'(z) - (S/I)*ghat)/r, S = sum(gu*silu'(z)*ghat).
    # S uses GHAT, not z -- it comes from d(ghat)/dg, which alpha does not enter.
    S = tl.sum(tl.where(is_norm, gu_ * df * gn, 0.0))
    # code 8 adds the d(r^p)/dg path: with A_j = r^p*SiLU(ghat_j) and T = sum(gu*SiLU(ghat)),
    #   dL/dg_i = r^(p-1) * [gu_i*silu'(ghat_i) - (ghat_i/I)*(S - p*T)]
    # which is the code-2 coupling with S -> (S - p*T). MUST be tested before is_norm: 8 is in it.
    T = tl.sum(tl.where(at == 8, gu_ * f, 0.0))
    grad_gate = tl.where(at == 8, rpm1 * (gu_ * df - (gn / I) * (S - p8 * T)),
                    tl.where(is_norm, aa * (gu_ * df - (S / I) * gn) / r, aa * gu_ * df))
    tl.store(GradGateUp_ptr + row * s_ggu_m + offs * s_ggu_i, grad_gate, mask=msk)
    tl.store(GradGateUp_ptr + row * s_ggu_m + (I + offs) * s_ggu_i, go * act, mask=msk)
    if WANT_AP:
        # dalpha = d/dalpha sum_j gu_j * f(alpha*x_j) = sum_j gu_j * silu'(z_j) * x_j, x = gn --
        # which for code 2 is exactly the S reduction above.
        # code 8: alpha is the exponent LOGIT, so d/dalpha = dp/dalpha * dL/dp with
        # dL/dp = T*r^p*ln(r) and dp/dalpha = p(1-p). Reuses T -- nothing extra to compute.
        da8 = p8 * (1.0 - p8) * rp * lr8 * T
        tl.store(DA_ptr + row, tl.where(at == 8, da8, tl.sum(gu_ * df * gn)))


@triton.jit
def _glu_fwd_kernel(GateUp_ptr, Act_ptr, Rms_ptr, Alpha_ptr, Out_ptr, M, I,
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
    aa = tl.load(Alpha_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    gn = gate / r                                        # r == 1 for code 0 (see _row_rms_kernel)
    z = aa * gn
    sig = 1.0 / (1.0 + tl.exp(-z))
    act = z * sig                                        # SiLU is the only nonlinearity left
    tl.store(Out_ptr + offs_m[:, None] * s_o_m + offs_i[None, :] * s_o_i,
             (act * up).to(Out_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _glu_bwd_kernel(GradOut_ptr, GateUp_ptr, Act_ptr, Rms_ptr, S_ptr, Alpha_ptr,
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
    aa = tl.load(Alpha_ptr + offs_m * s_ap, mask=mask_m, other=1.0).to(tl.float32)[:, None]   # 1.0 where at!=5
    r = tl.load(Rms_ptr + offs_m, mask=mask_m, other=1.0).to(tl.float32)[:, None]
    sv = tl.load(S_ptr + offs_m, mask=mask_m, other=0.0).to(tl.float32)[:, None]
    gn = gate / r
    z = aa * gn
    sig = 1.0 / (1.0 + tl.exp(-z))
    act = z * sig
    dsilu = sig * (1.0 + z * (1.0 - sig))
    # code 2 carries the RMS coupling; code 0 has r == 1 and sv == 0 so the same line serves both.
    grad_gate = tl.where(at == 2, (go * up * dsilu - (sv / I) * gn) / r, go * up * dsilu) * aa
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + offs_i[None, :] * s_ggu_i, grad_gate, mask=mask)
    tl.store(GradGateUp_ptr + offs_m[:, None] * s_ggu_m + (I + offs_i)[None, :] * s_ggu_i, go * act, mask=mask)


_ONES_CACHE = {}          # device -> a big fp32 ones buffer, sliced for the alpha default


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


def _alpha_tile_gap(row_alpha, I, code_hint):
    """Why per-expert alpha (and therefore RADIAL) is capped at I <= _ROWFUSE_MAX_I.

    NOT because p is high-dimensional -- p is one scalar per expert and costs a single load. The
    binding constraint is `r = rms(gate over I)`, a ROW REDUCTION. The row-fused kernel owns a whole
    row per program and gets r from one in-register tl.sum; above _ROWFUSE_MAX_I the row no longer
    fits in registers, so the tiled kernels split each row across programs and r must come from the
    _row_rms pre-pass. That part already works.

    What actually breaks on the tiled path:
      * FORWARD is fine -- _glu_fwd_kernel applies z = alpha*gn today.
      * BACKWARD is not: _row_s_kernel builds the coupling S = sum(gu*silu'(g_hat)*g_hat) from
        sig(gn), NOT from sig(alpha*gn), so S is silently WRONG whenever alpha != 1.
      * RADIAL additionally needs a SECOND row reduction T = sum(gu*silu(g_hat)) for both its
        gate coupling (S -> S - p*T) and its dalpha; no such pre-pass exists.

    To lift the cap: teach _row_s_kernel to read alpha, add a _row_t_kernel clone for T, and add
    the at==8 branches to _glu_fwd_kernel/_glu_bwd_kernel."""
    if row_alpha is None or I <= _ROWFUSE_MAX_I:
        return
    raise NotImplementedError(
        f"per-expert alpha needs the ROW-FUSED GLU kernels (I <= {_ROWFUSE_MAX_I}); got I={I}. "
        "The tiled FORWARD already applies alpha; the blocker is that _row_s_kernel computes the "
        "backward coupling S from sig(gate/r) instead of sig(alpha*gate/r), and radial (code 8) "
        "also needs a T = sum(gu*silu(ghat)) pre-pass that does not exist. See _alpha_tile_gap. "
        + ("Code 8 (radial) ALWAYS carries alpha, so radial is capped here too."
           if code_hint == 8 else ""))


def _glu_fwd(gate_up, row_act, code_hint=None, row_alpha=None):
    """code_hint: host-side int when EVERY row shares one act code (the per-expert path) — lets
    non-NormSiLU slices skip the _row_rms launch on the tiled fallback path (row-fused path needs
    no pre-pass at all). row_alpha: per-row fp32 scalar -- the input scale for codes 0/2, the
    exponent LOGIT theta for code 8 (p = sigmoid(theta)); None -> 1.0."""
    M, twoI = gate_up.shape; I = twoI // 2
    _alpha_tile_gap(row_alpha, I, code_hint)
    if code_hint == 8 and row_alpha is None:
        raise ValueError(
            "act code 8 (radial NormSiLU, r^p*SiLU(g/r)) requires row_alpha -- it carries the "
            "exponent LOGIT theta, p=sigmoid(theta). With alpha absent the kernel would default to "
            "theta=1 => p=0.731 instead of the intended 0.5 init. Pass act_params.")
    ra = _ones(M, gate_up.device) if row_alpha is None else row_alpha
    out = torch.empty(M, I, device=gate_up.device, dtype=gate_up.dtype)
    if I <= _ROWFUSE_MAX_I:
        if M > 0:
            BLOCK_I = max(16, triton.next_power_of_2(I))
            _glu_fwd_row_kernel[(M,)](gate_up, row_act, ra, out, I,
                                      gate_up.stride(0), gate_up.stride(1), out.stride(0), out.stride(1),
                                      _ap_stride(row_alpha),
                                      EPS=_NS_EPS, BLOCK_I=BLOCK_I, num_warps=(8 if BLOCK_I >= 1024 else 4))
        return out
    skip_ns = code_hint is not None and code_hint not in (2, 6, 7, 8)  # 2/6/7/8 need the row RMS
    rms = _ones(M, gate_up.device) if skip_ns else _row_rms(gate_up, row_act, I)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_fwd_kernel[grid](gate_up, row_act, rms, ra, out, M, I, gate_up.stride(0), gate_up.stride(1),
                          out.stride(0), out.stride(1), _ap_stride(row_alpha),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    return out


def _glu_bwd(grad_out, gate_up, row_act, code_hint=None, row_alpha=None, want_act_grads=False):
    """Returns ggu, or (ggu, da_rows) when want_act_grads (requires row_alpha). The row-fused path
    computes grad_gate_up AND the per-row alpha grad in the SAME pass. The tiled path has no alpha
    (row_alpha with I > _ROWFUSE_MAX_I raises), so want_act_grads is row-path only."""
    M, twoI = gate_up.shape; I = twoI // 2
    _alpha_tile_gap(row_alpha, I, code_hint)
    if code_hint == 8 and row_alpha is None:
        raise ValueError(
            "act code 8 (radial NormSiLU, r^p*SiLU(g/r)) requires row_alpha -- it carries the "
            "exponent LOGIT theta, p=sigmoid(theta). With alpha absent the kernel would default to "
            "theta=1 => p=0.731 instead of the intended 0.5 init. Pass act_params.")
    ra = _ones(M, gate_up.device) if row_alpha is None else row_alpha
    ggu = torch.empty_like(gate_up)
    if I <= _ROWFUSE_MAX_I:
        if want_act_grads:
            da = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        else:
            da = gate_up                      # dead pointer: WANT_AP=0 compiles the store out
        if M > 0:
            BLOCK_I = max(16, triton.next_power_of_2(I))
            _glu_bwd_row_kernel[(M,)](grad_out, gate_up, row_act, ra, ggu, da, I,
                                      grad_out.stride(0), grad_out.stride(1),
                                      gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                                      _ap_stride(row_alpha),
                                      EPS=_NS_EPS, WANT_AP=want_act_grads, BLOCK_I=BLOCK_I,
                                      num_warps=(8 if BLOCK_I >= 1024 else 4))
        return (ggu, da) if want_act_grads else ggu
    skip_ns = code_hint is not None and code_hint not in (2, 6, 7, 8)  # 2/6/7/8 need the row RMS
    if skip_ns:
        rms = _ones(M, gate_up.device)      # r=1 / S=0 semantics; values unread where at!=2
        sbuf = rms
    else:
        rms = _row_rms(gate_up, row_act, I)  # recompute (one gate-half read) — keeps ctx/signatures unchanged
        sbuf = torch.empty(M, device=gate_up.device, dtype=torch.float32)
        if M > 0:
            _row_s_kernel[(M,)](grad_out, gate_up, row_act, rms, sbuf, I,
                                grad_out.stride(0), grad_out.stride(1),
                                gate_up.stride(0), gate_up.stride(1), BLOCK_I=_NS_BLOCK_I)
    BLOCK_M = max(16, min(64, triton.next_power_of_2(M))); BLOCK_I = max(16, min(128, triton.next_power_of_2(I)))
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(I, BLOCK_I))
    _glu_bwd_kernel[grid](grad_out, gate_up, row_act, rms, sbuf, ra, ggu, M, I,
                          grad_out.stride(0), grad_out.stride(1),
                          gate_up.stride(0), gate_up.stride(1), ggu.stride(0), ggu.stride(1),
                          _ap_stride(row_alpha),
                          BLOCK_M=BLOCK_M, BLOCK_I=BLOCK_I)
    if want_act_grads:
        raise NotImplementedError(
            "per-expert act params are row-path only (I <= %d); the tiled path carries no alpha."
            % _ROWFUSE_MAX_I)
    return ggu


class BatchedGLU(torch.autograd.Function):
    """PolyGLU activation: out = act_{row}(gate) * up, with a per-row activation code.
    Optional per-row act scalar: grad returned PER ROW (fp32) — build row_alpha
    differentiably (e.g. repeat_interleave of an (E,) param) and autograd sums them per expert."""
    @staticmethod
    def forward(ctx, gate_up, row_act, row_alpha=None):
        ctx.save_for_backward(gate_up, row_act,
                              row_alpha if row_alpha is not None else torch.empty(0),
                              )
        ctx.has_ap = row_alpha is not None
        return _glu_fwd(gate_up, row_act, row_alpha=row_alpha)

    @staticmethod
    def backward(ctx, grad_out):
        gate_up, row_act, row_alpha = ctx.saved_tensors
        ra = row_alpha if ctx.has_ap else None
        go = grad_out   # kernels take explicit strides: broadcast/expanded grads work un-materialized
        if ctx.has_ap and ctx.needs_input_grad[2]:
            ggu, da = _glu_bwd(go, gate_up, row_act, row_alpha=ra, want_act_grads=True)
            return ggu, None, da
        return _glu_bwd(go, gate_up, row_act, row_alpha=ra), None, None


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
    Codes needing act_params (8) are rejected here — only the per-expert path carries them. Codes
    3/4 are ACCEPTED for legacy diagnostics (grouped_parity.py / bench.py) but produce the documented
    wrong grads on specials stacks; moe() never routes them here."""
    if _code_max(act_codes) > 4:
        raise ValueError("code 8 (radial) unsupported on the grouped path; use moe_per_expert(act_params=...)")
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
        raise ValueError("code 8 (radial) unsupported on the grouped-cublas path; use moe_per_expert(act_params=...)")
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
        # act_params fp32 (E,) or (E,2) -- only column 0 (alpha) is read; stays fp32.
        hidden, wt, gate_up_proj, down_proj = _amp_cast(hidden, wt, gate_up_proj, down_proj)
        N, H = hidden.shape
        E = act_codes.shape[0]                  # total routed experts (GLU + specials)
        codes = act_codes.tolist()              # GLU (weight slot e) = 0/1/2/5/6/7; 3 = +Identity, 4 = -Identity
        top_k = idx.shape[1]; dev = hidden.device
        st, sw, order, counts, bounds = _sort_by_expert(idx, wt, E)
        x_s = hidden.index_select(0, st)                                  # (M,H) contiguous gather
        counts_t = torch.tensor(counts, device=dev)
        M_rows = idx.numel()   # output_size: statically known -> repeat_interleave skips its host sync
        row_act = torch.repeat_interleave(act_codes, counts_t, output_size=M_rows).to(torch.int32)
        # act_params None => alpha = 1, the DEFAULT (and byte-identical to no-alpha). Passing it enables the
        # learnable variant; uniform slices broadcast the expert's scalars (stride-0, no row tensors).
        ap32 = act_params.float().contiguous() if act_params is not None else None
        # per-expert activations kept as LISTS — the GEMM outputs ARE the storage; no contiguous
        # buffer + slice-copy (that was a pure DtoD-memcpy + memory tax).
        gate_up_l = [None] * E; inter_l = [None] * E
        # ONE contiguous (M,H) expert-output buffer + ONE scatter at the end, instead of E scatters.
        # At 64 experts x 8 layers the per-expert scatter was 512 kernel launches per micro-batch for
        # the SAME total atomic traffic; the second GEMM now writes straight into its slice (out=),
        # so this also removes the per-expert output allocation.
        M_tot = st.numel()
        eo_all = torch.empty(M_tot, H, device=dev, dtype=hidden.dtype)
        sw_eff = sw                                    # -Identity folds its sign into the weight
        # BATCHED ROW-OPS. The activation is a pure ROW map, so it does not need to be called once
        # per expert -- with 64 experts x 8 layers that was 512 launches per micro-batch on slices of
        # one logical (M,2I) tensor. Give the GEMMs a contiguous gu buffer (out=) and run ONE _glu_fwd
        # over every row. Only valid when no special expert is in the stack: codes 3/4 rows must skip
        # the GLU entirely, and a whole-buffer call would run it on them.
        uniform = all(c <= 2 or c >= 6 for c in codes) and ap32 is None
        # _grouped_mm runs ALL experts' GEMMs in ONE launch. Its GPU time is the same as the
        # per-expert loop (measured 0.99x), but the loop costs 64 python-level torch.mm calls per
        # GEMM per layer -- 2048 per micro-batch across the four GEMMs -- and the step is partly
        # CPU-launch-bound: batching the row ops cut launches 511->8 and gained 3ms/micro-batch of
        # wall time while those kernels' own time got slightly WORSE. So this is a launch-count fix.
        use_gmm = (uniform and hasattr(torch, "_grouped_mm")
                   and hidden.dtype in (torch.bfloat16, torch.float16))
        offs = counts_t.cumsum(0).to(torch.int32) if use_gmm else None
        tile_map = None; tile_map_gg = None; tile_map_bw = None
        if use_gmm:
            hint = codes[0] if len(set(codes)) == 1 else None
            # FUSED gate_up GEMM + GLU epilogue when the activation is pointwise (codes 0/1).
            # _glu_fwd was measured sitting exactly at HBM bandwidth: its whole cost is re-reading
            # the (M,2I) gu that cuBLAS had to land in memory. Computing the activation while the
            # accumulators are still in registers removes that read. 1.32x on the real load spread.
            # The GENERIC grouped GEMMs (it@W2 here, grad_inter + dX-with-scatter in backward) are
            # activation-agnostic, so their tile map is built whenever the arch/dtype allows -- NOT
            # gated on fused_supported(). It used to be, which silently cost the RMS-normed codes
            # (2/6/7) three activation-independent wins at once, the dX scatter (and its 23.9 ms/step
            # index_add) included. Only the two FUSED-EPILOGUE kernels need a pointwise activation.
            # Each kernel autotuned to a different BM, so each needs its own map (a few small GPU ops).
            if _FUSED_GLU is not None and _FUSED_GLU.tiles_supported(x_s):
                tile_map_gg = _FUSED_GLU.build_tile_map(counts, counts_t, dev,
                                                        bm=_FUSED_GLU._GG[0])
            # The Triton gate_up GEMM beats cuBLAS by 1.16x on its own (1.903 -> 1.643 ms), so the
            # RMS-normed codes take it too -- with act=False, since only a POINTWISE activation fits
            # a tile-local epilogue. They then run _glu_fwd over gu exactly as before. Fusing their
            # activation into the down-projection GEMM's prologue instead was built and measured: it
            # loses (see fused_supported).
            gu_all = it_all = None
            if tile_map_gg is not None and _FUSED_GLU.gemm_supported(x_s, gate_up_proj, codes):
                tm = _FUSED_GLU.build_tile_map(counts, counts_t, dev)
                act = _FUSED_GLU.fused_supported(x_s, gate_up_proj, codes)
                gu_all, it_all = _FUSED_GLU.fused_gate_up_glu(x_s, gate_up_proj, tm, codes[0],
                                                              want_gu=True, act=act)
                if act:      # backward only fuses its epilogue for the pointwise codes
                    tile_map = tm
                    tile_map_bw = _FUSED_GLU.build_tile_map(counts, counts_t, dev,
                                                            bm=_FUSED_GLU._BBM)
            if gu_all is None:
                gu_all = torch._grouped_mm(x_s, gate_up_proj.transpose(1, 2), offs=offs)
            if it_all is None:
                it_all = _glu_fwd(gu_all, row_act, code_hint=hint)
            eo_all = None
            if tile_map_gg is not None:
                eo_all = _FUSED_GLU.grouped_gemm(it_all, down_proj.transpose(1, 2).contiguous()
                                                 if not down_proj.transpose(1, 2).is_contiguous()
                                                 else down_proj.transpose(1, 2), tile_map_gg)
            if eo_all is None:
                eo_all = torch._grouped_mm(it_all, down_proj.transpose(1, 2), offs=offs)
            gate_up_l = gu_all; inter_l = it_all
        elif uniform:
            gu_all = torch.empty(M_tot, 2 * gate_up_proj.shape[1] // 2, device=dev, dtype=hidden.dtype)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en > s:
                    torch.mm(x_s[s:en], gate_up_proj[e].t(), out=gu_all[s:en])
            hint = codes[0] if len(set(codes)) == 1 else None
            it_all = _glu_fwd(gu_all, row_act, code_hint=hint)        # ONE launch, all rows
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en > s:
                    torch.mm(it_all[s:en], down_proj[e].t(), out=eo_all[s:en])
            gate_up_l = gu_all; inter_l = it_all
        else:
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en == s:
                    continue
                if codes[e] == 3 or codes[e] == 4:      # ±Identity: signed weighted passthrough
                    eo_all[s:en].copy_(x_s[s:en])
                    if codes[e] == 4:
                        if sw_eff is sw:
                            sw_eff = sw.clone()
                        sw_eff[s:en].neg_()
                    continue
                gu = x_s[s:en] @ gate_up_proj[e].t()                     # GLU expert; weight slot = e
                # alpha applies to codes 0/2 as act(alpha_e*x) where x is the raw
                # gate (0/1) or gate/rms(gate) (2/6/7). Gating this on codes[e]==5 silently made the
                # feature inert for silu/normsilu/normsitu -- alpha stayed exactly 1.0 for a whole
                # 60-step run while still costing the uniform fast path (ap32 is not None -> per-expert).
                _has_ap = ap32 is not None
                it = _glu_fwd(gu, row_act[s:en], code_hint=codes[e],
                              row_alpha=(ap32[e, 0:1] if _has_ap else None),
                              )
                torch.mm(it, down_proj[e].t(), out=eo_all[s:en])
                gate_up_l[e] = gu; inter_l[e] = it
        out = torch.zeros(N, H, device=dev, dtype=torch.float32)          # fp32 accumulate (MiMo)
        _combine_scatter(eo_all, sw_eff, st, out)          # ONE fused (eo*w)->fp32 scatter over all rows
        ctx.sw_eff = sw_eff
        ctx.save_for_backward(x_s, st, sw, order, row_act, gate_up_proj, down_proj,
                              ap32 if ap32 is not None else torch.empty(0))
        ctx.lists = (gate_up_l, inter_l, eo_all); ctx.bounds = bounds; ctx.uniform = uniform
        ctx.offs = offs; ctx.shapes = (N, H, top_k, E); ctx.tile_map = tile_map; ctx.tile_map_gg = tile_map_gg; ctx.tile_map_bw = tile_map_bw
        ctx.codes = codes; ctx.has_ap = ap32 is not None
        return out.to(hidden.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (x_s, st, sw, order, row_act, gate_up_proj, down_proj, ap32) = ctx.saved_tensors
        gate_up_l, inter_l, eo_all = ctx.lists
        sw_eff = ctx.sw_eff
        N, H, top_k, E = ctx.shapes; bounds = ctx.bounds; codes = ctx.codes
        M = st.numel()
        if ctx.offs is not None:
            gu_all, it_all = gate_up_l, inter_l
            offs = ctx.offs
            ge_all, gw_all = _combine_bwd(grad_out, eo_all, sw, st)
            grad_down_proj = torch._grouped_mm(ge_all.t(), it_all, offs=offs)
            hint = codes[0] if len(set(codes)) == 1 else None
            # Fused grad_inter GEMM + GLU backward: computing grad_it in registers avoids
            # materializing (M,I) and re-reading (M,2I) gu. _glu_bwd was 42.5 ms/step, the largest
            # non-GEMM kernel, and sat exactly at HBM bandwidth.
            if ctx.tile_map is not None:
                grad_gate_up = _FUSED_GLU.fused_dinter_glu_bwd(
                    ge_all, down_proj, gu_all, ctx.tile_map_bw, codes[0])
            else:
                # No fused epilogue (RMS-normed act): grad_it must be materialized for _glu_bwd's row
                # reduction anyway, but the GEMM producing it is still activation-agnostic Triton.
                grad_inter = (_FUSED_GLU.grouped_gemm(ge_all, down_proj, ctx.tile_map_gg)
                              if ctx.tile_map_gg is not None else None)
                if grad_inter is None:
                    grad_inter = torch._grouped_mm(ge_all, down_proj, offs=offs)
                grad_gate_up = _glu_bwd(grad_inter, gu_all, row_act, code_hint=hint)
            grad_gate_up_proj = torch._grouped_mm(grad_gate_up.t(), x_s, offs=offs)
            grad_hidden = None
            if ctx.tile_map_gg is not None:                 # GEMM + scatter-add in one kernel
                gh32 = _FUSED_GLU.grouped_gemm_scatter(grad_gate_up, gate_up_proj, st,
                                                       ctx.tile_map_gg, N)
                if gh32 is not None:
                    grad_hidden = gh32.to(grad_out.dtype)
            if grad_hidden is None:
                grad_x = torch._grouped_mm(grad_gate_up, gate_up_proj, offs=offs)
                grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
                grad_hidden.index_add_(0, st, grad_x)
            grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
            grad_wt[order] = gw_all.to(grad_out.dtype)
            return (grad_hidden, None, grad_wt.view(N, top_k), grad_gate_up_proj, grad_down_proj,
                    None, None)
        if ctx.uniform:
            # Mirror of the forward's batched row-ops: _combine_bwd, the GLU backward and the dX
            # scatter are all row maps, so each runs ONCE over (M,*) instead of once per expert.
            # At 64 experts x 8 layers that is 512 launches -> 8 for each of the three.
            gu_all, it_all = gate_up_l, inter_l
            Icols = it_all.shape[1]
            ge_all, gw_all = _combine_bwd(grad_out, eo_all, sw, st)      # ONE call
            grad_w_s = gw_all.to(grad_out.dtype)
            grad_inter = torch.empty(M, Icols, device=grad_out.device, dtype=grad_out.dtype)
            grad_gate_up_proj = torch.zeros_like(gate_up_proj)
            grad_down_proj = torch.zeros_like(down_proj)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en == s:
                    continue
                torch.mm(ge_all[s:en], down_proj[e], out=grad_inter[s:en])
                torch.mm(ge_all[s:en].t(), it_all[s:en], out=grad_down_proj[e])
            hint = codes[0] if len(set(codes)) == 1 else None
            grad_gate_up = _glu_bwd(grad_inter, gu_all, row_act, code_hint=hint)   # ONE call
            grad_x = torch.empty(M, H, device=grad_out.device, dtype=grad_out.dtype)
            for e in range(E):
                s, en = bounds[e], bounds[e + 1]
                if en == s:
                    continue
                torch.mm(grad_gate_up[s:en].t(), x_s[s:en], out=grad_gate_up_proj[e])
                torch.mm(grad_gate_up[s:en], gate_up_proj[e], out=grad_x[s:en])
            grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
            grad_hidden.index_add_(0, st, grad_x)                        # ONE scatter
            grad_wt = torch.zeros(N * top_k, device=grad_out.device, dtype=grad_out.dtype)
            grad_wt[order] = grad_w_s
            return (grad_hidden, None, grad_wt.view(N, top_k), grad_gate_up_proj, grad_down_proj,
                    None, None)
        grad_w_s = torch.zeros(M, device=grad_out.device, dtype=grad_out.dtype)
        grad_gate_up_proj = torch.zeros_like(gate_up_proj)
        grad_down_proj = torch.zeros_like(down_proj)
        grad_hidden = torch.zeros(N, H, device=grad_out.device, dtype=grad_out.dtype)
        want_ap = ctx.has_ap and ctx.needs_input_grad[6]   # skip param-grad work when alpha is frozen
        grad_act_params = (torch.zeros(E, 2, device=grad_out.device, dtype=torch.float32)
                           if want_ap else None)
        for e in range(E):
            s, en = bounds[e], bounds[e + 1]
            if en == s:
                continue
            if codes[e] == 3 or codes[e] == 4:
                # ±Identity: out = (sgn*w)*x  ->  dx = (sgn*w)*go, dw = sgn*sum(go*x).
                # _combine_bwd(go, eo, w, tok) returns (go*w, sum_h(go*eo)), so feeding it the SIGNED
                # weight gives dx directly and d/d(sgn*w); one extra sgn recovers dw.
                sgn = 1.0 if codes[e] == 3 else -1.0
                ge, gw = _combine_bwd(grad_out, x_s[s:en], sw_eff[s:en], st[s:en])
                grad_w_s[s:en].copy_(gw * sgn)          # cast+copy in one, no temp
                grad_hidden.index_add_(0, st[s:en], ge)
                continue
            it = inter_l[e]                                            # (m,I)
            # fused combine bwd: gather grad_out[tok], emit grad_eo=go*w and grad_w=sum_h(go*eo)
            ge, gw = _combine_bwd(grad_out, eo_all[s:en], sw[s:en], st[s:en])  # (m,H), (m,) fp32
            grad_w_s[s:en].copy_(gw)                    # cast+copy in one, no temp
            grad_inter = ge @ down_proj[e]                              # (m,H)@(H,I) -> (m,I)
            torch.mm(ge.t(), it, out=grad_down_proj[e])                 # (H,m)@(m,I) -> (H,I)
            # mirrors the forward: alpha is live for every GLU code, so d_alpha must be too. The row
            # kernel writes DG=0 for non-SiTU codes (a per-expert OUTPUT gain is redundant with the
            # router weight), so gamma simply stays at its init there.
            if want_ap:
                grad_gate_up, da = _glu_bwd(grad_inter, gate_up_l[e], row_act[s:en],
                                                code_hint=codes[e], row_alpha=ap32[e, 0:1],
                                                want_act_grads=True)
                grad_act_params[e, 0] = da.sum()
            else:
                _has_ap = ctx.has_ap
                grad_gate_up = _glu_bwd(grad_inter, gate_up_l[e], row_act[s:en], code_hint=codes[e],
                                        row_alpha=(ap32[e, 0:1] if _has_ap else None),
                                        )   # (m,2I)
            torch.mm(grad_gate_up.t(), x_s[s:en], out=grad_gate_up_proj[e])  # (2I,m)@(m,H)->(2I,H)
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
    NOT implement the ±Identity (codes 3/4) special experts — it runs GLU over every expert
    uniformly — so it is correct ONLY for pure-GLU stacks. With a special expert present it produces
    wrong output and gradients (measured: grad rel ~1.6e+03 on the 9-GLU+2-special stack), so we
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
def _act_eager(gate, code, alpha=None):
    """Reference activation. THE ONLY THREE CODES: 0 silu, 2 normsilu, 8 radial normsilu.
    alpha is the per-expert input scale for 0/2 and the exponent LOGIT for 8 (p = sigmoid(alpha))."""
    a = 1.0 if alpha is None else alpha
    if code == 0:
        return F.silu(a * gate.float()).to(gate.dtype)
    if code == 8:
        # RADIAL NormSiLU: r^p * SiLU(g/r), r = rms(gate), p = sigmoid(alpha) bounded in (0,1).
        # p -> 0 IS normsilu and p -> 1 is full magnitude, so the learnable exponent interpolates
        # between the other two codes -- which is why it wins: measured p is a DEPTH RAMP.
        g = gate.float()
        r = torch.sqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
        pw = torch.sigmoid(alpha if torch.is_tensor(alpha) else torch.tensor(float(a), device=g.device))
        return (r.pow(pw) * F.silu(g / r)).to(gate.dtype)
    if code != 2:
        raise ValueError(f"unsupported act code {code}; only 0 (silu), 2 (normsilu), 8 (radial) exist")

    # NormSiLU: SiLU(alpha * rms-normed gate), matches BiBo eager (_NORMSILU_EPS). alpha sits
    # AFTER the rms -- rms is positively homogeneous, so scaling before it is exactly inert.
    g = gate.float()
    g = g * torch.rsqrt(g.square().mean(-1, keepdim=True) + _NS_EPS)
    return F.silu(a * g).to(gate.dtype)


def moe_eager(hidden, top_k_indices, top_k_weights, gate_up_proj, down_proj, act_codes,
              act_params=None):
    """Hand-written MoE: per-expert boolean mask (GPU sync each iter), unfused activation,
    E tiny GEMMs. Correct, and deliberately the slow baseline the fused paths beat."""
    N, H = hidden.shape
    twoI = gate_up_proj.shape[1]
    I = twoI // 2
    codes = act_codes.tolist()          # GLU (weight slot e) = 0/1/2/5/6/7; 3 = +Identity, 4 = -Identity
    E = len(codes)                       # total routed experts (GLU + specials)
    out = torch.zeros(N, H, device=hidden.device, dtype=torch.float32)   # fp32 accumulate (MiMo)
    for e in range(E):
        rows = (top_k_indices == e).any(-1)
        if not bool(rows.any()):
            continue
        w = (top_k_weights * (top_k_indices == e)).sum(-1)[rows]
        if codes[e] == 3 or codes[e] == 4:                               # ±Identity: signed passthrough
            sgn = 1.0 if codes[e] == 3 else -1.0
            out[rows] += (hidden[rows] * (sgn * w).unsqueeze(-1)).float()
            continue
        gate_up = hidden[rows] @ gate_up_proj[e].t()
        a, g = ((act_params[e, 0], act_params[e, 1]) if codes[e] == 5 and act_params is not None
                else (1.0, 1.0))
        inter = _act_eager(gate_up[:, :I], codes[e], a) * gate_up[:, I:]
        out[rows] += ((inter @ down_proj[e].t()) * w.unsqueeze(-1)).float()
    return out.to(hidden.dtype)
