from collections import defaultdict

import torch
import torch.optim as optim

from kernels.muon import muon_scaling as _scaling

_PE_COEFFS = (
    (8.28721201814563,   -23.595886519098837, 17.300387312530933),
    (4.107059111542203,   -2.9478499167379106,  0.5448431082926601),
    (3.9486908534822946,  -2.908902115962949,   0.5518191394370137),
    (3.3184196573706015,  -2.488488024314874,   0.51004894012372),
    (2.300652019954817,   -1.6689039845747493,  0.4188073119525673),
    (1.891301407787398,   -1.2679958271945868,  0.37680408948524835),
    (1.8750014808534479,  -1.2500016453999487,  0.3750001645474248),
    (1.875,               -1.25,                0.375),
)

_DSV4_COEFFS = ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2

_NS_DTYPE = torch.float16


def newton_schulz(G, coeffs=_DSV4_COEFFS, ns_dtype=_NS_DTYPE, eps=1e-7):
    orig_dtype = G.dtype
    squeeze = G.ndim == 2
    X = G.unsqueeze(0) if squeeze else G
    nrm = torch.linalg.vector_norm(X.flatten(1), dim=1, dtype=torch.float32).clamp_min(eps).view(-1, 1, 1)
    transposed = X.size(1) > X.size(2)
    if transposed:
        X = X.transpose(1, 2)
    X = X.to(ns_dtype) / nrm.to(ns_dtype)
    for a, b, c in coeffs:
        A = torch.bmm(X, X.transpose(1, 2))
        B = torch.baddbmm(A, A, A, beta=b, alpha=c)
        X = torch.baddbmm(X, B, X, beta=a, alpha=1.0)
    if transposed:
        X = X.transpose(1, 2)
    if squeeze:
        X = X.squeeze(0)
    return X.to(orig_dtype)


class FusedMuon(optim.Optimizer):

    DEFAULT_NS_DTYPE = _NS_DTYPE

    def __init__(self, params, lr=3e-4, momentum=0.95, nesterov=True, weight_decay=0.0,
                 coeffs=_DSV4_COEFFS, ns_dtype=None, scale_mode=_scaling.DEFAULT_MODE,
                 ns_batch_elems=4 * 1024 * 1024, use_graph=False, graph_warmup=3, aurora_k=None,
                 spectral_wd=0.0, swd_beta=0.99, xorth_post=0.0, xorth_backend="ns",
                 xorth_ns_iters=18, xorth_ema=0.95, xorth_gate_ref=0.3,
                 xorth_warmup_steps=0, xorth_where="post", cautious_decay=False):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, weight_decay=weight_decay,
                        xorth_post=float(xorth_post))
        super().__init__(params, defaults)
        self.cautious_decay = bool(cautious_decay)
        self.coeffs = coeffs
        self.ns_dtype = ns_dtype if ns_dtype is not None else self.DEFAULT_NS_DTYPE
        self.scale_mode = _scaling.validate(scale_mode)
        self.aurora_k = _scaling.AURORA_K if aurora_k is None else aurora_k
        self.spectral_wd = float(spectral_wd)
        self.swd_beta = float(swd_beta)
        self._swd_cov = None
        if xorth_backend not in ("ns", "eigh"):
            raise ValueError(f"xorth_backend must be 'ns' or 'eigh', got {xorth_backend!r}")
        self.xorth_backend = xorth_backend
        self.xorth_ns_iters = int(xorth_ns_iters)
        self.xorth_ema = float(xorth_ema)
        self.xorth_gate_ref = float(xorth_gate_ref)
        self.xorth_warmup_steps = int(xorth_warmup_steps)
        if xorth_where not in ("pre", "post"):
            raise ValueError(f"xorth_where must be 'pre' or 'post', got {xorth_where!r}")
        self.xorth_where = xorth_where
        self._xorth_step = 0
        self.ns_batch_elems = ns_batch_elems
        self.use_graph = use_graph
        self.graph_warmup = graph_warmup
        self._graph = None
        self._gwork = None
        self._gstep = 0
        self._graph_failed = False

    def _scale(self, p):
        return _scaling.scalar_scale(self.scale_mode, p.shape[-2], p.shape[-1])

    def _whiten_chunk(self, out, members, r, c, beta_max):
        byn = defaultdict(list)
        for p, o, n in members:
            if n > 1:
                byn[n].append((p, o))
        for n, po in byn.items():
            sel = torch.stack([out[o:o + n].reshape(n, -1) for _p, o in po]).float()
            if self.xorth_backend == "eigh":
                wht = _scaling.xorth_whiten_batch(sel, beta_max)
            else:
                cema = torch.stack([self._xorth_state(p, n) for p, _o in po])
                wht = _scaling.xorth_whiten_gated(sel, cema, beta_max,
                                                  rho=self.xorth_ema, gate_ref=self.xorth_gate_ref,
                                                  iters=self.xorth_ns_iters)
                for i, (p, _o) in enumerate(po):
                    self.state[p]["xorth_cema"].copy_(cema[i])
            for i, (_p, o) in enumerate(po):
                out[o:o + n] = wht[i].view(n, r, c).to(out.dtype)

    def _xorth_state(self, p, n):
        st = self.state[p]
        if "xorth_cema" not in st:
            st["xorth_cema"] = torch.eye(n, device=p.device, dtype=torch.float32)
        return st["xorth_cema"]

    def _polar(self, u):
        return newton_schulz(u, self.coeffs, self.ns_dtype)

    def _plan(self, group, params):
        cache = getattr(self, "_plan_cache", None)
        if cache is None:
            cache = self._plan_cache = {}
        key = id(group)
        if key in cache:
            return cache[key]
        buckets = defaultdict(list)
        for p in params:
            buckets[(p.shape[-2], p.shape[-1])].append(p)
        plan = []
        for (r, c), ps in buckets.items():
            members, off = [], 0
            for p in ps:
                n = p.numel() // (r * c)
                members.append((p, off, n)); off += n
            M = off
            anchor = ps[0]
            if "muon_mom" not in self.state[anchor]:
                self.state[anchor]["muon_mom"] = torch.zeros((M, r, c), device=anchor.device, dtype=self.ns_dtype)
            if _scaling.needs_perrow_state(self.scale_mode) and "scale_v" not in self.state[anchor]:
                self.state[anchor]["scale_v"] = _scaling.perrow_state(M, r, anchor.device)
            if self.spectral_wd > 0 and "swd_e" not in self.state[anchor]:
                self.state[anchor]["swd_e"] = torch.zeros((M, r), device=anchor.device)
            scale = (_scaling.scalar_scale(self.scale_mode, r, c)
                     if self.scale_mode in _scaling.SCALAR_MODES else 1.0)
            row_cap = max(1, self.ns_batch_elems // (r * c))
            chunks, cur, cur_rows, start = [], [], 0, 0
            for p, _o, n in members:
                if cur and cur_rows + n > row_cap:
                    chunks.append((cur, start, cur_rows)); start += cur_rows; cur, cur_rows = [], 0
                cur.append((p, cur_rows, n)); cur_rows += n
            if cur:
                chunks.append((cur, start, cur_rows))
            plan.append({"r": r, "c": c, "M": M, "chunks": chunks, "anchor": anchor, "scale": scale})
        cache[key] = plan
        return plan

    def _build_graph_work(self):
        if self._gwork is not None:
            return self._gwork
        work, decay = [], []
        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            plan = self._plan(group, params)
            if wd != 0:
                decay.append((params, 1.0 - lr * wd))
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                alpha = -lr * g["scale"]
                for members, start, crows in g["chunks"]:
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    work.append({"gbuf": gbuf, "dst": [gbuf[o:o + n] for _, o, n in members],
                                 "mom_c": mom[start:start + crows], "momentum": momentum,
                                 "nesterov": nesterov, "alpha": alpha, "members": members,
                                 "out_params": [p for p, _, _ in members], "r": r, "c": c})
        self._gwork = (work, decay)
        return self._gwork

    def _gather(self, work):
        for w in work:
            r, c = w["r"], w["c"]
            torch._foreach_copy_(w["dst"], [p.grad.reshape(n, r, c) for p, o, n in w["members"]])

    def _compute(self, work, decay):
        for params, f in decay:
            torch._foreach_mul_(params, f)
        for w in work:
            mom_c, gbuf = w["mom_c"], w["gbuf"]
            mom_c.mul_(w["momentum"]).add_(gbuf)
            u = gbuf.add_(mom_c, alpha=w["momentum"]) if w["nesterov"] else mom_c
            out = newton_schulz(u, self.coeffs, self.ns_dtype)
            r, c = w["r"], w["c"]
            torch._foreach_add_(w["out_params"],
                                [out[o:o + n].reshape(p.shape) for p, o, n in w["members"]],
                                alpha=w["alpha"])

    @torch.no_grad()
    def _graph_step(self):
        if self.cautious_decay and any(g["weight_decay"] != 0 for g in self.param_groups):
            raise NotImplementedError(
                "cautious_decay is not implemented on the CUDA-graph path: _build_graph_work plans the "
                "decay as a single pre-update _foreach_mul_, but the cautious mask needs the "
                "orthogonalized update. Run with use_graph=False (the default) or cautious_decay=False.")
        work, decay = self._build_graph_work()
        self._gather(work)
        if self._graph is not None:
            self._graph.replay()
            return
        self._gstep += 1
        if self._gstep <= self.graph_warmup or self._graph_failed:
            self._compute(work, decay)
            return
        try:
            g = torch.cuda.CUDAGraph()
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                self._compute(work, decay)
            self._graph = g
            self._graph.replay()
        except Exception as ex:
            self._graph, self._graph_failed = None, True
            print(f"  (FusedMuon CUDA-graph capture failed: {type(ex).__name__}: "
                  f"{str(ex).splitlines()[0]} — falling back to eager)")
            self._compute(work, decay)

    def set_graph(self, enabled):
        self.use_graph = enabled
        self._graph, self._gwork, self._gstep, self._graph_failed = None, None, 0, False

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        self._xorth_step += 1

        if (self.use_graph and self.spectral_wd == 0
                and not any(g.get("xorth_post", 0) > 0 for g in self.param_groups)
                and not (_scaling.is_perrow(self.scale_mode) or _scaling.is_aurora(self.scale_mode)
                         or _scaling.is_aurora_ema(self.scale_mode))):
            self._graph_step()
            return loss

        for group in self.param_groups:
            params = [p for p in group["params"] if p.grad is not None and p.ndim in (2, 3)]
            if not params:
                continue
            lr, momentum, wd, nesterov = (group["lr"], group["momentum"],
                                          group["weight_decay"], group["nesterov"])
            xp = group["xorth_post"]
            do_xorth = xp > 0 and self._xorth_step > self.xorth_warmup_steps

            plan = self._plan(group, params)
            spectral = self.spectral_wd > 0 and wd != 0
            cautious = self.cautious_decay and wd != 0 and not spectral
            if wd != 0 and not spectral and not cautious:
                torch._foreach_mul_(params, 1.0 - lr * wd)
            perrow = _scaling.is_perrow(self.scale_mode)
            aurora = _scaling.is_aurora(self.scale_mode)
            aurora_ema = _scaling.is_aurora_ema(self.scale_mode)
            for g in plan:
                r, c = g["r"], g["c"]
                mom = self.state[g["anchor"]]["muon_mom"]
                v_all = self.state[g["anchor"]].get("scale_v")
                e_all = self.state[g["anchor"]].get("swd_e") if spectral else None
                alpha = -lr * g["scale"]
                sc_alpha = -lr if _scaling.folds_scale(self.scale_mode) else alpha
                for members, start, crows in g["chunks"]:
                    mom_c = mom[start:start + crows]
                    gbuf = torch.empty((crows, r, c), device=mom.device, dtype=self.ns_dtype)
                    torch._foreach_copy_([gbuf[o:o + n] for _, o, n in members],
                                         [p.grad.reshape(n, r, c) for p, o, n in members])
                    mom_c.mul_(momentum).add_(gbuf)
                    u = gbuf.add_(mom_c, alpha=momentum) if nesterov else mom_c
                    if do_xorth and self.xorth_where == "pre":
                        if not nesterov:
                            u = u.clone()
                        self._whiten_chunk(u, members, r, c, xp)
                    if spectral:
                        s, cov = _scaling.spectral_wd_mult(u, e_all[start:start + crows], self.spectral_wd, self.swd_beta)
                        self._swd_cov = float(cov)
                        for p, o, n in members:
                            m = (s[o:o + n] if s is not None else 1.0)
                            pv = p.view(n, r, c)
                            pv.mul_(1.0 - lr * wd * (m.unsqueeze(-1) if s is not None else 1.0))
                    if aurora:
                        out = _scaling.aurora_update(u, self._polar, K=self.aurora_k)
                    elif aurora_ema:
                        v_c = v_all[start:start + crows]
                        if self.scale_mode == "aurora_ema_v2":
                            out = _scaling.aurora_ema_v2_update(u, self._polar, v_c, K=self.aurora_k)
                        else:
                            out = _scaling.aurora_ema_update(u, self._polar, v_c)
                    else:
                        out = newton_schulz(u, self.coeffs, self.ns_dtype)
                        if perrow:
                            out = _scaling.apply_perrow(self.scale_mode, out, v_all[start:start + crows])
                    if do_xorth and self.xorth_where == "post":
                        self._whiten_chunk(out, members, r, c, xp)
                    _pl = [p for p, _, _ in members]
                    _ul = [out[o:o + n].reshape(p.shape) for p, o, n in members]
                    if cautious:
                        for _p, _u in zip(_pl, _ul):
                            _p.sub_(_p * ((_u * _p) < 0), alpha=lr * wd)
                    torch._foreach_add_(_pl, _ul, alpha=sc_alpha)

        return loss


class DistributedMuon(FusedMuon):

    def __init__(self, params, *, process_group=None, **kwargs):
        super().__init__(params, **kwargs)
        if _scaling.is_aurora_ema(self.scale_mode):
            raise NotImplementedError("scale_mode 'aurora_ema' is only supported by FusedMuon, not DistributedMuon")
        self.pg = process_group
        self._owner = None

    def _ordered(self):
        out = []
        for g in self.param_groups:
            for p in g["params"]:
                if p.ndim in (2, 3):
                    out.append((p, g))
        return out

    def _plan(self, ordered, ws):
        load = [0] * ws
        owner = []
        for p, _ in ordered:
            r = min(range(ws), key=lambda i: load[i])
            owner.append(r); load[r] += p.numel()
        return owner

    @torch.no_grad()
    def step(self, closure=None):
        import torch.distributed as dist
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        ws, rank = dist.get_world_size(self.pg), dist.get_rank(self.pg)
        ordered = self._ordered()
        if self._owner is None or len(self._owner) != len(ordered):
            self._owner = self._plan(ordered, ws)

        upd = {}
        for i, (p, g) in enumerate(ordered):
            if self._owner[i] != rank or p.grad is None:
                continue
            gr = p.grad.to(self.ns_dtype)
            st = self.state[p]
            if "momentum_buffer" not in st:
                st["momentum_buffer"] = torch.zeros_like(gr)
            buf = st["momentum_buffer"]
            buf.mul_(g["momentum"]).add_(gr)
            u = gr.add(buf, alpha=g["momentum"]) if g["nesterov"] else buf
            if _scaling.is_aurora(self.scale_mode):
                upd[i] = _scaling.aurora_update(u.unsqueeze(0) if u.ndim == 2 else u,
                                                self._polar, K=self.aurora_k).reshape(p.shape).to(p.dtype)
            else:
                upd[i] = newton_schulz(u, self.coeffs, self.ns_dtype).to(p.dtype)

        for src in range(ws):
            idxs = [i for i in range(len(ordered))
                    if self._owner[i] == src and ordered[i][0].grad is not None]
            if not idxs:
                continue
            sizes = [ordered[i][0].numel() for i in idxs]
            ref = ordered[idxs[0]][0]
            if src == rank:
                blob = torch.cat([upd[i].reshape(-1) for i in idxs])
            else:
                blob = torch.empty(sum(sizes), device=ref.device, dtype=ref.dtype)
            dist.broadcast(blob, src=src, group=self.pg)
            off = 0
            perrow = _scaling.is_perrow(self.scale_mode)
            aurora = _scaling.is_aurora(self.scale_mode)
            for i, n in zip(idxs, sizes):
                p, g = ordered[i]
                u = blob[off:off + n].view_as(p); off += n
                lr, wd = g["lr"], g["weight_decay"]
                if wd != 0:
                    if self.cautious_decay:
                        p.sub_(p * ((u * p) < 0), alpha=lr * wd)
                    else:
                        p.mul_(1.0 - lr * wd)
                if aurora:
                    p.add_(u, alpha=-lr)
                elif perrow:
                    uu = u.unsqueeze(0) if u.ndim == 2 else u
                    st = self.state[p]
                    if "scale_v" not in st:
                        st["scale_v"] = _scaling.perrow_state(uu.shape[0], uu.shape[1], p.device)
                    p.add_(_scaling.apply_perrow(self.scale_mode, uu, st["scale_v"]).reshape(p.shape), alpha=-lr)
                else:
                    p.add_(u, alpha=-lr * self._scale(p))
        return loss
