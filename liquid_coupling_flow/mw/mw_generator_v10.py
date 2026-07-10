"""mW generator v10: identity toroidal transport on the complete trained v4 density.

Unlike v6-v9, v10 does not replace any learned v4 component.  The v4 global body
and 64-component full-covariance MDN form the base density.  An autoregressive
circular RQ-spline diffeomorphism is composed on top of the MDN output chart:

    u ~ q_v4(u | h),       y = T(u; h).

T is identity-initialized, so the initial model is exactly v4.  The correction is
trained alone first; later, the MDN and body are unfrozen at much smaller learning
rates.  The map is a diffeomorphism of the torus and preserves v4 multimodality.

Structure-aware fine-tune (``finetune_struct``)
-----------------------------------------------
v10 has the campaign's best likelihood (val 0.253) and exact density, but its SAMPLED
structure is stuck (TF pooled-predecessor g(r): shell1 ~1.2 vs data 2.18, shell2 ~1.07
vs 1.20) -- MLE is mass-covering and never purifies the smeared multimodal conditionals.
``finetune_struct`` adds a REPARAMETERIZED teacher-forced sample-energy term
``L = NLL/N + lam_t * L_struct`` that directly penalizes bad TF placements: at every step
a placement x_hat_j is drawn from the head with gradients flowing (mixture-component index
detached; the within-component draw + tanh + toroidal transport stay on the tape) and its
tanh-clipped Stillinger-Weber INSERTION energy against the true causal prefix is penalized.
The density code (log_prob/sample/head) is UNCHANGED -- only the training objective is new.

PRE-REGISTERED GATE (structure fine-tune).  GO iff:
    TF shell1 err < 10%  AND  TF shell2 >= 1.12 (halve the 1.07->1.20 deficit)
    AND  core_mass < 0.05  AND  val NLL <= 0.40 (calibration guard: bounded nats spent).
FR g(r) is recorded after a GO to size the rollout residual.
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn

from liquid_coupling_flow.transforms_spline import (
    CircularRQSplineElementwise, DEFAULT_MIN_DERIVATIVE,
)
from liquid_coupling_flow.mw.mw_energy import (
    A_CUT, COS0, GAMMA, LAMBDA3, RHO_STAR, _phi2,
)
from liquid_coupling_flow.mw.mw_generator import (
    ART, DEV, R_SHELL1, _augment_batch, canonical_order, load_training_bank,
    mw_scaffold, wrap_pm,
)
from liquid_coupling_flow.mw.mw_generator_v4 import FullCovTanhMDN, MWGlobalAR


class ToroidalResidualHead(nn.Module):
    """V4 MDN followed by an AR circular-spline transport."""

    def __init__(self, d_model=256, n_mix=64, num_bins=16, bound=4.0):
        super().__init__()
        self.n_mix = int(n_mix)
        self.num_bins = int(num_bins)
        self.bound = float(bound)
        self.period = 2 * self.bound
        self.base = FullCovTanhMDN(d_model=d_model, n_mix=n_mix)
        self.spline = CircularRQSplineElementwise(num_bins=num_bins, L=self.period)
        P = self.spline.params_per_dim
        self.transforms = nn.ModuleList([nn.Linear(d_model + 2 * i, P) for i in range(3)])
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for transform in self.transforms:
            nn.init.zeros_(transform.weight)
            with torch.no_grad():
                transform.bias.zero_(); transform.bias[2 * num_bins:] = const

    def _check_bound(self, bound):
        if abs(float(bound) - self.bound) > 1e-6:
            raise ValueError(f"runtime bound={bound} != residual-head bound={self.bound}; "
                             "v10 is the controlled N=64 experiment")

    def _circle(self, u):
        return torch.remainder(u + self.bound, self.period)

    def _embed(self, q):
        a = 2 * math.pi * q / self.period
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

    def forward_transport(self, h, u, bound):
        """Map v4 base coordinates u -> output coordinates y; return log|dy/du|."""
        self._check_bound(bound)
        ctx = h; ys = []; ld_total = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        for i in range(3):
            q = self._circle(u[..., i:i + 1])
            yq, ld = self.spline.forward(q, self.transforms[i](ctx))
            ys.append(yq - self.bound); ld_total += ld.squeeze(-1)
            # Triangular conditioning uses previous BASE coordinates.  The inverse
            # recovers those sequentially before constructing the next parameters.
            ctx = torch.cat([ctx, self._embed(q)], -1)
        return torch.cat(ys, -1), ld_total

    def inverse_transport(self, h, y, bound):
        """Map output y -> v4 base u sequentially; return log|du/dy|."""
        self._check_bound(bound)
        ctx = h; us = []; ld_total = torch.zeros(h.shape[:-1], device=h.device, dtype=h.dtype)
        for i in range(3):
            yq = self._circle(y[..., i:i + 1])
            q, ld = self.spline.inverse(yq, self.transforms[i](ctx))
            us.append(q - self.bound); ld_total += ld.squeeze(-1)
            ctx = torch.cat([ctx, self._embed(q)], -1)
        return torch.cat(us, -1), ld_total

    def log_prob(self, h, y, bound):
        u, inverse_ld = self.inverse_transport(h, y, bound)
        return self.base.log_prob(h, u, bound) + inverse_ld

    def sample(self, h, bound, gen=None):
        u, base_lp = self.base.sample(h, bound, gen=gen)
        y, forward_ld = self.forward_transport(h, u, bound)
        return y, base_lp - forward_ld


class MWV4ToroidalResidual(MWGlobalAR):
    """The complete v4 generator plus an identity-initialized torus correction."""

    def __init__(self, d_model=256, n_layers=4, n_heads=8, n_mix=64,
                 rail_k=8, num_bins=16, bound=4.0):
        super().__init__(d_model=d_model, n_layers=n_layers, n_heads=n_heads,
                         n_mix=n_mix, rail_k=rail_k)
        self.num_bins = int(num_bins); self.bound = float(bound)
        self.head = ToroidalResidualHead(d_model, n_mix, num_bins, bound)

    @torch.no_grad()
    def teacher_forced_structure(self, x, L, nbins=120, gen=None):
        x = torch.remainder(x, L); B, N = x.shape[:2]
        anchors, rank, R = mw_scaffold(N, L, x.device)
        perm = canonical_order(x, L, R, rank)
        xo = torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))
        s = (L / R) / 2
        u = wrap_pm(xo - anchors[None], L) / s
        h = self._hidden(u, xo, anchors, L)
        placed_u, _ = self.head.sample(h, float(R), gen=gen)
        placed = torch.remainder(anchors[None] + s * placed_u, L)
        d = wrap_pm(placed[:, :, None] - xo[:, None], L).norm(dim=-1)
        jj = torch.arange(N, device=x.device)
        causal = jj[None, :, None] > jj[None, None, :]
        vals = d[causal.expand(B, -1, -1)]
        edges = torch.linspace(0, L / 2, nbins + 1, device=x.device)
        counts = torch.histogram(vals.float().cpu(), bins=nbins,
                                 range=(0.0, L / 2))[0].to(x.device)
        r = (edges[1:] + edges[:-1]) / 2; dr = edges[1] - edges[0]
        norm = B * (N * (N - 1) / 2) * (4 * math.pi * r.square() * dr) / L ** 3
        gr = counts / norm.clamp_min(1e-12)
        peak_err = (gr[(r - R_SHELL1).abs().argmin()] - 2.18).abs() / 2.18
        core_coord = (vals < 1).sum().to(gr.dtype) / (B * N)
        return {"r": r, "gr": gr, "peak_err": peak_err,
                "core_coord": core_coord, "core_mass": core_coord,
                "composite": peak_err + core_coord}


_MODEL_KEYS = ("d_model", "n_layers", "n_heads", "n_mix", "rail_k", "num_bins", "bound")


def load_v4_exact(model, path, device):
    """Load every v4 tensor, remapping only ``head.*`` -> ``head.base.*``."""
    ck = torch.load(path, map_location=device, weights_only=False)
    remapped = {}
    for key, value in ck["state_dict"].items():
        remapped["head.base." + key[5:] if key.startswith("head.") else key] = value
    incompatible = model.load_state_dict(remapped, strict=False)
    missing = [k for k in incompatible.missing_keys if not k.startswith("head.transforms.")]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(f"v4 exact load mismatch: missing={missing}, "
                           f"unexpected={incompatible.unexpected_keys}")
    return ck


def _set_stage(model, correction_only):
    for name, p in model.named_parameters():
        p.requires_grad_(not correction_only or name.startswith("head.transforms."))


def _checkpoint(model, step, val_nll, struct, L, source, phase):
    return {"state_dict": model.state_dict(), "step": step, "val_nll": val_nll,
            "struct": {k: (float(v) if v.numel() == 1 else v.cpu()) for k, v in struct.items()},
            "train_N": round(RHO_STAR * L ** 3), "phase": phase, "warm_v4": source,
            "architecture": "v4_exact_plus_toroidal_residual_v10",
            **{k: getattr(model, k) for k in _MODEL_KEYS}}


def _evaluate(model, xva, L, N, gen):
    model.eval()
    with torch.no_grad():
        val = float(torch.stack([-model.log_prob(xva[i:i+64], L).sum()
                                 for i in range(0, len(xva), 64)]).sum() / (len(xva) * N))
        struct = model.teacher_forced_structure(xva[:64], L, gen=gen)
    model.train(); return val, struct


def train(steps=30_000, batch=32, correction_lr=2e-4, mdn_lr=2e-5,
          body_lr=1e-5, correction_only_steps=5_000, val_every=500, seed=109,
          out="mw_gen_N64_v10.pt", warm_v4="mw_gen_N64_v4.pt",
          primary_thin=2, val_frac=0.1, extension="mw_ref_N64_ext.pt",
          device=None, art_path=None, extra_banks=None, num_bins=16):
    torch.manual_seed(seed); device = device or DEV
    warm_path = warm_v4 if os.path.exists(warm_v4) else os.path.join(ART, warm_v4)
    source = torch.load(warm_path, map_location="cpu", weights_only=False)
    model = MWV4ToroidalResidual(
        d_model=source["d_model"], n_layers=source["n_layers"],
        n_heads=source["n_heads"], n_mix=source["n_mix"], rail_k=source["rail_k"],
        num_bins=num_bins, bound=4.0).to(device)
    load_v4_exact(model, warm_path, device)
    if extra_banks is None:
        ext = extension if os.path.exists(extension) else os.path.join(ART, extension)
        extra_banks = [(ext, 32, 1)]
    xtr, xva, L = load_training_bank(art_path=art_path, thin_events=primary_thin,
                                     val_frac=val_frac, extra_banks=extra_banks)
    xtr, xva = xtr.to(device), xva.to(device); N = xtr.shape[1]
    if N != 64: raise ValueError("v10 controlled v4 experiment requires N=64")
    _set_stage(model, correction_only=correction_only_steps > 0)
    correction = list(model.head.transforms.parameters())
    mdn = list(model.head.base.parameters())
    body = [p for name, p in model.named_parameters() if not name.startswith("head.")]
    opt = torch.optim.AdamW([
        {"params": correction, "lr": correction_lr},
        {"params": mdn, "lr": mdn_lr},
        {"params": body, "lr": body_lr}], weight_decay=1e-4)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    eval_gen = torch.Generator(device=device).manual_seed(seed + 2)
    out = out if os.path.isabs(out) else os.path.join(ART, out)
    stem = out[:-3] if out.endswith(".pt") else out
    paths = {"nll": stem + "_best_nll.pt", "struct": stem + "_best_struct.pt",
             "last": stem + "_last.pt"}
    # Save/evaluate the pristine identity model before any update: this is the
    # no-regression anchor and must reproduce v4.
    eval_gen.manual_seed(seed + 2)
    best_nll, struct0 = _evaluate(model, xva, L, N, eval_gen)
    best_struct = float(struct0["composite"])
    pristine = _checkpoint(model, -1, best_nll, struct0, L, warm_path, "identity-v4")
    for path in paths.values(): torch.save(pristine, path)
    print(f"v10 identity-v4: source_step={source.get('step')} source_val={source.get('val_nll')} "
          f"measured_val={best_nll:.4f} TF peak_err={float(struct0['peak_err']):.4f} "
          f"core_coord={float(struct0['core_coord']):.4f}", flush=True)
    for step in range(steps):
        if step == correction_only_steps and correction_only_steps > 0:
            _set_stage(model, correction_only=False)
            print(f"v10 step {step}: MDN/body unfrozen at lr={mdn_lr}/{body_lr}", flush=True)
        idx = torch.randint(len(xtr), (batch,), device=device)
        xb = _augment_batch(xtr[idx], L, aug_gen)
        loss = -model.log_prob(xb, L).mean() / N
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite v10 loss at {step}")
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5); opt.step()
        if step % val_every == 0 or step == steps - 1:
            eval_gen.manual_seed(seed + 2); val, struct = _evaluate(model, xva, L, N, eval_gen)
            phase = "correction-only" if step < correction_only_steps else "joint-low-lr"
            ck = _checkpoint(model, step, val, struct, L, warm_path, phase); torch.save(ck, paths["last"])
            if val < best_nll: best_nll = val; torch.save(ck, paths["nll"])
            score = float(struct["composite"])
            if score < best_struct: best_struct = score; torch.save(ck, paths["struct"])
            print(f"v10 {phase} step {step}: train {float(loss):.4f} val {val:.4f} "
                  f"TF peak_err {float(struct['peak_err']):.4f} core_coord "
                  f"{float(struct['core_coord']):.4f} composite {score:.4f}", flush=True)
    return {**paths, "best_nll": best_nll, "best_struct": best_struct}


def load_generator_v10(path, device=None):
    device = device or DEV
    ck = torch.load(path, map_location=device, weights_only=False)
    model = MWV4ToroidalResidual(**{k: ck[k] for k in _MODEL_KEYS})
    model.load_state_dict(ck["state_dict"]); return model.to(device).eval()


# ---------------------------------------------------------------------------
# Structure-aware fine-tune: reparameterized teacher-forced sample-energy penalty.
# ---------------------------------------------------------------------------

def _h3(dist):
    """SW three-body radial factor h(r) = exp(gamma/(r-a)) inside the cutoff.  Mirrors
    mw_energy._phi3_centers exactly: clamp_max keeps the (masked) out-of-cutoff branch finite
    (exp(gamma/-1e-9) underflows to 0, no inf/nan through torch.where)."""
    return torch.exp(GAMMA / (dist - A_CUT).clamp_max(-1e-9))


def _insertion_energy(placed, xo, L, three_body=True):
    """SW insertion energy ``U_ins[b,j]`` of a candidate placement ``placed[b,j]`` against the causal
    TRUE prefix ``{xo[b,i] : i < j}``, for every step j at once.  Equal (to float precision) to
    ``mw_energy(prefix + placement) - mw_energy(prefix)`` -- the ground-truth anchor asserted by
    ``test_insertion_energy_matches_du``.

    Full causal all-pairs with the within-cutoff mask are numerically identical to _phi3_centers'
    KMAX gather whenever the physical within-cutoff neighbour count <= KMAX -- exactly the regime in
    which mw_energy's own KMAX assert holds; we sum over every prefix neighbour (no topk) so a
    transient bad sample during training cannot trip an assert.

    placed, xo: [B,N,3].  Returns U_ins [B,N] (U_ins[:,0] = 0: empty prefix).
    """
    B, N = placed.shape[:2]
    dev = placed.device
    jj = torch.arange(N, device=dev)
    causal = jj[:, None] > jj[None, :]                              # [Nj,Ni]: neighbour i < step j

    # -- two-body: placement j against every prefix neighbour i<j (reuse mw_energy._phi2) --
    dA = wrap_pm(xo[:, None, :, :] - placed[:, :, None, :], L)      # [B,Nj,Ni,3] center placed_j -> i
    rA = dA.norm(dim=-1)                                            # [B,Nj,Ni]
    U = (_phi2(rA) * causal[None]).sum(-1)                          # [B,Nj]
    if not three_body:
        return U

    # -- 3-body A: placement j is the CENTER over prefix wings a<c (both < j) --
    within_A = (rA < A_CUT) & causal[None]                         # [B,Nj,Ni]
    hA = torch.where(within_A, _h3(rA), torch.zeros_like(rA))
    unitA = dA / rA.clamp_min(1e-12).unsqueeze(-1)                 # [B,Nj,Ni,3]
    cosA = torch.einsum("njad,njcd->njac", unitA, unitA)          # [B,Nj,Na,Nc]
    tri = torch.triu(torch.ones(N, N, dtype=torch.bool, device=dev), 1)
    termA = LAMBDA3 * (cosA - COS0) ** 2 * hA[..., :, None] * hA[..., None, :]
    maskA = within_A[..., :, None] & within_A[..., None, :] & tri[None, None]
    U = U + (termA * maskA).sum((-1, -2))                          # [B,Nj]

    # -- 3-body B: a prefix particle k<j is the CENTER; wings = placement j and prefix m<j, m!=k
    #    (captures BOTH vertex-k and vertex-m triplets of {j,k,m} as the distinct (center,wing) entries
    #    (k,m) and (m,k); vertex-j is 3-body A).  Angle is measured at center k. --
    dBj = wrap_pm(placed[:, :, None, :] - xo[:, None, :, :], L)     # [B,Nj,Nk,3] center k -> placement j
    rBj = dBj.norm(dim=-1)
    within_Bj = (rBj < A_CUT) & causal[None]                       # k < j
    hBj = torch.where(within_Bj, _h3(rBj), torch.zeros_like(rBj))
    unitBj = dBj / rBj.clamp_min(1e-12).unsqueeze(-1)

    dBm = wrap_pm(xo[:, None, :, :] - xo[:, :, None, :], L)         # [B,Nk,Nm,3] center k -> wing m
    rBm = dBm.norm(dim=-1)
    eye = torch.eye(N, dtype=torch.bool, device=dev)
    within_Bm = (rBm < A_CUT) & ~eye[None]                         # m != k, within cutoff
    hBm = torch.where(within_Bm, _h3(rBm), torch.zeros_like(rBm))
    unitBm = dBm / rBm.clamp_min(1e-12).unsqueeze(-1)

    cosB = torch.einsum("njkd,nkmd->njkm", unitBj, unitBm)         # [B,Nj,Nk,Nm] angle at center k
    mask_mj = jj[None, :] < jj[:, None]                            # [Nj,Nm]: wing m < step j
    termB = LAMBDA3 * (cosB - COS0) ** 2 * hBj[..., :, None] * hBm[:, None, :, :]
    maskB = within_Bj[..., :, None] & within_Bm[:, None, :, :] & mask_mj[None, :, None, :]
    U = U + (termB * maskB).sum((-1, -2))                          # [B,Nj]
    return U


def _smooth_clip(u_ins, clip_e):
    """Bounded, gradient-preserving clip clip_e*tanh(U/clip_e) in (-clip_e, clip_e).  tanh (NOT clamp,
    which zeroes gradients past the clip) so a core blow-up is bounded in value yet its gradient still
    steers the placement (and tanh' -> 0 tempers the huge r^-4 pair gradient of a hard clash)."""
    return clip_e * torch.tanh(u_ins / clip_e)


def _struct_loss(model, xo, t, s, bound, L, clip_e=10.0, three_body=True, h=None, gen=None):
    """Reparameterized teacher-forced sample-energy penalty.  Given canonical-ordered configs ``xo``
    and the shared teacher-forced hidden ``h`` (as in log_prob), SAMPLE a placement x_hat_j from the
    head at every step with GRADIENTS FLOWING, then penalize its SW insertion energy against the true
    causal prefix.

    Reparameterization: ``head.sample`` draws the MDN component index discretely -- and that multinomial
    is inherently DETACHED (used only to gather mu_k/L_k), so component-selection gradients are OMITTED;
    the mixture WEIGHTS keep training through the NLL term.  Everything downstream stays on the autograd
    tape: the within-component draw z = mu_k + L_k*eps, the u = bound*tanh(z) squash, the toroidal
    circular-spline transport, and the anchor+scale chart back to Cartesian.
    """
    if h is None:
        u = wrap_pm(xo - t[None], L) / s
        h = model._hidden(u, xo, t, L)
    placed_y, _ = model.head.sample(h, bound, gen=gen)             # reparam draw; component index detached
    placed = torch.remainder(t[None] + s * placed_y, L)           # chart -> Cartesian (differentiable)
    u_ins = _insertion_energy(placed, xo, L, three_body=three_body)
    u_pen = _smooth_clip(u_ins, clip_e)
    return u_pen.mean(), u_ins.detach()


def _evaluate_struct(model, xva, L, N, gen):
    """Full-val NLL/particle + TF pooled-predecessor structure, with shell2 (TF g at r=1.85) added."""
    model.eval()
    with torch.no_grad():
        val = float(torch.stack([-model.log_prob(xva[i:i + 64], L).sum()
                                 for i in range(0, len(xva), 64)]).sum() / (len(xva) * N))
        struct = model.teacher_forced_structure(xva[:64], L, gen=gen)
    r, gr = struct["r"], struct["gr"]
    struct["shell2"] = gr[(r - 1.85).abs().argmin()]              # second-shell TF g(r) (data ~1.20)
    model.train(); return val, struct


def _checkpoint_struct(model, step, val_nll, struct, L, warm):
    return {"state_dict": model.state_dict(), "step": step, "val_nll": val_nll,
            "struct": {k: (float(v) if torch.is_tensor(v) and v.numel() == 1
                           else (v.cpu() if torch.is_tensor(v) else v))
                       for k, v in struct.items()},
            "train_N": round(RHO_STAR * L ** 3), "phase": "struct-finetune", "warm": warm,
            "architecture": "v10_struct_finetune",
            **{k: getattr(model, k) for k in _MODEL_KEYS}}


def finetune_struct(steps=20_000, batch=24, lr=1e-4, lam=0.05, lam_warmup=1_000, clip_e=10.0,
                    three_body=True, val_every=500, warm="mw_gen_N64_v10_last.pt",
                    out="mw_gen_N64_v10s.pt", seed=71, primary_thin=2, val_frac=0.1,
                    extension="mw_ref_N64_ext.pt", device=None, art_path=None, extra_banks=None):
    """Structure-aware fine-tune of a warm v10 checkpoint: L = NLL/N + lam_t * L_struct, where lam_t
    ramps 0 -> lam linearly over lam_warmup steps (don't shock the warm density).  Multi-bank data,
    augmentation, and best_nll/best_struct/last checkpointing mirror v10.train(); the density code is
    untouched.  See the module docstring for the pre-registered GO gate."""
    torch.manual_seed(seed); device = device or DEV
    warm_path = warm if os.path.exists(warm) else os.path.join(ART, warm)
    model = load_generator_v10(warm_path, device=device); model.train()
    if extra_banks is None:
        ext = extension if os.path.exists(extension) else os.path.join(ART, extension)
        extra_banks = [(ext, 32, 1)]
    xtr, xva, L = load_training_bank(art_path=art_path, thin_events=primary_thin,
                                     val_frac=val_frac, extra_banks=extra_banks)
    xtr, xva = xtr.to(device), xva.to(device); N = xtr.shape[1]
    if N != 64: raise ValueError("v10 structure fine-tune is the controlled N=64 experiment")
    t, rank, R = mw_scaffold(N, L, device)
    s, bound = (L / R) / 2.0, float(R)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    aug_gen = torch.Generator(device=device).manual_seed(seed + 1)
    eval_gen = torch.Generator(device=device).manual_seed(seed + 2)
    out = out if os.path.isabs(out) else os.path.join(ART, out)
    stem = out[:-3] if out.endswith(".pt") else out
    paths = {"nll": stem + "_best_nll.pt", "struct": stem + "_best_struct.pt", "last": stem + "_last.pt"}
    # Warm baseline BEFORE any update: the no-regression anchor.
    eval_gen.manual_seed(seed + 2)
    best_nll, struct0 = _evaluate_struct(model, xva, L, N, eval_gen)
    best_struct = float(struct0["composite"])
    base = _checkpoint_struct(model, -1, best_nll, struct0, L, warm_path)
    for path in paths.values(): torch.save(base, path)
    print(f"v10s warm baseline from {warm_path}: val {best_nll:.4f} "
          f"TF peak_err {float(struct0['peak_err']):.4f} shell2 {float(struct0['shell2']):.4f} "
          f"core {float(struct0['core_coord']):.4f} (three_body={three_body} lam={lam})", flush=True)
    for step in range(steps):
        idx = torch.randint(len(xtr), (batch,), device=device)
        xb = _augment_batch(xtr[idx], L, aug_gen)
        x = torch.remainder(xb, L)
        perm = canonical_order(x, L, R, rank)
        xo = torch.gather(x, 1, perm[..., None].expand(-1, -1, 3))
        u = wrap_pm(xo - t[None], L) / s
        h = model._hidden(u, xo, t, L)                            # ONE teacher-forced pass, shared
        nll = -(model.head.log_prob(h, u, bound).sum(-1) - 3 * N * math.log(s)).mean() / N
        struct, _ = _struct_loss(model, xo, t, s, bound, L, clip_e=clip_e, three_body=three_body, h=h)
        lam_t = lam * min(1.0, step / max(1, lam_warmup))
        loss = nll + lam_t * struct
        if not torch.isfinite(loss): raise FloatingPointError(f"nonfinite v10s loss at {step}")
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % val_every == 0 or step == steps - 1:
            eval_gen.manual_seed(seed + 2); val, st = _evaluate_struct(model, xva, L, N, eval_gen)
            ck = _checkpoint_struct(model, step, val, st, L, warm_path); torch.save(ck, paths["last"])
            if val < best_nll: best_nll = val; torch.save(ck, paths["nll"])
            score = float(st["composite"])
            if score < best_struct: best_struct = score; torch.save(ck, paths["struct"])
            print(f"v10s step {step}: train_nll {float(nll):.4f} struct {float(struct):.4f} "
                  f"(lam_t {lam_t:.4f}) val {val:.4f} TF peak_err {float(st['peak_err']):.4f} "
                  f"shell2 {float(st['shell2']):.4f} core {float(st['core_coord']):.4f} "
                  f"composite {score:.4f}", flush=True)
    return {**paths, "best_nll": best_nll, "best_struct": best_struct}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="v10 toroidal residual: identity pretrain + structure fine-tune")
    p.add_argument("--finetune", action="store_true",
                   help="run the structure-aware fine-tune (finetune_struct) instead of train()")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--correction-only-steps", type=int, default=5_000)
    p.add_argument("--bins", type=int, default=16)
    p.add_argument("--out", default=None)
    p.add_argument("--seed", type=int, default=None)
    # finetune-only knobs
    p.add_argument("--batch", type=int, default=24)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lam", type=float, default=0.05)
    p.add_argument("--lam-warmup", type=int, default=1_000)
    p.add_argument("--clip-e", type=float, default=10.0)
    p.add_argument("--no-three-body", action="store_true")
    p.add_argument("--warm", default="mw_gen_N64_v10_last.pt")
    a = p.parse_args()
    if a.finetune:
        finetune_struct(steps=a.steps or 20_000, batch=a.batch, lr=a.lr, lam=a.lam,
                        lam_warmup=a.lam_warmup, clip_e=a.clip_e, three_body=not a.no_three_body,
                        warm=a.warm, out=a.out or "mw_gen_N64_v10s.pt", seed=a.seed or 71)
    else:
        train(steps=a.steps or 30_000, correction_only_steps=a.correction_only_steps,
              num_bins=a.bins, out=a.out or "mw_gen_N64_v10.pt", seed=a.seed or 109)
