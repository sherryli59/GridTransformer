"""JointBlockFlow: the joint continuous (x,u) block flow (poly Task J2).

Per docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md, the flow no longer
proposes a position-only relax after a discrete sigma-permutation (liquid_coupling_flow/poly/block_flow.py,
the position-only predecessor -- kept, refuted variant, NOT deleted). Instead each block particle carries a
state z = (x, u) in R^4, u = Phi^-1(F(sigma)) (see liquid_coupling_flow/poly/semigrand.py for the numba
u<->sigma maps this mirrors), and ONE velocity field outputs (dx/dt, du/dt) simultaneously, transporting
position AND composition together.

ARCHITECTURE INVESTIGATION (do first, per the task brief) -- findings:
liquid_coupling_flow/ipl44/learndiffeq/learndiffeq/particles/velocities/egnn_traceable.py's EGNN_dynamics
consumes species ONLY as an integer label: F.one_hot(a_nbr / a_central, n_species) is concatenated into the
node features `h` (in_node_nf = 1 + n_species, fixed at __init__ time) and into pot_model's input. There is
no continuous injection point: n_species is baked into the Linear layer widths at construction, and F.one_hot
requires an integer index tensor. We are NOT permitted to edit egnn_traceable.py or ka3d_cavity_egnn.py, so
route (A) (feed continuous u through a node-feature slot) is NOT achievable without modifying shared code.

We therefore use ROUTE (B): keep the existing discrete sigma_to_bin(sigma_of_u(u_t)) conditioning for the
POSITION field (CavityCondEGNN.vel_div, imported unmodified via CavityBlockFlow, exactly as block_flow.py
uses it) -- recomputed at EVERY RK4 sub-stage from the CURRENT u_t, so the position field's species
conditioning tracks the flowing composition. CAVEAT (documented per the brief): sigma_to_bin is piecewise
constant in u, so it contributes exactly zero to d(species)/du a.e. -- benign for exactness (a measure-zero
set of bin boundaries), but it means the position field only feels sigma through discrete bin jumps.

For the u-VELOCITY head we build a small, independent (not weight-shared with the position EGNN) per-particle
MLP `_UHead`, fed by a fixed-size k-NN feature vector constructed entirely inside this module (no shared-file
edit): for each block particle i, we take the K nearest OTHER real particles (movers ex-self, plus env) by
CURRENT position, and featurize [distance, neighbour u, validity] per neighbour slot, masked-mean-pooled and
combined with [u_i, t]. This gives u_dot_i genuine smooth dependence on x (via the pooled distances) and on
ALL real u_j (via the pooled neighbour-u channel), satisfying the "u_dot_i may depend on x and on all u_j"
requirement, while keeping the CavityCondEGNN position pathway completely untouched.

EXACTNESS: the divergence needed is div_x(vel_x) [existing analytical pathway, unchanged -- conditioning the
position field on u adds no new div_x terms since u is not a position coordinate the EGNN's autograd ever
differentiates] PLUS Sum_i d(u_dot_i)/d(u_i) [new]. We compute the latter EXACTLY via torch.func.jacrev,
vmapped over the batch: since k <= k_max <= 24, a full k x k Jacobian per batch row is cheap (no Hutchinson
estimator anywhere). We only ever read the DIAGONAL of that Jacobian (trace = sum of on-diagonal partials);
off-diagonal cross terms (d(u_dot_i)/d(u_j), i!=j; d(x_dot)/d(u); d(u_dot)/d(x)) are real but do not enter a
trace, exactly as the design brief specifies.

DENSITY DEFINITION (mirrors block_flow.py's, extended to the joint 4D state z=(x,u)): q(z_target | z_center,
env) is DEFINED by REVERSE fixed-grid RK4 (RK4_STEPS=24 steps), integrating the joint velocity field
(vel_x, vel_u) BACKWARD from z_target over the SAME grid t: 1 -> 0 used by the forward sampler, accumulating
the RAW (not negated) per-step (div_x + div_u) along the path, landing on the base point z0 at t=0:
    logq = logN(z0_x; x_center, BASE_W^2*I) + logN(z0_u; u_center, BASE_W_U^2*I) + sum_of_(div_x+div_u)
`propose` NEVER accumulates its own logq during the forward push -- it always calls `logq_of` on the sample
it just produced (single code path, not merely close), exactly as block_flow.py does. BASE_W=0.35 (x-channel,
fixed, matches block_flow.py); BASE_W_U is a constructor argument (default 0.3).

PADDING: x-dummies use the identical (DUMMY0 + DUMMY_D*(offset+j), 0, 0) spread as block_flow.py (imported
`_dummies` directly, byte-identical). u-dummies use pad value 0.0. Padded rows (mover index >= n_real) get
velocity forced to exactly zero in BOTH channels: the x-channel via block_flow.py's `_vel_div_masked` pattern
(ported here unchanged, same NaN-landmine fix: unfiltered top-k neighbour selection can otherwise give a
dummy row a large spurious self-divergence); the u-channel by construction -- `_udot_single` multiplies its
raw output by the (u-independent) real/pad mask BEFORE returning, so the masked output is IDENTICALLY zero
for padded rows regardless of upstream computation, which makes their row of the (u_dot vs u) Jacobian
identically zero too (a constant-zero function of u has a zero gradient everywhere) -- padded rows are
therefore automatically excluded from Sum_i d(u_dot_i)/d(u_i) with no separate slicing needed. Neighbour
SELECTION for the k-NN u-feature also explicitly excludes non-real candidates (their pairwise distance is
set to +inf before top-k), so a real particle's u_dot can never be contaminated by a dummy's pad-value u=0
sneaking in as a "neighbour" (this is the u-channel's mirror of `_vel_div_masked`'s exclusion).

BATCHING: `_integrate`/`_field` operate on true batched tensors [B, k_max, ...] (no `unsqueeze(0)` B=1
trick like block_flow.py's `_integrate` used) -- `propose`/`logq_of` are thin B=1 wrappers around
`propose_batch`/`logq_of_batch`. We assume the REAL mover count k and REAL env count (pre-truncation) are
uniform across a batch (the caller already hands in rectangular [B,k,3]/[B,m,3] tensors, which is the same
assumption block_flow.py's single-sample API makes, just carrying an explicit leading B dim) -- env
truncate-to-nearest-m_env is still evaluated per batch row independently (rows can pick different nearest
subsets), only the COUNT n/target_n comparison (and hence n_real) is shared across the batch.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.func import vmap, jacrev

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.poly.block_flow import (
    SIG_MIN, SIG_MAX, NSIG, BASE_W, RK4_STEPS, DUMMY0, DUMMY_D,  # noqa: F401  (DUMMY0/DUMMY_D re-exported)
    _dummies, _A, _B, _LOG_2PI,
)

U_PAD = 0.0
U_KNN_DEFAULT = 8
U_HIDDEN_DEFAULT = 64


def sigma_of_u_t(u):
    """Torch, batched, any shape: sigma(u) = (a - Phi(u)*(a-b))^-0.5, Phi = std normal CDF (erf form).
    Matches semigrand.py's numba `sigma_of_u` exactly (same A_CONST/B_CONST = SIG_MIN^-2/SIG_MAX^-2)."""
    phi = 0.5 * (1.0 + torch.erf(u * 0.7071067811865476))
    phi = phi.clamp(1e-12, 1.0 - 1e-12)
    inner = (_A - phi * (_A - _B)).clamp_min(1e-12)
    return inner.pow(-0.5)


def sigma_to_bin_t(sig):
    """Torch counterpart of block_flow.sigma_to_bin (same analytic quantile-bin formula), used to
    recompute the position field's species conditioning from a live (on-device, differentiating) u_t
    at every RK4 sub-stage without a host round-trip."""
    F = (_A - sig ** -2) / (_A - _B)
    F = F.clamp(0.0, 1.0)
    b = (F * NSIG).floor().clamp(max=float(NSIG - 1))
    return b.long()


def _to_tensor(a, dtype, device):
    if torch.is_tensor(a):
        return a.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


class _UHead(nn.Module):
    """Independent (not weight-shared with the position EGNN) per-particle u-velocity head. Input per
    particle i: masked-mean-pooled [distance, neighbour-u, validity] over its K nearest REAL neighbours
    (by current x), concatenated with [u_i, t]. Last layer zero-init (flow-matching identity convention,
    matches CavityBlockFlow's zero-init of pot_model[-1]): untrained -> u_dot == 0 everywhere."""

    def __init__(self, k_nn=U_KNN_DEFAULT, hidden=U_HIDDEN_DEFAULT):
        super().__init__()
        self.k_nn = int(k_nn)
        self.nbr_mlp = nn.Sequential(nn.Linear(3, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.out_mlp = nn.Sequential(nn.Linear(hidden + 2, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        with torch.no_grad():
            self.out_mlp[-1].weight.zero_()
            self.out_mlp[-1].bias.zero_()

    def forward(self, u_row, t, knn_d, knn_u, valid):
        """u_row[k_max], t: python float, knn_d/knn_u/valid[k_max,K] -> u_dot_raw[k_max] (UNMASKED by
        real/pad -- caller applies the pad mask)."""
        feat = torch.stack([knn_d, knn_u, valid], dim=-1)          # [k_max,K,3]
        enc = self.nbr_mlp(feat) * valid.unsqueeze(-1)              # zero contribution from invalid slots
        cnt = valid.sum(-1, keepdim=True).clamp_min(1.0)
        pooled = enc.sum(-2) / cnt                                  # [k_max,hidden] masked mean
        t_col = torch.full_like(u_row, float(t)).unsqueeze(-1)
        z = torch.cat([pooled, u_row.unsqueeze(-1), t_col], dim=-1)
        return self.out_mlp(z).squeeze(-1)                          # [k_max]


class JointBlockFlow(nn.Module):
    """Fixed-size (k_max movers, m_env env slots) joint continuous (x,u) block flow. See module docstring
    for the density definition, route-B conditioning choice, and padding/masking contract."""

    def __init__(self, k_max, m_env, hidden_nf=128, n_layers=4, base_w_u=0.3,
                 u_knn=U_KNN_DEFAULT, u_hidden=U_HIDDEN_DEFAULT):
        super().__init__()
        self.k_max = int(k_max)
        self.m_env = int(m_env)
        self.base_w_u = float(base_w_u)
        self.u_knn = int(u_knn)
        # ORDER MATTERS for RNG-init reproducibility across differing k_max (test 4, dummy inertness):
        # u_head is built FIRST so its random init never depends on how many random draws _cbf's
        # construction below consumes. CavityCondEGNN's LAST-constructed submodule (pot_com_model) has
        # a k_max/m_env-DEPENDENT shape (its input width includes max_nb_neighbors, which depends on the
        # total cloud size P=k_max+m_env) even though pot_com_model is UNUSED by vel_div/vel_div_masked
        # (see ka3d_cavity_egnn.py's CAVEAT docstring) -- if u_head were built AFTER _cbf, two flows that
        # differ only in k_max would draw different numbers of random values for pot_com_model and hence
        # start u_head's own init from a shifted RNG stream, breaking the "dummy padding is inert" bit-
        # exactness test even with correct masking. Building u_head first sidesteps this entirely: its
        # init and every _cbf submodule actually used by vel_div (egnn.*, pot_model) are constructed
        # before pot_com_model in both variants, so same-seed instances get IDENTICAL real weights.
        self.u_head = _UHead(k_nn=self.u_knn, hidden=u_hidden)
        self._cbf = CavityBlockFlow(n_cage=self.m_env, k=self.k_max, r_c=2.5,
                                     hidden_nf=hidden_nf, n_layers=n_layers,
                                     n_species=NSIG, max_neighbors=16)

    @property
    def ce(self):
        """The exact field the RK4 integrator uses for the POSITION channel -- CavityCondEGNN.vel_div's
        underlying egnn, unmodified (identical to block_flow.PolyBlockFlow.ce)."""
        return self._cbf.ce

    def _device(self):
        return next(self.parameters()).device

    # ---- padding (x,u jointly; batched, uniform real-count across the batch) -------------------------
    def _pad_xu_batch(self, x, u, target_n, offset, truncate_nearest):
        """x[B,n,3], u[B,n] -> (x[B,target_n,3], u[B,target_n], n_real). Ports block_flow.PolyBlockFlow's
        `_pad` (truncate-to-nearest-origin for env, dummy-pad otherwise) to a genuine batch leading dim;
        n (real count pre-truncation) and hence n_real are assumed uniform across the batch -- see module
        docstring's BATCHING note -- only the per-row NEAREST SUBSET (when truncating) varies by row."""
        B, n, _ = x.shape
        if n > target_n:
            if not truncate_nearest:
                raise ValueError(f"{n} movers > k_max={target_n}")
            d2 = (x ** 2).sum(-1)                                        # [B,n]
            idx = torch.argsort(d2, dim=1)[:, :target_n]                 # [B,target_n]
            x = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            u = torch.gather(u, 1, idx)
            n = target_n
        n_real = n
        if n < target_n:
            pad_x = _dummies(target_n - n, x.device, offset=offset).unsqueeze(0).expand(B, -1, -1)
            pad_u = torch.full((B, target_n - n), U_PAD, dtype=u.dtype, device=u.device)
            x = torch.cat([x, pad_x], dim=1)
            u = torch.cat([u, pad_u], dim=1)
        return x, u, n_real

    def _prep_movers(self, x, u):
        return self._pad_xu_batch(x, u, self.k_max, offset=0, truncate_nearest=False)

    def _prep_env(self, env_x, env_u):
        x, u, n_real = self._pad_xu_batch(env_x, env_u, self.m_env, offset=self.k_max, truncate_nearest=True)
        return x, u, n_real

    # ---- position channel: ported verbatim from block_flow.PolyBlockFlow._vel_div_masked -------------
    def _vel_div_masked(self, cloud, t, sp, n_real):
        """Identical computation/rationale to block_flow.PolyBlockFlow._vel_div_masked (see that
        docstring for the NaN-landmine this masking fixes) -- ported here (not imported) because `sp`
        must be recomputed from the CURRENT u_t at every RK4 sub-stage, which block_flow's fixed-label
        API does not support."""
        diff = torch.is_grad_enabled()
        vel_full, divpp = self.ce.egnn.forward_and_perparticle_divergence(cloud, t, sp, differentiable=diff)
        vel = vel_full[:, :self.k_max]
        if n_real < self.k_max:
            vel = vel.clone()
            vel[:, n_real:, :] = 0.0
        div = divpp[:, :n_real].sum(-1) if n_real > 0 else torch.zeros(
            cloud.shape[0], device=cloud.device, dtype=cloud.dtype)
        return (vel, div) if diff else (vel.detach(), div.detach())

    # ---- u channel: independent k-NN featurizer + _UHead, EXACT diagonal Jacobian via jacrev/vmap -----
    def _udot_single(self, u_row, x_row, envx_row, envu_row, mask_row, t):
        """UNBATCHED (no leading B dim -- called under torch.func.vmap, which supplies B): u_row[k_max],
        x_row[k_max,3], envx_row[m_env,3], envu_row[m_env], mask_row[k_max+m_env] bool (True = real,
        i.e. not a dummy-pad row), t: python float -> u_dot[k_max]. jacrev differentiates this w.r.t.
        u_row (argnums=0 default); x_row/envx_row/envu_row/mask_row are held constant, matching "u_dot_i
        may depend on x and all u_j -- only d(u_dot)/d(u) is needed for the divergence"."""
        x_all = torch.cat([x_row, envx_row], dim=0)                      # [P,3]
        u_all = torch.cat([u_row, envu_row], dim=0)                      # [P]
        d = torch.cdist(x_row, x_all)                                     # [k_max,P]
        idx = torch.arange(self.k_max, device=x_row.device)
        inf_row = torch.full((self.k_max,), float("inf"), device=x_row.device, dtype=d.dtype)
        d = d.index_put((idx, idx), inf_row)                              # exclude self
        d = d.masked_fill(~mask_row.unsqueeze(0), float("inf"))           # exclude non-real candidates
        K = min(self.u_knn, d.shape[-1])
        knn_d, knn_idx = torch.topk(d, k=K, dim=-1, largest=False)
        valid = torch.isfinite(knn_d).to(u_row.dtype)
        knn_d = torch.where(torch.isfinite(knn_d), knn_d, torch.zeros_like(knn_d))
        knn_u = torch.gather(u_all.unsqueeze(0).expand(self.k_max, -1), 1, knn_idx) * valid
        u_dot_raw = self.u_head(u_row, t, knn_d, knn_u, valid)
        mover_real = mask_row[:self.k_max].to(u_row.dtype)
        return u_dot_raw * mover_real                                     # padded rows -> IDENTICALLY 0

    def _field(self, x_t, u_t, env_x, env_bin, env_u, t, n_real, n_env_real):
        """One evaluation of the joint velocity field at (x_t,u_t,t): returns (vel_x,div_x,vel_u,div_u),
        all [B,...] / [B]. div_x via the existing analytical EGNN pathway (unchanged by u-conditioning);
        div_u = Sum_i d(u_dot_i)/d(u_i), computed EXACTLY (full k_max x k_max Jacobian, diagonal only) via
        vmap(jacrev(_udot_single)) over the batch -- no Hutchinson estimator anywhere."""
        sp_movers = sigma_to_bin_t(sigma_of_u_t(u_t))                     # [B,k_max], recomputed live
        sp = torch.cat([sp_movers, env_bin], dim=1)                       # [B,P]
        cloud = torch.cat([x_t, env_x], dim=1)                            # [B,P,3]
        vel_x, div_x = self._vel_div_masked(cloud, t, sp, n_real)

        B = x_t.shape[0]
        device = x_t.device
        mover_mask = torch.arange(self.k_max, device=device) < n_real
        env_mask = torch.arange(self.m_env, device=device) < n_env_real
        real_mask = torch.cat([mover_mask, env_mask]).unsqueeze(0).expand(B, -1)   # [B,P]

        def udot_fn(u_row, x_row, envx_row, envu_row, mask_row):
            return self._udot_single(u_row, x_row, envx_row, envu_row, mask_row, t)

        vel_u = vmap(udot_fn)(u_t, x_t, env_x, env_u, real_mask)                    # [B,k_max]
        jac = vmap(jacrev(udot_fn))(u_t, x_t, env_x, env_u, real_mask)              # [B,k_max,k_max]
        div_u = torch.diagonal(jac, dim1=-2, dim2=-1).sum(-1)                       # [B]
        return vel_x, div_x, vel_u, div_u

    def _integrate(self, x0, u0, env_x, env_bin, env_u, reverse, n_real, n_env_real):
        """Fixed-grid RK4 (RK4_STEPS steps) of the joint state (x,u), dl/dt = div_x + div_u (RAW, not
        negated -- see module docstring). Batched: x0[B,k_max,3], u0[B,k_max]. forward (reverse=False):
        t 0->1; reverse=True: t 1->0. Mirrors block_flow.PolyBlockFlow._integrate's structure exactly,
        generalized off its B=1 `unsqueeze(0)` trick to a true batch dim."""
        dt = (-1.0 if reverse else 1.0) / RK4_STEPS
        x, u = x0, u0
        t = 1.0 if reverse else 0.0
        l = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)

        def f(xx, uu, tt):
            vx, dx, vu, du = self._field(xx, uu, env_x, env_bin, env_u, tt, n_real, n_env_real)
            return vx, dx, vu, du

        for _ in range(RK4_STEPS):
            k1x, k1dx, k1u, k1du = f(x, u, t)
            k2x, k2dx, k2u, k2du = f(x + 0.5 * dt * k1x, u + 0.5 * dt * k1u, t + 0.5 * dt)
            k3x, k3dx, k3u, k3du = f(x + 0.5 * dt * k2x, u + 0.5 * dt * k2u, t + 0.5 * dt)
            k4x, k4dx, k4u, k4du = f(x + dt * k3x, u + dt * k3u, t + dt)
            x = x + (dt / 6.0) * (k1x + 2.0 * k2x + 2.0 * k3x + k4x)
            u = u + (dt / 6.0) * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
            l = l + (dt / 6.0) * ((k1dx + k1du) + 2.0 * (k2dx + k2du) + 2.0 * (k3dx + k3du) + (k4dx + k4du))
            t = t + dt
        return x, u, l

    def _prep_env_bin(self, env_x_t, env_u_t):
        """Pad/truncate env once, compute its (fixed -- env never moves) species bin once, and force
        padded env rows to bin 0 (matches block_flow.PolyBlockFlow._prep_env's explicit dummy label=0
        convention -- here `sigma_to_bin_t(sigma_of_u_t(U_PAD))` would give an arbitrary nonzero bin, so
        we zero it explicitly rather than rely on an accident of U_PAD's placement in [SIG_MIN,SIG_MAX])."""
        env_xp, env_up, n_env_real = self._prep_env(env_x_t, env_u_t)
        env_bin = sigma_to_bin_t(sigma_of_u_t(env_up))
        if n_env_real < self.m_env:
            env_bin = env_bin.clone()
            env_bin[:, n_env_real:] = 0
        return env_xp, env_up, env_bin, n_env_real

    # ---- public batched API -----------------------------------------------------------------------
    @torch.no_grad()
    def propose_batch(self, x_old, u_old, env_x, env_u, gen):
        """x_old[B,k,3], u_old[B,k], env_x[B,m,3], env_u[B,m], gen: CPU torch.Generator (a cuda Generator
        crashes torch.randn -- landmine) -> (x_new[B,k,3] np.float32, u_new[B,k] np.float32,
        logq[B] np.float64). base z = (x_old,u_old) + N(0,BASE_W^2)/N(0,BASE_W_U^2) noise generated on
        CPU via `gen` then moved to device (mirrors block_flow.PolyBlockFlow.propose), forward RK4 t:0->1,
        then re-evaluate the produced sample's density via `logq_of_batch` -- one code path, never two."""
        device = self._device()
        x_old_t = _to_tensor(x_old, torch.float32, device)
        u_old_t = _to_tensor(u_old, torch.float32, device)
        env_x_t = _to_tensor(env_x, torch.float32, device)
        env_u_t = _to_tensor(env_u, torch.float32, device)
        B, k, _ = x_old_t.shape

        noise_x = torch.randn(B, k, 3, generator=gen, dtype=torch.float32).to(device)
        noise_u = torch.randn(B, k, generator=gen, dtype=torch.float32).to(device)
        z_x = x_old_t + BASE_W * noise_x
        z_u = u_old_t + self.base_w_u * noise_u

        movers_x, movers_u, n_real = self._prep_movers(z_x, z_u)
        env_xp, env_up, env_bin, n_env_real = self._prep_env_bin(env_x_t, env_u_t)

        x1, u1, _ = self._integrate(movers_x, movers_u, env_xp, env_bin, env_up,
                                     reverse=False, n_real=n_real, n_env_real=n_env_real)
        x_new = x1[:, :k].detach().cpu().numpy().astype(np.float32)
        u_new = u1[:, :k].detach().cpu().numpy().astype(np.float32)

        logq = self.logq_of_batch(x_new, u_new, x_old, u_old, env_x, env_u)
        return x_new, u_new, logq

    @torch.no_grad()
    def logq_of_batch(self, x_target, u_target, x_center, u_center, env_x, env_u):
        """Density of producing z_target=(x_target,u_target)[B,k,{3,1}] when the base is centered at
        z_center (the reverse MH leg). Reverse RK4 t:1->0 from z_target lands at the base point z0;
        logq = logN(z0_x;x_center,BASE_W^2) + logN(z0_u;u_center,BASE_W_U^2) + path's raw (div_x+div_u)
        integral. Returns np.ndarray[B] (float64)."""
        device = self._device()
        x_target_t = _to_tensor(x_target, torch.float32, device)
        u_target_t = _to_tensor(u_target, torch.float32, device)
        x_center_t = _to_tensor(x_center, torch.float32, device)
        u_center_t = _to_tensor(u_center, torch.float32, device)
        env_x_t = _to_tensor(env_x, torch.float32, device)
        env_u_t = _to_tensor(env_u, torch.float32, device)
        B, k, _ = x_target_t.shape

        movers_x, movers_u, n_real = self._prep_movers(x_target_t, u_target_t)
        env_xp, env_up, env_bin, n_env_real = self._prep_env_bin(env_x_t, env_u_t)

        zx_full, zu_full, path_sum = self._integrate(movers_x, movers_u, env_xp, env_bin, env_up,
                                                       reverse=True, n_real=n_real, n_env_real=n_env_real)
        zx_real = zx_full[:, :k]
        zu_real = zu_full[:, :k]

        dx = zx_real - x_center_t
        du = zu_real - u_center_t
        log_nx = (-0.5 * (dx * dx).sum(dim=(1, 2)) / (BASE_W ** 2)
                  - k * 3 * 0.5 * (_LOG_2PI + 2.0 * math.log(BASE_W)))
        log_nu = (-0.5 * (du * du).sum(dim=1) / (self.base_w_u ** 2)
                  - k * 1 * 0.5 * (_LOG_2PI + 2.0 * math.log(self.base_w_u)))
        logq = log_nx + log_nu + path_sum
        return logq.detach().cpu().numpy().astype(np.float64)

    # ---- single-sample wrappers (thin B=1 shims over the batched API -- one code path) --------------
    def propose(self, x_old, u_old, env_x, env_u, gen):
        """x_old[k,3], u_old[k], env_x[m,3], env_u[m], gen: torch.Generator -> (x_new[k,3], u_new[k],
        logq: float)."""
        x_new, u_new, logq = self.propose_batch(
            np.asarray(x_old)[None], np.asarray(u_old)[None],
            np.asarray(env_x)[None], np.asarray(env_u)[None], gen)
        return x_new[0], u_new[0], float(logq[0])

    def logq_of(self, x_target, u_target, x_center, u_center, env_x, env_u):
        """-> float. See logq_of_batch."""
        logq = self.logq_of_batch(
            np.asarray(x_target)[None], np.asarray(u_target)[None],
            np.asarray(x_center)[None], np.asarray(u_center)[None],
            np.asarray(env_x)[None], np.asarray(env_u)[None])
        return float(logq[0])
