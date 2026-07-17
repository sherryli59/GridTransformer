"""SwapBlockFlow: the swap-endpoint sigma(t)-conditioned block flow (poly Task J3a, PHASE 1 of
docs/superpowers/specs/2026-07-17-joint-sigma-x-continuous-flow-amendment.md's DECISION 2026-07-17).

PHASE 1 walks back from the joint (x,u) semi-grand design (joint_flow.py, kept -- phase-2 asset,
gated on phase-1 success) to the CANONICAL fixed-composition ensemble: sigma never becomes a flowed
state variable. Instead, a classical swap MC endpoint (pick pair (i,j) inside a k-NN block, exchange
sigma_i <-> sigma_j exactly) is treated as CONTINUOUS TIME-DEPENDENT CONDITIONING of a position-only
flow: sigma(t) = (1-t)*sigma_start + t*sigma_end interpolates linearly from the pre-swap to the
post-swap per-particle sigma assignment (only the two swapped movers' sigma actually changes along
this path; every other mover and the whole env is constant in t). Positions flow to accommodate the
endpoint swap; sigma itself is never a continuous physical state, only a conditioning signal -- this
recovers the classical two-particle swap kernel's target ensemble exactly (no mu, no composition
change), unlike joint_flow.py's semi-grand pi(x,u) ~ exp(-beta U) * prod phi(u_i).

Reuses block_flow.py's validated position-only machinery UNCHANGED: BASE_W Gaussian base, fixed-grid
RK4 (RK4_STEPS steps), CavityBlockFlow.vel_div as the analytical field, and the _vel_div_masked
padded-mover-inertness fix (block_flow.py's module docstring documents the NaN landmine this guards
against; ported here verbatim rather than imported because species labels must be RECOMPUTED at every
RK4 sub-stage from the live t, which block_flow's fixed-label API does not support -- same reason
joint_flow.py re-implements its own _vel_div_masked instead of importing PolyBlockFlow's).

TIME-DEPENDENT CONDITIONING (the new piece): `labels_at_t(sig_start, sig_end, t, n_sig_bins)` linearly
interpolates the two CONTINUOUS sigma assignments and re-bins with the SAME analytic quantile-CDF
formula block_flow.sigma_to_bin uses (generalized off block_flow's hardcoded NSIG=8 to an arbitrary
n_sig_bins, since n_sig_bins sets CavityBlockFlow's n_species at construction time here). Labels are
PIECEWISE CONSTANT in t (sigma(t) moves continuously but its bin only hops at bin-boundary crossings)
-- this is EXACTLY FINE for exactness, not merely a tolerated approximation: `propose`'s forward leg
and `logq_of`'s reverse leg both evaluate species conditioning off the SAME fixed RK4_STEPS-step grid
and the SAME (sig_start, sig_end) path (propose never accumulates its own logq -- it always calls
logq_of on the sample it just produced, one code path, exactly as block_flow.py does), so whatever
finite-step discretization the bin hops introduce is baked into the single density definition below,
never an inconsistency between two differently-discretized densities. See block_flow.py's module
docstring for the full CNF change-of-variables derivation this mirrors:
    q(x_target | x_center, sig_start, sig_end, env_x, env_sig)
is DEFINED by REVERSE fixed-grid RK4, integrating v = flow.ce.vel_div BACKWARD from x_target over the
grid t: 1 -> 0, accumulating the raw per-step divergence, landing on the base point z at t=0:
    logq = logN(z; x_center, BASE_W^2 * I) + sum_of_div_over_reverse_path
with mover species labels at each sub-stage's t given by labels_at_t(sig_start, sig_end, t, n_sig_bins)
and env species labels FIXED (env never swaps; env_sig enters once via labels_at_t(env_sig, env_sig, ...
), any t, since sig_start==sig_end there gives a t-independent result).

MH USAGE (external to this module -- SwapBlockFlow only conditions on a GIVEN path, it never decides
forward vs. reverse): to accept/reject a proposed swap old->new with a classical Metropolis-Hastings
ratio, the caller computes q_fwd = logq_of(x_new, x_old, sig_start=sig_old, sig_end=sig_swapped, ...)
and q_rev = logq_of(x_old, x_new, sig_start=sig_swapped, sig_end=sig_old, ...) -- i.e. the REVERSE MH
leg is just another forward-in-t (t:1->0 as always) call of logq_of with the sigma path direction
reversed by the CALLER, never by re-reversing anything inside this module.

PADDING: identical convention to block_flow.py/joint_flow.py (dummy coordinates via the shared
`_dummies` helper, offset=0 for movers / offset=k_max for env so the two dummy coordinate ranges never
collide). Padded rows get sig_start = sig_end = SIG_MIN (not an arbitrary placeholder that then needs a
separate force-to-bin-0 step downstream, unlike joint_flow.py's env_bin masking): SIG_MIN is the exact
left edge of the quantile CDF (F(SIG_MIN) = (a - SIG_MIN^-2)/(a-b) = 0/(a-b) = 0), so labels_at_t gives
bin 0 for a padded row at EVERY t, by the same formula real rows use, with no special-casing.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path("/mnt/ssd/GridTransformer")
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.poly.block_flow import (
    SIG_MIN, SIG_MAX, NSIG, BASE_W, RK4_STEPS, DUMMY0, DUMMY_D,  # noqa: F401  (re-exported for callers)
    _dummies, _A, _B, _LOG_2PI, sigma_to_bin,
)


def _sigma_to_bin_n(sig, n_bins):
    """numpy: block_flow.sigma_to_bin's exact formula, generalized off the hardcoded module-level
    NSIG=8 to an arbitrary n_bins (SwapBlockFlow's n_sig_bins is a constructor argument, not a fixed
    global, since it sets CavityBlockFlow's n_species)."""
    sig = np.asarray(sig, dtype=np.float64)
    F = (_A - sig ** -2) / (_A - _B)
    F = np.clip(F, 0.0, 1.0)
    return np.minimum((F * n_bins).astype(np.int64), n_bins - 1)


def _sigma_to_bin_n_t(sig, n_bins):
    """torch counterpart of _sigma_to_bin_n (on-device, no host round-trip), used inside the RK4 loop
    at every sub-stage -- mirrors joint_flow.sigma_to_bin_t, generalized to n_bins."""
    F = (_A - sig ** -2) / (_A - _B)
    F = F.clamp(0.0, 1.0)
    b = (F * n_bins).floor().clamp(max=float(n_bins - 1))
    return b.long()


def labels_at_t(sig_start, sig_end, t, n_sig_bins):
    """Pure function: sigma(t) = (1-t)*sig_start + t*sig_end, then quantile-bin with n_sig_bins bins
    (see module docstring's TIME-DEPENDENT CONDITIONING section). Accepts numpy arrays (dataset/test
    call sites) or torch tensors (the RK4 loop's on-device call sites) transparently -- same formula,
    dispatched on input type so the RK4 loop never round-trips through numpy every sub-stage.

    At t=0 this equals sigma_to_bin(sig_start) (generalized to n_sig_bins); at t=1 it equals
    sigma_to_bin(sig_end); labels are piecewise-constant in t between those endpoints (bin hops at
    quantile-CDF boundary crossings of the linearly-interpolated sigma)."""
    if torch.is_tensor(sig_start) or torch.is_tensor(sig_end):
        s0 = sig_start if torch.is_tensor(sig_start) else torch.as_tensor(sig_start)
        s1 = sig_end if torch.is_tensor(sig_end) else torch.as_tensor(sig_end)
        sig_t = (1.0 - t) * s0 + t * s1
        return _sigma_to_bin_n_t(sig_t, n_sig_bins)
    s0 = np.asarray(sig_start, dtype=np.float64)
    s1 = np.asarray(sig_end, dtype=np.float64)
    sig_t = (1.0 - t) * s0 + t * s1
    return _sigma_to_bin_n(sig_t, n_sig_bins)


def _to_tensor(a, dtype, device):
    if torch.is_tensor(a):
        return a.to(dtype=dtype, device=device)
    return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)


class SwapBlockFlow(nn.Module):
    """Fixed-size (k_max movers, m_env env slots) swap-endpoint sigma(t)-conditioned block flow. Wraps
    CavityBlockFlow exactly like block_flow.PolyBlockFlow does (position-only field; NO u-channel, NO
    u-divergence -- the sole divergence term is the existing analytical position divergence, via the
    same _vel_div_masked pattern with padded rows kept inert). See module docstring for the density
    definition and the sigma(t) conditioning contract."""

    def __init__(self, k_max, m_env, hidden_nf=128, n_layers=4, n_sig_bins=8, base_w=BASE_W):
        super().__init__()
        self.k_max = int(k_max)
        self.m_env = int(m_env)
        self.n_sig_bins = int(n_sig_bins)
        self.base_w = float(base_w)   # base-noise width; 0.35 legacy, ~cage-scale for glass blocks
        self._cbf = CavityBlockFlow(n_cage=self.m_env, k=self.k_max, r_c=2.5,
                                     hidden_nf=hidden_nf, n_layers=n_layers,
                                     n_species=self.n_sig_bins, max_neighbors=16)

    @property
    def ce(self):
        """The exact field the RK4 integrator uses -- CavityCondEGNN.vel_div (identical property to
        block_flow.PolyBlockFlow.ce / joint_flow.JointBlockFlow.ce)."""
        return self._cbf.ce

    def _device(self):
        return next(self.parameters()).device

    # ---- padding (batched; x plus a sig_start/sig_end pair, or a single sig channel for env) --------
    def _pad_batch(self, x, sig_start, sig_end, target_n, offset, truncate_nearest):
        """x[B,n,3], sig_start[B,n], sig_end[B,n] -> padded/truncated to target_n. Mirrors
        joint_flow.JointBlockFlow._pad_xu_batch's structure with (sig_start, sig_end) standing in for
        its (x,u) pair; env callers pass sig_start=sig_end=env_sig, so the two padded/truncated outputs
        are identical and the caller keeps only one. n (real count pre-truncation) and hence n_real are
        assumed uniform across the batch (same assumption block_flow.py's single-sample API makes, now
        carrying an explicit leading B dim); only the per-row NEAREST SUBSET varies by row."""
        B, n, _ = x.shape
        if n > target_n:
            if not truncate_nearest:
                raise ValueError(f"{n} movers > k_max={target_n}")
            d2 = (x ** 2).sum(-1)
            idx = torch.argsort(d2, dim=1)[:, :target_n]
            x = torch.gather(x, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
            sig_start = torch.gather(sig_start, 1, idx)
            sig_end = torch.gather(sig_end, 1, idx)
            n = target_n
        n_real = n
        if n < target_n:
            pad_x = _dummies(target_n - n, x.device, offset=offset).unsqueeze(0).expand(B, -1, -1)
            pad_sig = torch.full((B, target_n - n), SIG_MIN, dtype=sig_start.dtype, device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            sig_start = torch.cat([sig_start, pad_sig], dim=1)
            sig_end = torch.cat([sig_end, pad_sig], dim=1)
        return x, sig_start, sig_end, n_real

    def _prep_movers(self, x, sig_start, sig_end):
        return self._pad_batch(x, sig_start, sig_end, self.k_max, offset=0, truncate_nearest=False)

    def _prep_env(self, env_x, env_sig):
        x, sig, _, n_real = self._pad_batch(env_x, env_sig, env_sig, self.m_env,
                                             offset=self.k_max, truncate_nearest=True)
        return x, sig, n_real

    # ---- position channel: ported from block_flow.PolyBlockFlow._vel_div_masked -----------------------
    def _vel_div_masked(self, cloud, t, sp, n_real):
        """Identical computation/rationale to block_flow.PolyBlockFlow._vel_div_masked (see that
        docstring for the padded-dummy-row spurious-self-divergence NaN landmine this masking fixes) --
        ported here (not imported) because `sp` must be recomputed from the CURRENT sigma(t) at every
        RK4 sub-stage, which block_flow's fixed-label API does not support (same reason joint_flow.py
        re-implements this rather than importing it)."""
        diff = torch.is_grad_enabled()
        vel_full, divpp = self.ce.egnn.forward_and_perparticle_divergence(cloud, t, sp, differentiable=diff)
        vel = vel_full[:, :self.k_max]
        if n_real < self.k_max:
            vel = vel.clone()
            vel[:, n_real:, :] = 0.0
        div = divpp[:, :n_real].sum(-1) if n_real > 0 else torch.zeros(
            cloud.shape[0], device=cloud.device, dtype=cloud.dtype)
        return (vel, div) if diff else (vel.detach(), div.detach())

    def _integrate(self, x0, sig_start, sig_end, env_x, env_sig, reverse, n_real, n_env_real):
        """Fixed-grid RK4 (RK4_STEPS steps) of dx/dt = v(x,t | sigma(t)-conditioned labels, env),
        dl/dt = div(x,t | ...) (RAW, not negated -- see module docstring). Batched: x0[B,k_max,3].
        forward (reverse=False): t 0->1, dt=+1/RK4_STEPS; reverse=True: t 1->0, dt=-1/RK4_STEPS.
        Mover species labels are RECOMPUTED at every RK4 sub-stage's own t via labels_at_t(sig_start,
        sig_end, t, n_sig_bins) -- both k1/k2/k3/k4 sub-stages of every big step get their own t, so the
        label field genuinely varies within a single RK4 step, not just once per step. env labels are
        computed once (t-independent, sig_start==sig_end there) and reused for every sub-stage."""
        dt = (-1.0 if reverse else 1.0) / RK4_STEPS
        x = x0
        t = 1.0 if reverse else 0.0
        l = torch.zeros(x0.shape[0], device=x0.device, dtype=x0.dtype)

        env_labels = labels_at_t(env_sig, env_sig, 0.0, self.n_sig_bins)   # [B,m_env], t-independent

        def f(xx, tt):
            cloud = torch.cat([xx, env_x], dim=1)
            mover_labels = labels_at_t(sig_start, sig_end, tt, self.n_sig_bins)   # [B,k_max]
            sp = torch.cat([mover_labels, env_labels], dim=1)
            v, div = self._vel_div_masked(cloud, tt, sp, n_real)
            return v, div

        for _ in range(RK4_STEPS):
            k1v, k1d = f(x, t)
            k2v, k2d = f(x + 0.5 * dt * k1v, t + 0.5 * dt)
            k3v, k3d = f(x + 0.5 * dt * k2v, t + 0.5 * dt)
            k4v, k4d = f(x + dt * k3v, t + dt)
            x = x + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            l = l + (dt / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
            t = t + dt
        return x, l

    # ---- public batched API -----------------------------------------------------------------------
    @torch.no_grad()
    def propose_batch(self, x_old, sig_start, sig_end, env_x, env_sig, gen):
        """x_old[B,k,3], sig_start[B,k], sig_end[B,k], env_x[B,m,3], env_sig[B,m], gen: CPU
        torch.Generator (a cuda Generator crashes torch.randn -- landmine) -> (x_new[B,k,3] np.float32,
        logq[B] np.float64). base z = x_old + N(0, BASE_W^2) noise generated on CPU via `gen` then moved
        to device (mirrors block_flow.PolyBlockFlow.propose), forward RK4 t:0->1 walking sigma_start ->
        sigma_end, then re-evaluate the produced sample's density via `logq_of_batch` on the SAME path
        -- one code path, never two."""
        device = self._device()
        x_old_t = _to_tensor(x_old, torch.float32, device)
        sig_start_t = _to_tensor(sig_start, torch.float64, device)
        sig_end_t = _to_tensor(sig_end, torch.float64, device)
        env_x_t = _to_tensor(env_x, torch.float32, device)
        env_sig_t = _to_tensor(env_sig, torch.float64, device)
        B, k, _ = x_old_t.shape

        noise = torch.randn(B, k, 3, generator=gen, dtype=torch.float32).to(device)
        z_real = x_old_t + self.base_w * noise

        movers_x, movers_s0, movers_s1, n_real = self._prep_movers(z_real, sig_start_t, sig_end_t)
        env_xp, env_sp, n_env_real = self._prep_env(env_x_t, env_sig_t)

        x1, _ = self._integrate(movers_x, movers_s0, movers_s1, env_xp, env_sp,
                                 reverse=False, n_real=n_real, n_env_real=n_env_real)
        x_new = x1[:, :k].detach().cpu().numpy().astype(np.float32)

        logq = self.logq_of_batch(x_new, x_old, sig_start, sig_end, env_x, env_sig)
        return x_new, logq

    @torch.no_grad()
    def logq_of_batch(self, x_target, x_center, sig_start, sig_end, env_x, env_sig):
        """Density of producing x_target[B,k,3] when the base is centered at x_center[B,k,3], AT TIME t
        USING sigma(t) OF THE STATED (sig_start, sig_end) PATH -- the caller supplies the path (both MH
        legs use this same call, see module docstring's MH USAGE section); this function never reverses
        a path internally. Reverse RK4 t:1->0 from x_target lands at the base point z; logq = logN(z;
        x_center, BASE_W^2) + the path's accumulated raw divergence. Returns np.ndarray[B] (float64)."""
        device = self._device()
        x_target_t = _to_tensor(x_target, torch.float32, device)
        x_center_t = _to_tensor(x_center, torch.float32, device)
        sig_start_t = _to_tensor(sig_start, torch.float64, device)
        sig_end_t = _to_tensor(sig_end, torch.float64, device)
        env_x_t = _to_tensor(env_x, torch.float32, device)
        env_sig_t = _to_tensor(env_sig, torch.float64, device)
        B, k, _ = x_target_t.shape

        movers_x, movers_s0, movers_s1, n_real = self._prep_movers(x_target_t, sig_start_t, sig_end_t)
        env_xp, env_sp, n_env_real = self._prep_env(env_x_t, env_sig_t)

        z_full, path_sum = self._integrate(movers_x, movers_s0, movers_s1, env_xp, env_sp,
                                            reverse=True, n_real=n_real, n_env_real=n_env_real)
        z_real = z_full[:, :k]

        d = z_real - x_center_t
        logN = (-0.5 * (d * d).sum(dim=(1, 2)) / (self.base_w ** 2)
                - k * 3 * 0.5 * (_LOG_2PI + 2.0 * np.log(self.base_w)))
        logq = logN + path_sum
        return logq.detach().cpu().numpy().astype(np.float64)

    # ---- single-sample wrappers (thin B=1 shims over the batched API -- one code path) --------------
    def propose(self, x_old, sig_start, sig_end, env_x, env_sig, gen):
        """x_old[k,3], sig_start[k], sig_end[k], env_x[m,3], env_sig[m], gen: torch.Generator ->
        (x_new[k,3] np.float32, logq: float)."""
        x_new, logq = self.propose_batch(
            np.asarray(x_old)[None], np.asarray(sig_start)[None], np.asarray(sig_end)[None],
            np.asarray(env_x)[None], np.asarray(env_sig)[None], gen)
        return x_new[0], float(logq[0])

    def logq_of(self, x_target, x_center, sig_start, sig_end, env_x, env_sig):
        """-> float. See logq_of_batch."""
        logq = self.logq_of_batch(
            np.asarray(x_target)[None], np.asarray(x_center)[None],
            np.asarray(sig_start)[None], np.asarray(sig_end)[None],
            np.asarray(env_x)[None], np.asarray(env_sig)[None])
        return float(logq[0])
