"""sigma-conditioned block flow: wraps CavityBlockFlow (liquid_coupling_flow/ka3d_cavity_egnn.py) as the
polydisperse block corrector's proposal. sigma is CONTINUOUS; NSIG=8 quantile bins of P(sigma) ~ sigma^-3
on [SIG_MIN, SIG_MAX] (matches poly/model.py's draw_sigmas inverse-CDF, run forward instead of inverted)
become the EGNN's integer species labels for movers, env, and dummy padding (species 0). The flow moves
POSITIONS only -- sigma enters purely through the species labels of movers and env, exactly as
CavityCondEGNN.vel_div expects (cloud = cat([movers_x, cage_x], dim=1), sp = integer species per row).

DENSITY DEFINITION (read before touching propose/logq_of):
    q(x_target | x_center, labels, env)
is DEFINED by REVERSE fixed-grid RK4: integrate the velocity field v = flow.ce.vel_div BACKWARD from
x_target over the SAME RK4_STEPS=24-step grid t: 1 -> 0 used by the forward sampler, accumulating the raw
(not negated) per-step divergence along the path, landing on the base point z at t=0:
    logq = logN(z; x_center, BASE_W^2 * I) + sum_of_div_over_reverse_path
This is the standard continuous-normalizing-flow change of variables, log p1(x1) = log p0(x0) -
integral_0^1 div(v) dt, arranged so a NATURAL backward-in-time RK4 walk (same rate div(x,t), no sign
flip inserted by hand, genuinely negative dt) sums to exactly -integral_0^1 div(v) dt -- the term the CNF
formula needs -- with no extra bookkeeping.

Forward (propose) and reverse (logq_of, i.e. the MH reverse leg) walk the SAME fixed grid, so both
directions share one density definition by construction: `propose` never accumulates its own logq during
the forward push -- it always calls `logq_of` on the sample it just produced, so the two are literally the
same code path, not merely close. Any mismatch between "forward pushforward density" and "reverse-defined
density" for a FINITE-step CNF is a real, O(dt^4) effect of truncating the ODE solve to RK4_STEPS steps --
but because we never evaluate the forward-pushforward density directly (only ever the reverse-defined one,
for both legs of MH), this mismatch never appears as an inconsistency: it is baked into the single density
definition above, which is the controlled approximation this gate accepts.
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

SIG_MIN = 0.725
SIG_MAX = 1.61
NSIG = 8
BASE_W = 0.35
RK4_STEPS = 24
DUMMY0 = 50.0
DUMMY_D = 5.0

_A = SIG_MIN ** -2
_B = SIG_MAX ** -2
_LOG_2PI = float(np.log(2.0 * np.pi))


def sigma_to_bin(sig):
    """Quantile bin (0..NSIG-1) of P(sigma) ~ sigma^-3 on [SIG_MIN, SIG_MAX]. Analytic CDF
    F(sigma) = (a - sigma^-2) / (a - b), a = SIG_MIN^-2, b = SIG_MAX^-2 -- the exact inverse of
    poly/model.py's draw_sigmas (sigma = 1/sqrt(a - u*(a-b))), run forward instead of inverted, so bins
    are equal-probability under the same P(sigma) the data generator uses."""
    sig = np.asarray(sig, dtype=np.float64)
    F = (_A - sig ** -2) / (_A - _B)
    F = np.clip(F, 0.0, 1.0)
    b = np.minimum((F * NSIG).astype(np.int64), NSIG - 1)
    return b


def _dummies(m, device, offset=0):
    """m inert points spread (not stacked) at (DUMMY0 + DUMMY_D*(offset+j), 0, 0), j=0..m-1 -- identical
    pattern to reports/logs-2026-07-15/train_cavity_ersi.py's dummies(). `offset` lets two independently
    dummy-padded groups (movers, env) share one cloud without landing on the same coordinates (which would
    put two rows at r=0 of each other -> NaN in the EGNN's 1/r divergence term)."""
    d = torch.zeros(m, 3, device=device)
    if m > 0:
        d[:, 0] = DUMMY0 + DUMMY_D * (offset + torch.arange(m, device=device, dtype=torch.float32))
    return d


class PolyBlockFlow(nn.Module):
    """Fixed-size (k_max movers, m_env env/cage slots) sigma-conditioned block flow. Movers smaller than
    k_max are dummy-padded (offset=0); env larger than m_env is truncated to the m_env slots nearest the
    block centroid (positions are pre-centered by poly_block_data.make_block_pair), env smaller than m_env
    is dummy-padded (offset=k_max, so mover- and env-dummy coordinate ranges never collide)."""

    def __init__(self, k_max, m_env, hidden_nf=128, n_layers=4):
        super().__init__()
        self.k_max = int(k_max)
        self.m_env = int(m_env)
        self._cbf = CavityBlockFlow(n_cage=self.m_env, k=self.k_max, r_c=2.5,
                                     hidden_nf=hidden_nf, n_layers=n_layers,
                                     n_species=NSIG, max_neighbors=16)

    @property
    def ce(self):
        """The exact field the RK4 integrator (and the FM trainer) both use -- CavityCondEGNN.vel_div."""
        return self._cbf.ce

    def _device(self):
        return next(self.parameters()).device

    @staticmethod
    def _to_tensor(a, dtype, device):
        if torch.is_tensor(a):
            return a.to(dtype=dtype, device=device)
        return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

    def _pad(self, x, labels, target_n, offset, truncate_nearest):
        """x[n,3] float, labels[n] long -> (x[target_n,3], labels[target_n], n_real). truncate_nearest=True
        keeps the target_n rows nearest the origin (block-centered coords) when n > target_n (env case);
        movers are never truncated (n > k_max is a caller error). n_real = min(n, target_n) is the count of
        genuine (non-dummy) rows occupying the FIRST n_real slots -- callers use it to keep padded rows
        inert (see _vel_div_masked)."""
        n = x.shape[0]
        if n > target_n:
            if not truncate_nearest:
                raise ValueError(f"{n} movers > k_max={target_n}")
            d2 = (x ** 2).sum(-1)
            idx = torch.argsort(d2)[:target_n]
            x, labels = x[idx], labels[idx]
            n = target_n
        n_real = n
        if n < target_n:
            pad_x = _dummies(target_n - n, x.device, offset=offset)
            pad_l = torch.zeros(target_n - n, dtype=torch.long, device=x.device)
            x = torch.cat([x, pad_x], dim=0)
            labels = torch.cat([labels, pad_l], dim=0)
        return x, labels, n_real

    def _prep_movers(self, x, labels):
        return self._pad(x, labels, self.k_max, offset=0, truncate_nearest=False)

    def _prep_env(self, env_x, env_labels):
        x, labels, _ = self._pad(env_x, env_labels, self.m_env, offset=self.k_max, truncate_nearest=True)
        return x, labels

    def _vel_div_masked(self, cloud, t, sp, n_real):
        """Same computation as CavityCondEGNN.vel_div (liquid_coupling_flow/ka3d_cavity_egnn.py, NOT
        modified here) -- forward_and_perparticle_divergence over the full cloud, mover rows = cloud[:,:k_max]
        -- but with padded mover rows (slots n_real..k_max-1; real movers occupy the FIRST n_real slots per
        _pad's convention) EXCLUDED from the returned velocity and divergence.

        WHY: CavityCondEGNN's neighbour selection (_compute_common_terms) is an UNFILTERED top-k by distance
        with no radius cutoff. A padded dummy mover row (far from every real particle by construction, see
        _dummies) still gets `max_neighbors` neighbours assigned -- the farthest real/dummy rows available --
        so once the net has nonzero weights it can emit a large spurious velocity AND self-divergence on that
        row. `vel_div` sums divpp[:, :k_max] over ALL k_max mover slots before returning, so that spurious
        self-divergence poisons the scalar logq the instant a block has fewer real movers than k_max (the
        NaN bug fixed here). Masking here -- not in the shared CavityCondEGNN/EGNN_dynamics module -- keeps
        the fix local to the polydisperse wrapper without touching code shared by other campaigns.

        Zeroing padded-row velocity also makes the RK4 integration leave those rows exactly at their dummy
        coordinates for the whole path (they never move), so this one mask serves both requirements: inert
        padded movers AND a logq unpolluted by their spurious self-divergence."""
        diff = torch.is_grad_enabled()
        vel_full, divpp = self.ce.egnn.forward_and_perparticle_divergence(cloud, t, sp, differentiable=diff)
        vel = vel_full[:, :self.k_max]
        if n_real < self.k_max:
            vel = vel.clone()
            vel[:, n_real:, :] = 0.0
        div = divpp[:, :n_real].sum(-1) if n_real > 0 else torch.zeros(
            cloud.shape[0], device=cloud.device, dtype=cloud.dtype)
        return (vel, div) if diff else (vel.detach(), div.detach())

    def _integrate(self, x0, cage_x, sp, reverse, n_real):
        """Fixed-grid RK4 (RK4_STEPS steps) of dx/dt = v(x,t | cage,sp), dl/dt = div(x,t | cage,sp) (RAW,
        not negated -- see module docstring). forward (reverse=False): t 0->1, dt=+1/RK4_STEPS; the
        returned l is unused by `propose` (it re-evaluates via `logq_of` instead -- one density definition,
        never two). reverse=True: t 1->0, dt=-1/RK4_STEPS; l is exactly `logq_of`'s path integral term.
        n_real = number of genuine (non-dummy) mover rows (the first n_real of k_max) -- passed through to
        _vel_div_masked so padded rows stay inert (zero velocity, excluded from the divergence sum) on
        BOTH legs identically, preserving forward/reverse self-consistency."""
        dt = (-1.0 if reverse else 1.0) / RK4_STEPS
        x = x0.unsqueeze(0)                                  # [1,k_max,3]
        cage = cage_x.unsqueeze(0)                            # [1,m_env,3]
        sp_b = sp.unsqueeze(0)                                # [1,k_max+m_env]
        t = 1.0 if reverse else 0.0
        l = torch.zeros(1, device=x0.device, dtype=x0.dtype)

        def f(xx, tt):
            cloud = torch.cat([xx, cage], dim=1)
            v, div = self._vel_div_masked(cloud, tt, sp_b, n_real)
            return v, div

        for _ in range(RK4_STEPS):
            k1v, k1d = f(x, t)
            k2v, k2d = f(x + 0.5 * dt * k1v, t + 0.5 * dt)
            k3v, k3d = f(x + 0.5 * dt * k2v, t + 0.5 * dt)
            k4v, k4d = f(x + dt * k3v, t + dt)
            x = x + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
            l = l + (dt / 6.0) * (k1d + 2.0 * k2d + 2.0 * k3d + k4d)
            t = t + dt
        return x.squeeze(0), float(l.squeeze(0))

    @torch.no_grad()
    def propose(self, x_old, sig_labels, env_x, env_labels, gen):
        """x_old[k,3], sig_labels[k] (from sigma_to_bin), env_x[n_env,3], env_labels[n_env], gen:
        torch.Generator -> (x_new[k,3] np.float32, logq_fwd float). base z = x_old + N(0, BASE_W^2)
        (short-transport prior, CPU-generated via `gen` for reproducibility then moved to device),
        forward RK4 t:0->1, then re-evaluate the produced sample's density via logq_of -- so propose and
        the reverse MH leg are byte-identical code, never two divergent density definitions."""
        device = self._device()
        k = np.asarray(x_old).shape[0]
        x_old_t = self._to_tensor(x_old, torch.float32, device)
        noise = torch.randn(k, 3, generator=gen, dtype=torch.float32).to(device)
        z_real = x_old_t + BASE_W * noise

        sig_t = self._to_tensor(sig_labels, torch.long, device)
        env_x_t = self._to_tensor(env_x, torch.float32, device)
        env_labels_t = self._to_tensor(env_labels, torch.long, device)

        movers_full, sp_movers, n_real = self._prep_movers(z_real, sig_t)
        cage_full, sp_cage = self._prep_env(env_x_t, env_labels_t)
        sp = torch.cat([sp_movers, sp_cage], dim=0)

        x1_full, _ = self._integrate(movers_full, cage_full, sp, reverse=False, n_real=n_real)
        x_new = x1_full[:k].cpu().numpy().astype(np.float32)

        logq = self.logq_of(x_new, x_old, sig_labels, env_x, env_labels)
        return x_new, logq

    @torch.no_grad()
    def logq_of(self, x_target, x_center, sig_labels, env_x, env_labels):
        """Density of producing x_target[k,3] when the base is centered at x_center[k,3] (the reverse leg
        of MH). Reverse RK4 t:1->0 from x_target lands at base point z; logq = logN(z; x_center, BASE_W^2)
        + the path's accumulated raw divergence (see module docstring for the sign derivation)."""
        device = self._device()
        k = np.asarray(x_target).shape[0]
        x_target_t = self._to_tensor(x_target, torch.float32, device)
        x_center_t = self._to_tensor(x_center, torch.float32, device)
        sig_t = self._to_tensor(sig_labels, torch.long, device)
        env_x_t = self._to_tensor(env_x, torch.float32, device)
        env_labels_t = self._to_tensor(env_labels, torch.long, device)

        movers_full, sp_movers, n_real = self._prep_movers(x_target_t, sig_t)
        cage_full, sp_cage = self._prep_env(env_x_t, env_labels_t)
        sp = torch.cat([sp_movers, sp_cage], dim=0)

        z_full, path_sum = self._integrate(movers_full, cage_full, sp, reverse=True, n_real=n_real)
        z_real = z_full[:k]

        d = z_real - x_center_t
        logN = float(-0.5 * (d * d).sum() / (BASE_W ** 2) - k * 3 * 0.5 * (_LOG_2PI + 2.0 * np.log(BASE_W)))
        return logN + path_sum
