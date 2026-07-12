"""3D isolated (non-periodic) conditional-EGNN velocity field for the cavity block-flow corrector.
See docs/superpowers/specs/2026-07-12-egnn-cavity-block-flow-corrector-design.md (Task 1: foundation).

Mirrors liquid_coupling_flow.ka_cluster_egnn.ConditionalEGNN's contract exactly (movers message-pass with
each other AND the frozen cage; cage never moves and is NOT in the divergence) but for a 3D, isolated
(non-periodic) cavity instead of a 2D periodic box: no minimum-image, no `L`, raw Euclidean distances. We
never pass `L`, so only the isolated (L=None) branch of EGNN_dynamics is used. NOTE: this wrapper is the
FIRST isolated-branch consumer of forward_and_perparticle_divergence, and building it exposed a latent sign
bug in that branch (it returned the divergence of -vel; fixed 2026-07-12, commit 79c91ed). Do NOT assume the
isolated branch is more battle-tested than the periodic one -- it was effectively unexercised before this.

Exactness contract: `vel_div` returns (vel, div) from `EGNN_dynamics.forward_and_perparticle_divergence`,
which is the SAME field both used for the velocity and differentiated (via autograd on the radial pot) for
the divergence -- so (vel, div) are self-consistent by construction: div is the exact analytic trace of
d(vel)/d(cloud) restricted to the mover block, to floating-point precision. See
reports/logs-2026-07-12/test_egnn3d_exact.py for the brute-force verification.

CAVEAT (velocity completeness): `vel` here is the PAIRWISE central-force term only. In isolated mode
EGNN_dynamics.forward() additionally adds a `com_pot * source_xs_com` COM-directed term that
forward_and_perparticle_divergence does NOT include. This is intentional and exactness-preserving -- the flow
uses vel_div consistently for BOTH sampling and scoring, so self-consistency (not a match to forward()) is
what the log-det needs -- but it means the velocity field is pairwise-only. If the trained flow underfits the
cavity structure, adding a differentiable per-particle COM term is the first expressiveness lever.
"""
from __future__ import annotations
import torch
import torch.nn as nn

from liquid_coupling_flow.ipl44.learndiffeq.learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics


class CavityCondEGNN(nn.Module):
    """Wraps an isolated (non-periodic) 3D EGNN_dynamics over the fixed cloud (k movers + n_cage cage).
    vel_div returns ONLY the mover velocities and the mover-restricted divergence (cage is fixed context ->
    not in the log-det), exactly ConditionalEGNN's contract but n_dimension=3 and L=None (no minimum-image)."""

    def __init__(self, n_cage, k, r_c=2.5, hidden_nf=64, n_layers=4, n_species=2, max_neighbors=None,
                 rep_prior=False):
        super().__init__()
        P = k + n_cage
        mn = P - 1 if max_neighbors is None else min(int(max_neighbors), P - 1)   # None -> full cloud; else k-NN
        self.egnn = EGNN_dynamics(n_particles=P, n_dimension=3, cutoff=r_c, max_neighbors=mn,
                                  L=None, n_species=n_species, hidden_nf=hidden_nf, n_layers=n_layers,
                                  rep_prior=rep_prior)

    def vel_div(self, cloud, t, sp, k):
        # Under inference (@torch.no_grad) do NOT build the second-order graph the divergence's inner autograd
        # would otherwise retain (create_graph/retain_graph=True): wasted memory in the integrator loop. Detach
        # the outputs so no graph is carried through. Grad-enabled callers (the exactness test, which takes the
        # Jacobian of vel; flow-matching training) keep differentiability. Mirrors ConditionalEGNN.vel_div.
        diff = torch.is_grad_enabled()
        vel, divpp = self.egnn.forward_and_perparticle_divergence(cloud, t, sp, differentiable=diff)  # [B,P,3],[B,P]
        vel, div = vel[:, :k], divpp[:, :k].sum(-1)                                # mover velocities + div
        return (vel, div) if diff else (vel.detach(), div.detach())


class CavityBlockFlow(nn.Module):
    """Task 2: augmented-ODE (dopri5) integrator + composed exact log-q on top of Task-1's CavityCondEGNN.
    Wraps the [x_block, logdet] ODE (adapted from EGNNClusterFlow._dopri5_integrate in ka_cluster_egnn.py,
    lines ~164-186) for the 3D ISOLATED (non-periodic) cavity: no torch.remainder / minimum-image anywhere
    (raw coordinates), cage frozen (only the first k cloud rows -- the movers -- are integrated).

    Sign convention: `logdet` returned by `flow()` is defined so that the composed log-q is a plain SUM
    (matching the design spec's `logq_composed = logq_AR(x0) + integral_0^1 -div v dt` and this task's
    brief verbatim: `logq = logq_ar + logdet`) -- i.e. the auxiliary ODE state accumulates dl/dt = -div v
    (the NEGATIVE of vel_div's raw divergence), not the raw ∫div v dt that EGNNClusterFlow.sample subtracts.
    Forward (t: 0->1, base->corrected) and reverse (t: 1->0, corrected->base) integrate the SAME func, so
    running both directions on a round trip makes the two logdets cancel to ~ode_rtol (time-reversal of the
    dopri5 trajectory), independent of this sign choice.

    R (cavity radius) handling: CavityCondEGNN.vel_div / EGNN_dynamics.forward(t, xs, a=None) (see
    egnn_traceable.py) has NO global-feature argument beyond (t, positions, species) -- Task 1 did not add
    one. Per the brief, we do NOT invent an interface Task 1 doesn't have: `flow`/`composed_logq` accept
    `R` for forward-compatibility (and so callers don't need an if-branch) but currently ignore it -- the
    AR base already conditions on R, so the flow only needs to correct RELATIVE structure. Wiring R into
    the velocity (e.g. as an extra global feature) is a follow-up if the trained flow underfits across R."""

    def __init__(self, n_cage, k, r_c=2.5, hidden_nf=64, n_layers=4, n_species=2, max_neighbors=None,
                 rep_prior=False, ode_rtol=1e-6, ode_atol=1e-6, max_steps=10000):
        super().__init__()
        self.n_cage = n_cage; self.k = k
        # rtol/atol default 1e-6: tight enough that the composed log-q is exact far below the AR base's ~8e-3
        # leak, but robust. AVOID 1e-8: an untrained/stiff velocity field has high-frequency content dopri5
        # cannot resolve to 1e-8, so its adaptive step-size collapses and the solve never terminates (measured).
        # max_steps BOUNDS the solver (odeint raises AssertionError when exceeded) so a stiff proposal can't hang
        # forever. IMPORTANT (verified by review): rtol=1e-6 does NOT *eliminate* the step-collapse for a
        # sufficiently rough field -- it only raises the roughness threshold; max_steps is the real backstop.
        # "Bounded" != "fail-fast": exceeding max_steps can still take tens of seconds. DEPLOYMENT (Task 7) must
        # wrap flow moves in try/except -> reject the proposal (weight -> -inf) on AssertionError, and consider a
        # wall-clock budget. A trained smooth field integrates in a few hundred steps (<< max_steps).
        self.ode_rtol = ode_rtol; self.ode_atol = ode_atol; self.max_steps = int(max_steps)
        self.ce = CavityCondEGNN(n_cage=n_cage, k=k, r_c=r_c, hidden_nf=hidden_nf, n_layers=n_layers,
                                  n_species=n_species, max_neighbors=max_neighbors, rep_prior=rep_prior)
        with torch.no_grad():   # flow-matching init: untrained velocity == 0 -> identity flow (x1==x0, logdet==0)
            self.ce.egnn.pot_model[-1].weight.zero_()
            self.ce.egnn.pot_model[-1].bias.zero_()

    def _dopri5_integrate(self, cloud, sp, k, reverse):
        """Adapted from EGNNClusterFlow._dopri5_integrate (ka_cluster_egnn.py) with the periodic
        torch.remainder(x, L) wrap dropped (isolated cavity, raw coordinates). Augmented ODE [x_block, l]:
        dx/dt = v(x,t | cage), dl/dt = -div v(x,t | cage) (see class docstring for the sign convention)."""
        from torchdiffeq import odeint
        dev = cloud.device; B = cloud.shape[0]
        cl = cloud[:, :k]; cage = cloud[:, k:]

        def func(t, y):
            x, _ = y
            v, div = self.ce.vel_div(torch.cat([x, cage], 1), t, sp, k)
            return (v, -div)

        t_span = torch.tensor([1.0, 0.0] if reverse else [0.0, 1.0], device=dev, dtype=cloud.dtype)
        ld0 = torch.zeros(B, device=dev, dtype=cloud.dtype)
        xf, ldf = odeint(func, (cl, ld0), t_span, method="dopri5", rtol=self.ode_rtol, atol=self.ode_atol,
                         options={"max_num_steps": self.max_steps})
        return xf[-1], ldf[-1]

    @torch.no_grad()   # inference path (MTM/SMC weight); training is flow-matching, never through this ODE
    def flow(self, x0_block, cage_x, sp_block, sp_cage, R=None, reverse=False):
        """x0_block[B,k,3], cage_x[B,m,3], sp_block[B,k], sp_cage[B,m] -> (x1[B,k,3], logdet[B]). Cage rows
        are concatenated into the cloud for message passing but only the first k (movers) are ever advanced
        by the ODE -- cage_x itself is never read back out or mutated, so it is unchanged by construction.
        reverse=False: t 0->1 (base->corrected); reverse=True: t 1->0 (corrected->base)."""
        cloud = torch.cat([x0_block, cage_x], dim=1)
        sp = torch.cat([sp_block, sp_cage], dim=1)
        x1, logdet = self._dopri5_integrate(cloud, sp, self.k, reverse)
        return x1, logdet

    @torch.no_grad()
    def composed_logq(self, x0_block, logq_ar, cage_x, sp_block, sp_cage, R=None):
        """Carry-the-latent composed log-q: logq(x1) = logq_AR(x0) + integral_0^1 -div v dt. Always runs the
        FORWARD flow (t 0->1) on the given x0 -- no inversion needed. Returns (x1[B,k,3], logq[B])."""
        x1, logdet = self.flow(x0_block, cage_x, sp_block, sp_cage, R=R, reverse=False)
        return x1, logq_ar + logdet


@torch.no_grad()
def sample_corrected_block(ar_model, flow_model, xo, so, block_mask, bnd, s_bnd, R, gen=None, pos_temp=1.0):
    """Task 3: wire the frozen AR base (`KA3DScaffoldEBMBatched`) to a `CavityBlockFlow` block corrector.

    xo[M,n,3], so[M,n] -- full configs (`block_mask[n]` bool, shared across the batch, marks the movers).
    bnd[m,3], s_bnd[m] -- frozen exterior boundary shell (never touched).

    Steps: (1) AR-sample the block (`ar_model.sample_block_b`), which gives an updated full config
    `xo_ar/so_ar` (retained slots pass through unchanged) plus the exact AR block log-density `logq_ar`.
    (2) Assemble the flow's cage = boundary + the RETAINED interior slots of `xo_ar` (frozen context, never
    entered into the divergence -- matches `CavityBlockFlow`'s contract). (3) Extract the AR-sampled block
    (the `block_mask==True` slots) and run it through `flow_model.composed_logq`, which integrates the
    augmented ODE and returns the corrected block position + `logq_ar + logdet` (exact composed log-q; no
    inversion needed since the AR pre-image `x0` is carried through, per the design spec's carry-the-latent
    scheme). (4) Write the corrected block back into the full config; retained slots and species are
    untouched by the flow (species are FIXED from the AR, per the design spec).

    Species order within the block/cage does not matter for correctness (EGNN message passing is
    permutation-equivariant per particle) as long as positions and species stay slot-aligned, which the
    boolean masking below preserves by construction; only the mover COUNT (`block_mask.sum()`) and cage size
    (`bnd.shape[0] + (~block_mask).sum()`) must match `flow_model`'s fixed `k`/`n_cage`.

    dtype/device: the AR base runs in float32 (its native precision); the flow may run in a different dtype
    (e.g. double, for a tight dopri5 solve) -- the block+cage positions are cast to `flow_model`'s parameter
    dtype for the ODE and the corrected block is cast back to the AR's dtype on write-back. Device is left
    alone (AR and flow are expected to already share a device, e.g. both on CUDA).

    Returns (x_full[M,n,3], s_full[M,n], logq_composed[M]).
    """
    xo_ar, so_ar, logq_ar = ar_model.sample_block_b(xo, so, block_mask, bnd, s_bnd, R, gen=gen, pos_temp=pos_temp)
    M = xo_ar.shape[0]
    x0_block = xo_ar[:, block_mask]                                    # [M,k,3] AR-sampled movers
    sp_block = so_ar[:, block_mask]                                    # [M,k]
    ret_x = xo_ar[:, ~block_mask]                                      # [M,n_ret,3] retained interior (unchanged)
    ret_s = so_ar[:, ~block_mask]
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), ret_x], dim=1)    # [M,n_cage,3] frozen: boundary + retained
    sp_cage = torch.cat([s_bnd[None].expand(M, -1), ret_s], dim=1)     # [M,n_cage]

    flow_dtype = next(flow_model.parameters()).dtype
    x1_block, logq_composed = flow_model.composed_logq(
        x0_block.to(flow_dtype), logq_ar.to(flow_dtype), cage_x.to(flow_dtype), sp_block, sp_cage, R=R)

    x_full = xo_ar.clone()
    x_full[:, block_mask] = x1_block.to(xo_ar.dtype)
    return x_full, so_ar, logq_composed.to(xo_ar.dtype)
