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
