import torch

from liquid_coupling_flow.mw.mw_ersi_common import _add_learndiffeq_path


def test_egnn_3d_forward_and_divergence_runs():
    _add_learndiffeq_path()
    from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics

    N, D, K, L, B = 8, 3, 4, 4.0, 2
    m = EGNN_dynamics(n_particles=N, n_dimension=D, hidden_nf=32, n_layers=2,
                      max_neighbors=K, L=L, n_species=1)
    xs = torch.rand(B, N, D) * L
    t = torch.zeros(B)
    a = torch.zeros(B, N, dtype=torch.long)
    vel, div = m.forward_and_divergence(xs, t, a)
    assert vel.shape == (B, N, D)
    assert div.shape == (B,) and torch.isfinite(div).all()
