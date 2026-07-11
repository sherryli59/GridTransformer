import pytest
import torch

from liquid_coupling_flow.mw.mw_ersi_common import _add_learndiffeq_path


def _bruteforce_div(m, xs, t, a):
    """div v = sum_i d v_i / d x_i via a direct autograd trace."""
    B, N, D = xs.shape
    xs = xs.clone().requires_grad_(True)
    vel, _ = m.forward_and_divergence(xs, t, a, differentiable=True)
    div = torch.zeros(B, device=xs.device, dtype=xs.dtype)
    for i in range(N):
        for d in range(D):
            grad = torch.autograd.grad(vel[:, i, d].sum(), xs, retain_graph=True)[0]
            div = div + grad[:, i, d]
    return div


@pytest.mark.parametrize("K", [4, 6, 7])
def test_analytical_div_matches_bruteforce_3d(K):
    _add_learndiffeq_path()
    from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics

    torch.manual_seed(0)
    N, D, L, B = 8, 3, 4.0, 3
    m = EGNN_dynamics(n_particles=N, n_dimension=D, hidden_nf=32, n_layers=2,
                      max_neighbors=K, L=L, n_species=1).double()
    xs = torch.rand(B, N, D, dtype=torch.float64) * L
    t = torch.full((B,), 0.37, dtype=torch.float64)
    a = torch.zeros(B, N, dtype=torch.long)
    _, div_analytical = m.forward_and_divergence(xs, t, a)
    div_brute = _bruteforce_div(m, xs, t, a)
    assert torch.allclose(div_analytical, div_brute, atol=1e-6), (
        f"K={K}: max|d| {float((div_analytical - div_brute).abs().max()):.2e}"
    )
