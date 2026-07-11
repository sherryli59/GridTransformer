import torch

from liquid_coupling_flow.mw.mw_ersi import load_ersi, normalization_check, solver_convergence
from liquid_coupling_flow.mw.mw_ersi_data import L_for_N
from liquid_coupling_flow.mw.mw_ersi_train import _make_flow, train


def _tiny_ckpt(tmp_path, N=8):
    L = L_for_N(N)
    bank = tmp_path / f"b{N}.pt"
    torch.save({"cfgs": torch.rand(256, N, 3) * L}, bank)
    out = tmp_path / f"m{N}.pt"
    train(N=N, K=min(4, N - 1), hidden_nf=16, n_layers=1, steps=20, batch=16,
          out=str(out), bank_paths=[str(bank)], seed=0)
    return str(out)


def test_sample_logq_roundtrip(tmp_path):
    m = load_ersi(_tiny_ckpt(tmp_path), "cpu")
    g = torch.Generator().manual_seed(0)
    X, logq_gen = m.sample_and_logq(6, g)
    assert X.shape == (6, m.N, 3) and (X >= 0).all() and (X < m.L).all()
    logq_score = m.log_q(X)
    assert torch.allclose(logq_gen, logq_score, atol=1e-3), (
        f"round-trip max|d| {float((logq_gen - logq_score).abs().max()):.2e}"
    )


def test_normalization_small_n_zero_velocity(tmp_path):
    """The quadrature itself is a full N=2 joint integral, not a fake N=8 conditional."""
    N = 2
    L = L_for_N(N)
    flow = _make_flow(N=N, K=1, hidden_nf=8, n_layers=1, L=L, lr=1e-3, ot=True)
    for parameter in flow.b.parameters():
        parameter.data.zero_()
    path = tmp_path / "zero.pt"
    torch.save({"architecture": "mw_ersi_traceable_rfm_v1", "state_dict": flow.b.state_dict(),
                "N": N, "K": 1, "hidden_nf": 8, "n_layers": 1, "L": L,
                "dim_phys": 3, "n_species": 1, "ot": True,
                "solver": {"method": "rk4", "n_steps": 8}}, path)
    model = load_ersi(str(path), "cpu")
    assert abs(normalization_check(model, n_quad=3) - 1.0) < 1e-5
    scores = solver_convergence(model, torch.rand(2, N, 3) * L, [4, 8])
    assert torch.allclose(scores[4], scores[8])
