import numpy as np
import torch
from liquid_coupling_flow.poly.block_flow import PolyBlockFlow, sigma_to_bin


def test_propose_logq_selfconsistent():
    torch.manual_seed(0)
    f = PolyBlockFlow(k_max=16, m_env=32, hidden_nf=32, n_layers=2)      # tiny untrained net is fine
    k = 8
    rng = np.random.default_rng(0)
    x_old = rng.standard_normal((k, 3)) * 0.5
    sig = rng.uniform(0.725, 1.61, k)
    env_x = rng.standard_normal((40, 3)) * 2 + 3.0
    env_sig = rng.uniform(0.725, 1.61, 40)
    gen = torch.Generator().manual_seed(1)
    x_new, logq = f.propose(x_old, sigma_to_bin(sig), env_x, sigma_to_bin(env_sig), gen)
    assert x_new.shape == (k, 3) and np.isfinite(logq)
    logq2 = f.logq_of(x_new, x_old, sigma_to_bin(sig), env_x, sigma_to_bin(env_sig))
    assert abs(logq - logq2) < 1e-4


def test_sigma_to_bin_range_and_monotone():
    sig = np.array([0.725, 0.9, 1.2, 1.61])
    b = sigma_to_bin(sig)
    assert b.dtype == np.int64
    assert (b >= 0).all() and (b < 8).all()
    assert (np.diff(b) >= 0).all()          # monotone non-decreasing with sigma


def test_propose_shapes_and_finite_with_padding():
    # env larger than m_env (truncated to nearest) and movers smaller than k_max (dummy-padded)
    torch.manual_seed(0)
    f = PolyBlockFlow(k_max=12, m_env=16, hidden_nf=16, n_layers=2)
    k = 5
    rng = np.random.default_rng(1)
    x_old = rng.standard_normal((k, 3)) * 0.3
    sig = rng.uniform(0.725, 1.61, k)
    env_x = rng.standard_normal((30, 3)) * 2 + 2.0     # > m_env=16, exercises truncation
    env_sig = rng.uniform(0.725, 1.61, 30)
    gen = torch.Generator().manual_seed(2)
    x_new, logq = f.propose(x_old, sigma_to_bin(sig), env_x, sigma_to_bin(env_sig), gen)
    assert x_new.shape == (k, 3)
    assert np.isfinite(x_new).all()
    assert np.isfinite(logq)
