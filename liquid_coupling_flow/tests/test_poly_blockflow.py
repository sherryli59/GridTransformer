import numpy as np
import torch
from liquid_coupling_flow.poly.block_flow import PolyBlockFlow, sigma_to_bin


def _perturb(f):
    """Un-zero the net so velocities/divergences are actually nonzero -- at zero-init (the __init__
    default) pot_model[-1].weight/.bias are zeroed, so v==0 everywhere trivially and both the
    self-consistency check and the dummy-padding bug are DORMANT (any masking bug is invisible when
    everything is already zero). Perturbing makes both tests non-vacuous."""
    torch.manual_seed(42)
    for p in f.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    # __init__ explicitly zeroed these (flow-matching identity init) -- randn already un-zeroed them
    # above via f.parameters(), but re-assert with an explicit nonzero add so intent is unambiguous.
    f.ce.egnn.pot_model[-1].weight.data.add_(torch.randn_like(f.ce.egnn.pot_model[-1].weight) * 0.1)
    f.ce.egnn.pot_model[-1].bias.data.add_(torch.randn_like(f.ce.egnn.pot_model[-1].bias) * 0.1)


def test_propose_logq_selfconsistent():
    torch.manual_seed(0)
    f = PolyBlockFlow(k_max=16, m_env=32, hidden_nf=32, n_layers=2)      # tiny untrained net is fine
    _perturb(f)                                                          # nonzero net -- non-vacuous check
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
    _perturb(f)     # nonzero net: this is the exact reviewer repro (5 real movers, k_max=12, nonzero
                     # weights) that produced logq=NaN on ee28229 via dummy-mover self-divergence poisoning
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


def test_dummy_padding_is_inert():
    """Plan Task 4 Step 1's mandated test (omitted by the ee28229 implementer): with a PERTURBED
    (nonzero-velocity) net and a block using k < k_max -- forcing mover padding -- (a) propose() must
    return a FINITE logq, not NaN (the reviewer's exact repro: unfiltered top-k neighbour selection gives
    a padded dummy row a large spurious self-divergence that, unmasked, poisons the summed logq), and
    (b) a real-mover-only proposal must be UNAFFECTED by the presence/count of dummy rows: running the
    SAME real block+env through two flows that differ only in k_max (i.e. differ only in how much dummy
    padding is present) must give identical x_new and logq to near machine precision. This must FAIL on
    ee28229 (NaN) and PASS after masking padded mover rows out of the velocity update and divergence sum."""
    k = 5
    rng = np.random.default_rng(7)
    x_old = rng.standard_normal((k, 3)) * 0.3
    sig = rng.uniform(0.725, 1.61, k)
    env_x = rng.standard_normal((20, 3)) * 2 + 2.0
    env_sig = rng.uniform(0.725, 1.61, 20)
    sig_bin, env_bin = sigma_to_bin(sig), sigma_to_bin(env_sig)

    torch.manual_seed(3)
    f_a = PolyBlockFlow(k_max=k, m_env=16, hidden_nf=16, n_layers=2)          # k_max == k: NO padding
    _perturb(f_a)
    torch.manual_seed(3)
    f_b = PolyBlockFlow(k_max=k + 4, m_env=16, hidden_nf=16, n_layers=2)      # k_max > k: 4 dummy movers
    _perturb(f_b)

    gen_a = torch.Generator().manual_seed(9)
    gen_b = torch.Generator().manual_seed(9)
    x_new_a, logq_a = f_a.propose(x_old, sig_bin, env_x, env_bin, gen_a)
    x_new_b, logq_b = f_b.propose(x_old, sig_bin, env_x, env_bin, gen_b)

    assert np.isfinite(logq_a) and np.isfinite(logq_b)     # the reviewer's NaN repro must not recur
    assert np.abs(x_new_a - x_new_b).max() < 1e-6
    assert abs(logq_a - logq_b) < 1e-6
