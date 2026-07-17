import numpy as np
import torch

from liquid_coupling_flow.poly.joint_flow import JointBlockFlow, sigma_of_u_t, sigma_to_bin_t


def _perturb(f):
    """Un-zero every net (position EGNN's pot_model[-1] AND u_head's out_mlp[-1] are both zero-inited
    for the flow-matching identity convention) so both channels' velocities/divergences are actually
    nonzero -- mirrors block_flow.py's test _perturb helper (see its comment: at zero-init, every
    exactness/masking check is vacuously true)."""
    torch.manual_seed(42)
    for p in f.parameters():
        p.data.add_(torch.randn_like(p) * 0.1)
    f._cbf.ce.egnn.pot_model[-1].weight.data.add_(torch.randn_like(f._cbf.ce.egnn.pot_model[-1].weight) * 0.1)
    f._cbf.ce.egnn.pot_model[-1].bias.data.add_(torch.randn_like(f._cbf.ce.egnn.pot_model[-1].bias) * 0.1)
    f.u_head.out_mlp[-1].weight.data.add_(torch.randn_like(f.u_head.out_mlp[-1].weight) * 0.1)
    f.u_head.out_mlp[-1].bias.data.add_(torch.randn_like(f.u_head.out_mlp[-1].bias) * 0.1)


def _rand_block(rng, k, m):
    x_old = rng.standard_normal((k, 3)) * 0.4
    u_old = rng.standard_normal(k) * 0.4
    env_x = rng.standard_normal((m, 3)) * 2 + 3.0
    env_u = rng.standard_normal(m) * 0.4
    return x_old, u_old, env_x, env_u


def test_propose_logq_selfconsistent():
    torch.manual_seed(0)
    f = JointBlockFlow(k_max=10, m_env=16, hidden_nf=32, n_layers=2, u_knn=4)
    _perturb(f)
    rng = np.random.default_rng(0)
    x_old, u_old, env_x, env_u = _rand_block(rng, 8, 20)
    gen = torch.Generator().manual_seed(1)
    x_new, u_new, logq = f.propose(x_old, u_old, env_x, env_u, gen)
    assert x_new.shape == (8, 3) and u_new.shape == (8,)
    assert np.isfinite(logq) and np.isfinite(x_new).all() and np.isfinite(u_new).all()
    logq2 = f.logq_of(x_new, u_new, x_old, u_old, env_x, env_u)
    assert abs(logq - logq2) < 1e-4, f"{logq} vs {logq2}"


def test_u_divergence_exact_vs_bruteforce():
    """Sum_i d(u_dot_i)/d(u_i) from our vmap(jacrev(...)) pathway vs a brute-force
    torch.autograd.functional.jacobian trace, on ONE field evaluation with a perturbed net."""
    torch.manual_seed(2)
    k, m = 6, 9
    f = JointBlockFlow(k_max=k, m_env=m, hidden_nf=16, n_layers=2, u_knn=4)
    _perturb(f)
    rng = np.random.default_rng(3)
    x, u, envx, envu = _rand_block(rng, k, m)
    x_t = torch.tensor(x[None], dtype=torch.float32)
    u_t = torch.tensor(u[None], dtype=torch.float32)
    envx_t = torch.tensor(envx[None], dtype=torch.float32)
    envu_t = torch.tensor(envu[None], dtype=torch.float32)
    mask = torch.ones(k + m, dtype=torch.bool)
    t = 0.37

    def udot_of_u(u_row):
        return f._udot_single(u_row, x_t[0], envx_t[0], envu_t[0], mask, t)

    jac_bf = torch.autograd.functional.jacobian(udot_of_u, u_t[0])
    div_bf = float(torch.diagonal(jac_bf).sum())

    env_bin = sigma_to_bin_t(sigma_of_u_t(envu_t))
    _, _, _, div_u = f._field(x_t, u_t, envx_t, env_bin, envu_t, t, n_real=k, n_env_real=m)
    div_ours = float(div_u[0])

    assert abs(div_ours - div_bf) < 1e-6, f"{div_ours} vs {div_bf}"


def test_path_consistency_forward_vs_reverse():
    """Forward RK4's raw accumulated integral (negated) vs reverse RK4's raw path-sum on the SAME
    endpoint, no padding (k==k_max, m==m_env), perturbed net."""
    torch.manual_seed(4)
    k, m = 5, 6
    f = JointBlockFlow(k_max=k, m_env=m, hidden_nf=16, n_layers=2, u_knn=3)
    _perturb(f)
    rng = np.random.default_rng(11)
    x0, u0, envx, envu = _rand_block(rng, k, m)
    device = f._device()
    x0_t = torch.tensor(x0[None], dtype=torch.float32, device=device)
    u0_t = torch.tensor(u0[None], dtype=torch.float32, device=device)
    envx_t = torch.tensor(envx[None], dtype=torch.float32, device=device)
    envu_t = torch.tensor(envu[None], dtype=torch.float32, device=device)

    movers_x, movers_u, n_real = f._prep_movers(x0_t, u0_t)
    env_xp, env_up, env_bin, n_env_real = f._prep_env_bin(envx_t, envu_t)
    assert n_real == k and n_env_real == m          # no padding in this test

    x1, u1, l_fwd = f._integrate(movers_x, movers_u, env_xp, env_bin, env_up,
                                  reverse=False, n_real=n_real, n_env_real=n_env_real)
    _, _, l_rev = f._integrate(x1, u1, env_xp, env_bin, env_up,
                                reverse=True, n_real=n_real, n_env_real=n_env_real)

    assert abs(float(-l_fwd[0]) - float(l_rev[0])) < 1e-3, f"{float(-l_fwd[0])} vs {float(l_rev[0])}"


def test_dummy_inertness_both_channels():
    """k_max=k (no mover padding) vs k_max=k+4 (4 dummy movers) on IDENTICAL real data -> x_new, u_new,
    logq must agree to near machine precision and logq must be finite in both (the block_flow.py
    NaN-landmine repro, extended to the u channel)."""
    k, m = 5, 14
    rng = np.random.default_rng(7)
    x_old, u_old, env_x, env_u = _rand_block(rng, k, m)

    torch.manual_seed(3)
    f_a = JointBlockFlow(k_max=k, m_env=16, hidden_nf=16, n_layers=2, u_knn=3)
    _perturb(f_a)
    torch.manual_seed(3)
    f_b = JointBlockFlow(k_max=k + 4, m_env=16, hidden_nf=16, n_layers=2, u_knn=3)
    _perturb(f_b)

    gen_a = torch.Generator().manual_seed(9)
    gen_b = torch.Generator().manual_seed(9)
    x_new_a, u_new_a, logq_a = f_a.propose(x_old, u_old, env_x, env_u, gen_a)
    x_new_b, u_new_b, logq_b = f_b.propose(x_old, u_old, env_x, env_u, gen_b)

    assert np.isfinite(logq_a) and np.isfinite(logq_b)
    assert np.abs(x_new_a - x_new_b).max() < 1e-6, np.abs(x_new_a - x_new_b).max()
    assert np.abs(u_new_a - u_new_b).max() < 1e-6, np.abs(u_new_a - u_new_b).max()
    # logq tolerance is looser than block_flow.py's analogous 1e-6 (position-only, one channel):
    # the joint flow accumulates div_x AND div_u over the SAME 24-step reverse RK4 (twice the chained
    # float32 ops per stage), so ~1e-6-scale float32 rounding noise in the padded-vs-unpadded comparison
    # is expected, not a bug -- CONCRETELY VERIFIED by rerunning this exact scenario in float64 (both
    # models .double()'d, same base noise fed to both), which gives logq_a == logq_b to machine epsilon
    # (diff 0.0) and x/u diffs of 1.1e-16. See reports/logs-2026-07-17 diagnostic notes.
    assert abs(logq_a - logq_b) < 5e-6, f"{logq_a} vs {logq_b}"


def test_batch_consistency_logq_of_batch_vs_loop():
    torch.manual_seed(5)
    k, m = 7, 12
    f = JointBlockFlow(k_max=10, m_env=16, hidden_nf=16, n_layers=2, u_knn=4)
    _perturb(f)
    B = 3
    rng = np.random.default_rng(13)
    x_t = np.stack([_rand_block(rng, k, m)[0] for _ in range(B)])
    u_t = np.stack([rng.standard_normal(k) * 0.4 for _ in range(B)])
    x_c = np.stack([rng.standard_normal((k, 3)) * 0.4 for _ in range(B)])
    u_c = np.stack([rng.standard_normal(k) * 0.4 for _ in range(B)])
    env_x = np.stack([rng.standard_normal((m, 3)) * 2 + 3.0 for _ in range(B)])
    env_u = np.stack([rng.standard_normal(m) * 0.4 for _ in range(B)])

    logq_batch = f.logq_of_batch(x_t, u_t, x_c, u_c, env_x, env_u)
    assert logq_batch.shape == (B,)
    logq_loop = np.array([
        f.logq_of(x_t[b], u_t[b], x_c[b], u_c[b], env_x[b], env_u[b]) for b in range(B)
    ])
    assert np.abs(logq_batch - logq_loop).max() < 1e-5, np.abs(logq_batch - logq_loop)
