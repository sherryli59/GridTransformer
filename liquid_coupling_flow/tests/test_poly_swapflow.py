import numpy as np
import torch

from liquid_coupling_flow.poly.swap_flow import SwapBlockFlow, labels_at_t
from liquid_coupling_flow.poly.block_flow import sigma_to_bin, SIG_MIN, SIG_MAX


PERTURB_SCALE = 0.05
# Measured (scale_probe, T=2026-07-17): scale 0.1 (block_flow.py's/joint_flow.py's own convention)
# sends BOTH this module's SwapBlockFlow AND the validated PolyBlockFlow baseline to NaN on some
# random configs (24-step fixed-grid RK4, untrained/perturbed field -- generic RK4-blowup risk, not
# a sigma(t)-path-specific bug: confirmed by reproducing the same NaN on PolyBlockFlow with identical
# geometry/seed/scale). 0.05 is finite for both (logq -299.9 vs -282.4 on the repro config) while
# still fully un-zeroing the net -- non-vacuous, not seed-mined.


def _perturb(f):
    """Un-zero the net so velocities/divergences are actually nonzero -- at zero-init (the
    flow-matching identity __init__ default) pot_model[-1].weight/.bias are zeroed, so v==0
    everywhere trivially and every exactness/masking/conditioning check below would be vacuously
    true. Mirrors block_flow.py's/joint_flow.py's test _perturb helper, at PERTURB_SCALE (see above
    for why 0.1 is unsafe here)."""
    torch.manual_seed(42)
    for p in f.parameters():
        p.data.add_(torch.randn_like(p) * PERTURB_SCALE)
    f.ce.egnn.pot_model[-1].weight.data.add_(
        torch.randn_like(f.ce.egnn.pot_model[-1].weight) * PERTURB_SCALE)
    f.ce.egnn.pot_model[-1].bias.data.add_(
        torch.randn_like(f.ce.egnn.pot_model[-1].bias) * PERTURB_SCALE)


def _rand_block(rng, k, m):
    x_old = rng.standard_normal((k, 3)) * 0.4
    sig = rng.uniform(SIG_MIN, SIG_MAX, k)
    env_x = rng.standard_normal((m, 3)) * 2 + 3.0
    env_sig = rng.uniform(SIG_MIN, SIG_MAX, m)
    return x_old, sig, env_x, env_sig


def _swap_two_most_dissimilar(sig):
    """sig[k] -> sig_end[k] with the pair (a,b) of maximally-different sigma values exchanged, plus
    the local indices (a,b). Used to build a genuinely large-|dsigma| swap path for test 3."""
    k = sig.shape[0]
    best = (-1.0, 0, 1)
    for i in range(k):
        for j in range(i + 1, k):
            d = abs(sig[i] - sig[j])
            if d > best[0]:
                best = (d, i, j)
    _, a, b = best
    sig_end = sig.copy()
    sig_end[a], sig_end[b] = sig[b], sig[a]
    return sig_end, a, b


# ---- Test 1: propose/logq_of self-consistency (perturbed net) -------------------------------------

def test_propose_logq_selfconsistent():
    torch.manual_seed(0)
    f = SwapBlockFlow(k_max=8, m_env=16, hidden_nf=32, n_layers=2, n_sig_bins=8)
    _perturb(f)
    k = 8
    rng = np.random.default_rng(0)
    x_old, sig, env_x, env_sig = _rand_block(rng, k, 20)
    sig_end, a, b = _swap_two_most_dissimilar(sig)

    gen = torch.Generator().manual_seed(1)
    x_new, logq = f.propose(x_old, sig, sig_end, env_x, env_sig, gen)
    assert x_new.shape == (k, 3) and np.isfinite(logq)

    logq2 = f.logq_of(x_new, x_old, sig, sig_end, env_x, env_sig)
    assert abs(logq - logq2) < 1e-4, f"{logq} vs {logq2}"


# ---- Test 2: dummy-mover-padding inertness ---------------------------------------------------------

def test_dummy_padding_is_inert():
    """k_max == k (no padding) vs k_max == k+4 (4 dummy movers) on IDENTICAL real data -> x_new/logq
    agree to near machine precision and logq is finite in both -- block_flow.py's NaN-landmine repro,
    carried over verbatim since padded rows must stay inert under sigma(t)-conditioning too."""
    k, m = 5, 16
    rng = np.random.default_rng(7)
    x_old, sig, env_x, env_sig = _rand_block(rng, k, m)
    sig_end, _, _ = _swap_two_most_dissimilar(sig)

    torch.manual_seed(3)
    f_a = SwapBlockFlow(k_max=k, m_env=16, hidden_nf=16, n_layers=2, n_sig_bins=8)
    _perturb(f_a)
    torch.manual_seed(3)
    f_b = SwapBlockFlow(k_max=k + 4, m_env=16, hidden_nf=16, n_layers=2, n_sig_bins=8)
    _perturb(f_b)

    gen_a = torch.Generator().manual_seed(9)
    gen_b = torch.Generator().manual_seed(9)
    x_new_a, logq_a = f_a.propose(x_old, sig, sig_end, env_x, env_sig, gen_a)
    x_new_b, logq_b = f_b.propose(x_old, sig, sig_end, env_x, env_sig, gen_b)

    assert np.isfinite(logq_a) and np.isfinite(logq_b)
    assert np.abs(x_new_a - x_new_b).max() < 1e-6, np.abs(x_new_a - x_new_b).max()
    assert abs(logq_a - logq_b) < 1e-6, f"{logq_a} vs {logq_b}"


# ---- Test 3: conditioning is LIVE (not frozen at t=0) ----------------------------------------------

def test_conditioning_is_live():
    """Same perturbed net, same base noise (identically-seeded generators): a swap path (endpoints =
    the two most dissimilar sigmas in the block, exchanged) must produce a DIFFERENT x_new than the
    identity path (sig_end == sig_start) -- guards against labels_at_t being evaluated once at t=0 and
    then frozen for the rest of the RK4 walk (the landmine this test exists to catch)."""
    torch.manual_seed(0)
    f = SwapBlockFlow(k_max=8, m_env=16, hidden_nf=32, n_layers=2, n_sig_bins=8)
    _perturb(f)
    k = 8
    rng = np.random.default_rng(2)
    x_old, sig, env_x, env_sig = _rand_block(rng, k, 20)
    sig_end, a, b = _swap_two_most_dissimilar(sig)
    assert abs(sig[a] - sig[b]) > 1e-3          # sanity: the two picked sigmas really differ

    gen_swap = torch.Generator().manual_seed(5)
    gen_id = torch.Generator().manual_seed(5)
    x_new_swap, _ = f.propose(x_old, sig, sig_end, env_x, env_sig, gen_swap)
    x_new_id, _ = f.propose(x_old, sig, sig, env_x, env_sig, gen_id)

    dist = float(np.sqrt(((x_new_swap - x_new_id) ** 2).sum()))
    assert dist > 1e-3, dist


# ---- Test 4: labels_at_t unit behavior --------------------------------------------------------------

def test_labels_at_t_endpoints_and_monotone_hops():
    rng = np.random.default_rng(11)
    k = 8
    n_sig_bins = 8
    sig_start = np.linspace(SIG_MIN, SIG_MAX, k)                # spread across the full range
    sig_end, a, b = _swap_two_most_dissimilar(sig_start)
    # the two swapped entries must indeed span >= 2 bins
    assert abs(int(sigma_to_bin(sig_start[a])) - int(sigma_to_bin(sig_start[b]))) >= 2

    lbl_t0 = labels_at_t(sig_start, sig_end, 0.0, n_sig_bins)
    lbl_t1 = labels_at_t(sig_start, sig_end, 1.0, n_sig_bins)
    np.testing.assert_array_equal(lbl_t0, sigma_to_bin(sig_start))
    np.testing.assert_array_equal(lbl_t1, sigma_to_bin(sig_end))

    ts = np.linspace(0.0, 1.0, 25)
    hist = np.stack([labels_at_t(sig_start, sig_end, float(t), n_sig_bins) for t in ts])  # [25,k]
    for i in range(k):
        col = hist[:, i]
        if i in (a, b):
            # the swapped entries move from bin(sig_start[i]) to bin(sig_end[i]) monotonically
            diffs = np.diff(col)
            if sig_end[i] >= sig_start[i]:
                assert (diffs >= 0).all(), (i, col)
            else:
                assert (diffs <= 0).all(), (i, col)
            assert col[0] == sigma_to_bin(sig_start[i]) and col[-1] == sigma_to_bin(sig_end[i])
        else:
            # every other (unswapped) entry has sig_start == sig_end -> label constant over t
            assert (col == col[0]).all(), (i, col)


def test_labels_at_t_torch_matches_numpy():
    rng = np.random.default_rng(13)
    sig_start = rng.uniform(SIG_MIN, SIG_MAX, 6)
    sig_end = rng.uniform(SIG_MIN, SIG_MAX, 6)
    for t in (0.0, 0.3, 0.71, 1.0):
        np_lbl = labels_at_t(sig_start, sig_end, t, 8)
        t_lbl = labels_at_t(torch.as_tensor(sig_start), torch.as_tensor(sig_end), t, 8)
        np.testing.assert_array_equal(np_lbl, t_lbl.numpy())


# ---- Test 5: batch consistency ----------------------------------------------------------------------

def test_batch_consistency_logq_of_batch_vs_loop():
    torch.manual_seed(5)
    k, m = 7, 12
    f = SwapBlockFlow(k_max=k, m_env=m, hidden_nf=16, n_layers=2, n_sig_bins=8)
    _perturb(f)
    B = 3
    rng = np.random.default_rng(17)

    x_t, x_c, sig_start, sig_end, env_x, env_sig = [], [], [], [], [], []
    for _ in range(B):
        xo, sig, ex, esig = _rand_block(rng, k, m)
        se, _, _ = _swap_two_most_dissimilar(sig)
        x_c.append(rng.standard_normal((k, 3)) * 0.4)
        x_t.append(xo)
        sig_start.append(sig)
        sig_end.append(se)
        env_x.append(ex)
        env_sig.append(esig)
    x_t, x_c = np.stack(x_t), np.stack(x_c)
    sig_start, sig_end = np.stack(sig_start), np.stack(sig_end)
    env_x, env_sig = np.stack(env_x), np.stack(env_sig)

    logq_batch = f.logq_of_batch(x_t, x_c, sig_start, sig_end, env_x, env_sig)
    assert logq_batch.shape == (B,)
    logq_loop = np.array([
        f.logq_of(x_t[i], x_c[i], sig_start[i], sig_end[i], env_x[i], env_sig[i]) for i in range(B)
    ])
    assert np.abs(logq_batch - logq_loop).max() < 1e-5, np.abs(logq_batch - logq_loop)
