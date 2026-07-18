"""Tests for the FLOW-ESCORTED NCMC propagator (poly ML item 2):
reports/logs-2026-07-17/poly_escort_ncmc.py's flow-MH move (SwapBlockFlow at a fixed-lambda IDENTITY
sigma path) and its ledger separation from the surrounding NCMC work bookkeeping (poly_ncmc_v2.py,
imported not modified).

Test 1: identity-path propose/logq self-consistency (<1e-4) -- mirrors test_poly_swapflow.py's Test 1
        pattern (propose then logq_of on the SAME forward path must reproduce propose's own returned
        density), specialized to sig_end == sig_start (the escort's actual operating regime, where
        labels_at_t is degenerate/constant in t -- worth an explicit check, not just inherited from a
        dissimilar-path test).
Test 2: a rejected flow-MH move restores the block's positions BITWISE (exact array equality, not
        merely "close") -- forced via a fake rng whose .random() always returns 1.0 (>= any
        Metropolis ratio in [0,1], so the move is deterministically rejected regardless of dU/logq).
Test 3: with the flow disabled (escort_every=None, flow=None), poly_escort_ncmc.
        ncmc_work_local_escorted's W matches poly_ncmc_v2.ncmc_work_local's W (imported, unmodified)
        on the same seed/config to 1e-10 -- proves the escort's own bookkeeping never leaks into W.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "reports/logs-2026-07-17"))

from liquid_coupling_flow.poly.swap_flow import SwapBlockFlow  # noqa: E402
from liquid_coupling_flow.poly.block_flow import SIG_MIN, SIG_MAX  # noqa: E402
from liquid_coupling_flow.poly.model import draw_sigmas, seed_numba  # noqa: E402
from poly_ncmc_v2 import ncmc_work_local  # noqa: E402  (IMPORT ONLY, never modify)
from poly_escort_ncmc import ncmc_work_local_escorted, _flow_mh_move  # noqa: E402


PERTURB_SCALE = 0.05   # matches test_poly_swapflow.py's landmine note: 0.1 sends this RK4 integrator
                        # to NaN on some random configs even for the validated PolyBlockFlow baseline.


def _perturb(f):
    """Un-zero the net (flow-matching identity __init__ zeros the last layer) so velocities/
    divergences are genuinely nonzero -- otherwise every check below is vacuously true."""
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


def _safe_positions(rng, n, L, min_d=0.3):
    for _ in range(10_000):
        x = rng.random((n, 3)) * L
        d = x[:, None, :] - x[None, :, :]
        d -= L * np.round(d / L)
        r = np.sqrt((d ** 2).sum(-1)) + np.eye(n) * 1e9
        if r.min() >= min_d:
            return x
    raise RuntimeError("could not draw safe positions")


class _AlwaysReject:
    """Fake rng: .random() always returns 1.0, which is >= min(1, exp(...)) for ANY dU/logq (the
    Metropolis ratio A is always in [0, 1]) -- so `rng.random() < A` is deterministically False, i.e.
    the move is ALWAYS rejected, regardless of what the flow actually proposed."""
    def random(self):
        return 1.0


# ---- Test 1: identity-path propose/logq self-consistency -------------------------------------------

def test_identity_path_propose_logq_selfconsistent():
    torch.manual_seed(0)
    f = SwapBlockFlow(k_max=8, m_env=16, hidden_nf=32, n_layers=2, n_sig_bins=8)
    _perturb(f)
    k = 8
    rng = np.random.default_rng(0)
    x_old, sig, env_x, env_sig = _rand_block(rng, k, 20)

    gen = torch.Generator().manual_seed(1)
    x_new, logq = f.propose(x_old, sig, sig, env_x, env_sig, gen)   # IDENTITY PATH: sig_end == sig_start
    assert x_new.shape == (k, 3) and np.isfinite(logq)

    logq2 = f.logq_of(x_new, x_old, sig, sig, env_x, env_sig)       # same forward path, external recompute
    assert abs(logq - logq2) < 1e-4, f"{logq} vs {logq2}"

    # sanity: identity path really is a no-op for labels_at_t (guards against a degenerate-path bug
    # silently producing NaN/garbage instead of a genuinely finite, nonzero-noise proposal)
    assert not np.allclose(x_new, x_old), "identity-path proposal collapsed to a no-op displacement"


# ---- Test 2: reject-restore is bitwise ---------------------------------------------------------------

def test_reject_restore_bitwise():
    torch.manual_seed(2)
    f = SwapBlockFlow(k_max=8, m_env=16, hidden_nf=16, n_layers=2, n_sig_bins=8, base_w=0.05)
    _perturb(f)
    N, k = 40, 8
    rng_np = np.random.default_rng(3)
    L = 20.0
    x = np.mod(rng_np.standard_normal((N, 3)) * 2.0 + 5.0, L)
    sig = rng_np.uniform(SIG_MIN, SIG_MAX, N)
    idx = np.arange(k, dtype=np.int64)
    x_before = x.copy()
    sig_before = sig.copy()

    gen = torch.Generator().manual_seed(4)
    accepted, dU, logq_fwd, logq_rev = _flow_mh_move(
        x, sig, L, beta=1.0 / 0.085, idx=idx, flow=f, gen=gen, rng=_AlwaysReject())

    assert accepted is False
    assert np.isfinite(dU) and np.isfinite(logq_fwd) and np.isfinite(logq_rev)
    # bitwise: exact array equality, not merely np.allclose
    assert np.array_equal(x[idx], x_before[idx]), "rejected move did not restore block positions bitwise"
    assert np.array_equal(x, x_before), "rejected move touched positions outside the block"
    assert np.array_equal(sig, sig_before), "flow-MH move must never touch sigma (identity path only)"


# ---- Test 3: ledger separation ------------------------------------------------------------------------

def test_ledger_separation_matches_ncmc_work_local():
    n = 40
    rng = np.random.default_rng(9)
    L = (n / 0.8) ** (1.0 / 3.0)
    x0 = _safe_positions(rng, n, L, min_d=0.3)
    sig0 = draw_sigmas(n, seed=10)
    i, j = 2, 11
    beta = 1.0 / 0.2
    n_steps = 50
    schedule = np.linspace(1.0 / n_steps, 1.0, n_steps)
    r_loc, step = 2.0, 0.15

    seed_numba(0)
    x = x0.copy(); sig = sig0.copy()
    dW = np.zeros(n_steps); nS = np.zeros(n_steps)
    W_escort, fcalls, faccepts, fdU = ncmc_work_local_escorted(
        x, sig, L, beta, i, j, schedule, 1, r_loc, step, dW, nS,
        flow=None, gen=None, rng=np.random.default_rng(0), k_block=8, escort_every=None)

    seed_numba(0)
    x_ref = x0.copy(); sig_ref = sig0.copy()
    dW_ref = np.zeros(n_steps); nS_ref = np.zeros(n_steps)
    W_ref = ncmc_work_local(x_ref, sig_ref, L, beta, i, j, schedule, 1, r_loc, step, dW_ref, nS_ref)

    assert fcalls == 0 and faccepts == 0 and fdU == []
    assert abs(W_escort - W_ref) < 1e-10, f"{W_escort} vs {W_ref}"
    assert np.array_equal(dW, dW_ref)
    assert np.array_equal(nS, nS_ref)
    assert np.array_equal(x, x_ref)
    assert np.array_equal(sig, sig_ref)


def test_ledger_separation_holds_even_with_escort_firing():
    """Stronger form: even when the flow IS wired in and DOES fire (escort_every small), the
    RETURNED W (and the per-step dW_step/nS_step profile) is unaffected -- only the flow_calls/
    flow_accepts/flow_dU side channel differs. Uses a perturbed net so the escort move is a real,
    nonzero-probability decision (not a trivially-always-rejected untrained net)."""
    n = 40
    rng = np.random.default_rng(21)
    L = (n / 0.8) ** (1.0 / 3.0)
    x0 = _safe_positions(rng, n, L, min_d=0.3)
    sig0 = draw_sigmas(n, seed=22)
    i, j = 5, 19
    beta = 1.0 / 0.2
    n_steps = 30
    schedule = np.linspace(1.0 / n_steps, 1.0, n_steps)
    r_loc, step = 2.0, 0.15

    torch.manual_seed(6)
    flow = SwapBlockFlow(k_max=8, m_env=16, hidden_nf=16, n_layers=2, n_sig_bins=8, base_w=0.05)
    _perturb(flow)
    gen = torch.Generator().manual_seed(7)
    mh_rng = np.random.default_rng(8)

    seed_numba(1)
    x = x0.copy(); sig = sig0.copy()
    dW = np.zeros(n_steps); nS = np.zeros(n_steps)
    W_escorted, fcalls, faccepts, fdU = ncmc_work_local_escorted(
        x, sig, L, beta, i, j, schedule, 1, r_loc, step, dW, nS,
        flow=flow, gen=gen, rng=mh_rng, k_block=8, escort_every=10)

    seed_numba(1)
    x_ref = x0.copy(); sig_ref = sig0.copy()
    dW_ref = np.zeros(n_steps); nS_ref = np.zeros(n_steps)
    W_ref = ncmc_work_local(x_ref, sig_ref, L, beta, i, j, schedule, 1, r_loc, step, dW_ref, nS_ref)

    assert fcalls == 3   # escort_every=10 over 30 steps -> fires at k=10,20,30
    assert abs(W_escorted - W_ref) < 1e-10, f"{W_escorted} vs {W_ref}"
    assert np.array_equal(dW, dW_ref)
    assert np.array_equal(nS, nS_ref)
    # sigma trajectory (lambda-switch driven) must be identical regardless of escort moves (the flow
    # never touches sigma) -- only POSITIONS may differ (the escort moves the block).
    assert np.array_equal(sig, sig_ref)
