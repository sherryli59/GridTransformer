"""Phase 1 likelihood instrumentation (reports/action-plan-variants-equivariance-likelihood.md).

Covers:
- arc_logp_position_correction: the closed-form (N-1)*log(rho) arc->position
  change of measure (resolves review finding 3.6)
- _arc_decode_positions diagnostics: clamp-hit (L3) and fine-wrap (L2) events
- arc_canonical_stats: canonical-order violation (L4) and duplicate-code detection
- sampler outputs: logp_position + per-sample exactness masks
- particle-0 seeding: empirical_p0_positions helper and the arc_p0_positions kwarg
"""
import math

import numpy as np
import pytest
import torch

from grid_transformer.data.lj_transferable import RelativeDeltaTokenizer
from grid_transformer.models.transformer import GraphormerAR


# ---------------------------------------------------------------------------
# Closed-form correction
# ---------------------------------------------------------------------------

def test_arc_logp_position_correction_closed_form():
    """log p(positions) = logp_continuous + (N-1) * log(N / V): the -log X
    code-interval mass and the -3 log(cell) fine Jacobian collapse to log(rho),
    independent of the Hilbert resolution R."""
    from sample_lj import arc_logp_position_correction

    n, box = 10, [2.0, 3.0, 4.0]
    expected = (n - 1) * math.log(n / (2.0 * 3.0 * 4.0))
    assert arc_logp_position_correction(n, box) == pytest.approx(expected, rel=1e-12)

    # At rho = 1 the correction vanishes.
    assert arc_logp_position_correction(27, [3.0, 3.0, 3.0]) == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Decode diagnostics (L2 fine-wrap, L3 clamp)
# ---------------------------------------------------------------------------

def test_arc_decode_diagnostics_flags_and_codes():
    from sample_lj import _arc_decode_positions

    R, L, N = 8, 2.0, 8
    X = R**3 // N  # 64
    box_t = torch.tensor([[L, L, L]], dtype=torch.float32)
    # All three rows start at the code-0 cell center (cell (0,0,0)).
    curr = torch.full((3, 3), L / R / 2.0, dtype=torch.float32)
    delta = torch.tensor(
        [
            [100.0, 0.0, 0.0, 0.0],  # code 6400 >> R^3-1=511 -> clamp
            [1.0, 0.7, 0.0, 0.0],    # |fine_x| > 0.5 -> wrapped
            [1.0, 0.1, 0.1, 0.1],    # clean step
        ],
        dtype=torch.float32,
    )

    pos_plain = _arc_decode_positions(curr, delta, box_t, R, N)
    pos, diag = _arc_decode_positions(curr, delta, box_t, R, N, return_diagnostics=True)

    # Diagnostics must not change the decoded positions.
    torch.testing.assert_close(pos, pos_plain)

    np.testing.assert_array_equal(diag["c_curr"].numpy(), [0, 0, 0])
    np.testing.assert_array_equal(diag["c_next"].numpy(), [R**3 - 1, X, X])
    np.testing.assert_array_equal(diag["clamp_hit"].numpy(), [True, False, False])
    np.testing.assert_array_equal(diag["fine_wrap"].numpy(), [False, True, False])


# ---------------------------------------------------------------------------
# Canonical-order stats (L4)
# ---------------------------------------------------------------------------

def test_arc_canonical_stats():
    from sample_lj import arc_canonical_stats

    codes = torch.tensor(
        [
            [1, 5, 9],   # strictly increasing: canonical
            [4, 4, 8],   # duplicate code: tie -> noncanonical too
            [5, 3, 9],   # decreasing step: order violation
        ],
        dtype=torch.long,
    )
    stats = arc_canonical_stats(codes)
    np.testing.assert_array_equal(stats["noncanonical"].numpy(), [False, True, True])
    np.testing.assert_array_equal(stats["duplicate_code"].numpy(), [False, True, False])


# ---------------------------------------------------------------------------
# Sampler integration
# ---------------------------------------------------------------------------

def _tiny_arc_model(K: int, sos_id: int) -> GraphormerAR:
    torch.manual_seed(2)
    return GraphormerAR(
        K=K,
        d_model=32,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        sos_id=sos_id,
        use_edge_bias=True,
        use_dir_bias=True,
        dist_bins=8,
        edge_max_dist=4.0,
        ida_spatial_dim=3,
        torus=True,
        use_continuous_head=True,
        num_mixtures=3,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
    ).eval()


def _sample_kwargs(tok, **over):
    kw = dict(
        n_particles=8,
        box_lengths=[3.0, 3.0, 3.0],
        nsamples=4,
        tokenizer=tok,
        seed=7,
        sample_mode="multinomial",
        use_continuous_head=True,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
        periodic=True,
        hilbert_resolution=8,
    )
    kw.update(over)
    return kw


def test_sampler_emits_logp_position_and_exactness_masks():
    from sample_lj import arc_logp_position_correction, autoregressive_relative_delta_sample

    tok = RelativeDeltaTokenizer(window=3.0, bins=4, dim=3)
    model = _tiny_arc_model(K=int(tok.vocab_size), sos_id=int(tok.vocab_size) - 1)
    out = autoregressive_relative_delta_sample(model, **_sample_kwargs(tok))

    n, box = 8, [3.0, 3.0, 3.0]
    expected = out["logp_continuous"] + arc_logp_position_correction(n, box)
    torch.testing.assert_close(out["logp_position"], expected)

    for key in ("arc_clamp_any", "arc_fine_wrap_any", "arc_noncanonical", "arc_duplicate_code"):
        assert key in out, f"sampler output missing {key}"
        assert out[key].shape == (4,)
        assert out[key].dtype == torch.bool

    # A duplicate code is a tie, which is by definition non-canonical.
    dup_but_canonical = out["arc_duplicate_code"] & ~out["arc_noncanonical"]
    assert not bool(dup_but_canonical.any())


def test_sampler_uses_arc_p0_positions_pool():
    from sample_lj import autoregressive_relative_delta_sample

    tok = RelativeDeltaTokenizer(window=3.0, bins=4, dim=3)
    model = _tiny_arc_model(K=int(tok.vocab_size), sos_id=int(tok.vocab_size) - 1)

    p0 = torch.tensor([[0.5, 0.6, 0.7]], dtype=torch.float32)
    out = autoregressive_relative_delta_sample(
        model, **_sample_kwargs(tok, nsamples=3, arc_p0_positions=p0)
    )
    np.testing.assert_allclose(
        out["x_base"][:, 0, :].numpy(),
        np.repeat(p0.numpy(), 3, axis=0),
        atol=1e-6,
        err_msg="particle 0 must come from the provided empirical pool",
    )


# ---------------------------------------------------------------------------
# Empirical particle-0 marginal helper
# ---------------------------------------------------------------------------

def test_empirical_p0_positions_picks_lowest_code_particle():
    from sample_lj import empirical_p0_positions

    R, L = 8, 2.0
    high = [1.8, 1.8, 1.8]  # cell (7,7,7): code != 0
    low0 = [0.10, 0.11, 0.12]  # cell (0,0,0): code 0 (the curve start)
    low1 = [0.05, 0.20, 0.10]
    configs = np.array(
        [
            [high, high, low0, high],
            [low1, high, high, high],
        ],
        dtype=np.float64,
    )
    out = empirical_p0_positions(configs, box_lengths=[L, L, L], R=R)
    np.testing.assert_allclose(out, np.array([low0, low1]), atol=1e-12)


# ---------------------------------------------------------------------------
# CLI / save plumbing
# ---------------------------------------------------------------------------

def test_arc_exactness_rates_from_masks():
    from sample_lj import arc_exactness_rates

    out = {
        "arc_clamp_any": torch.tensor([True, False, False, False]),
        "arc_fine_wrap_any": torch.tensor([False, False, False, False]),
        "arc_noncanonical": torch.tensor([True, True, False, False]),
        "arc_duplicate_code": torch.tensor([False, True, False, False]),
    }
    rates = arc_exactness_rates(out)
    assert rates["arc_clamp_rate"] == pytest.approx(0.25)
    assert rates["arc_fine_wrap_rate"] == pytest.approx(0.0)
    assert rates["arc_noncanonical_rate"] == pytest.approx(0.5)
    assert rates["arc_duplicate_code_rate"] == pytest.approx(0.25)


def test_arc_p0_file_flag(tmp_path):
    from sample_lj import build_parser

    argv = ["--ckpt", "x.ckpt", "--Lx", "3", "--Ly", "3"]
    assert build_parser().parse_args(argv).arc_p0_file is None
    args = build_parser().parse_args(argv + ["--arc_p0_file", str(tmp_path / "p.npz")])
    assert args.arc_p0_file == str(tmp_path / "p.npz")


def test_load_arc_p0_pool_from_npz(tmp_path):
    from sample_lj import _load_arc_p0_pool

    R, L = 8, 2.0
    high = [1.8, 1.8, 1.8]
    low = [0.10, 0.11, 0.12]
    x_base = np.array([[high, low, high], [low, high, high]], dtype=np.float32)
    path = tmp_path / "samples.npz"
    np.savez(path, x_base=x_base)

    pool = _load_arc_p0_pool(str(path), box_lengths=[L, L, L], R=R)
    assert isinstance(pool, torch.Tensor)
    np.testing.assert_allclose(pool.numpy(), np.array([low, low], dtype=np.float32), atol=1e-6)
