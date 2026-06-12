"""Phase 2 equivariance diagnostic (action-plan §2.1): octahedral rotation NLL spread.

The box symmetry group is the 24-element proper octahedral group (signed permutation
matrices with det +1). The diagnostic rotates val configs about the box center,
re-Hilbert-sorts, and evaluates the arc-repr NLL per rotation.
"""
import numpy as np
import torch

from grid_transformer.models.transformer import GraphormerAR


def test_proper_octahedral_rotations_is_the_24_element_group():
    from octahedral_nll_diagnostic import proper_octahedral_rotations

    Rs = proper_octahedral_rotations()
    assert Rs.shape == (24, 3, 3)

    eye = np.eye(3)
    seen = set()
    for R in Rs:
        np.testing.assert_allclose(R @ R.T, eye, atol=1e-12)  # orthogonal
        assert np.linalg.det(R) == 1.0  # proper rotation
        assert np.abs(R).sum() == 3.0  # signed permutation matrix
        seen.add(tuple(R.astype(np.int64).ravel().tolist()))
    assert len(seen) == 24  # all distinct
    assert tuple(eye.astype(np.int64).ravel().tolist()) in seen  # identity included


def test_rotate_wrap_preserves_box_and_min_image_distances():
    from octahedral_nll_diagnostic import proper_octahedral_rotations, rotate_wrap

    L = 3.0
    rng = np.random.default_rng(0)
    configs = rng.uniform(0.0, L, size=(5, 27, 3))
    box = np.array([L, L, L])

    def min_image_dists(c):
        d = c[:, :, None, :] - c[:, None, :, :]
        d -= box * np.round(d / box)
        return np.linalg.norm(d, axis=-1)

    ref = min_image_dists(configs)
    for R in proper_octahedral_rotations()[:6]:
        out = rotate_wrap(configs, R, box)
        assert out.min() >= 0.0 and out.max() < L
        np.testing.assert_allclose(min_image_dists(out), ref, atol=1e-9)


def test_arc_nll_for_configs_finite_and_deterministic():
    from octahedral_nll_diagnostic import arc_nll_for_configs

    torch.manual_seed(2)
    model = GraphormerAR(
        K=65,
        d_model=32,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        sos_id=64,
        use_edge_bias=True,
        ida_spatial_dim=3,
        torus=True,
        use_continuous_head=True,
        num_mixtures=3,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
    ).eval()

    L, R = 2.0, 8
    rng = np.random.default_rng(1)
    configs = rng.uniform(0.0, L, size=(4, 8, 3))

    nll = arc_nll_for_configs(model, configs, box_lengths=[L, L, L], R=R, batch_size=2)
    assert nll.shape == (4,)
    assert np.all(np.isfinite(nll))
    nll2 = arc_nll_for_configs(model, configs, box_lengths=[L, L, L], R=R, batch_size=4)
    np.testing.assert_allclose(nll, nll2, atol=1e-4)


def test_arc_nll_ordering_param():
    """ordering="hilbert" is byte-identical to the default; ordering="gilbert"
    with a non-pow2-compatible R runs without error and returns finite NLL."""
    from octahedral_nll_diagnostic import arc_nll_for_configs

    torch.manual_seed(3)
    model = GraphormerAR(
        K=65,
        d_model=32,
        n_layer=2,
        n_head=4,
        dropout=0.0,
        sos_id=64,
        use_edge_bias=True,
        ida_spatial_dim=3,
        torus=True,
        use_continuous_head=True,
        num_mixtures=3,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
    ).eval()

    L, R = 2.0, 8
    rng = np.random.default_rng(7)
    configs = rng.uniform(0.0, L, size=(4, 8, 3))

    # Default (hilbert implicit) vs explicit ordering="hilbert" must be byte-identical.
    nll_default = arc_nll_for_configs(model, configs, box_lengths=[L, L, L], R=R, batch_size=4)
    nll_hilbert = arc_nll_for_configs(model, configs, box_lengths=[L, L, L], R=R, batch_size=4, ordering="hilbert")
    np.testing.assert_array_equal(nll_default, nll_hilbert)

    # ordering="gilbert" with a non-pow-2 R (e.g. R=6) must run and return finite NLL.
    R_gilbert = 6  # not a power of two
    nll_gilbert = arc_nll_for_configs(model, configs, box_lengths=[L, L, L], R=R_gilbert, batch_size=4, ordering="gilbert")
    assert nll_gilbert.shape == (4,)
    assert np.all(np.isfinite(nll_gilbert))
