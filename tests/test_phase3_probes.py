"""Phase 3 warm-start probe infrastructure (action-plan Phase 3).

Covers:
- mdn_loss per-token NLL exposure (needed for jump-stratified logging)
- stratified_nll_means: jump vs local NLL split by the |Δs| stratum
- JointEdgeBias (variant A): zero at init (warm-start safe), KV bias_row parity
- additive near-contact RBF refinement (variant F): zero at init
- kNN attention mask (variant G): bias masking
- GraphormerAR wiring: forward == baseline at init; forward_step parity (KV cache)
- train.py --warm_start_ckpt loading (strict=False with new modules)
"""
import math

import numpy as np
import pytest
import torch

from grid_transformer.models.transformer import GraphormerAR
from grid_transformer.training.ar import MDNHead, mdn_loss


# ---------------------------------------------------------------------------
# Shared infrastructure: per-token NLL + stratification
# ---------------------------------------------------------------------------

def _mdn_outputs(B=2, T=5, M=3, D=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    log_pi = torch.log_softmax(torch.randn(B, T, M, generator=g), dim=-1)
    mu = torch.randn(B, T, M, D, generator=g)
    sigma = torch.rand(B, T, M, D, generator=g) + 0.5
    target = torch.randn(B, T, D, generator=g)
    return log_pi, mu, sigma, target


def test_mdn_loss_returns_point_nll_consistent_with_seq_nll():
    log_pi, mu, sigma, target = _mdn_outputs()
    loss, seq_nll, count = mdn_loss(log_pi, mu, sigma, target)
    loss2, seq_nll2, count2, point_nll = mdn_loss(
        log_pi, mu, sigma, target, return_point_nll=True
    )
    torch.testing.assert_close(loss, loss2)
    torch.testing.assert_close(seq_nll, seq_nll2)
    assert point_nll.shape == (2, 5)
    torch.testing.assert_close(point_nll.sum(dim=1), seq_nll)


def test_stratified_nll_means_splits_by_jump_threshold():
    from grid_transformer.training.ar import stratified_nll_means

    point_nll = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    delta_s = torch.tensor([[1.0, 10.0, 1.5], [0.9, 1.1, 7.0]])
    out = stratified_nll_means(point_nll, delta_s, threshold=4.0)
    # jumps: (0,1)=2.0 and (1,2)=6.0 -> mean 4.0; locals: 1,3,4,5 -> mean 3.25
    assert out["nll_jump"] == pytest.approx(4.0)
    assert out["nll_local"] == pytest.approx(3.25)
    assert out["jump_fraction"] == pytest.approx(2.0 / 6.0)

    # All-local edge case: jump mean must be NaN-free (returns None).
    out2 = stratified_nll_means(point_nll, torch.ones_like(delta_s), threshold=4.0)
    assert out2["nll_jump"] is None
    assert out2["nll_local"] == pytest.approx(3.5)


# ---------------------------------------------------------------------------
# Variant A: JointEdgeBias (Euclidean distance x arc separation)
# ---------------------------------------------------------------------------

def test_joint_edge_bias_zero_at_init_and_shapes():
    from grid_transformer.models.joint_arc_bias import JointEdgeBias

    torch.manual_seed(0)
    jb = JointEdgeBias(n_head=4)
    B, T = 2, 6
    coords = torch.rand(B, T, 3) * 3.0
    arc_s = torch.cumsum(torch.rand(B, T), dim=1)
    box = torch.full((B, 3), 3.0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)

    bias = jb(coords, arc_s, causal, box_size=box, torus=True)
    assert bias.shape == (B, 4, T, T)
    # Zero-init output layer: warm start must be exactly the baseline.
    allowed = causal[:, None, :, :].expand_as(bias)
    assert torch.all(bias[allowed] == 0.0)
    assert torch.all(torch.isinf(bias[~allowed]) & (bias[~allowed] < 0))

    # After perturbing the output layer the bias must be nonzero and depend on arc_s.
    with torch.no_grad():
        jb.mlp[-1].weight.normal_(std=0.1)
        jb.mlp[-1].bias.normal_(std=0.1)
    b1 = jb(coords, arc_s, causal, box_size=box, torus=True)
    b2 = jb(coords, arc_s * 3.0, causal, box_size=box, torus=True)
    assert not torch.allclose(b1[allowed], b2[allowed])


def test_joint_edge_bias_row_matches_full():
    from grid_transformer.models.joint_arc_bias import JointEdgeBias

    torch.manual_seed(1)
    jb = JointEdgeBias(n_head=4)
    with torch.no_grad():
        jb.mlp[-1].weight.normal_(std=0.1)
        jb.mlp[-1].bias.normal_(std=0.1)
    B, T = 2, 6
    coords = torch.rand(B, T, 3) * 3.0
    arc_s = torch.cumsum(torch.rand(B, T), dim=1)
    box = torch.full((B, 3), 3.0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)

    full = jb(coords, arc_s, causal, box_size=box, torus=True)
    for t in range(1, T + 1):
        row = jb.bias_row(coords[:, :t, :], arc_s[:, :t], box_size=box, torus=True)
        assert row.shape == (B, 4, 1, t)
        torch.testing.assert_close(row, full[:, :, t - 1 : t, :t], atol=1e-6, rtol=1e-5)


# ---------------------------------------------------------------------------
# GraphormerAR wiring: probes must be exact no-ops at init (zero-init outputs)
# ---------------------------------------------------------------------------

def _arc_model(**kw):
    torch.manual_seed(2)
    defaults = dict(
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
    )
    defaults.update(kw)
    return GraphormerAR(**defaults).eval()


def _arc_batch(B=2, T=6, seed=3):
    g = torch.Generator().manual_seed(seed)
    seq = torch.full((B, T), 64, dtype=torch.long)
    deltas = torch.randn(B, T, 4, generator=g) * 0.3
    deltas[:, 0] = 0.0
    deltas[:, :, 0] = deltas[:, :, 0].abs() + 0.5  # positive arc steps
    coords = torch.rand(B, T, 3, generator=g) * 3.0
    box = torch.full((B, 3), 3.0)
    return seq, deltas, coords, box


def test_joint_arc_bias_is_noop_at_init_and_active_after_perturbation():
    base = _arc_model()
    probe = _arc_model(use_joint_arc_bias=True)
    # Copy shared weights so outputs are comparable.
    missing, unexpected = probe.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all("joint_arc_bias" in k for k in missing)

    seq, deltas, coords, box = _arc_batch()
    with torch.no_grad():
        out_b = base(seq, coords=coords, box_size=box, input_deltas=deltas)
        out_p = probe(seq, coords=coords, box_size=box, input_deltas=deltas)
    for got, want in zip(out_p, out_b):
        torch.testing.assert_close(got, want)

    with torch.no_grad():
        probe.joint_arc_bias.mlp[-1].weight.normal_(std=0.1)
        out_p2 = probe(seq, coords=coords, box_size=box, input_deltas=deltas)
    assert not torch.allclose(out_p2[1], out_b[1])


def test_joint_arc_bias_forward_step_matches_full():
    model = _arc_model(use_joint_arc_bias=True)
    with torch.no_grad():
        model.joint_arc_bias.mlp[-1].weight.normal_(std=0.1)
        model.joint_arc_bias.mlp[-1].bias.normal_(std=0.1)
    seq, deltas, coords, box = _arc_batch()
    B, T = seq.shape
    with torch.no_grad():
        full = model(seq, coords=coords, box_size=box, input_deltas=deltas)
        cache = model.new_generation_cache()
        for t in range(T):
            out = model.forward_step(
                seq[:, t : t + 1],
                cache=cache,
                coords=coords[:, : t + 1, :],
                box_size=box,
                input_deltas=deltas[:, t : t + 1, :],
            )
            for got, want in zip(out, full):
                torch.testing.assert_close(got, want[:, t : t + 1], atol=1e-5, rtol=1e-4)


def test_refine_rbf_bias_is_noop_at_init():
    base = _arc_model()
    probe = _arc_model(use_refine_rbf_bias=True)
    missing, unexpected = probe.load_state_dict(base.state_dict(), strict=False)
    assert not unexpected
    assert all("refine_rbf_bias" in k for k in missing)

    seq, deltas, coords, box = _arc_batch()
    with torch.no_grad():
        out_b = base(seq, coords=coords, box_size=box, input_deltas=deltas)
        out_p = probe(seq, coords=coords, box_size=box, input_deltas=deltas)
    for got, want in zip(out_p, out_b):
        torch.testing.assert_close(got, want)


def test_knn_mask_restricts_attention():
    base = _arc_model()
    probe = _arc_model(knn_mask_k=2)
    probe.load_state_dict(base.state_dict(), strict=True)  # no new params

    seq, deltas, coords, box = _arc_batch()
    with torch.no_grad():
        out_b = base(seq, coords=coords, box_size=box, input_deltas=deltas)
        out_p = probe(seq, coords=coords, box_size=box, input_deltas=deltas)
    # k=2 over T=6 prefix: behavior must differ from full attention.
    assert not torch.allclose(out_p[1], out_b[1])


# ---------------------------------------------------------------------------
# Warm start
# ---------------------------------------------------------------------------

def test_warm_start_load_ignores_new_probe_modules(tmp_path):
    from train import warm_start_from_checkpoint

    base = _arc_model()
    path = tmp_path / "base.ckpt"
    torch.save({"state_dict": base.state_dict()}, path)

    probe = _arc_model(use_joint_arc_bias=True)
    report = warm_start_from_checkpoint(probe, str(path))
    assert report["n_loaded"] > 0
    assert all("joint_arc_bias" in k for k in report["missing"])
    assert report["unexpected"] == []

    seq, deltas, coords, box = _arc_batch()
    with torch.no_grad():
        out_b = base(seq, coords=coords, box_size=box, input_deltas=deltas)
        out_p = probe(seq, coords=coords, box_size=box, input_deltas=deltas)
    for got, want in zip(out_p, out_b):
        torch.testing.assert_close(got, want)
