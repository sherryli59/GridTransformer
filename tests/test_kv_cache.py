"""KV-cache parity tests: incremental generation must equal the full-prefix forward.

The cache is a pure speedup — every test here asserts numerical equality against the
legacy path (no RoPE, per-token embeddings, eval-mode dropout => caching is lossless).
"""
import numpy as np
import pytest
import torch

from grid_transformer.models.transformer import CausalSelfAttnWithBias, EdgeBias, GraphormerAR

B, T, D, NH = 3, 7, 32, 4


def _rand_coords(box=3.0, t=T):
    g = torch.Generator().manual_seed(0)
    return torch.rand((B, t, 3), generator=g) * box


def test_edge_bias_row_matches_full():
    torch.manual_seed(0)
    eb = EdgeBias(n_head=NH, n_bins=8, use_dir=True, max_dist=4.0).eval()
    coords = _rand_coords()
    box = torch.full((B, 3), 3.0)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool)).unsqueeze(0).expand(B, -1, -1)

    for torus in (False, True):
        full = eb(coords, causal, box_size=box if torus else None, torus=torus)
        for t in range(1, T + 1):
            row = eb.bias_row(coords[:, :t, :], box_size=box if torus else None, torus=torus)
            assert row.shape == (B, NH, 1, t)
            torch.testing.assert_close(row, full[:, :, t - 1 : t, :t], atol=1e-6, rtol=1e-5)


def test_attn_incremental_matches_full():
    torch.manual_seed(1)
    attn = CausalSelfAttnWithBias(d_model=D, n_head=NH, dropout=0.0).eval()
    x = torch.randn(B, T, D)
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    bias = torch.randn(B, NH, T, T).masked_fill(~causal[None, None], float("-inf"))

    full = attn(x, bias)

    past = None
    for t in range(T):
        y_t, past = attn(x[:, t : t + 1, :], bias[:, :, t : t + 1, : t + 1], past_kv=past, return_kv=True)
        torch.testing.assert_close(y_t, full[:, t : t + 1, :], atol=1e-5, rtol=1e-4)


def _tiny_model(**kw) -> GraphormerAR:
    defaults = dict(
        K=17,
        d_model=D,
        n_layer=2,
        n_head=NH,
        dropout=0.0,
        sos_id=16,
        use_edge_bias=True,
        use_dir_bias=True,
        dist_bins=8,
        edge_max_dist=4.0,
        ida_spatial_dim=3,
        torus=True,
        use_pos_emb=False,
    )
    defaults.update(kw)
    torch.manual_seed(2)
    return GraphormerAR(**defaults).eval()


def test_forward_step_matches_full_discrete():
    model = _tiny_model()
    g = torch.Generator().manual_seed(3)
    seq = torch.randint(0, 16, (B, T), generator=g)
    coords = _rand_coords()
    box = torch.full((B, 3), 3.0)

    with torch.no_grad():
        full = model(seq, coords=coords, box_size=box)
        cache = model.new_generation_cache()
        for t in range(T):
            out = model.forward_step(
                seq[:, t : t + 1], cache=cache, coords=coords[:, : t + 1, :], box_size=box
            )
            torch.testing.assert_close(out, full[:, t : t + 1, :], atol=1e-5, rtol=1e-4)


def test_forward_step_matches_full_arc_continuous():
    """The arc_repr checkpoint config: continuous 4D input, MDN full-covariance head."""
    model = _tiny_model(
        use_continuous_head=True,
        num_mixtures=3,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
    )
    g = torch.Generator().manual_seed(4)
    seq = torch.full((B, T), 16, dtype=torch.long)  # SOS ids; content comes from deltas
    deltas = torch.randn(B, T, 4, generator=g) * 0.3
    deltas[:, 0] = 0.0  # position 0 is SOS
    coords = _rand_coords()
    box = torch.full((B, 3), 3.0)

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


def test_sampler_kv_cache_matches_legacy():
    """End to end: argmax sampling with the cache reproduces the legacy path exactly."""
    from grid_transformer.data.lj_transferable import RelativeDeltaTokenizer
    from sample_lj import autoregressive_relative_delta_sample

    tok = RelativeDeltaTokenizer(window=3.0, bins=4, dim=3)
    model = _tiny_model(
        K=int(tok.vocab_size),
        sos_id=int(tok.vocab_size) - 1,
        use_continuous_head=True,
        num_mixtures=3,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
    )

    kw = dict(
        n_particles=8,
        box_lengths=[3.0, 3.0, 3.0],
        nsamples=4,
        tokenizer=tok,
        seed=7,
        sample_mode="argmax",
        use_continuous_head=True,
        full_covariance=True,
        continuous_input=True,
        arc_repr=True,
        periodic=True,
        hilbert_resolution=8,
    )
    legacy = autoregressive_relative_delta_sample(model, use_kv_cache=False, **kw)
    cached = autoregressive_relative_delta_sample(model, use_kv_cache=True, **kw)
    np.testing.assert_allclose(
        cached["x_base"], legacy["x_base"], atol=1e-4,
        err_msg="KV-cached sampling diverged from the legacy full-prefix path",
    )
