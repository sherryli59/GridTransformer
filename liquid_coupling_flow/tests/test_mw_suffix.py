import os

import pytest
import torch

from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR

N, L = 27, 3.9  # 27 = 3^3 -> R=3 scaffold


def tiny_model():
    torch.manual_seed(0)
    return MWGlobalAR(d_model=32, n_layers=1, n_heads=2, n_mix=4, rail_k=4).eval()


def test_suffix_logprob_full_equals_preordered_logprob():
    m = tiny_model()
    with torch.no_grad():
        x, _ = m.sample(2, N, L, gen=torch.Generator().manual_seed(1))
        assert torch.allclose(m.suffix_log_prob(x, N, L),
                              m.log_prob(x, L, preordered=True), atol=1e-4)


def test_sample_suffix_prefix_frozen_and_score_consistent():
    m = tiny_model()
    with torch.no_grad():
        x, _ = m.sample(3, N, L, gen=torch.Generator().manual_seed(2))
        xs, lqf = m.sample_suffix(x, 10, L, gen=torch.Generator().manual_seed(3))
        assert torch.equal(xs[:, :N - 10], x[:, :N - 10])
        assert not torch.allclose(xs[:, N - 10:], x[:, N - 10:])
        assert torch.allclose(lqf, m.suffix_log_prob(xs, 10, L), atol=1e-4)


def test_sample_suffix_full_matches_sample_density():
    m = tiny_model()
    with torch.no_grad():
        x0 = torch.rand(2, N, 3) * L   # arbitrary prefix content is irrelevant at m=N
        xs, lqf = m.sample_suffix(x0, N, L, gen=torch.Generator().manual_seed(4))
        assert torch.allclose(lqf, m.log_prob(xs, L, preordered=True), atol=1e-4)


def test_sample_suffix_does_not_mutate_input():
    m = tiny_model()
    with torch.no_grad():
        x, _ = m.sample(3, N, L, gen=torch.Generator().manual_seed(2))
        x_orig = x.clone()
        xs, _ = m.sample_suffix(x, 10, L, gen=torch.Generator().manual_seed(3))
        assert torch.equal(x, x_orig)


V10 = "liquid_coupling_flow/mw/artifacts/mw_gen_N64_v10_best_nll.pt"


@pytest.mark.skipif(not (torch.cuda.is_available() and os.path.exists(V10)), reason="needs GPU+ckpt")
def test_suffix_consistency_v10_checkpoint():
    from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    model = load_generator_v10(V10, "cuda")
    model = model[0] if isinstance(model, tuple) else model
    model.eval()
    N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0)
    with torch.no_grad():
        x, _ = model.sample(2, N, L, gen=torch.Generator(device="cuda").manual_seed(0))
        xs, lqf = model.sample_suffix(x, 24, L, gen=torch.Generator(device="cuda").manual_seed(1))
        assert torch.allclose(lqf, model.suffix_log_prob(xs, 24, L), atol=1e-3, rtol=0)
