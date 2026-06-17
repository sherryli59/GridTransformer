import torch
from liquid_coupling_flow.ka_flow_coupling import KACouplingFlow
from liquid_coupling_flow.ka_mcmc import make_species


def test_invertibility_and_species():
    torch.manual_seed(0)
    N, L = 24, (24 / 1.2) ** 0.5
    s = make_species(N, 0.35)
    f = KACouplingFlow(N=N, L=L, n_cycles=3, num_bins=16, hidden=64, cutoff=1.8)
    x, lp_s = f.sample(16, s, device="cpu")
    assert (lp_s - f.log_prob(x, s)).abs().max() < 1e-4      # exact invertibility
    assert x.min() >= 0 and x.max() < L
    s2 = (~s.bool()).to(torch.int8)
    assert (f.log_prob(x, s) - f.log_prob(x, s2)).abs().mean() > 1e-3   # species active
    (-f.log_prob(x, s).mean()).backward()
    assert f.sp_emb.weight.grad.norm() > 0


if __name__ == "__main__":
    test_invertibility_and_species()
    print("KA COUPLING FLOW TESTS PASSED")
