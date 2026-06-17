import torch
from liquid_coupling_flow.ka_flow_ar import KAARFlow
from liquid_coupling_flow.ka_mcmc import make_species


def test_invertibility_and_species_conditioning():
    torch.manual_seed(0)
    N, L = 24, (24 / 1.2) ** 0.5
    s = make_species(N, 0.35)
    f = KAARFlow(N=N, L=L, d=2, num_bins=16, hidden=64, cutoff=1.8)
    # exact triangular log q: logp(sample) == logp(eval)
    x, lp_s = f.sample(32, s, device="cpu")
    assert (lp_s - f.log_prob(x, s)).abs().max() < 1e-4
    assert x.min() >= 0 and x.max() < L
    # species are ACTIVE conditioning: flipping labels changes log q
    s2 = (~s.bool()).to(torch.int8)
    assert (f.log_prob(x, s) - f.log_prob(x, s2)).abs().mean() > 1e-3
    # species embedding is wired into the loss
    (-f.log_prob(x, s).mean()).backward()
    assert f.sp_emb.weight.grad.norm() > 0


if __name__ == "__main__":
    test_invertibility_and_species_conditioning()
    print("KA AR FLOW TESTS PASSED")
