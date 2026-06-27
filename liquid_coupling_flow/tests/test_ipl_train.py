import torch
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model


def test_ipl_exactness_gate():
    # the IPL model must still satisfy sample logq == log_prob (required for the IS weight)
    torch.manual_seed(0)
    m = make_ipl_model(knn=8, device="cpu"); m.eval()       # knn<N=44 fine; small for speed
    pos, sp, logq = m.sample(4, 44, n_B=22, device="cpu", return_logq=True)
    assert torch.allclose(logq, m.log_prob(pos, sp), atol=1e-4)


if __name__ == "__main__":
    test_ipl_exactness_gate(); print("IPL TRAIN TEST PASSED")
