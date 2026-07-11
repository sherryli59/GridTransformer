import torch
from liquid_coupling_flow.ka3d_block_corrector import AffineCoupling

def test_affine_coupling_roundtrip_and_logdet():
    torch.manual_seed(0)
    ac = AffineCoupling()
    u = torch.randn(5, 3)
    params = torch.randn(5, 6) * 0.3
    u_out, logdet = ac.forward(u, params)
    u_back = ac.inverse(u_out, params)
    assert torch.allclose(u_back, u, atol=1e-5)
    # logdet == sum of log-scales; scale = exp(tanh-bounded raw)
    ls = torch.tanh(params[:, :3]) * ac.smax
    assert torch.allclose(logdet, ls.sum(-1), atol=1e-5)

def test_affine_coupling_identity_when_params_zero():
    ac = AffineCoupling()
    u = torch.randn(4, 3)
    u_out, logdet = ac.forward(u, torch.zeros(4, 6))
    assert torch.allclose(u_out, u, atol=1e-6)
    assert torch.allclose(logdet, torch.zeros(4), atol=1e-6)
