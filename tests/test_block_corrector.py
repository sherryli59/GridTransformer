import torch
from liquid_coupling_flow.ka3d_block_corrector import AffineCoupling, CageConditioner

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

def test_cage_conditioner_shape_and_zero_init():
    torch.manual_seed(0)
    cc = CageConditioner()
    M, A, C = 4, 3, 20
    ax = torch.randn(M, A, 3); asp = torch.randint(0, 2, (M, A))
    cx = torch.randn(M, C, 3); csp = torch.randint(0, 2, (M, C))
    p = cc.params(ax, asp, cx, csp, R=2.0)
    assert p.shape == (M, A, 6)
    # zero-init head => params all ~0 (identity coupling at init)
    assert p.abs().max() < 1e-6

from liquid_coupling_flow.ka3d_block_corrector import BlockCorrector

def test_block_corrector_roundtrip_and_identity_init():
    torch.manual_seed(0)
    bc = BlockCorrector(n_layers=6)
    M, K, C = 3, 8, 30; R = 2.0
    u = torch.randn(M, K, 3) * 0.5
    s = torch.randint(0, 2, (M, K)); ay = torch.randn(K, 3) * 0.5
    cx = torch.randn(M, C, 3); cs = torch.randint(0, 2, (M, C))
    # identity at init (zero-init heads): forward == input, logdet == 0
    u_out, logdet = bc.forward(u, s, ay, cx, cs, R)
    assert torch.allclose(u_out, u, atol=1e-5)
    assert logdet.abs().max() < 1e-5
    # perturb params so it is non-trivial, then check invertibility + logdet sign
    for p in bc.parameters():
        p.data += torch.randn_like(p) * 0.05
    u_out, ld_f = bc.forward(u, s, ay, cx, cs, R)
    u_back, ld_i = bc.inverse(u_out, s, ay, cx, cs, R)
    assert torch.allclose(u_back, u, atol=1e-4)
    assert torch.allclose(ld_f, -ld_i, atol=1e-4)
