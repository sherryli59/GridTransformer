import torch
from liquid_coupling_flow.ka3d_block_corrector import AffineCoupling, CageConditioner
from liquid_coupling_flow.ka3d_block_corrector import CorrectedBlockModel, load_base_and_cavity

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


def test_composition_exactness_and_identity_reproduces_base():
    dev = "cuda"
    base, cav = load_base_and_cavity(dev, ci=900, R=2.0, K=6, M=6)  # helper builds one cavity + block
    from liquid_coupling_flow.ka3d_block_corrector import BlockCorrector
    corr = BlockCorrector(n_layers=6).to(dev)
    model = CorrectedBlockModel(base, corr)
    xo, so, blk, bnd, sb, R = cav
    # Outer no_grad: the two compared calls in each part below must share grad-mode (the frozen base's
    # forward pass is not exactly run-to-run invariant to grad-tracking mode, ~2e-3 magnitude) -- this is
    # a property of the comparison, not of block_log_prob_corrected itself (which must stay grad-transparent
    # for training; see test_corrector_receives_base_gradient).
    with torch.no_grad():
        # (a) identity-init: corrected block_log_prob == base block_log_prob_b (to fp)
        lp_corr = model.block_log_prob_corrected(xo, so, blk, bnd, sb, R)
        lp_base = base.block_log_prob_b(xo, so, blk, bnd, sb, R)
        assert (lp_corr - lp_base).abs().max() < 1e-3
        # (b) exactness round-trip: sample logq == score logq (inherits base 8e-3)
        xn, sn, lq_s = model.sample_block_corrected(xo, so, blk, bnd, sb, R, gen=torch.Generator(dev).manual_seed(1))
        lq_score = model.block_log_prob_corrected(xn, sn, blk, bnd, sb, R)
        assert (lq_s - lq_score).abs().max() < 1e-2


def test_composition_exactness_perturbed_corrector():
    # The identity-init test above cannot discriminate the ball-map logdet sign convention: at coincident
    # points (identity corrector) the unsquash/squash logdet terms cancel regardless of sign. Perturb the
    # corrector so F is non-identity -- a sign bug would fail this by ~2*logdet, not float noise.
    dev = "cuda"
    base, cav = load_base_and_cavity(dev, ci=900, R=2.0, K=6, M=6)
    from liquid_coupling_flow.ka3d_block_corrector import BlockCorrector
    corr = BlockCorrector(n_layers=6).to(dev)
    for p in corr.parameters():
        p.data += torch.randn_like(p) * 0.05
    model = CorrectedBlockModel(base, corr)
    xo, so, blk, bnd, sb, R = cav
    with torch.no_grad():
        xn, sn, lq_sample = model.sample_block_corrected(xo, so, blk, bnd, sb, R, gen=torch.Generator(dev).manual_seed(2))
        lq_score = model.block_log_prob_corrected(xn, sn, blk, bnd, sb, R)
        assert (lq_sample - lq_score).abs().max() < 1e-2


def test_corrector_receives_base_gradient():
    # Proves Finding 1 is fixed: the base-density term (through x0 = corrector.inverse(...)) must
    # propagate a gradient into the corrector's trainable params, or the MLE loss can never teach the
    # corrector to map data onto base-mass.
    dev = "cuda"
    base, cav = load_base_and_cavity(dev, ci=900, R=2.0, K=6, M=6)
    from liquid_coupling_flow.ka3d_block_corrector import BlockCorrector
    corr = BlockCorrector(n_layers=6).to(dev)
    for p in corr.parameters():
        p.data += torch.randn_like(p) * 0.05
    model = CorrectedBlockModel(base, corr)
    xo, so, blk, bnd, sb, R = cav
    loss = -model.block_log_prob_corrected(xo, so, blk, bnd, sb, R).sum()
    loss.backward()
    grads = [p.grad for p in corr.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().max() > 0 for g in grads)
