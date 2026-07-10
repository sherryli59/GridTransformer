"""JointSpeciesFlow dimensionality: 2D default is unchanged (regression guard) and 3D is a working forward/
sample/train path. The 3D flow is the bulk generator for the transferable 3D reference (larger-N zero-shot)."""
import torch

from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, joint_loss, random_22_labeling


def test_2d_default_is_unchanged():
    torch.manual_seed(0); N, L, B = 12, 4.0, 3
    m = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=16, n_layers=2)     # default n_dimension=2
    assert m.n_dim == 2
    x = torch.rand(B, N, 2) * L; s = random_22_labeling(B, N, N // 2, "cpu")
    v, logits = m(torch.zeros(B, 1, 1), x, s)
    assert v.shape == (B, N, 2) and logits.shape == (B, N, 2)


def test_3d_forward_shapes_and_finite():
    torch.manual_seed(1); N, L, B = 12, (12 / 1.2) ** (1 / 3), 3
    m = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=16, n_layers=2, n_dimension=3)
    assert m.n_dim == 3
    v, logits = m(torch.zeros(B, 1, 1), torch.rand(B, N, 3) * L, random_22_labeling(B, N, N // 2, "cpu"))
    assert v.shape == (B, N, 3) and torch.isfinite(v).all() and logits.shape == (B, N, 2)


def test_3d_sample_preserves_species_count():
    torch.manual_seed(2); N, L, B = 12, (12 / 1.2) ** (1 / 3), 3
    m = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=16, n_layers=2, n_dimension=3)
    s0 = random_22_labeling(B, N, N // 2, "cpu"); x0 = torch.rand(B, N, 3) * L
    x1, s1 = m.sample(x0, s0, n_steps=3)
    assert x1.shape == (B, N, 3) and torch.equal(s1.sum(1), s0.sum(1))


def test_3d_joint_loss_backprops():
    torch.manual_seed(3); N, L, B = 12, (12 / 1.2) ** (1 / 3), 3
    m = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=16, n_layers=2, n_dimension=3)
    x0 = torch.rand(B, N, 3) * L; x1 = torch.rand(B, N, 3) * L
    s0 = random_22_labeling(B, N, N // 2, "cpu"); s1 = random_22_labeling(B, N, N // 2, "cpu")
    loss, _, _ = joint_loss(m, x0, x1, s0, s1)
    assert torch.isfinite(loss)
    loss.backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
