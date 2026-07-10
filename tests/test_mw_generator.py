import math, torch, pytest
from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order, build_frames

def test_scaffold_covers_box():
    t, rank, R = mw_scaffold(64, 5.198, "cpu")
    assert R == 4 and t.shape == (64, 3) and t.min() > 0 and t.max() < 5.198
    assert sorted(rank.flatten().tolist()) == list(range(64))

def test_canonical_permutation_invariant():
    g = torch.Generator().manual_seed(0); L = 5.198
    x = torch.rand(3, 64, 3, generator=g) * L
    t, rank, R = mw_scaffold(64, L, "cpu")
    p = canonical_order(x, L, R, rank)
    xs = torch.gather(x, 1, p[..., None].expand(-1, -1, 3))
    perm = torch.randperm(64, generator=g)
    p2 = canonical_order(x[:, perm], L, R, rank)
    xs2 = torch.gather(x[:, perm], 1, p2[..., None].expand(-1, -1, 3))
    assert torch.allclose(xs, xs2, atol=1e-7)           # same canonical sequence from any storage order

def test_frames_orthonormal_and_covariant():
    g = torch.Generator().manual_seed(1)
    d = torch.randn(50, 12, 3, generator=g)
    n_valid = torch.full((50,), 12)
    Rf = build_frames(d, n_valid)
    eye = torch.eye(3).expand(50, 3, 3)
    assert torch.allclose(Rf @ Rf.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.det(Rf), torch.ones(50), atol=1e-5)
    # covariance: rotate all displacements by R0 -> frame coords of any rotated vector unchanged
    th = 0.7; c, s = math.cos(th), math.sin(th)
    R0 = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    Rf2 = build_frames(d @ R0.T, n_valid)
    v = torch.randn(50, 3, generator=g)
    assert torch.allclose(torch.einsum("bij,bj->bi", Rf, v),
                          torch.einsum("bij,bj->bi", Rf2, v @ R0.T), atol=1e-4)

def test_frames_fallbacks():
    d = torch.zeros(3, 12, 3); d[1, 0] = torch.tensor([1., 0., 0.])
    d[2, 0] = torch.tensor([1., 0., 0.]); d[2, 1] = torch.tensor([2., 0., 0.])   # collinear pair
    Rf = build_frames(d, torch.tensor([0, 1, 2]))
    for b in range(3):
        assert torch.allclose(Rf[b] @ Rf[b].T, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(Rf[b]), torch.tensor(1.), atol=1e-5)

def test_frames_degenerate_valid_neighbor():
    # nominally-valid but exactly-zero rows must not silently break orthonormality
    d = torch.zeros(2, 12, 3)
    d[1, 1] = torch.tensor([0., 1., 0.])                 # row 1: zero d1 but a genuine 2nd neighbor
    Rf = build_frames(d, torch.tensor([1, 2]))           # row 0: n_valid=1 with d1 = 0
    for b in range(2):
        assert torch.allclose(Rf[b] @ Rf[b].T, torch.eye(3), atol=1e-5)
        assert torch.allclose(torch.det(Rf[b]), torch.tensor(1.), atol=1e-5)
    assert not torch.isnan(Rf).any()

def test_canonical_order_rejects_out_of_domain():
    L = 5.198
    t, rank, R = mw_scaffold(64, L, "cpu")
    g = torch.Generator().manual_seed(2)
    x = torch.rand(1, 64, 3, generator=g) * L
    x[0, 0, 0] = -1e-7                                   # .long() would silently truncate into cell 0
    with pytest.raises(AssertionError):
        canonical_order(x, L, R, rank)

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_geometry():
    dev, L = "cuda", 4.0
    t, rank, R = mw_scaffold(27, L, dev)
    assert t.device.type == "cuda" and R == 3
    assert sorted(rank.flatten().tolist()) == list(range(27))
    g = torch.Generator().manual_seed(3)
    x = (torch.rand(2, 27, 3, generator=g) * L).to(dev)
    p = canonical_order(x, L, R, rank)
    xs = torch.gather(x, 1, p[..., None].expand(-1, -1, 3))
    perm = torch.randperm(27, generator=g).to(dev)
    p2 = canonical_order(x[:, perm], L, R, rank)
    xs2 = torch.gather(x[:, perm], 1, p2[..., None].expand(-1, -1, 3))
    assert torch.allclose(xs, xs2, atol=1e-7)            # permutation invariance on device
    d = torch.randn(8, 6, 3, generator=g).to(dev)
    d[0] = 0                                             # 0-valid fallback row
    nv = torch.full((8,), 6, device=dev); nv[0] = 0
    Rf = build_frames(d, nv)
    eye = torch.eye(3, device=dev).expand(8, 3, 3)
    assert torch.allclose(Rf @ Rf.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.det(Rf), torch.ones(8, device=dev), atol=1e-5)
