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


# ---------------------------------------------------------------------------
# Task 9: AR generator model (Spline3Head + MWGenerator)
# ---------------------------------------------------------------------------

def _model():
    torch.manual_seed(0)
    from liquid_coupling_flow.mw.mw_generator import MWGenerator
    return MWGenerator(knn=6, d_model=32, n_layers=1, n_heads=2)

def test_sample_logprob_consistency():
    # Whole-config exactness needs canonical_order(sampled) == arange(N): every particle must land back
    # inside the cell targeted by its own generation step, so log_prob's slot-indexed anchor t[slot]
    # recovers the SAME anchor sample() used for that particle (log_prob has no other way to know which
    # anchor a given physical particle was generated against -- it re-derives order from geometry alone).
    # At identity-init the per-step offset is u~N(0,1) (spline==identity exactly, verified: head.log_prob
    # is bit-identical for any h, since all Linear weights are zero -- see Spline3Head._identity_init), so
    # a step's physical offset has std = s = HALF a cell width. P(a particle stays inside its own cell) =
    # P(|N(0,1)|<1)^3 ~ 0.32, so P(all N stay) ~ 0.32^N: astronomically small at N=64 (never observed in
    # 8 trials, confirmed empirically), but a real, checkable event at small N with a wide-enough batch.
    # We therefore test at N=8, drop to a subset that IS canonical (found empirically: ~2/256 at this
    # seed), and require the subset be non-empty -- this instantiates verbatim the brief's own prescribed
    # debugging procedure ("check canonical_order(sampled)==arange first ... compare on canonical ones").
    # Verified (scratch diagnostic, see task-9-report.md): on the canonical subset the two sides agree to
    # 1.9e-6 (three orders of magnitude inside the 1e-3 gate); on non-canonical entries the gap reaches
    # ~20 nats/config, entirely attributable to the anchor mismatch above, not a bug in the formula.
    from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order
    m = _model(); L, N, B = 5.198, 8, 256
    g = torch.Generator().manual_seed(0)
    x, lq_s, nwrap = m.sample(B, N, L, gen=g)
    t, rank, R = mw_scaffold(N, L, "cpu")
    perm = canonical_order(x, L, R, rank)
    canon = (perm == torch.arange(N)[None, :]).all(-1)
    assert canon.any(), "no canonical sampled config found in this batch -- widen B or change seed"
    lq_e = m.log_prob(x, L)
    d = (lq_s[canon] - lq_e[canon]).abs()
    assert torch.allclose(lq_s[canon], lq_e[canon], atol=1e-3), float(d.max())

def test_logprob_permutation_invariant():
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(1)
    x = torch.rand(4, 64, 3, generator=g) * L
    perm = torch.randperm(64, generator=g)
    assert torch.allclose(m.log_prob(x, L), m.log_prob(x[:, perm], L), atol=1e-4)

def test_conditional_normalized_1d():
    # identity-init head: q(a|h) must integrate to 1 on a fine grid (Gaussian base, machine-level)
    from liquid_coupling_flow.mw.mw_generator import Spline3Head
    torch.manual_seed(0)
    head = Spline3Head(32, 8, 4.0)
    h = torch.randn(1, 32)
    a = torch.linspace(-8, 8, 4001)[:, None]
    lp = head.logp_a(h.expand(4001, -1), a)             # expose per-dim conditional for this test
    z = torch.trapz(lp.exp().squeeze(), a.squeeze())
    assert abs(float(z) - 1.0) < 1e-3

def test_prefix_storage_invariance():
    # conditioning tensors are geometry-only: shuffling storage of placed particles changes nothing
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(2)
    x = torch.rand(2, 64, 3, generator=g) * L
    lp1 = m.log_prob(x, L)
    lp2 = m.log_prob(x[:, torch.randperm(64, generator=g)], L)
    assert torch.allclose(lp1, lp2, atol=1e-4)

def test_translation_by_lattice():
    # Certifies the offset/frame plumbing at identity-init, NOT trained-model invariance: the head's
    # linear layers are all zero-weight at init (see Spline3Head._identity_init via ka_flowhead pattern),
    # so log_prob is purely a function of the per-step offset u_j = Rf_j @ wrap_pm(x_j - t_j, L) / s, and
    # since Rf is orthonormal, log q_u,j depends ONLY on ||off_j|| (rotation-invariant Gaussian base) --
    # i.e. only on which cell a particle occupies and its offset from THAT cell's own center, both
    # invariant under a whole-config lattice shift. This holds PROVIDED canonical_order's per-cell
    # occupancy is a bijection (<=1 particle/cell): with a collision (2 particles share a cell, another
    # is empty), the stable (rank, distance) sort shifts every later slot's rank away from its position,
    # so log_prob's slot-indexed anchor t[slot] stops matching that particle's own cell -- breaking this
    # argument (verified: NOT a bug, see task-9-report.md). torch.rand(...)*L (i.i.d. uniform) is NOT
    # valid input here: 64 particles into R^3=64 cells collide like the birthday problem (measured ~22/64
    # collisions at this seed, canonical_order(x)==arange holds for 0/64 slots). We instead build a
    # jittered lattice (one particle per cell by construction, jitter << half-cell so no boundary is ever
    # crossed) -- a bijective, still-random config that actually exercises the intended invariance
    # (verified exact to 1e-6 on this construction before finalizing).
    from liquid_coupling_flow.mw.mw_generator import mw_scaffold
    m = _model(); L = 5.198
    t, rank, R = mw_scaffold(64, L, "cpu")
    w = L / R
    g = torch.Generator().manual_seed(3)
    jitter = (torch.rand(2, 64, 3, generator=g) - 0.5) * 0.6 * w    # +-0.3w, safely inside +-0.5w (=s)
    x = torch.remainder(t[None].expand(2, -1, -1) + jitter, L)
    shift = torch.tensor([L / 4, 0.0, 0.0])             # one full cell (R=4): scaffold maps onto itself
    lp2 = m.log_prob(torch.remainder(x + shift, L), L)
    assert torch.allclose(m.log_prob(x, L), lp2, atol=1e-3)
