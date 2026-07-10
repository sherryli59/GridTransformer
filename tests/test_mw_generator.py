import math, os, torch, pytest
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

def test_sample_logprob_consistency_preordered():
    # THE exactness gate for the SMC-base role: a sample's true density is the AR density in GENERATION
    # order, and sample() returns x with storage slot j == generation step j. preordered=True scores
    # exactly that factorization (slot j = step j, anchor t_j, causal prefix = slots < j), so it must
    # match the accumulated sample logq for EVERY config -- no canonical-subset gating. (Canonical-lift
    # evaluation of samples mis-states their density on noncanonical configs; see the renamed diagnostic
    # test below.) Only wrapping is a legitimate out: an |off| component > L/2 makes wrap_pm(x_j - t_j)
    # differ from the generated offset -- a separate, counted effect, hence the nwrap==0 gate.
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(0)
    x, lq_s, nwrap = m.sample(8, 64, L, gen=g)
    lq_e = m.log_prob(x, L, preordered=True)
    # Loud wrap guard: the seed is fixed and known to give nwrap==0, so if a future change produces
    # wraps this gate must FAIL (pick a new nwrap==0 seed deliberately), never silently skip.
    assert nwrap == 0, f"expected nwrap==0 at this fixed seed, got {nwrap}"
    assert torch.allclose(lq_s, lq_e, atol=1e-4), float((lq_s - lq_e).abs().max())

def test_sample_logprob_consistency_perturbed():
    # At identity init the head's Linear weights are ZERO, so the ENTIRE conditioning path (kNN
    # mirror, frames, encoder, padding mask, ctx chaining, both logabsdet signs) is invisible to
    # the density -- e.g. a transposed frame or desynchronized neighbor set would still pass the
    # identity-init gate above (the Gaussian base is rotation-invariant, so any orthonormal-frame
    # mixup preserves ||u|| and is invisible there; verified by mutation: transposing log_prob's
    # frame einsum blows THIS test up to max|delta|=57.8 while the identity-init gate stays green).
    # Perturbing ALL parameters makes every pathway load-bearing: the sequential sample-side and the
    # vectorized teacher-forced side must construct identical conditioning for the assert to hold.
    m = _model(); L = 5.198
    torch.manual_seed(5)                                 # fixed perturbation seed, chosen for nwrap==0
                                                          # (re-picked when the 14-dim pair features
                                                          # changed embed's shape: old seed 4 wrapped 2
                                                          # offsets; seeds 0-11 scanned, all non-wrap
                                                          # seeds consistent at <=7.6e-5)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(0.05 * torch.randn_like(p))
    g = torch.Generator().manual_seed(0)
    x, lq_s, nwrap = m.sample(8, 64, L, gen=g)
    assert nwrap == 0, f"expected nwrap==0 at this fixed seed, got {nwrap}"
    lq_e = m.log_prob(x, L, preordered=True)
    assert torch.allclose(lq_s, lq_e, atol=1e-4), float((lq_s - lq_e).abs().max())

def test_logprob_canonical_mode_on_canonical_configs():
    # DIAGNOSTIC (not the exactness gate -- that is the preordered test above): canonical-mode log_prob
    # agrees with sample logq only where canonical_order(sampled) == arange(N), i.e. every particle
    # landed back inside the cell targeted by its own generation step so the slot-indexed anchor t[slot]
    # recovers the anchor sample() used. At identity-init the per-step offset is u~N(0,1) (spline ==
    # identity exactly; head.log_prob is bit-identical for any h since all Linear weights are zero), so
    # a step's physical offset has std = s = HALF a cell width and P(all N particles stay in their own
    # cells) ~ 0.32^N: astronomically small at N=64, a real checkable event at N=8 with a wide batch
    # (~2/256 canonical at this seed). On that subset the agreement is ~1.9e-6; on noncanonical entries
    # the gap reaches ~20 nats/config -- which is exactly why the SMC-base role must use preordered=True.
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
    # q(a|h) must integrate to 1 on a fine grid, BOTH at identity init (Gaussian base, machine-level)
    # and with mildly perturbed weights -- the perturbed case pins the ABSOLUTE sign of the inverse
    # logabsdet in logp_a (a global sign flip would keep sample/log_prob mirror agreement, which uses
    # forward and inverse ld with opposite signs, but breaks normalization: int q da != 1).
    from liquid_coupling_flow.mw.mw_generator import Spline3Head
    torch.manual_seed(0)
    head = Spline3Head(32, 8, 4.0)
    h = torch.randn(1, 32)
    a = torch.linspace(-8, 8, 4001)[:, None]
    lp = head.logp_a(h.expand(4001, -1), a)             # expose per-dim conditional for this test
    z = torch.trapz(lp.exp().squeeze(), a.squeeze())
    assert abs(float(z) - 1.0) < 1e-3
    torch.manual_seed(4)
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.05 * torch.randn_like(p))
    lp = head.logp_a(h.expand(4001, -1), a)
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


# ---------------------------------------------------------------------------
# Task 10: training CLI + Phase-2 entry diagnostics
# ---------------------------------------------------------------------------

def test_training_bank(tmp_path):
    # Synthetic event-major bank: every particle/coord in a config equals that config's event value
    # (chain doesn't matter for this arithmetic check). Values stay well inside (0,L) so
    # torch.remainder is a no-op and the event index is exactly recoverable from the stored value.
    from liquid_coupling_flow.mw.mw_generator import load_training_bank, CHAINS_B
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    N, n_events, thin_events, val_frac = 4, 20, 6, 0.2
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    spacing = L / (n_events + 2)
    ev = torch.arange(n_events).repeat_interleave(CHAINS_B).float()      # [n_events*CHAINS_B]
    cfgs = ((ev + 1) * spacing)[:, None, None].expand(-1, N, 3).clone()
    bank_path = tmp_path / "fake_bank.pt"
    torch.save({"ref": {"cfgs": cfgs, "step": 0.05}}, bank_path)

    x_train, x_val, L_out = load_training_bank(art_path=str(bank_path), thin_events=thin_events,
                                                 val_frac=val_frac)
    assert abs(L_out - L) < 1e-6

    # thinned events (cfgs[::6] of 20) = 0,6,12,18 (4 events); val_frac=0.2 -> n_val=round(4*0.2)=1
    # -> train keeps the first 3 (0,6,12), val gets the last (18): strictly later-in-time.
    assert x_train.shape == (3 * CHAINS_B, N, 3)
    assert x_val.shape == (1 * CHAINS_B, N, 3)

    def events_of(x):
        return sorted(set((x[:, 0, 0] / spacing - 1).round().long().tolist()))
    assert events_of(x_train) == [0, 6, 12]
    assert events_of(x_val) == [18]

    assert bool((x_train >= 0).all() and (x_train < L_out).all())
    assert bool((x_val >= 0).all() and (x_val < L_out).all())

def test_training_bank_wraps_domain(tmp_path):
    from liquid_coupling_flow.mw.mw_generator import load_training_bank, CHAINS_B
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    N = 4
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    cfgs = torch.full((2 * CHAINS_B, N, 3), L + 0.37)                    # deliberately out of [0,L)
    bank_path = tmp_path / "fake_bank.pt"
    torch.save({"ref": {"cfgs": cfgs, "step": 0.05}}, bank_path)

    x_train, x_val, L_out = load_training_bank(art_path=str(bank_path), thin_events=1, val_frac=0.5)
    assert bool((x_train >= 0).all() and (x_train < L_out).all())
    assert bool((x_val >= 0).all() and (x_val < L_out).all())
    assert torch.allclose(x_train, torch.full_like(x_train, 0.37), atol=1e-5)

def test_token_pair_features():
    # feature-vector layout guard: [frame(3), r(1), sincos(8), min(1/r^2,4)(1), r/1.19(1)] = 14;
    # the excluded-volume tail must clamp at r=0.5 (cap 4.0) and stay finite at the zeroed
    # invalid-slot rows (r=0), where the clamped branch must also carry zero gradient (no inf/nan).
    from liquid_coupling_flow.mw.mw_generator import R_SHELL1, EV_RMIN
    m = _model()
    Rf = torch.eye(3)[None]
    nbr = torch.tensor([[[2.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.0, 0.0, 0.0]]], requires_grad=True)
    tok = m._tokens(Rf, nbr)
    assert tok.shape == (1, 3, 14)
    assert torch.allclose(tok[0, :, 12], torch.tensor([0.25, 4.0, 4.0]))      # 1/2^2; clamped; clamped
    assert torch.allclose(tok[0, :, 13], torch.tensor([2.0, 0.25, 0.0]) / R_SHELL1)
    tok[0, :, 12].sum().backward()
    assert torch.isfinite(nbr.grad).all()                                      # clamp kills the 1/r^3 inf
    assert EV_RMIN == 0.5 and abs(1.0 / EV_RMIN ** 2 - 4.0) < 1e-9

def test_v1_ckpt_backcompat(tmp_path):
    # Checkpoint schema back-compat, all three load paths:
    #   (a) pre-schema v1 ckpt (12-dim tokens, NO n_pair_feat key)      -> shape-inferred 0
    #   (b) key-less ckpt from an in-flight v2 process (14-dim, no key) -> shape-inferred 2
    #   (c) v2 ckpt with the key stored                                  -> key wins, roundtrips 2
    from liquid_coupling_flow.mw.mw_generator import MWGenerator, load_generator
    L = 5.198

    def _save(m, n_pair_key, path):
        ck = {"state_dict": m.state_dict(), "knn": 6, "d_model": 32, "n_layers": 1, "n_heads": 2,
              "num_bins": 8, "tail_bound": 4.0, "periods": (0.5, 1.0, 2.0, 4.0), "val_nll": 3.0,
              "step": 0, "train_N": 64}
        if n_pair_key is not None:
            ck["n_pair_feat"] = n_pair_key
        torch.save(ck, path)
        return path

    torch.manual_seed(0)
    m_v1 = MWGenerator(knn=6, d_model=32, n_layers=1, n_heads=2, n_pair_feat=0)
    assert m_v1.embed.in_features == 12
    m1 = load_generator(_save(m_v1, None, str(tmp_path / "v1.pt")), device="cpu")     # (a)
    assert m1.n_pair_feat == 0 and m1.embed.in_features == 12

    m_v2 = MWGenerator(knn=6, d_model=32, n_layers=1, n_heads=2)                      # default 2
    m2 = load_generator(_save(m_v2, None, str(tmp_path / "v2_keyless.pt")), device="cpu")  # (b)
    assert m2.n_pair_feat == 2 and m2.embed.in_features == 14
    m3 = load_generator(_save(m_v2, 2, str(tmp_path / "v2.pt")), device="cpu")        # (c)
    assert m3.n_pair_feat == 2 and m3.embed.in_features == 14

    g = torch.Generator().manual_seed(0)
    x = torch.rand(2, 64, 3, generator=g) * L
    for m in (m1, m2, m3):
        lp = m.log_prob(x, L)
        assert lp.shape == (2,) and torch.isfinite(lp).all()
    # loaded v1 must reproduce the source 12-dim model's density exactly (state carried, not re-init)
    assert torch.allclose(m1.log_prob(x, L), m_v1.log_prob(x, L), atol=1e-5)

def test_augment_exactness():
    # Ground-truth check that _augment_batch is an exact symmetry of U (torus translation + cubic
    # O_h) -- catches sign/center/wrap bugs directly at the energy level: any wrong center, sign
    # convention, or non-symmetry element changes pair distances and hence mw_energy.
    from liquid_coupling_flow.mw.mw_generator import _augment_batch
    from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR
    N = 27
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    g = torch.Generator().manual_seed(0)
    x = torch.rand(16, N, 3, generator=g) * L
    xa = _augment_batch(x, L, g)
    assert xa.shape == x.shape
    assert bool((xa >= 0).all() and (xa < L).all())
    assert not torch.allclose(xa, x)                      # actually transformed, not identity
    U0, U1 = mw_energy(x, L), mw_energy(xa, L)
    rel = ((U1 - U0).abs() / U0.abs().clamp_min(1.0)).max()
    assert float(rel) < 1e-4, f"augmentation changed the energy: max rel diff {float(rel):.2e}"

def test_train_smoke(tmp_path):
    # 30 steps on a tiny model config, tiny synthetic bank ("200-config slice"): checks the training
    # loop plumbing (finite + decreasing loss, both checkpoint files, load_generator round-trip) --
    # not a convergence test.
    from liquid_coupling_flow.mw.mw_generator import train, load_generator, CHAINS_B
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    torch.manual_seed(0)
    N, n_events = 64, 13                                  # 13*16=208 configs ~ "200-config slice"
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    g = torch.Generator().manual_seed(0)
    cfgs = torch.rand(n_events * CHAINS_B, N, 3, generator=g) * L
    bank_path = tmp_path / "fake_bank.pt"
    torch.save({"ref": {"cfgs": cfgs, "step": 0.0783}}, bank_path)

    out_path = str(tmp_path / "smoke.pt")
    result = train(steps=30, batch=8, lr=1e-2, val_every=10, out=out_path, seed=0,
                    art_path=str(bank_path), thin_events=1, val_frac=0.1, device="cpu",
                    knn=6, d_model=32, n_layers=1, n_heads=2)

    history = result["history"]
    assert len(history) >= 2
    assert all(math.isfinite(h["train_loss"]) and math.isfinite(h["val_nll"]) for h in history)
    assert history[-1]["val_nll"] < history[0]["val_nll"]

    last_path = result["last_path"]
    assert os.path.exists(out_path) and os.path.exists(last_path)

    model = load_generator(out_path, device="cpu")
    xb = torch.rand(4, N, 3) * L
    lp = model.log_prob(xb, L)
    assert lp.shape == (4,) and torch.isfinite(lp).all()

def test_train_warm_smoke(tmp_path):
    # warm-continue: second run constructs from the saved ckpt's hyperparams (no model_kw passed --
    # the tiny architecture must round-trip through the ckpt), loads its weights, and carries
    # best_val, so its history starts near the first run's end rather than from scratch.
    from liquid_coupling_flow.mw.mw_generator import train, CHAINS_B
    from liquid_coupling_flow.mw.mw_energy import RHO_STAR
    torch.manual_seed(0)
    N, n_events = 64, 13
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    g = torch.Generator().manual_seed(0)
    cfgs = torch.rand(n_events * CHAINS_B, N, 3, generator=g) * L
    bank_path = tmp_path / "fake_bank.pt"
    torch.save({"ref": {"cfgs": cfgs, "step": 0.0783}}, bank_path)

    out1 = str(tmp_path / "warm_a.pt")
    r1 = train(steps=10, batch=8, lr=1e-2, val_every=5, out=out1, seed=0,
                art_path=str(bank_path), thin_events=1, val_frac=0.1, device="cpu",
                knn=6, d_model=32, n_layers=1, n_heads=2)
    out2 = str(tmp_path / "warm_b.pt")
    r2 = train(steps=10, batch=8, lr=1e-3, val_every=5, out=out2, seed=1,
                art_path=str(bank_path), thin_events=1, val_frac=0.1, device="cpu", warm=out1)

    # starts near the loaded model's val (not fresh-init: scratch step-0 val is ~4.69 and nearly
    # seed-independent here -- the identity-init head is a pure standard-normal base)
    assert abs(r2["history"][0]["val_nll"] - r1["best_val"]) < 0.2
    assert r2["history"][0]["val_nll"] < r1["history"][0]["val_nll"] - 0.05
    # best_val carried from the ckpt: the continuation's best can only be <= the loaded best
    assert r2["best_val"] <= r1["best_val"] + 1e-6
    assert all(math.isfinite(h["val_nll"]) for h in r2["history"])
    assert os.path.exists(r2["last_path"])                # last always written; best only on improvement
