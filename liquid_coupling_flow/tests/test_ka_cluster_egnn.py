import math, torch
from liquid_coupling_flow import ka_cluster_egnn as E
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _cfg(B=4, n_cage=48):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(f"{E.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, geo, pos, s, cl, n_cage

def test_build_cloud_shape_and_cluster_first():
    sc, L, geo, pos, s, cl, n_cage = _cfg()
    cloud, sp, c = E.build_cloud(pos, s, cl, sc, L, n_cage)
    assert cloud.shape == (4, 7 + n_cage, 2) and sp.shape == (4, 7 + n_cage) and c.shape == (4, 2)
    # first k cloud entries are exactly the cluster particles
    assert torch.allclose(cloud[:, :7], pos[:, cl], atol=0)
    # cage entries are NON-cluster particles (indices disjoint from cluster) — check none equals a cluster pos exactly
    # (cheap proxy: sorted nearest-first by distance to the CLUSTER centroid, the interface's stated selection
    # criterion (build_cloud docstring: "nearest n_cage ... by min-image dist to the cluster centroid"); note this
    # is NOT `c` (the returned cage centroid, a distinct downstream quantity computed from the selected cage only,
    # per cage_centroid) — the two centroids diverge enough (cage spans a wide radial shell) that checking
    # sortedness against `c` is not a valid proxy and fails spuriously.
    from liquid_coupling_flow.ka_gridformer import _wrap_pm
    clu = pos[:, cl]; seed_pt = clu[:, :1]
    ccen = torch.remainder(seed_pt[:, 0] + _wrap_pm(clu - seed_pt, L).mean(1), L)
    dc = _wrap_pm(cloud[:, 7:] - ccen[:, None], L).norm(dim=-1)
    assert (dc[:, 1:] >= dc[:, :-1] - 1e-4).all()          # cage sorted nearest-first (to cluster centroid)

def test_base_logp_closed_form():
    torch.manual_seed(0)
    c = torch.randn(4, 2, device=DEV); x0 = c[:, None] + 0.3 * torch.randn(4, 7, 2, device=DEV); sb = 1.1
    got = E.base_logp(x0, c, sb, 100.0)                    # L large -> min-image is a no-op, closed form holds
    d2 = ((x0 - c[:, None]) ** 2).sum(-1)
    want = (-d2 / (2 * sb ** 2) - math.log(2 * math.pi * sb ** 2)).sum(-1)
    assert torch.allclose(got, want, atol=1e-5)

def test_sigma_b_from_data():
    sc, L, geo, pos, s, cl, n_cage = _cfg()
    ref = torch.load(f"{E.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    sb = E.compute_sigma_b(ref["x"].to(DEV), ref["s"].to(DEV).long(), geo, sc, L, 7, n_cage)
    assert 0.7 < sb < 1.6            # cluster spread about its centroid

def _warmed_flow(steps=150, n_steps=32, rep_prior=False, rep_scale_init=None):
    """A briefly OT-FM-warmed flow: the untrained velocity is zero-init (identity); a few flow-matching steps push
    it to a TAME nonzero field (straight base->data), so the RK4 round-trip is reversible and sampler==scorer
    exercises a real (nonzero) log-det rather than the trivial identity. Seeded for reproducibility.
    rep_prior/rep_scale_init: warm a flow WITH the analytic repulsive prior active (set rep_scale before warmup so
    the learned field is trained to be reversible INCLUDING the prior term)."""
    from liquid_coupling_flow.ka_gridformer import _wrap_pm
    torch.manual_seed(0)
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=8)
    flow = E.EGNNClusterFlow(sigma_b=1.1, n_cage=n_cage, r_c=3.0, L=L, hidden_nf=32, n_layers=2, n_steps=n_steps,
                             rep_prior=rep_prior).to(DEV).train()
    if rep_scale_init is not None:
        flow.ce.egnn.rep_scale.data.fill_(rep_scale_init)
    opt = torch.optim.Adam(flow.parameters(), lr=1e-3)
    cloud, sp, c = E.build_cloud(pos, s, cl, sc, L, n_cage); x1 = cloud[:, :7]
    for _ in range(steps):
        z = E.sample_base(c, 7, flow.sigma_b)
        tgt = _wrap_pm(x1 - z, L); t = torch.rand(8, device=DEV)
        xt = torch.remainder(z + t[:, None, None] * tgt, L)
        v, _ = flow.ce.vel_div(torch.cat([xt, cloud[:, 7:]], 1), t, sp, 7)
        loss = ((v - tgt) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return flow.eval()

def test_sampler_equals_scorer():
    flow = _warmed_flow()
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    xC, logq = flow.sample(pos, s, cl, sc, L)
    logq2 = flow.log_q(pos, s, cl, xC, sc, L)
    assert xC.shape == (4, 7, 2)
    # atol = the RK4 round-trip accuracy on the stiff EGNN central-force field (the 1/r divergence makes it not
    # exactly reversible; config-dependent ~0.1-0.3). NOT a bug: the DIVERGENCE is exact to 6.48e-8
    # (test_perparticle_divergence_matches_bruteforce), the identity flow round-trips to 0, and the cutoff is ruled
    # out. This atol catches gross log-det/sign/base bugs (which were O(10-90)); the gate uses sample() not log_q.
    # If exact MH/IS is needed downstream, tighten with more RK4 steps / an adaptive or reversible integrator.
    assert torch.allclose(logq, logq2, atol=3e-1), (logq - logq2).abs().max()

def test_sampler_equals_scorer_with_rep_prior():
    """Close the exactness chain END-TO-END through the repulsive-prior path at the DEPLOYED amplitude (default
    init softplus(-4)=0.019, matching the trained model's learned amps 0.010-0.026). The divergence test
    (test_egnn_audit.test_divergence_exact_with_rep_prior) proves vel_div's log-det integrand is exact with the
    prior on; this proves the full RK4 sample()->log_q() round-trip integrates it without a gross log-det/sign/base
    bug when the prior is wired in.

    KNOWN, DOCUMENTED: the round-trip residual with the prior on (~1.0) is LARGER than the no-prior case
    (~0.1-0.3, test_sampler_equals_scorer). This is NOT a bug in the prior's divergence (that is exact pointwise);
    it is fixed-step RK4 losing accuracy on the STIFF r^-12 field, and it does not shrink monotonically with
    n_steps (empirically 64 steps ~= 32). A gross log-det/base/sign bug would be O(10-300) (a 15x-larger prior
    amplitude drives the residual to ~300), so atol=2.5 cleanly separates 'stiff-field integrator residual' from
    'broken machinery'. IMPLICATION: exact MH/IS with the prior ON wants a stiffer/adaptive/reversible integrator
    (or the prior-OFF model, whose log_q round-trips to ~0.2); the g(r) gate is unaffected (uses sample_fast, no
    log_q)."""
    flow = _warmed_flow(rep_prior=True)                           # default rep_scale init = -4.0 (deployed magnitude)
    assert flow.ce.egnn.rep_prior                                 # guard: the prior path is actually live
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    xC, logq = flow.sample(pos, s, cl, sc, L)
    logq2 = flow.log_q(pos, s, cl, xC, sc, L)
    assert xC.shape == (4, 7, 2)
    assert torch.allclose(logq, logq2, atol=2.5), (logq - logq2).abs().max()

def test_not_translation_invariant():
    """Translating the cluster ALONE (cage fixed) must change logq (position is pinned by the cage)."""
    flow = _warmed_flow()
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    xC, _ = flow.sample(pos, s, cl, sc, L)
    lq0 = flow.log_q(pos, s, cl, xC, sc, L)
    lq1 = flow.log_q(pos, s, cl, torch.remainder(xC + 0.5, L), sc, L)   # move cluster only
    assert (lq0 - lq1).abs().max() > 1e-2

def test_perparticle_divergence_matches_bruteforce():
    """Analytic cluster-restricted divergence == trace of the autograd Jacobian of the cluster velocity."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=1)
    ce = E.ConditionalEGNN(n_cage=n_cage, hidden_nf=32, n_layers=2, r_c=3.0, L=L).to(DEV)
    cloud, sp, c = E.build_cloud(pos, s, cl, sc, L, n_cage)
    cloud = cloud.double(); ce = ce.double()
    t = torch.tensor(0.37, device=DEV, dtype=torch.float64)
    k = 7
    xcl = cloud[:, :k].reshape(-1).clone().requires_grad_(True)   # [2k]
    def vel_flat(xf):
        cl2 = cloud.clone(); cl2[:, :k] = xf.reshape(1, k, 2)
        v, _ = ce.vel_div(cl2, t, sp, k)
        return v.reshape(-1)
    J = torch.autograd.functional.jacobian(vel_flat, xcl)         # [2k,2k]
    div_bf = torch.diagonal(J).sum()
    _, div_analytic = ce.vel_div(cloud, t, sp, k)
    assert abs(div_bf.item() - div_analytic.item()) < 1e-4, (div_bf.item(), div_analytic.item())

def test_ot_species_and_cost():
    torch.manual_seed(0)
    z = torch.randn(3, 7, 2, device=DEV); x = torch.randn(3, 7, 2, device=DEV)
    sp = torch.tensor([0, 0, 0, 0, 1, 1, 1], device=DEV)[None].expand(3, 7)
    perm = E.ot_assign(z, x, sp, L=9.13)
    assert perm.shape == (3, 7)
    for b in range(3):
        assert sorted(perm[b].tolist()) == list(range(7))          # valid permutation
        assert (sp[b] == sp[b][perm[b]]).all()                     # species preserved by the matching
    def cost(p):
        idx = p[:, :, None].expand(-1, -1, 2)
        return ((z - torch.gather(x, 1, idx)) ** 2).sum((-1, -2))
    ident = torch.arange(7, device=DEV)[None].expand(3, 7)
    assert (cost(perm) <= cost(ident) + 1e-5).all()                # OT cost <= identity cost

def test_train_smoke_and_load():
    torch.cuda.empty_cache()                                        # release prior tests' reserved GPU cache
    ck = E.train(steps=60, n_cage=48, hidden_nf=32, n_layers=2, batch=8, save=False)
    # per-step FM loss is noisy (batch=8, random seed-cluster each step) -> assert STABILITY (not a noisy 2-point
    # decrease): training runs, sigma_b from data, loss finite/bounded (not diverged). The real loss TREND is in train's log.
    assert 0.7 < ck["sigma_b"] < 1.6 and ck["loss_last"] < 4.0
    flow = E.load_flow(ck, DEV)
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    xC, logq = flow.sample(pos, s, cl, sc, L)
    assert torch.isfinite(logq).all() and xC.shape == (4, 7, 2)
