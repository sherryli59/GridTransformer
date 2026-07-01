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
    got = E.base_logp(x0, c, sb)
    d2 = ((x0 - c[:, None]) ** 2).sum(-1)
    want = (-d2 / (2 * sb ** 2) - math.log(2 * math.pi * sb ** 2)).sum(-1)
    assert torch.allclose(got, want, atol=1e-5)

def test_sigma_b_from_data():
    sc, L, geo, pos, s, cl, n_cage = _cfg()
    ref = torch.load(f"{E.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    sb = E.compute_sigma_b(ref["x"].to(DEV), ref["s"].to(DEV).long(), geo, sc, L, 7, n_cage)
    assert 0.7 < sb < 1.6            # cluster spread about its centroid

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
