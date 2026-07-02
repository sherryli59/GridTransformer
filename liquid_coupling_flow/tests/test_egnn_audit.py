"""Numerical audits of the traceable-EGNN periodic velocity backend (invariants the KA cluster flow relies on).
BUG-HUNT tests (never-refute-bug-hypothesis): each asserts the INTENDED semantics; a red test is a confirmed
backend defect, not a test bug. Run: pytest liquid_coupling_flow/tests/test_egnn_audit.py -v -s"""
import torch, torch.nn as nn
from liquid_coupling_flow import ka_cluster as KC, ka_cluster_egnn as E
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _cloud(B=4, n_cage=48):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(f"{E.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    cloud, sp, c = E.build_cloud(pos, s, cl, sc, L, n_cage)
    return cloud, sp, L


def _rand_ce(L, max_neighbors=24, seed=0):
    torch.manual_seed(seed)
    # randomly initialized (NOT zero-init: a zero velocity field passes any invariance test vacuously)
    return E.ConditionalEGNN(n_cage=48, k=7, r_c=3.0, L=L, hidden_nf=32, n_layers=2,
                             max_neighbors=max_neighbors).to(DEV)


def test_velocity_translation_invariance_random_init():
    cloud, sp, L = _cloud()
    ce = _rand_ce(L)
    t = torch.tensor(0.7, device=DEV)
    v0 = ce.egnn.forward(t, cloud, sp)
    v1 = ce.egnn.forward(t, torch.remainder(cloud + torch.tensor([1.7, -2.3], device=DEV), L), sp)
    assert torch.allclose(v0, v1, atol=1e-3), f"translation breaks velocity: max diff {(v0 - v1).abs().max()}"


def test_velocity_translation_invariance_trained_ckpt():
    ck = torch.load(f"{E.ART}/ka_cluster_egnn_N100.pt", map_location=DEV, weights_only=False)
    P = E.load_flow(ck, DEV)
    cloud, sp, L = _cloud()
    t = torch.tensor(0.7, device=DEV)
    v0 = P.ce.egnn.forward(t, cloud, sp)
    v1 = P.ce.egnn.forward(t, torch.remainder(cloud + torch.tensor([1.7, -2.3], device=DEV), L), sp)
    assert torch.allclose(v0, v1, atol=1e-3), f"trained field not shift-invariant: max diff {(v0 - v1).abs().max()}"


def test_velocity_D4_equivariance():
    cloud, sp, L = _cloud()
    ce = _rand_ce(L)
    t = torch.tensor(0.7, device=DEV)
    R = torch.tensor([[0.0, -1.0], [1.0, 0.0]], device=DEV)          # 90 deg about the box center (torus-safe)
    xr = torch.remainder((cloud - L / 2) @ R.T + L / 2, L)
    v0 = ce.egnn.forward(t, cloud, sp)
    v1 = ce.egnn.forward(t, xr, sp)
    assert torch.allclose(v1, v0 @ R.T, atol=1e-3), f"rotation breaks velocity: {(v1 - v0 @ R.T).abs().max()}"


class _OnePot(nn.Module):
    def forward(self, z):
        return torch.ones(z.shape[0], 1, device=z.device, dtype=z.dtype)


def test_central_force_semantics_pot_one():
    """With pot forced to 1 and the inner coord updates zeroed, the periodic velocity MUST equal
    sum_j min-image(x_j - x_i) over the mn nearest neighbours (the documented central-force form).
    This pins the diffij frame. Under the frame-mixing hypothesis the code returns ~ -mn * x_i instead."""
    cloud, sp, L = _cloud()
    ce = _rand_ce(L)
    dyn = ce.egnn                                                     # EGNN_dynamics
    for i in range(dyn.egnn.n_layers):                                # dyn.egnn = inner EGNN
        gcl = dyn.egnn._modules[f"gcl_{i}"]
        lin = [m for m in gcl.coord_mlp if isinstance(m, nn.Linear)][-1]
        lin.weight.data.zero_()                                       # x_final == xs_centered (no inner coord update)
    dyn.pot_model = _OnePot()
    t = torch.tensor(0.5, device=DEV)
    v = dyn.forward(t, cloud, sp)                                     # [B,P,2]
    B, P, D = cloud.shape
    d = cloud[:, None, :, :] - cloud[:, :, None, :]                   # d[b,i,j] = x_j - x_i
    d = d - L * torch.round(d / L)
    dist = d.norm(dim=-1) + torch.eye(P, device=DEV)[None] * 1e6
    idx = dist.argsort(-1)[:, :, :dyn.max_nb_neighbors]
    exp = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, D)).sum(2)
    assert torch.allclose(v, exp, atol=1e-3), f"central-force frame wrong: max diff {(v - exp).abs().max()}"


def _production_phantom_count(cloud, L, mn, cutoff=3.0):
    """Count edges the PRODUCTION compute_edges selects whose RAW cloud-frame separation exceeds the cutoff.
    The per-particle clouds are contiguous (min-imaged about the central particle upstream), so any such edge
    is a phantom; guards against re-introducing box-L folding in compute_edges."""
    ce = _rand_ce(L, max_neighbors=mn)
    dyn = ce.egnn
    B, P, D = cloud.shape
    d = cloud[:, None, :, :] - cloud[:, :, None, :]
    d = d - L * torch.round(d / L)
    dist = d.norm(dim=-1) + torch.eye(P, device=cloud.device)[None] * 1e6
    idx = dist.argsort(-1)[:, :, :dyn.max_nb_neighbors]
    nbr = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, D))    # contiguated cloud, as in _compute_common_terms
    xs_c = (nbr - nbr.mean(2, keepdim=True)).reshape(B * P, dyn.max_nb_neighbors, D)
    rows, cols = dyn.compute_edges(xs_c, cutoff=cutoff)
    flat = xs_c.reshape(-1, D)
    raw = (flat[rows] - flat[cols]).norm(dim=-1)
    return int((raw >= cutoff).sum())


def test_no_phantom_edges_knn24():
    cloud, sp, L = _cloud()
    n = _production_phantom_count(cloud, L, 24)
    assert n == 0, f"{n} phantom inner edges at kNN-24"


def test_no_phantom_edges_fullcloud():
    cloud, sp, L = _cloud()
    n = _production_phantom_count(cloud, L, cloud.shape[1] - 1)
    assert n == 0, f"{n} phantom inner edges at full cloud"
