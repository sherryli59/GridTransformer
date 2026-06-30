import torch, pytest
from liquid_coupling_flow import ka_cluster as C
from liquid_coupling_flow.ka_noncausal import NonCausalLF
ART = "liquid_coupling_flow/artifacts"; DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _model(N=100):
    ck = torch.load(f"{ART}/ka_noncausal_N100.pt", map_location=DEV, weights_only=False)
    m = NonCausalLF(rho=1.2, n_bins=192, knn=ck["knn"], canonical=False).to(DEV).eval()
    m.load_state_dict(ck["state_dict"])
    return m


def test_cluster_slots_deterministic_and_seeded():
    m = _model(); N = 100; sc = m.geo._scaffold(N, DEV); L = m.geo._Lof(N); k = 7
    cl = C.cluster_slots(5, sc, k, L)
    assert cl.numel() == k and cl[0].item() == 5                     # contains seed first
    assert torch.equal(cl, C.cluster_slots(5, sc, k, L))            # deterministic
    assert len(set(cl.tolist())) == k                               # distinct slots
    # nearest-neighbour sanity: all cluster slots are within the k-th nearest scaffold distance of the seed
    d = (C._wrap_pm(sc - sc[5][None], L) ** 2).sum(-1)
    assert d[cl].max() <= d.topk(k, largest=False).values.max() + 1e-6


def test_frame_independent_of_xC_bit_exact():
    """KEYSTONE: proposal-A's Jacobian-1 exactness holds iff the frame is bit-exactly independent of x_C.
    Move the whole cluster a lot, recompute -> frame must be identical to the bit."""
    m = _model(); N = 100; sc = m.geo._scaffold(N, DEV); L = m.geo._Lof(N)
    ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos = ref["x"][0].to(DEV).clone(); cl = C.cluster_slots(5, sc, 7, L)
    o0, R0 = C.cluster_frame(pos, cl, sc, L)
    p2 = pos.clone()
    p2[cl] = torch.remainder(p2[cl] + torch.tensor([1.7, -1.1], device=DEV), L)       # move the WHOLE cluster
    o1, R1 = C.cluster_frame(p2, cl, sc, L)
    assert (o1 - o0).abs().max().item() == 0.0, "frame origin depends on x_C!"
    assert (R1 - R0).abs().max().item() == 0.0, "frame rotation depends on x_C!"


def test_frame_roundtrip_rigid():
    m = _model(); N = 100; sc = m.geo._scaffold(N, DEV); L = m.geo._Lof(N)
    ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos = ref["x"][0].to(DEV).clone(); cl = C.cluster_slots(5, sc, 7, L)
    o, R = C.cluster_frame(pos, cl, sc, L)
    u = C.to_frame(pos[cl], o, R, L); back = C.from_frame(u, o, R, L)
    assert torch.allclose(back, pos[cl], atol=1e-5)                                   # rigid map invertible
    assert torch.allclose(R @ R.T, torch.eye(2, device=DEV), atol=1e-5)              # R orthonormal


def test_cluster_roundtrip_sampler_equals_evaluator():
    """PRIMARY density guard: the density the sampler realizes equals log_q at the sampled point (untrained
    weights are fine — this checks sample/score consistency, not quality)."""
    from liquid_coupling_flow.ka_cluster_flow import ClusterProposal
    m = _model(); N = 100; sc = m.geo._scaffold(N, DEV); L = m.geo._Lof(N)
    ref = torch.load(f"{ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos = ref["x"][0:4].to(DEV); s = ref["s"].to(DEV).long(); cl = C.cluster_slots(5, sc, 7, L)  # B=4 chains
    P = ClusterProposal().to(DEV).eval()
    xC_new, logq = P.sample(pos, s, cl, sc, L)
    logq_eval = P.log_q(pos, s, cl, xC_new, sc, L)
    assert xC_new.shape == (4, 7, 2)
    assert torch.allclose(logq, logq_eval, atol=1e-4), (logq - logq_eval).abs().max()
