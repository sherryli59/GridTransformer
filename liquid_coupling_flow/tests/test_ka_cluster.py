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
