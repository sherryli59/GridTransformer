"""Stage-B GB0 tests: the PT-ladder event miner (ka_events.py).

Synthetic-truth tests: build pairs with KNOWN character (pure vibration / one localized hop cluster / a
whole-config replacement) and check the classifier recovers exactly that. Layout test uses the real N=256
ladder artifact if present.
"""
import os, torch, pytest

DEV = "cpu"
L = (100 / 1.2) ** 0.5


def _base(B=1, N=100, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(B, N, 2, generator=g) * L


def test_pair_index_layout():
    """Adjacent-time same-replica pairs within each seed block; never across the seed boundary."""
    from liquid_coupling_flow.ka_events import pair_index
    ia, ib = pair_index(n_seeds=2, nc=5, n_rep=3)              # per seed: 5 times x 3 replicas = 15
    assert len(ia) == 2 * 4 * 3                                 # (nc-1) * n_rep * n_seeds
    # first seed-block pair for replica 1: (0*3+1, 1*3+1)
    assert (ia[0], ib[0]) == (0, 3) or (0, 3) in set(zip(ia.tolist(), ib.tolist()))
    assert (1, 4) in set(zip(ia.tolist(), ib.tolist()))
    # no pair crosses the seed boundary (last time of seed0 = idx 12..14; first of seed1 = 15..17)
    assert (12, 15) not in set(zip(ia.tolist(), ib.tolist()))


def test_classify_vibration():
    """Small jitter everywhere -> clean pair, NO event."""
    from liquid_coupling_flow.ka_events import classify_pair
    xa = _base()[0]
    xb = torch.remainder(xa + 0.05 * torch.randn_like(xa), L)
    r = classify_pair(xa, xb, L)
    assert r["kind"] == "quiet" and r["mobile"].numel() == 0


def test_classify_event_localized_hop():
    """A 4-particle localized hop (~0.8) on a quiet background -> event with exactly that mobile set."""
    from liquid_coupling_flow.ka_events import classify_pair
    xa = _base()[0]
    xb = torch.remainder(xa + 0.03 * torch.randn_like(xa), L)
    movers = torch.tensor([10, 11, 12, 13])
    xb[movers] = torch.remainder(xa[movers] + 0.8, L)          # coherent ~0.8*sqrt(2) hop
    r = classify_pair(xa, xb, L)
    assert r["kind"] == "event"
    assert set(r["mobile"].tolist()) == set(movers.tolist())
    assert r["k"] == 4 and torch.isfinite(torch.tensor(r["extent"]))


def test_classify_exchange_rejected():
    """A completely different config (PT exchange replacement) -> rejected, not an event."""
    from liquid_coupling_flow.ka_events import classify_pair
    xa = _base(seed=0)[0]; xb = _base(seed=1)[0]
    r = classify_pair(xa, xb, L)
    assert r["kind"] == "exchange"


def test_min_image_wrap():
    """A hop across the periodic boundary is measured min-image (small), not box-sized."""
    from liquid_coupling_flow.ka_events import classify_pair
    xa = _base()[0]
    xb = xa.clone()
    xa[0] = torch.tensor([0.05, 0.05]); xb[0] = torch.tensor([L - 0.05, L - 0.05])   # true disp ~0.14
    r = classify_pair(xa, xb, L)
    assert r["kind"] == "quiet", "wrap-crossing vibration misread as a hop (min-image bug)"


@pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(__file__), "..", "artifacts",
                                                    "pt_ladder_N256.pt")), reason="no ladder artifact")
def test_mine_ladder_runs_on_real_artifact():
    """End-to-end smoke on the real N=256 ladder: stats structure sane, counts add up."""
    from liquid_coupling_flow.ka_events import mine_ladder
    stats = mine_ladder("pt_ladder_N256.pt", rungs=[0], max_pairs=200)
    s = stats[0]
    assert s["n_pairs"] == 200
    assert s["n_exchange"] + s["n_quiet"] + s["n_event"] == 200
    for ev in s["events"][:3]:
        assert 1 <= ev["k"] and ev["rung"] == 0 and "beta" in ev
