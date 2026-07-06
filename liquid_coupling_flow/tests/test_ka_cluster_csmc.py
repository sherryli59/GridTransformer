"""Stage A1 tests: (1) refactor safety net for ClusterProposal's per-step API — byte-identical golden +
sample<->log_q consistency; (2) the cSMC cluster-move kernel (conditional SMC / particle Gibbs, PGAS).

Runs at N=100 (scaffold fixed by the ka_noncausal checkpoint). CPU-deterministic for the golden.
"""
import os, pytest, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART

DEV = "cpu"                                   # golden byte-identical => force deterministic CPU
GOLD = os.path.join(os.path.dirname(__file__), "test_ka_cluster_csmc_golden.pt")
CKPT = os.path.join(ART, "ka_cluster_flow_full_N100.pt")
REF = os.path.join(ART, "ka_reference_N100.pt")


def _env(B=4, seed=0):
    if not (os.path.exists(CKPT) and os.path.exists(REF)):
        pytest.skip("ARM-FULL ckpt or N=100 reference unavailable")
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(REF, map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    P = _load(torch.load(CKPT, map_location=DEV, weights_only=False), DEV)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, geo, pos, s, P, cl


# ---------- refactor safety net ----------

def test_refactor_byte_identical():
    """sample()/log_q() outputs are bit-identical to a golden captured from the pre-refactor code.
    Green now (captures golden), MUST stay green after the _step extraction."""
    sc, L, geo, pos, s, P, cl = _env()
    torch.manual_seed(1234)
    xC, logq = P.sample(pos, s, cl, sc, L)
    lq = P.log_q(pos, s, cl, xC, sc, L)
    if not os.path.exists(GOLD):
        torch.save({"xC": xC, "logq": logq, "lq": lq}, GOLD)
        pytest.skip("golden captured (pre-refactor baseline) — rerun to compare")
    g = torch.load(GOLD, map_location=DEV, weights_only=False)
    assert torch.equal(xC, g["xC"]), "sample() positions drifted from golden"
    assert torch.equal(logq, g["logq"]), "sample() logq drifted from golden"
    assert torch.equal(lq, g["lq"]), "log_q() drifted from golden"


def test_sample_logq_roundtrip_consistent():
    """logq accumulated during sample() equals log_q() re-scoring the same coords (exactness of the AR density)."""
    sc, L, geo, pos, s, P, cl = _env()
    torch.manual_seed(7)
    xC, logq = P.sample(pos, s, cl, sc, L)
    lq = P.log_q(pos, s, cl, xC, sc, L)
    assert torch.allclose(logq, lq, atol=1e-4), f"sample/log_q mismatch max {(logq-lq).abs().max():.2e}"


# ---------- cSMC kernel (new behavior) ----------

def test_incremental_energy_telescopes():
    """Sum of per-step in-frame incremental potentials over a full placement == the cluster's total energy
    (cluster-cluster + cluster-env). Guards the in-frame DeltaU decomposition."""
    from liquid_coupling_flow.ka_cluster_csmc import cluster_energy_incremental, cluster_energy_total
    sc, L, geo, pos, s, P, cl = _env()
    xC = pos[:, cl]                                                  # the reference cluster itself
    inc = cluster_energy_incremental(xC, pos, s, cl, L)             # [B,k] per-step increments
    tot = cluster_energy_total(xC, pos, s, cl, L)                   # [B] direct
    assert torch.allclose(inc.sum(1), tot, atol=1e-4), \
        f"telescope mismatch max {(inc.sum(1)-tot).abs().max():.2e}"


def test_csmc_M1_is_identity():
    """Conditional SMC with M=1: the single particle IS the retained reference path -> output == input exactly."""
    from liquid_coupling_flow.ka_cluster_csmc import csmc_cluster_move
    sc, L, geo, pos, s, P, cl = _env()
    torch.manual_seed(0)
    pos2, s2, info = csmc_cluster_move(P, pos, s, cl, sc, L, beta=2.0, M=1)
    assert torch.equal(pos2, pos), "M=1 cSMC must be identity on positions"
    assert torch.equal(s2, s), "species must be untouched"


def test_csmc_invariances():
    """Species untouched everywhere; non-cluster positions untouched; counts preserved."""
    from liquid_coupling_flow.ka_cluster_csmc import csmc_cluster_move
    sc, L, geo, pos, s, P, cl = _env()
    torch.manual_seed(0)
    counts0 = s.sum(1).clone()
    pos2, s2, info = csmc_cluster_move(P, pos, s, cl, sc, L, beta=2.0, M=8)
    assert torch.equal(s2, s), "species must be untouched by a positional move"
    assert torch.equal(s2.sum(1), counts0)
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(pos2[:, mask], pos[:, mask]), "non-cluster positions must be untouched"


def test_csmc_moves_and_finite():
    """M>1 actually resamples the cluster (not a no-op) and everything stays finite."""
    from liquid_coupling_flow.ka_cluster_csmc import csmc_cluster_move
    sc, L, geo, pos, s, P, cl = _env(B=8)
    torch.manual_seed(0)
    pos2, s2, info = csmc_cluster_move(P, pos, s, cl, sc, L, beta=2.0, M=16)
    assert torch.isfinite(pos2).all()
    moved = (pos2[:, cl] - pos[:, cl]).abs().sum((-1, -2)) > 1e-6
    assert moved.any(), "with M=16 at least some rows should adopt a resampled cluster"
    assert 0.0 <= info["ref_survival"] <= 1.0


def test_pgas_branch_runs():
    """Ancestor sampling (PGAS) path is valid: finite, species untouched, counts preserved."""
    from liquid_coupling_flow.ka_cluster_csmc import csmc_cluster_move
    sc, L, geo, pos, s, P, cl = _env(B=4)
    torch.manual_seed(0)
    pos2, s2, info = csmc_cluster_move(P, pos, s, cl, sc, L, beta=2.0, M=8, ancestor=True)
    assert torch.isfinite(pos2).all() and torch.equal(s2, s)
    assert torch.equal(s2.sum(1), s.sum(1))
