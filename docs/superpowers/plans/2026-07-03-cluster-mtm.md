# MTM Cluster Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Spec: docs/superpowers/specs/2026-07-03-cluster-mtm-design.md (READ FIRST — K1 kernel, validity conditions, K3 trap).

**Goal:** Implement the K1 Barker/Tjelmeland ensemble independence-MTM cluster kernel over the ARM-FULL spline proposal, with the vectorized cluster-involving energy; gate by exactness-by-stationarity.

## Global Constraints
- NEVER `git add -A`. fp32. β=2.0 (T*=0.5). Long jobs: `> log 2>&1`, mtime liveness.
- Spline-head checkpoints ONLY (full support) — assert at kernel entry; bins must be rejected.
- The kernel's exactness rests on x_C-independent conditioning — assert structurally (see `assert_mtm_valid`).
- K3 (SIR without the current state) is implemented ONLY as the negative control inside the gate script, never exported for production use.

---

### Task 1: `ka_cluster_mtm.py` + tests (single task, one review)

**Files:** Create `liquid_coupling_flow/ka_cluster_mtm.py`, `liquid_coupling_flow/tests/test_ka_cluster_mtm.py`.

**Complete implementation:**

```python
"""Multiple-try Metropolis cluster kernel (K1: Barker/Tjelmeland ensemble independence-MTM).
Spec: docs/superpowers/specs/2026-07-03-cluster-mtm-design.md. Exactness rests on ClusterProposal's
x_C-INDEPENDENT conditioning (frame + tokens from x_R only) -> pure independence sampler on the block ->
no reverse draws needed. Spline-head checkpoints only (full support; the bins head is hard-zero outside
the box and violates the independence-sampler support condition)."""
from __future__ import annotations
import torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR


def cluster_energy(xC, pos, cluster_idx, s, L):
    """Cluster-involving shifted-KA energy: cross (cluster-rest) + intra (cluster-cluster) pair terms.
    xC [B,M,k,2] candidate cluster coords; pos [B,N,2] current config (its cluster columns are IGNORED —
    the rest columns supply the cross term); s [N] or [B,N]. Returns [B,M]. The rest-rest term is omitted
    (identical across the candidate set -> cancels in all MTM weight ratios)."""
    B, M, k, _ = xC.shape; N = pos.shape[1]; dev = xC.device; dtype = xC.dtype
    sig_t = torch.tensor(SIGMA, device=dev, dtype=dtype); eps_t = torch.tensor(EPS, device=dev, dtype=dtype)
    s = s.expand(B, N) if s.dim() == 1 else s
    mask = torch.ones(N, dtype=torch.bool, device=dev); mask[cluster_idx] = False
    rest = pos[:, mask]                                              # [B,R,2]
    s_rest = s[:, mask]; s_cl = s[:, cluster_idx]                    # [B,R],[B,k]

    def pair_e(d2, si, sj):
        sig = sig_t[si, sj]; eps = eps_t[si, sj]; rc = RCUT_FACTOR * sig
        inv6 = (sig * sig / d2) ** 3
        e = 4 * eps * (inv6 * inv6 - inv6)
        src6 = (sig / rc) ** 6
        return torch.where(d2 < rc * rc, e - 4 * eps * (src6 * src6 - src6), torch.zeros_like(e))

    R = rest.shape[1]
    d = xC[:, :, :, None, :] - rest[:, None, None, :, :]; d = d - L * torch.round(d / L)
    d2 = (d ** 2).sum(-1).clamp_min(1e-12)
    e_cross = pair_e(d2, s_cl[:, None, :, None].expand(B, M, k, R),
                     s_rest[:, None, None, :].expand(B, M, k, R)).sum((-1, -2))
    di = xC[:, :, :, None, :] - xC[:, :, None, :, :]; di = di - L * torch.round(di / L)
    d2i = (di ** 2).sum(-1).clamp_min(1e-12)
    iu = torch.triu(torch.ones(k, k, dtype=torch.bool, device=dev), 1)
    e_intra = (pair_e(d2i, s_cl[:, None, :, None].expand(B, M, k, k),
                      s_cl[:, None, None, :].expand(B, M, k, k)) * iu).sum((-1, -2))
    return e_cross + e_intra


def assert_mtm_valid(P, pos, s, cluster_idx, sc, L):
    """Structural validity: spline head (full support) + x_C-independent conditioning (frame/tokens identical
    when the cluster is displaced). Cheap; call once per kernel construction."""
    assert getattr(P, "head_mode", "bins") == "spline", "MTM requires the full-support spline head"
    B, N, _ = pos.shape
    s2 = s.expand(B, N) if s.dim() == 1 else s
    pos2 = pos.clone(); pos2[:, cluster_idx] = torch.remainder(pos2[:, cluster_idx] + 1.234, L)
    o1, R1, t1, q1, u1, sp1 = P._ctx_tokens(pos, s2, cluster_idx, sc, L)
    o2, R2, t2, q2, u2, sp2 = P._ctx_tokens(pos2, s2, cluster_idx, sc, L)
    for a, b in ((o1, o2), (R1, R2), (t1, t2), (q1, q2), (u1, u2)):
        assert torch.equal(a, b), "conditioning depends on x_C — MTM independence assumption violated"


@torch.no_grad()
def mtm_move(P, pos, s, cluster_idx, sc, L, M=32, beta=2.0, gen=None, chunk_rows=4096):
    """One K1 ensemble move on the given cluster for all B chains.
    Returns (pos_new [B,N,2], moved [B] bool, info dict)."""
    B, N, _ = pos.shape; dev = pos.device; k = cluster_idx.shape[0]
    s2 = s.expand(B, N) if s.dim() == 1 else s
    rows = B * M
    if rows <= chunk_rows:
        pos_rep = pos.repeat_interleave(M, 0); s_rep = s2.repeat_interleave(M, 0)
        yC, logq_y = P.sample(pos_rep, s_rep, cluster_idx, sc, L)
        yC = yC.view(B, M, k, 2); logq_y = logq_y.view(B, M)
    else:                                                            # chunk along the M axis
        ys, ls = [], []
        Mc = max(1, chunk_rows // B)
        for m0 in range(0, M, Mc):
            m = min(Mc, M - m0)
            yc, lq = P.sample(pos.repeat_interleave(m, 0), s2.repeat_interleave(m, 0), cluster_idx, sc, L)
            ys.append(yc.view(B, m, k, 2)); ls.append(lq.view(B, m))
        yC = torch.cat(ys, 1); logq_y = torch.cat(ls, 1)
    logq_x = P.log_q(pos, s2, cluster_idx, pos[:, cluster_idx], sc, L)                    # [B]
    U_y = cluster_energy(yC, pos, cluster_idx, s2, L)                                     # [B,M]
    U_x = cluster_energy(pos[:, cluster_idx].unsqueeze(1), pos, cluster_idx, s2, L).squeeze(1)
    ell = torch.cat([(-beta * U_x - logq_x)[:, None], -beta * U_y - logq_y], 1)           # [B,M+1]; j=0 = current
    w = torch.softmax(ell, -1)
    J = torch.multinomial(w, 1, generator=gen).squeeze(1)                                 # Barker-ensemble select
    moved = J > 0
    pick = (J - 1).clamp_min(0)
    xC_sel = yC[torch.arange(B, device=dev), pick]                                        # [B,k,2]
    xC_new = torch.where(moved[:, None, None], xC_sel, pos[:, cluster_idx])
    pos_new = pos.clone(); pos_new[:, cluster_idx] = xC_new
    info = {"move_prob": float(w[:, 1:].sum(-1).mean()),
            "w_ess": float((1.0 / (w ** 2).sum(-1)).mean()),
            "moved_frac": float(moved.float().mean())}
    return pos_new, moved, info


@torch.no_grad()
def mtm_sweep(P, pos, s, sc, L, M=32, beta=2.0, k=7, gen=None):
    """One sweep: an MTM move at every seed slot in random order. Returns (pos, mean move_prob)."""
    N = pos.shape[1]; probs = []
    order = torch.randperm(N, generator=gen).tolist()
    for seed in order:
        cl = KC.cluster_slots(seed, sc, k, L)
        pos, _, info = mtm_move(P, pos, s, cl, sc, L, M=M, beta=beta, gen=gen)
        probs.append(info["move_prob"])
    return pos, sum(probs) / len(probs)
```

**Tests** (`test_ka_cluster_mtm.py`):

```python
import torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import cluster_energy, mtm_move, assert_mtm_valid
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
import os
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _env(B=4):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, pos, s, cl

def test_cluster_energy_matches_full_difference():
    """U_clu(new)-U_clu(cur) must equal ka_energy(new)-ka_energy(cur) (rest-rest cancels)."""
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    xC_new = torch.remainder(pos[:, cl] + 0.3 * torch.randn_like(pos[:, cl]), L)
    pos_new = pos.clone(); pos_new[:, cl] = xC_new
    dU_full = ka_energy(pos_new, s[0], L) - ka_energy(pos, s[0], L)
    dU_clu = (cluster_energy(xC_new.unsqueeze(1), pos, cl, s, L)
              - cluster_energy(pos[:, cl].unsqueeze(1), pos, cl, s, L)).squeeze(1)
    assert torch.allclose(dU_full, dU_clu, atol=1e-3), (dU_full - dU_clu).abs().max()

def test_assert_mtm_valid_spline_pass_bins_fail():
    sc, L, pos, s, cl = _env()
    ck = torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    assert_mtm_valid(P, pos, s, cl, sc, L)                      # spline: passes
    ckb = torch.load(os.path.join(ART, "ka_cluster_flow_N100.pt"), map_location=DEV, weights_only=False)
    Pb = _load(ckb, DEV)
    import pytest
    with pytest.raises(AssertionError):
        assert_mtm_valid(Pb, pos, s, cl, sc, L)                 # bins: rejected

def test_mtm_move_semantics():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env()
    ck = torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    pos_new, moved, info = mtm_move(P, pos, s, cl, sc, L, M=8)
    assert pos_new.shape == pos.shape and torch.isfinite(pos_new).all()
    # non-cluster columns NEVER change
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(pos_new[:, mask], pos[:, mask])
    # unmoved chains keep their cluster exactly
    if (~moved).any():
        assert torch.equal(pos_new[~moved][:, cl], pos[~moved][:, cl])
    assert 0.0 <= info["move_prob"] <= 1.0 and info["w_ess"] >= 1.0

def test_mtm_move_chunked_equals_unchunked_shapes():
    torch.manual_seed(0)
    sc, L, pos, s, cl = _env(B=2)
    ck = torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV, weights_only=False)
    P = _load(ck, DEV)
    pos_new, moved, info = mtm_move(P, pos, s, cl, sc, L, M=8, chunk_rows=4)   # forces chunking path
    assert pos_new.shape == pos.shape and torch.isfinite(pos_new).all()
```

Steps: write tests → fail (module missing) → implement → all green (+ existing `test_cluster_spline_head.py` 7/7 still green) → commit both files: `feat(cluster-mtm): K1 Barker-ensemble independence-MTM kernel + vectorized cluster energy`.

### Task 2 (controller): stationarity gate + performance curve
- `mtm_gate.py` (scratch): B=64 chains from the reference; 200 sweeps at M=32; track ⟨U⟩/N (full ka_energy) + g_BB every 10 sweeps; PASS if ⟨U⟩/N stays within the reference plateau band (|drift| < 0.02) and g_BB peak within ±0.15 of data. NEGATIVE CONTROL: K3 (softmax over candidates ONLY, current state excluded, always move) at M=2 for 50 sweeps — must visibly drift (proves test sensitivity).
- Performance: move_prob + wall-clock/sweep at M ∈ {8, 32, 128}.
- Record in report + ledger + memory; commit.
