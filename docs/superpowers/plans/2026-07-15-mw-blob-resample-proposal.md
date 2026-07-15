# MW Blob-Resample Proposal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `MWBlobProposal`, a cage-anchored blob-resample proposal for the mW transition kernel — exact two-state transition density, centroid-anchored, β-conditioned, size-transferable — replacing the repurposed scaffold model as the `two_blob`/bridge proposal and supplying the K=1 single-site heat-bath.

**Architecture:** Pure-geometry primitives (reversible selection/cage, cage-vector frame, sites, centroid anchor) + a labeled Hungarian assignment + a Cat3-headed AR module that places each blob particle as `centroid(cage ∪ prefix) + offset`, with slot order fixed by cage-derived sites and the head seeing ordinal-only slot features. The MH move is genuinely two-state: forward `sample_block(x, selection)` uses the source-x assignment; reverse `transition_log_prob(x, y, selection)` recomputes the assignment at y. Bridge acceptance uses the full geometric-bridge ratio.

**Tech Stack:** PyTorch (CUDA), `scipy.optimize.linear_sum_assignment` (1.13.0), existing `liquid_coupling_flow/mw` modules (`mw_energy`, `mw_kernels`, `mw_smc_portfolio`, `mw_reference`), `ka3d_scaffold_ar.Cat3Head`, `ka3d_smc_bridge.geometric_bridge_log_accept`, pytest.

**Spec:** `docs/superpowers/specs/2026-07-15-mw-blob-resample-proposal-design.md`

## Global Constraints

- **Exactness is the product.** `sample_block(x, selection)`'s accumulated log-density MUST equal `transition_log_prob(target=y, source=x, selection)` to ≤ 1e-4/particle. It must NOT equal a self-score `transition_log_prob(y, y, selection)`. Every exactness test compares the two-state transition at (y ← x), never a self-score.
- **Anchor is always `centroid(cage ∪ x_{<j})`.** Sites set order ONLY; they never enter the positional head and never replace the anchor. The head receives ordinal/pattern slot features only — never absolute site coordinates.
- **Cage is keyed to the state-independent selection center**, never the blob centroid. `selection = {"centers", "block_idx", "cage_idx"}` is computed once and threaded through forward and reverse unchanged. Re-deriving it at y must reproduce identical `block_idx` and `cage_idx` (reject in-kernel otherwise).
- **Bridge acceptance uses the full ratio** `ka3d_smc_bridge.geometric_bridge_log_accept` with `log_r_reverse = transition_log_prob(x, y, selection)`, `log_r_forward = logq_fwd`; the `(1−λ)[log q0(y)−log q0(x)]` term never drops. Proposal head gets **β_eff = λ·β**.
- **Hungarian always** (never nearest-site-with-rejection): cost = min-image squared distance in float64, solver `scipy.optimize.linear_sum_assignment` pinned, deterministic tie-break lexicographic by `(slot index, particle label)`.
- **Frame** = cage-vector Gram–Schmidt (deterministic), NOT raw PCA. `e1 = normalize(cage_nearest − c0)`, `e2` = Gram–Schmidt of the largest-rejection cage vector, `e3 = e1 × e2`; ties broken by `cage_idx`.
- **CLAUDE.md directives:** long training/eval runs are launched by the CONTROLLER (never a subagent's background — it dies at teardown); both streams to a log (`> log.out 2>&1`); per-unit incremental checkpoints; raw logs + one-off scripts committed to `reports/logs-2026-07-15/` immediately; every plot's full path printed; judge by full distributions, not scalars; kill/check runs by exact PID.
- **System constants** (`liquid_coupling_flow/mw/mw_energy.py`): `T_STAR=0.09632`, `RHO_STAR=0.4564`, `A_CUT=1.8`; N=64 → L=(64/RHO_STAR)^(1/3)≈5.1953; β=1/T*. Run pytest from `/mnt/ssd/GridTransformer` as `python -m pytest <path> -v`. GPU is shared (D0/other runs active) — unit tests run CPU with `CUDA_VISIBLE_DEVICES=""`.

## File Structure

- `liquid_coupling_flow/mw/mw_blob_geom.py` — pure-geometry primitives (Task 1): `select`, `cage_frame`, `blob_sites`, `blob_anchor`. Deterministic, no nn.
- `liquid_coupling_flow/mw/mw_blob_assign.py` — labeled Hungarian assignment (Task 2): `hungarian_assign`.
- `liquid_coupling_flow/mw/mw_blob.py` — `MWBlobProposal` nn.Module (Tasks 3–5): context encoder + Cat3 ordinal head, `sample_block`, `transition_log_prob`, two-blob mode.
- `liquid_coupling_flow/mw/mw_blob_train.py` — training + `load_model` (Task 7).
- `liquid_coupling_flow/mw/mw_kernels.py` — MODIFY (Task 6): add `single_site_move`, rewrite `two_blob_move` to the two-state/selection API.
- `liquid_coupling_flow/tests/test_mw_blob_geom.py`, `test_mw_blob_assign.py`, `test_mw_blob.py`, `test_mw_blob_kernel.py` — unit/exactness tests (Tasks 1–6).
- `reports/logs-2026-07-15/gate_*.py` — measurement-gate scripts (Task 8).

---

### Task 1: Geometry primitives — reversible selection, cage, cage-vector frame, sites, anchor

**Files:**
- Create: `liquid_coupling_flow/mw/mw_blob_geom.py`
- Test: `liquid_coupling_flow/tests/test_mw_blob_geom.py`

**Interfaces:**
- Consumes: `mw_kernels.recanonicalize` is not needed here; uses only torch + `mw_energy` constants indirectly (L passed in).
- Produces:
  - `select(x, centers, K, M, L) -> dict` with keys `centers [n_c,3]`, `block_idx [Kb] long` (Kb=K for one center, 2K for two), `cage_idx [M] long`. `block_idx` = K nearest non-overlapping to each center; `cage_idx` = M nearest non-blob to the center set. Deterministic; a pure function of `centers` and the NON-block coordinates.
  - `cage_frame(cage_xyz, c0, L) -> (Frame [3,3], spacing float)` — deterministic Gram–Schmidt frame from cage vectors + cage-only spacing.
  - `blob_sites(cage_xyz, L, K) -> (sites [K,3], order [K] long)` — K ordered sites `c0 + Frame·(spacing·pattern[K])`, Morton order.
  - `blob_anchor(cage_xyz, placed_xyz, L) -> [3]` — PBC soft centroid of `cage ∪ placed` (SIGMA-weighted about the cage centroid).

- [ ] **Step 1: Write the failing tests**

```python
import torch, math
from liquid_coupling_flow.mw.mw_blob_geom import select, cage_frame, blob_sites, blob_anchor

L = 5.1953
torch.manual_seed(0)


def _cfg(n=64):
    return torch.remainder(torch.rand(n, 3) * L, L)


def test_select_is_reversible_under_block_change():
    # cage/block keyed to the CENTER + non-block coords -> re-deriving after moving the block reproduces indices
    x = _cfg()
    c = torch.tensor([2.0, 2.0, 2.0])
    sel = select(x, c[None], K=4, M=24, L=L)
    y = x.clone()
    y[sel["block_idx"]] += 0.3 * torch.randn(sel["block_idx"].numel(), 3)  # move ONLY the block
    sel2 = select(y, c[None], K=4, M=24, L=L)
    assert torch.equal(sel["block_idx"].sort().values, sel2["block_idx"].sort().values)
    assert torch.equal(sel["cage_idx"].sort().values, sel2["cage_idx"].sort().values)


def test_select_block_and_cage_disjoint():
    x = _cfg()
    sel = select(x, torch.tensor([[1.0, 1.0, 1.0]]), K=4, M=24, L=L)
    assert set(sel["block_idx"].tolist()).isdisjoint(set(sel["cage_idx"].tolist()))


def test_frame_orthonormal_right_handed_deterministic():
    x = _cfg(); c = torch.tensor([[2.5, 2.5, 2.5]])
    sel = select(x, c, K=4, M=24, L=L)
    cage = x[sel["cage_idx"]]
    c0 = torch.remainder(c[0] + _pbc_mean(cage, c[0], L), L)
    F1, s1 = cage_frame(cage, c0, L)
    F2, s2 = cage_frame(cage, c0, L)
    assert torch.allclose(F1, F2) and s1 == s2                      # deterministic
    assert torch.allclose(F1 @ F1.T, torch.eye(3), atol=1e-5)       # orthonormal
    assert torch.det(F1) > 0                                        # right-handed
    assert s1 > 0


def test_sites_are_cage_only_invariant_to_blob():
    x = _cfg(); c = torch.tensor([[2.5, 2.5, 2.5]])
    sel = select(x, c, K=4, M=24, L=L)
    cage = x[sel["cage_idx"]]
    s_a, o_a = blob_sites(cage, L, K=4)
    x2 = x.clone(); x2[sel["block_idx"]] += 0.5 * torch.randn(4, 3)   # move blob only; cage unchanged
    s_b, o_b = blob_sites(x2[sel["cage_idx"]], L, K=4)
    assert torch.allclose(s_a, s_b) and torch.equal(o_a, o_b)


def test_anchor_depends_only_on_cage_and_prefix():
    x = _cfg(); c = torch.tensor([[2.5, 2.5, 2.5]])
    sel = select(x, c, K=4, M=24, L=L)
    cage = x[sel["cage_idx"]]
    placed = x[sel["block_idx"][:2]]
    a1 = blob_anchor(cage, placed, L)
    a2 = blob_anchor(cage, placed, L)
    assert torch.allclose(a1, a2)
    a3 = blob_anchor(cage, x[sel["block_idx"][:1]], L)               # fewer placed -> different anchor
    assert not torch.allclose(a1, a3)


def _pbc_mean(pts, ref, L):
    d = pts - ref
    d = d - L * torch.round(d / L)
    return d.mean(0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_geom.py -v`
Expected: FAIL with `ModuleNotFoundError: liquid_coupling_flow.mw.mw_blob_geom`.

- [ ] **Step 3: Implement `mw_blob_geom.py`**

```python
"""Deterministic geometry primitives for the cage-anchored blob-resample proposal (spec 2026-07-15).

Every function here is a PURE function of the frozen cage + selection center + already-placed
prefix -- never of the blob positions being (re)generated. That invariant is what makes the MH
move exact (sites/order/anchor identical forward and reverse). See spec sections 2.1-2.3.
"""
from __future__ import annotations
import torch

SIGMA = 1.2  # anchor locator width (sigma units); soft-centroid weighting scale


def _wrap_pm(d, L):
    return d - L * torch.round(d / L)


def _pbc_centroid(pts, ref, L):
    """Min-image mean of pts about a stable reference ref (PBC-safe). pts [n,3], ref [3]."""
    return torch.remainder(ref + _wrap_pm(pts - ref[None], L).mean(0), L)


def select(x, centers, K, M, L):
    """Reversible selection: block = K nearest non-overlapping particles to each center;
    cage = M nearest NON-block particles to the center set. Pure fn of centers + non-block coords."""
    N = x.shape[0]
    centers = centers.to(x.dtype)
    # distance of every particle to the nearest center
    d = _wrap_pm(x[:, None, :] - centers[None, :, :], L).norm(dim=-1)  # [N, n_c]
    block = []
    for ci in range(centers.shape[0]):
        order = torch.argsort(d[:, ci])
        picked = [i for i in order.tolist() if i not in block][:K]
        block.extend(picked)
    block_idx = torch.tensor(sorted(block), dtype=torch.long, device=x.device)
    is_block = torch.zeros(N, dtype=torch.bool, device=x.device); is_block[block_idx] = True
    dmin = d.min(dim=1).values.clone()
    dmin[is_block] = float("inf")
    cage_idx = torch.argsort(dmin)[:M].sort().values
    return {"centers": centers, "block_idx": block_idx, "cage_idx": cage_idx}


def cage_frame(cage_xyz, c0, L):
    """Deterministic orthonormal right-handed frame from cage vectors (Gram-Schmidt), cage-only
    spacing. Robust where PCA degenerates. Ties broken by cage row order (a fixed label)."""
    rel = _wrap_pm(cage_xyz - c0[None], L)                       # [M,3] cage vectors
    r = rel.norm(dim=1)
    i1 = int(torch.argmin(torch.where(r > 1e-6, r, torch.full_like(r, 1e9))))
    e1 = rel[i1] / rel[i1].norm().clamp_min(1e-12)
    rej = rel - (rel @ e1)[:, None] * e1[None]                   # reject e1 component
    rn = rej.norm(dim=1); rn[i1] = -1.0
    i2 = int(torch.argmax(rn))
    e2 = rej[i2] / rej[i2].norm().clamp_min(1e-12)
    e3 = torch.linalg.cross(e1, e2)
    Frame = torch.stack([e1, e2, e3], 0)                        # rows = axes
    if torch.det(Frame) < 0:                                     # force right-handed
        Frame[2] = -Frame[2]
    nn = torch.cdist(cage_xyz, cage_xyz)
    nn.fill_diagonal_(float("inf"))
    spacing = float(nn.min(dim=1).values.median())
    return Frame, spacing


def _unit_pattern(K):
    """K fixed unit points (Morton-ordered), centered. Fibonacci sphere for K>1, origin for K=1."""
    if K == 1:
        return torch.zeros(1, 3)
    idx = torch.arange(K, dtype=torch.float64)
    phi = torch.acos(1 - 2 * (idx + 0.5) / K)
    ga = math.pi * (3 - 5 ** 0.5)
    theta = ga * idx
    pts = torch.stack([torch.sin(phi) * torch.cos(theta),
                       torch.sin(phi) * torch.sin(theta),
                       torch.cos(phi)], -1).to(torch.float32)
    return pts  # already index-ordered (deterministic)


import math  # noqa: E402  (used by _unit_pattern)


def blob_sites(cage_xyz, L, K):
    """K ordered sites c0 + Frame*(spacing*pattern). Cage-only -> exact order both directions."""
    c0 = _pbc_centroid(cage_xyz, cage_xyz[0], L)
    Frame, spacing = cage_frame(cage_xyz, c0, L)
    pat = _unit_pattern(K).to(cage_xyz.device)
    sites = torch.remainder(c0[None] + (pat * spacing) @ Frame, L)   # [K,3]
    order = torch.arange(K, device=cage_xyz.device)                  # pattern index = Morton order
    return sites, order


def blob_anchor(cage_xyz, placed_xyz, L):
    """PBC soft centroid of (cage union placed) -> unit-Jacobian anchor a_j. Depends only on the
    frozen cage + already-placed prefix. placed_xyz may be [0,3]."""
    pts = torch.cat([cage_xyz, placed_xyz], 0) if placed_xyz.numel() else cage_xyz
    ref = cage_xyz.mean(0) if cage_xyz.shape[0] else pts[0]
    d = _wrap_pm(pts - ref[None], L)
    w = torch.exp(-(d ** 2).sum(-1) / (2 * SIGMA ** 2))
    w = w / w.sum().clamp_min(1e-12)
    return torch.remainder(ref + (w[:, None] * d).sum(0), L)
```

(Note: move `import math` to the top of the file when implementing — it is shown inline here only to flag the dependency.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_geom.py -v`
Expected: 6 passed. If `test_frame_orthonormal...` fails on right-handedness, check the `det<0` sign flip; if `test_select_is_reversible...` fails, the block selection is leaking blob-position dependence — it must rank by distance to the CENTER only.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_blob_geom.py liquid_coupling_flow/tests/test_mw_blob_geom.py
git commit -m "feat(mw): reversible selection/cage + cage-vector frame + sites + centroid anchor"
```

---

### Task 2: Labeled Hungarian assignment (deterministic, tie-broken)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_blob_assign.py`
- Test: `liquid_coupling_flow/tests/test_mw_blob_assign.py`

**Interfaces:**
- Consumes: `blob_sites` (Task 1) for the sites to match against.
- Produces: `hungarian_assign(blob_xyz, sites, L) -> sigma [K] long` — `sigma[t]` = index into `blob_xyz` assigned to slot t. Total deterministic bijection; min-image squared-distance cost in float64; `scipy.optimize.linear_sum_assignment`; ties broken lexicographically by `(slot, blob-row)`.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from liquid_coupling_flow.mw.mw_blob_assign import hungarian_assign

L = 5.1953


def test_is_total_bijection():
    torch.manual_seed(1)
    blob = torch.rand(6, 3) * L
    sites = torch.rand(6, 3) * L
    sigma = hungarian_assign(blob, sites, L)
    assert sorted(sigma.tolist()) == list(range(6))     # permutation of all K


def test_deterministic_under_repeat_and_exact_ties():
    # symmetric configuration with exact cost ties must resolve identically every call
    blob = torch.tensor([[0.0, 0, 0], [1.0, 0, 0]])
    sites = torch.tensor([[0.5, 0, 0], [0.5, 0, 0]])    # both sites equidistant from both blobs
    a = hungarian_assign(blob, sites, L)
    b = hungarian_assign(blob, sites, L)
    assert torch.equal(a, b)


def test_min_image_cost_used():
    # a blob particle just across the periodic boundary from its site must still match it
    blob = torch.tensor([[0.01, 0, 0]])
    sites = torch.tensor([[L - 0.01, 0, 0]])            # min-image distance 0.02, raw distance ~L
    sigma = hungarian_assign(blob, sites, L)
    assert sigma.tolist() == [0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_assign.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `mw_blob_assign.py`**

```python
"""Deterministic labeled Hungarian assignment of blob particles to cage-derived sites (spec 2.3).

Total bijection (never nearest-site-with-rejection). Cost = min-image squared distance (float64).
Deterministic tie-break: add a tiny lexicographic ramp epsilon*(slot*K + blob_row) so exact and
float-noise cost ties resolve identically on every call, without changing any strict ordering."""
from __future__ import annotations
import torch
from scipy.optimize import linear_sum_assignment

_TIE_EPS = 1e-9


def hungarian_assign(blob_xyz, sites, L):
    K = blob_xyz.shape[0]
    d = blob_xyz[:, None, :] - sites[None, :, :]
    d = d - L * torch.round(d / L)
    cost = (d ** 2).sum(-1).to(torch.float64)                 # [K blob, K site]
    ramp = torch.arange(K, dtype=torch.float64)[:, None] * K + torch.arange(K, dtype=torch.float64)[None, :]
    cost = cost + _TIE_EPS * ramp                              # deterministic tie-break
    rows, cols = linear_sum_assignment(cost.cpu().numpy())     # rows sorted 0..K-1
    sigma = torch.empty(K, dtype=torch.long, device=blob_xyz.device)
    # slot t == site index t; sigma[t] = the blob row assigned to site t
    inv = {int(c): int(r) for r, c in zip(rows, cols)}
    for t in range(K):
        sigma[t] = inv[t]
    return sigma
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_assign.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_blob_assign.py liquid_coupling_flow/tests/test_mw_blob_assign.py
git commit -m "feat(mw): deterministic labeled Hungarian assignment (min-image cost, tie-broken)"
```

---

### Task 3: `MWBlobProposal` module — cage context encoder + β embedding + Cat3 ordinal-only head

**Files:**
- Create: `liquid_coupling_flow/mw/mw_blob.py`
- Test: `liquid_coupling_flow/tests/test_mw_blob.py`

**Interfaces:**
- Consumes: `ka3d_scaffold_ar.Cat3Head` (head; `log_prob(h,u)`, `sample(h,gen)->u`), Task-1 geometry.
- Produces:
  - `class MWBlobProposal(nn.Module)` with `__init__(d_model=128, num_bins=64, u_range=2.5, n_head=4, n_layer=2, knn_cage=24)`.
  - `_slot_context(cage_xyz, placed_xyz, anchor, slot_ord, K, beta, L) -> h [d_model]` — the per-slot conditioning vector: cage kNN-transformer context (relative to `anchor`), placed-member features (relative to `anchor`), ORDINAL slot features `[slot_ord/K, K/16]`, and a β embedding. **No absolute site coordinates.**
  - `_head_logprob(h, u) -> scalar` and `_head_sample(h, gen) -> (u [3], lp scalar)` wrapping `Cat3Head` so sample and score share the identical density (sample draws u, then lp = head.log_prob(h,u)).

- [ ] **Step 1: Write the failing tests**

```python
import torch
from liquid_coupling_flow.mw.mw_blob import MWBlobProposal

L = 5.1953
torch.manual_seed(0)


def _m():
    return MWBlobProposal(d_model=32, num_bins=32, n_head=2, n_layer=1, knn_cage=8).eval()


def test_slot_context_ignores_absolute_site_but_uses_cage():
    m = _m()
    cage = torch.rand(8, 3) * L
    placed = torch.rand(2, 3) * L
    anchor = cage.mean(0)
    h1 = m._slot_context(cage, placed, anchor, slot_ord=2, K=4, beta=10.0, L=L)
    # ordinal slot features only: same cage/anchor/beta but a different slot ordinal -> h changes
    h2 = m._slot_context(cage, placed, anchor, slot_ord=3, K=4, beta=10.0, L=L)
    assert not torch.allclose(h1, h2)
    # moving the cage changes context (cage IS used)
    h3 = m._slot_context(cage + 0.4, placed, anchor, slot_ord=2, K=4, beta=10.0, L=L)
    assert not torch.allclose(h1, h3)


def test_head_sample_equals_score():
    m = _m()
    h = torch.randn(m.d_model)
    u, lp = m._head_sample(h, gen=torch.Generator().manual_seed(3))
    assert torch.allclose(lp, m._head_logprob(h, u), atol=1e-5)


def test_beta_enters_context():
    m = _m()
    cage = torch.rand(8, 3) * L; placed = torch.zeros(0, 3); anchor = cage.mean(0)
    hlo = m._slot_context(cage, placed, anchor, 0, 4, beta=2.0, L=L)
    hhi = m._slot_context(cage, placed, anchor, 0, 4, beta=40.0, L=L)
    assert not torch.allclose(hlo, hhi)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob.py -v`
Expected: FAIL with `ModuleNotFoundError` / `AttributeError`.

- [ ] **Step 3: Implement the module core in `mw_blob.py`**

```python
"""Cage-anchored blob-resample proposal (spec 2026-07-15). Two-state transition density for MH.

Placement: x_j = centroid(cage union placed_{<j}) + offset_j, offset from a Cat3 categorical head.
Slot ORDER from cage-derived sites (mw_blob_geom); particle<->slot assignment by labeled Hungarian
(mw_blob_assign). The head sees ordinal-only slot features -- never absolute site coords -- so sites
set order, the centroid sets position (spec 2.5)."""
from __future__ import annotations
import math
import torch
import torch.nn as nn

from ka3d_scaffold_ar import Cat3Head
from liquid_coupling_flow.mw.mw_blob_geom import (select, blob_sites, blob_anchor, _wrap_pm, SIGMA)
from liquid_coupling_flow.mw.mw_blob_assign import hungarian_assign


class MWBlobProposal(nn.Module):
    def __init__(self, d_model=128, num_bins=64, u_range=2.5, n_head=4, n_layer=2, knn_cage=24):
        super().__init__()
        self.d_model = d_model
        self.u_range = float(u_range)
        self.knn_cage = int(knn_cage)
        # per-neighbour token: relative xyz(3) + distance(1); placed tokens flagged by a kind bit(1)
        self.tok = nn.Linear(5, d_model)
        self.slot_emb = nn.Sequential(nn.Linear(2, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.beta_emb = nn.Sequential(nn.Linear(1, d_model), nn.SiLU(), nn.Linear(d_model, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        layer = nn.TransformerEncoderLayer(d_model, n_head, 2 * d_model, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, n_layer)
        self.head = Cat3Head(d_model, num_bins=num_bins, u_range=u_range)

    def _tokens(self, cage_xyz, placed_xyz, anchor, L):
        cg = _wrap_pm(cage_xyz - anchor[None], L)
        cg = torch.cat([cg, cg.norm(dim=-1, keepdim=True), torch.zeros(cg.shape[0], 1, device=cg.device)], -1)
        if placed_xyz.numel():
            pl = _wrap_pm(placed_xyz - anchor[None], L)
            pl = torch.cat([pl, pl.norm(dim=-1, keepdim=True), torch.ones(pl.shape[0], 1, device=pl.device)], -1)
            toks = torch.cat([cg, pl], 0)
        else:
            toks = cg
        # keep nearest knn_cage tokens (by distance already in col 3)
        if toks.shape[0] > self.knn_cage:
            keep = torch.argsort(toks[:, 3])[: self.knn_cage]
            toks = toks[keep]
        return toks

    def _slot_context(self, cage_xyz, placed_xyz, anchor, slot_ord, K, beta, L):
        toks = self._tokens(cage_xyz, placed_xyz, anchor, L)          # [m,5]
        emb = self.tok(toks)[None]                                    # [1,m,d]
        seq = torch.cat([self.cls, emb], 1)                           # [1,m+1,d]
        h = self.enc(seq)[0, 0]                                       # [d]
        slot = self.slot_emb(torch.tensor([slot_ord / max(K, 1), K / 16.0], device=h.device))
        b = self.beta_emb(torch.tensor([beta / 10.0], device=h.device))
        return h + slot + b

    def _head_logprob(self, h, u):
        return self.head.log_prob(h, u)

    def _head_sample(self, h, gen=None):
        out = self.head.sample(h, gen=gen)
        u = out[0] if isinstance(out, tuple) else out                # Cat3Head.sample returns u (or (u,lp))
        return u, self.head.log_prob(h, u)
```

(When implementing, open `ka3d_scaffold_ar.py:297` to confirm `Cat3Head.sample`'s exact return; wrap so `_head_sample` always returns `(u, lp)` with `lp` recomputed via `log_prob` for guaranteed sample==score.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_blob.py liquid_coupling_flow/tests/test_mw_blob.py
git commit -m "feat(mw): MWBlobProposal context encoder + Cat3 ordinal-only head + beta embedding"
```

---

### Task 4: Two-state API — `sample_block` + `transition_log_prob` (labeled forward/reverse), K≥1

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_blob.py` (add methods to `MWBlobProposal`)
- Test: `liquid_coupling_flow/tests/test_mw_blob.py` (extend)

**Interfaces:**
- Consumes: Task-1 `blob_sites`/`blob_anchor`, Task-2 `hungarian_assign`, Task-3 `_slot_context`/`_head_*`.
- Produces (methods on `MWBlobProposal`):
  - `sample_block(x, selection, L, beta, gen) -> (y, logq_fwd)` — Hungarian at source x → σ_x; generate blob particles in slot order; `y` is `x` with `block_idx` replaced; `logq_fwd` the accumulated head log-density under σ_x.
  - `transition_log_prob(target_x, source_x, selection, L, beta) -> logq` — Hungarian at `source_x` → σ; score `target_x`'s block coords in that order/anchoring.
  - Both single-config `[N,3]`. `sample_block(x,...)`'s `logq_fwd == transition_log_prob(y, x, selection, ...)`.

- [ ] **Step 1: Write the failing exactness tests**

```python
import torch
from liquid_coupling_flow.mw.mw_blob import MWBlobProposal
from liquid_coupling_flow.mw.mw_blob_geom import select

L = 5.1953


def _setup(K):
    torch.manual_seed(4)
    m = MWBlobProposal(d_model=32, num_bins=32, n_head=2, n_layer=1, knn_cage=8).eval()
    x = torch.remainder(torch.rand(40, 3) * L, L)
    sel = select(x, torch.tensor([[2.5, 2.5, 2.5]]), K=K, M=12, L=L)
    return m, x, sel


def test_sample_equals_two_state_transition_at_y_from_x():
    for K in (1, 3, 5):
        m, x, sel = _setup(K)
        with torch.no_grad():
            y, lqf = m.sample_block(x, sel, L, beta=10.0, gen=torch.Generator().manual_seed(7))
            lq_check = m.transition_log_prob(y, x, sel, L, beta=10.0)   # target=y, source=x
        assert torch.allclose(lqf, lq_check, atol=1e-4), f"K={K}: {lqf} vs {lq_check}"


def test_transition_is_two_state_not_self_score():
    m, x, sel = _setup(4)
    with torch.no_grad():
        y, _ = m.sample_block(x, sel, L, beta=10.0, gen=torch.Generator().manual_seed(8))
        two_state = m.transition_log_prob(x, y, sel, L, beta=10.0)      # q_C(x|y): source=y
        self_score = m.transition_log_prob(x, x, sel, L, beta=10.0)     # q_C(x|x): source=x
    assert not torch.allclose(two_state, self_score, atol=1e-3)         # sigma_y != sigma_x in general


def test_prefix_and_cage_frozen_in_sample():
    m, x, sel = _setup(4)
    with torch.no_grad():
        y, _ = m.sample_block(x, sel, L, beta=10.0, gen=torch.Generator().manual_seed(9))
    mask = torch.ones(x.shape[0], dtype=torch.bool); mask[sel["block_idx"]] = False
    assert torch.equal(y[mask], x[mask])                                # only the block moved


def test_k1_degenerate_no_ordering():
    m, x, sel = _setup(1)
    with torch.no_grad():
        y, lqf = m.sample_block(x, sel, L, beta=10.0, gen=torch.Generator().manual_seed(1))
        assert torch.allclose(lqf, m.transition_log_prob(y, x, sel, L, beta=10.0), atol=1e-4)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob.py -v -k "sample or transition or prefix or k1"`
Expected: FAIL with `AttributeError: 'MWBlobProposal' object has no attribute 'sample_block'`.

- [ ] **Step 3: Implement the two-state methods**

```python
    # --- add to MWBlobProposal ---
    def _generate(self, cage_xyz, sigma, placed_seed, L, beta, K, gen):
        """Generate/collect K blob offsets in slot order. Returns (positions[K,3] in slot order, logq)."""
        placed = placed_seed.new_zeros(0, 3)
        pos = []
        logq = cage_xyz.new_zeros(())
        for t in range(K):
            a = blob_anchor(cage_xyz, placed, L)
            h = self._slot_context(cage_xyz, placed, a, slot_ord=t, K=K, beta=beta, L=L)
            u, lp = self._head_sample(h, gen=gen)
            xj = torch.remainder(a + u, L)
            pos.append(xj); placed = torch.cat([placed, xj[None]], 0); logq = logq + lp
        return torch.stack(pos, 0), logq

    def sample_block(self, x, selection, L, beta, gen=None):
        cage = x[selection["cage_idx"]]
        blk = selection["block_idx"]; K = blk.numel()
        sites, _ = blob_sites(cage, L, K)
        sigma = hungarian_assign(x[blk], sites, L) if K > 1 else torch.zeros(1, dtype=torch.long)
        labels = blk[sigma]                                   # slot t -> particle label (source=x)
        pos_slot, logq = self._generate(cage, sigma, x[blk], L, beta, K, gen)
        y = x.clone()
        y[labels] = pos_slot                                  # slot t position -> its label
        return y, logq

    def transition_log_prob(self, target_x, source_x, selection, L, beta):
        cage = source_x[selection["cage_idx"]]                # cage identical at x and y (frozen)
        blk = selection["block_idx"]; K = blk.numel()
        sites, _ = blob_sites(cage, L, K)
        sigma = hungarian_assign(source_x[blk], sites, L) if K > 1 else torch.zeros(1, dtype=torch.long)
        labels = blk[sigma]                                   # ordering induced at SOURCE
        placed = cage.new_zeros(0, 3); logq = cage.new_zeros(())
        for t in range(K):
            a = blob_anchor(cage, placed, L)
            h = self._slot_context(cage, placed, a, slot_ord=t, K=K, beta=beta, L=L)
            xj = target_x[labels[t]]                          # score TARGET's coord in this slot
            u = _wrap_pm(xj - a, L)
            logq = logq + self._head_logprob(h, u)
            placed = torch.cat([placed, xj[None]], 0)
        return logq
```

**Implementer note (exactness):** the cage used in `transition_log_prob` is indexed from `source_x`, but cage particles are non-block so `source_x[cage_idx] == target_x[cage_idx]` whenever the selection is reversible (Task 6 enforces the reverse-check). Both `sample_block` and `transition_log_prob` build `placed` from the SAME slot order and the SAME anchor recursion, so the head sees identical `(cage, placed, anchor, slot_ord, beta)` at every slot when target=y,source=x → densities match to head precision.

- [ ] **Step 4: Run tests to verify they pass**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob.py -v`
Expected: all pass (3 from Task 3 + 4 new). If `test_sample_equals_two_state...` fails, the offset scoring in `transition_log_prob` (`u = wrap_pm(xj - a)`) must mirror `_generate` exactly (same `blob_anchor` call sequence, same `placed` accumulation order).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_blob.py liquid_coupling_flow/tests/test_mw_blob.py
git commit -m "feat(mw): two-state sample_block/transition_log_prob (labeled fwd/rev, exact)"
```

---

### Task 5: Two-blob 2K-site mode (count redistribution)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_blob.py` (extend `select`-consumption to two centers), `liquid_coupling_flow/mw/mw_blob_geom.py` (two-center sites)
- Test: `liquid_coupling_flow/tests/test_mw_blob.py` (extend)

**Interfaces:**
- Consumes: Task-1 `select` already accepts `centers [n_c,3]` and returns 2K `block_idx` for two centers.
- Produces: `blob_sites_union(cage_xyz, centers, L, K) -> (sites [2K,3], order)` — K sites per center from that center's local frame, concatenated in a single Morton order. `sample_block`/`transition_log_prob` already accept the union `selection` unchanged (K derived from `block_idx.numel()`), but must call `blob_sites_union` when `selection["centers"].shape[0] == 2`.

- [ ] **Step 1: Write the failing tests**

```python
def test_two_blob_exact_and_count_can_redistribute():
    import torch
    from liquid_coupling_flow.mw.mw_blob import MWBlobProposal
    from liquid_coupling_flow.mw.mw_blob_geom import select
    L = 5.1953; torch.manual_seed(5)
    m = MWBlobProposal(d_model=32, num_bins=32, n_head=2, n_layer=1, knn_cage=12).eval()
    x = torch.remainder(torch.rand(60, 3) * L, L)
    centers = torch.tensor([[1.3, 1.3, 1.3], [3.9, 3.9, 3.9]])
    sel = select(x, centers, K=3, M=16, L=L)
    assert sel["block_idx"].numel() == 6                          # 2K union block
    with torch.no_grad():
        y, lqf = m.sample_block(x, sel, L, beta=8.0, gen=torch.Generator().manual_seed(2))
        assert torch.allclose(lqf, m.transition_log_prob(y, x, sel, L, beta=8.0), atol=1e-4)
    # count redistribution POSSIBLE: at least one generated particle can be nearer the other center
    dA = (torch.remainder(y[sel["block_idx"]] - centers[0], L) - L * torch.round((y[sel["block_idx"]] - centers[0]) / L)).norm(dim=-1)
    dB = (torch.remainder(y[sel["block_idx"]] - centers[1], L) - L * torch.round((y[sel["block_idx"]] - centers[1]) / L)).norm(dim=-1)
    assert dA.shape[0] == 6 and dB.shape[0] == 6                  # union scored; per-lobe count is emergent
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob.py -v -k two_blob`
Expected: FAIL (either `blob_sites_union` missing, or `sample_block` uses single-center `blob_sites` and mis-sizes sites vs 2K block).

- [ ] **Step 3: Implement two-center sites + dispatch**

Add to `mw_blob_geom.py`:

```python
def blob_sites_union(cage_xyz, centers, L, K):
    """2K ordered sites: K around each center from the SHARED cage frame, single Morton order."""
    c0 = _pbc_centroid(cage_xyz, cage_xyz[0], L)
    Frame, spacing = cage_frame(cage_xyz, c0, L)
    pat = _unit_pattern(K).to(cage_xyz.device)
    all_sites = []
    for ci in range(centers.shape[0]):
        ctr = torch.remainder(centers[ci], L)
        all_sites.append(torch.remainder(ctr[None] + (pat * spacing) @ Frame, L))
    sites = torch.cat(all_sites, 0)                              # [2K,3]
    return sites, torch.arange(sites.shape[0], device=cage_xyz.device)
```

In `mw_blob.py`, replace the `blob_sites(cage, L, K)` calls inside `sample_block` and `transition_log_prob` with a dispatch:

```python
        centers = selection["centers"]
        if centers.shape[0] == 1:
            sites, _ = blob_sites(cage, L, K // 1)
        else:
            from liquid_coupling_flow.mw.mw_blob_geom import blob_sites_union
            sites, _ = blob_sites_union(cage, centers, L, K // centers.shape[0])
```

(K here is `block_idx.numel()`; per-lobe pattern size = K / n_centers.)

- [ ] **Step 4: Run to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob.py -v`
Expected: all pass including two-blob.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_blob.py liquid_coupling_flow/mw/mw_blob_geom.py liquid_coupling_flow/tests/test_mw_blob.py
git commit -m "feat(mw): two-blob 2K-site union mode (emergent count redistribution)"
```

---

### Task 6: Kernel edit — two-state `single_site_move` + `two_blob_move` (full bridge ratio)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_kernels.py`
- Test: `liquid_coupling_flow/tests/test_mw_blob_kernel.py`

**Interfaces:**
- Consumes: `MWBlobProposal.sample_block/transition_log_prob` (Tasks 4–5), `mw_blob_geom.select`, `mw_kernels.draw_centers`, `ka3d_smc_bridge.geometric_bridge_log_accept`, `mw_energy`.
- Produces:
  - `blob_single_site_move(x, U, lq0, q0_model, blob_model, lam, beta, L, gen) -> (x, U, lq0, stats)` — K=1 heat-bath: draw a center, `select K=1`, propose+accept via the full bridge ratio.
  - `blob_two_blob_move(x, U, lq0, q0_model, blob_model, K, lam, beta, L, min_sep, gen, regions=None) -> (x, U, lq0, stats)` — union move using `select` two centers + `transition_log_prob` reverse. Reverse-check: re-`select` at the proposed state must reproduce `block_idx`+`cage_idx`, else reject.

- [ ] **Step 1: Write the failing stationarity test (mutation-verified)**

```python
import math, torch
import liquid_coupling_flow.mw.mw_kernels as mk
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import mc_run
from liquid_coupling_flow.mw.mw_blob import MWBlobProposal
from liquid_coupling_flow.mw.mw_kernels import blob_single_site_move

N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0)


def _equil(beta, B=8):
    r = mc_run(N, L, beta, n_equil=1500, n_collect=1500, every=50, seed=8, B=B)
    return r["cfgs"][-B:].clone(), r["U"]


def test_single_site_stationarity_and_mutation_sensitivity():
    torch.manual_seed(0)
    m = MWBlobProposal(d_model=32, num_bins=32, n_head=2, n_layer=1, knn_cage=12).eval()
    beta = 2.0
    x0, U_ref = _equil(beta)
    def run(bridge_fn):
        mk.geometric_bridge_log_accept = bridge_fn
        x = x0.clone(); U = mw_energy(x, L); acc = 0.0; us = []
        gen = torch.Generator().manual_seed(1)
        for i in range(200):
            for b in range(x.shape[0]):
                xb, Ub, _, st = blob_single_site_move(x[b][None], U[b][None], None, None, m,
                                                      lam=1.0, beta=beta, L=L, gen=gen)
                x[b] = xb[0]; U[b] = Ub[0]; acc += st["acc"]
            if i >= 100: us.append(U.clone())
        return torch.stack(us), acc
    mk._orig_bridge = mk.geometric_bridge_log_accept
    u_honest, acc = run(mk._orig_bridge)
    # mutation: swap reverse/forward -> must break stationarity
    def swapped(**kw):
        kw["log_r_reverse"], kw["log_r_forward"] = kw["log_r_forward"], kw["log_r_reverse"]
        return mk._orig_bridge(**kw)
    u_bad, _ = run(swapped)
    mk.geometric_bridge_log_accept = mk._orig_bridge
    def z(u):
        mb = u.mean(0); sem = math.sqrt(mb.var(unbiased=True) / u.shape[1] + (U_ref / N).var() / u.shape[1])
        return abs(float(mb.mean() / N - (U_ref / N).mean())) / max(float(sem), 1e-9)
    assert acc > 20, f"single-site never fired (acc={acc})"
    assert z(u_honest) < 4.0, f"honest chain drifted z={z(u_honest):.1f}"
    assert z(u_bad) > 6.0, f"swapped-density mutant NOT caught z={z(u_bad):.1f}"
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_kernel.py -v`
Expected: FAIL with `ImportError: cannot import name 'blob_single_site_move'`.

- [ ] **Step 3: Implement the two kernels in `mw_kernels.py`**

```python
# --- add near the top of mw_kernels.py ---
from liquid_coupling_flow.mw.mw_blob_geom import select as _blob_select


def _bridge_accept(lq0, lq0p, U, Up, lqr, lqf, lam, beta):
    return geometric_bridge_log_accept(
        log_q0_current=lq0 if lq0 is not None else torch.zeros_like(U),
        log_q0_proposed=lq0p, energy_current=U, energy_proposed=Up,
        log_r_reverse=lqr, log_r_forward=lqf, lam=lam, beta=beta)


def blob_single_site_move(x, U, lq0, q0_model, blob_model, lam, beta, L, gen):
    """K=1 full-cage heat-bath via the two-state blob proposal. x [1,N,3]."""
    beta_eff = lam * beta
    c = torch.rand(3, device=x.device, generator=gen) * L
    sel = _blob_select(x[0], c[None], K=1, M=blob_model.knn_cage, L=L)
    y0, lqf = blob_model.sample_block(x[0], sel, L, beta_eff, gen=gen)
    sel_rev = _blob_select(y0, c[None], K=1, M=blob_model.knn_cage, L=L)
    if not (torch.equal(sel_rev["block_idx"], sel["block_idx"]) and
            torch.equal(sel_rev["cage_idx"], sel["cage_idx"])):
        return x, U, lq0, {"acc": 0.0, "reject_reverse": 1.0}
    lqr = blob_model.transition_log_prob(x[0], y0, sel, L, beta_eff)
    Up = mw_energy(y0[None], L)
    lq0p = None if lam >= 1.0 else q0_model.log_prob(y0[None], L)
    la = _bridge_accept(lq0, lq0p, U, Up, lqr, lqf, lam, beta)
    acc = torch.rand(1, device=x.device, generator=gen).clamp_min(1e-38).log() < la
    if bool(acc):
        return y0[None], Up, (lq0p if lq0p is not None else lq0), {"acc": 1.0, "reject_reverse": 0.0}
    return x, U, lq0, {"acc": 0.0, "reject_reverse": 0.0}


def blob_two_blob_move(x, U, lq0, q0_model, blob_model, K, lam, beta, L, min_sep, gen, regions=None):
    """Two-blob union redistribution via the two-state blob proposal. x [1,N,3]."""
    beta_eff = lam * beta
    cA, cB = draw_centers(L, min_sep, gen, x.device, regions)
    centers = torch.stack([cA, cB], 0)
    sel = _blob_select(x[0], centers, K=K, M=blob_model.knn_cage, L=L)
    if sel["block_idx"].numel() != 2 * K:
        return x, U, lq0, {"acc": 0.0, "reject_reverse": 1.0}
    y0, lqf = blob_model.sample_block(x[0], sel, L, beta_eff, gen=gen)
    sel_rev = _blob_select(y0, centers, K=K, M=blob_model.knn_cage, L=L)
    if not (torch.equal(sel_rev["block_idx"], sel["block_idx"]) and
            torch.equal(sel_rev["cage_idx"], sel["cage_idx"])):
        return x, U, lq0, {"acc": 0.0, "reject_reverse": 1.0}
    lqr = blob_model.transition_log_prob(x[0], y0, sel, L, beta_eff)
    Up = mw_energy(y0[None], L)
    lq0p = None if lam >= 1.0 else q0_model.log_prob(y0[None], L)
    la = _bridge_accept(lq0, lq0p, U, Up, lqr, lqf, lam, beta)
    acc = torch.rand(1, device=x.device, generator=gen).clamp_min(1e-38).log() < la
    if bool(acc):
        return y0[None], Up, (lq0p if lq0p is not None else lq0), {"acc": 1.0, "reject_reverse": 0.0}
    return x, U, lq0, {"acc": 0.0, "reject_reverse": 0.0}
```

- [ ] **Step 4: Run to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_kernel.py -v`
Expected: PASS. If the honest chain drifts (z>4) STOP and debug the ratio — a stationarity failure is a real bug (never-refute-bug-hypothesis); if the swapped mutant is NOT caught (z_bad<6), the test is not gating the acceptance formula.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/mw/mw_kernels.py liquid_coupling_flow/tests/test_mw_blob_kernel.py
git commit -m "feat(mw): two-state blob single-site + two-blob kernels (full bridge ratio, mutation-verified)"
```

---

### Task 7: Training — `mw_blob_train.py` (MLE + free-run energy REINFORCE), `load_model`

**Files:**
- Create: `liquid_coupling_flow/mw/mw_blob_train.py`
- Test: `liquid_coupling_flow/tests/test_mw_blob_train.py`

**Interfaces:**
- Consumes: `MWBlobProposal`, `mw_blob_geom.select`, `mw_energy`, D0 per-T banks `liquid_coupling_flow/mw/artifacts/d0_probe_T{tag}_N64.pt` (keys `cfgs [n,64,3]`, `tstar`, `L`) + `mw_bank_ambient_N64.pt`.
- Produces:
  - `carve_pool(banks, K_choices, M, L, n, gen) -> list[dict]` — sampled `(x, selection, beta)` training tuples across temperatures.
  - `mle_rl_step(model, batch, L, lam_rl, gen) -> dict(loss, nll, rl)` — one optimization step; `nll` = −Σ `transition_log_prob(x, x, sel)` self-score on DATA blobs (data is its own canonical target: source=target=x); `rl` = `mean(adv.detach() * logq_freerun)`, `adv = standardize(−clamp(insertion energy of the free-run blob, 0, cap))`.
  - `load_model(path, device) -> (MWBlobProposal, ckpt)`.
  - `train(...)` CLI writing `liquid_coupling_flow/mw/artifacts/mw_blob_runs/mw_blob.pt` (+ `_last.pt`), checkpointing every val interval.

- [ ] **Step 1: Write the failing tests**

```python
import torch
from liquid_coupling_flow.mw.mw_blob_train import carve_pool, mle_rl_step, load_model
from liquid_coupling_flow.mw.mw_blob import MWBlobProposal

L = 5.1953


def _fake_bank(n=32, N=64, tstar=0.09632):
    return {"cfgs": torch.remainder(torch.rand(n, N, 3) * L, L), "tstar": tstar, "L": L}


def test_carve_pool_shapes_and_beta():
    pool = carve_pool([_fake_bank(), _fake_bank(tstar=0.065)], K_choices=[1, 4], M=16, L=L,
                      n=8, gen=torch.Generator().manual_seed(0))
    assert len(pool) == 8
    for t in pool:
        assert t["selection"]["block_idx"].numel() in (1, 4)
        assert t["beta"] > 0


def test_step_reduces_loss_on_overfit_batch():
    torch.manual_seed(0)
    m = MWBlobProposal(d_model=32, num_bins=32, n_head=2, n_layer=1, knn_cage=8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    pool = carve_pool([_fake_bank()], K_choices=[4], M=12, L=L, n=4, gen=torch.Generator().manual_seed(1))
    first = None
    for _ in range(40):
        opt.zero_grad()
        out = mle_rl_step(m, pool, L, lam_rl=0.0, gen=torch.Generator().manual_seed(2))
        out["loss"].backward(); opt.step()
        first = first if first is not None else float(out["loss"])
    assert float(out["loss"]) < first        # MLE overfits the tiny pool


def test_load_model_roundtrip(tmp_path):
    import os
    m = MWBlobProposal(d_model=32, num_bins=32, n_head=2, n_layer=1, knn_cage=8)
    p = tmp_path / "m.pt"
    torch.save({"state_dict": m.state_dict(), "d_model": 32, "num_bins": 32,
                "n_head": 2, "n_layer": 1, "knn_cage": 8}, p)
    m2, ck = load_model(str(p), "cpu")
    assert ck["num_bins"] == 32
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.allclose(a, b)
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_train.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 3: Implement `mw_blob_train.py`**

```python
"""Train MWBlobProposal: MLE (data blobs are their own canonical target, source=target=x) + free-run
energy REINFORCE (handoff lever). beta sampled per tuple across the training range; SO(3) rotation
aug. See spec 2026-07-15 section 4."""
from __future__ import annotations
import os, math, torch
from liquid_coupling_flow.mw.mw_blob import MWBlobProposal
from liquid_coupling_flow.mw.mw_blob_geom import select
from liquid_coupling_flow.mw.mw_energy import mw_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts", "mw_blob_runs")


def _rand_rot(gen):
    q = torch.randn(4, generator=gen); q = q / q.norm()
    w, xq, yq, zq = q
    return torch.tensor([[1 - 2 * (yq * yq + zq * zq), 2 * (xq * yq - zq * w), 2 * (xq * zq + yq * w)],
                         [2 * (xq * yq + zq * w), 1 - 2 * (xq * xq + zq * zq), 2 * (yq * zq - xq * w)],
                         [2 * (xq * zq - yq * w), 2 * (yq * zq + xq * w), 1 - 2 * (xq * xq + yq * yq)]])


def carve_pool(banks, K_choices, M, L, n, gen):
    pool = []
    for _ in range(n):
        bank = banks[int(torch.randint(len(banks), (1,), generator=gen))]
        cfgs = bank["cfgs"]; x = torch.remainder(cfgs[int(torch.randint(cfgs.shape[0], (1,), generator=gen))], L)
        K = K_choices[int(torch.randint(len(K_choices), (1,), generator=gen))]
        c = torch.rand(3, generator=gen) * L
        sel = select(x, c[None], K=K, M=M, L=L)
        pool.append({"x": x, "selection": sel, "beta": 1.0 / bank["tstar"]})
    return pool


def _insertion_energy(x, block_idx, L):
    """Local energy attributable to the block: full - energy with block removed is expensive; use the
    total mw_energy delta of the block vs a reference is not needed for the reward -- reward on the
    CAPPED total energy of the proposed config (repulsive-dominated). Simplest exact proxy: mw_energy."""
    return mw_energy(x[None], L)[0]


def mle_rl_step(model, pool, L, lam_rl, gen, cap=50.0):
    nll = torch.zeros(())
    rl = torch.zeros(())
    for t in pool:
        R = _rand_rot(gen)
        x = torch.remainder((t["x"] - L / 2) @ R.T + L / 2, L)     # SO(3) aug about box center
        sel = t["selection"]; beta = t["beta"]
        nll = nll - model.transition_log_prob(x, x, sel, L, beta)  # data blob: source=target=x
        if lam_rl > 0:
            y, lqf = model.sample_block(x, sel, L, beta, gen=gen)
            with torch.no_grad():
                u = torch.clamp(_insertion_energy(y, sel["block_idx"], L), 0.0, cap)
                adv = -(u - u.mean())
            rl = rl + adv * lqf
    n = max(len(pool), 1)
    loss = nll / n + lam_rl * (rl / n)
    return {"loss": loss, "nll": nll / n, "rl": rl / n}


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    m = MWBlobProposal(d_model=ck["d_model"], num_bins=ck["num_bins"], n_head=ck["n_head"],
                       n_layer=ck["n_layer"], knn_cage=ck["knn_cage"]).to(device)
    m.load_state_dict(ck["state_dict"]); return m, ck


def train(banks_glob="liquid_coupling_flow/mw/artifacts/d0_probe_T*_N64.pt", steps=20000,
          batch=8, lr=2e-4, lam_rl=0.1, rl_warmup=2000, d_model=128, num_bins=64, n_head=4,
          n_layer=2, knn_cage=24, K_choices=(1, 2, 4, 8, 16), M=24, out="mw_blob.pt",
          seed=0, device=None, val_every=500):
    import glob
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(ART, exist_ok=True)
    banks = [torch.load(p, map_location="cpu", weights_only=False) for p in sorted(glob.glob(banks_glob))]
    banks = [b for b in banks if "cfgs" in b and b["cfgs"].numel()]
    L = float(banks[0]["L"])
    model = MWBlobProposal(d_model, num_bins, 2.5, n_head, n_layer, knn_cage).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    gen = torch.Generator().manual_seed(seed)
    best = float("inf"); out_p = os.path.join(ART, out)
    meta = dict(d_model=d_model, num_bins=num_bins, n_head=n_head, n_layer=n_layer, knn_cage=knn_cage, L=L)
    for step in range(1, steps + 1):
        pool = carve_pool(banks, list(K_choices), M, L, batch, gen)
        pool = [{"x": t["x"].to(device), "selection": {k: (v.to(device) if torch.is_tensor(v) else v)
                 for k, v in t["selection"].items()}, "beta": t["beta"]} for t in pool]
        lam = lam_rl if step > rl_warmup else 0.0
        opt.zero_grad(); out = mle_rl_step(model, pool, L, lam, gen); out["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
        if step % val_every == 0:
            v = float(out["nll"])
            torch.save({"state_dict": model.state_dict(), "step": step, "nll": v, **meta},
                       out_p[:-3] + "_last.pt")
            if v < best:
                best = v; torch.save({"state_dict": model.state_dict(), "step": step, "nll": v, **meta}, out_p)
            print(f"step {step}: nll {v:.4f} rl {float(out['rl']):.3f} best {best:.4f}", flush=True)


if __name__ == "__main__":
    train()
```

**Implementer note:** the reward `_insertion_energy` uses the full `mw_energy` of the proposed config (repulsive-dominated capped); if training pushes over-packing, switch to a block-local insertion energy (per handoff §5). Keep the MLE anchor always on so the density cannot collapse.

- [ ] **Step 4: Run to verify it passes**

Run: `CUDA_VISIBLE_DEVICES="" python -m pytest liquid_coupling_flow/tests/test_mw_blob_train.py -v`
Expected: 3 passed. `test_step_reduces_loss` proves the MLE path learns.

- [ ] **Step 5: Commit (code only — the real run is controller-launched)**

```bash
git add liquid_coupling_flow/mw/mw_blob_train.py liquid_coupling_flow/tests/test_mw_blob_train.py
git commit -m "feat(mw): MWBlobProposal training (MLE + free-run energy REINFORCE) + load_model"
```

- [ ] **Step 6: Hand the real training run to the controller**

Report to the controller: the training CLI is `python -m liquid_coupling_flow.mw.mw_blob_train` (or `train(...)`), needs the D0 per-T banks present (ambient + supercooled rungs), ~hours on GPU, saves per-`val_every` to `mw_blob_runs/mw_blob.pt`. The controller launches it in the background (subagent-launched long runs die at teardown) after Task 1b's banks exist.

---

### Task 8: Gate battery — G-K1, G-transfer, G-Kgt1, G-energy, G-mismatch-rate, G-pca-gap

**Files:**
- Create: `reports/logs-2026-07-15/gate_blob_proposal.py`
- Output: `reports/logs-2026-07-15/gate_blob_*.png` + `reports/2026-07-15-mw-blob-proposal-gates.md`

**Interfaces:**
- Consumes: `mw_blob_train.load_model`, `mw_blob_geom.select`, `mw_energy`, the trained `mw_blob.pt`, banks at N=64 and N=512.
- Produces: verdict per spec §5.2 — every gate reported as a full distribution, not a scalar (evaluate-distributions-not-scalars directive).

- [ ] **Step 1: Write the gate script** (runs after training; measurement, not pytest)

```python
"""Post-training gate battery for MWBlobProposal (spec 2026-07-15 section 5.2). Distributions, not scalars.
Baselines to beat: scaffold +55 @K~4, +86 @K~7, +14 @K=1 (measured 2026-07-15)."""
import sys, torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR
from liquid_coupling_flow.mw.mw_blob_train import load_model
from liquid_coupling_flow.mw.mw_blob_geom import select

CK = sys.argv[1]; TSTAR = float(sys.argv[2]); DEV = "cuda" if torch.cuda.is_available() else "cpu"
model, meta = load_model(CK, DEV); model.eval(); beta = 1.0 / TSTAR


def bdu(bankfile, N, K):
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    X = torch.remainder(torch.load(bankfile, map_location=DEV, weights_only=False)["cfgs"][:60].to(DEV).float(), L)
    g = torch.Generator(device=DEV).manual_seed(0); out = []
    with torch.no_grad():
        for c in range(60):
            x = X[c]; ctr = torch.rand(3, device=DEV, generator=g) * L
            sel = select(x, ctr[None], K=K, M=meta["knn_cage"], L=L)
            y, _ = model.sample_block(x, sel, L, beta, gen=g)
            out.append(float((mw_energy(y[None], L) - mw_energy(x[None], L))[0]) * beta)
    return sorted(out)


N64 = "liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt"
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
rows = []
for K in (1, 2, 4, 8):
    d = bdu(N64, 64, K); med = d[len(d) // 2]; frac = sum(1 for v in d if v < 3) / len(d)
    rows.append((K, med, frac)); ax[0].hist(d, bins=40, histtype="step", label=f"K={K} (med {med:+.1f})")
    print(f"G-Kgt1 K={K}: median beta*dU {med:+.1f}  frac<3 {frac:.2f}")
ax[0].set(xlabel="beta*dU", title=f"blob proposal N64 T*={TSTAR}"); ax[0].legend()
# G-transfer: SAME model (trained on N64 beta-range banks) tested at N=64 vs N=512.
# Prereq: an N=512 equilibrium bank at T*_work -- reuse an existing N512 scaffold bank
# (liquid_coupling_flow/mw/artifacts/mw_scaffold_runs/ banks carry N512 cfgs) or generate one
# via mw_reference.mc_run(512, L512, beta, ...); pass its path as argv[3].
d64 = bdu(N64, 64, 4)
N512_BANK = sys.argv[3]                      # N=512 equilibrium bank at T*_work
d512 = bdu(N512_BANK, 512, 4)
ax[1].hist(d64, bins=40, histtype="step", label="N64"); ax[1].hist(d512, bins=40, histtype="step", label="N512")
ax[1].set(xlabel="beta*dU", title="G-transfer"); ax[1].legend()
fig.tight_layout(); p = "reports/logs-2026-07-15/gate_blob_kdu.png"; fig.savefig(p, dpi=140)
print(f"PLOT: /mnt/ssd/GridTransformer/{p}")
```

- [ ] **Step 2: Controller runs the gate script after training**

Run (controller, once `mw_blob.pt` exists): `python reports/logs-2026-07-15/gate_blob_proposal.py liquid_coupling_flow/mw/artifacts/mw_blob_runs/mw_blob.pt <TSTAR_WORK> > reports/logs-2026-07-15/gate_blob.out 2>&1`
Expected: per-K β·ΔU table + plot path.

- [ ] **Step 3: Add the exactness/mismatch/PCA gate probes**

Extend the script with: G-mismatch-rate (fraction of moves where σ_x≠σ_y, distribution over K), and G-pca-gap (cage-frame `e2`-rejection magnitude vs per-move β·ΔU — scatter). Print distributions. G-exact / G-cage-reversible / G-hungarian-determinism / G-involution / G-stationarity are already covered by the pytest suites (Tasks 1–6); reference them in the verdict rather than re-running.

- [ ] **Step 4: Write the verdict**

`reports/2026-07-15-mw-blob-proposal-gates.md`: per-gate PASS/FAIL vs §5.2 thresholds (G-K1 median β·ΔU ≤ ~1–2 & acc ≥ 20%; G-Kgt1 large improvement vs +55/+86; G-transfer flat across N; G-energy RL vs MLE; G-mismatch-rate distribution; G-pca-gap). Link every plot by full path. If G-K1 or G-Kgt1 fails, the design's §6 fallback (B canonical-lift ordering) or a longer/retuned train is the next step — record which.

- [ ] **Step 5: Commit**

```bash
git add reports/logs-2026-07-15/gate_blob_proposal.py reports/logs-2026-07-15/gate_blob*.png reports/logs-2026-07-15/gate_blob.out reports/2026-07-15-mw-blob-proposal-gates.md
git commit -m "measure(mw): MWBlobProposal gate battery + verdict vs scaffold baselines"
```

---

## Execution notes

- **Task order:** 1→2→3→4→5→6 are the code stream (each depends on the prior); 7 (training) needs 1–5; 8 (gates) needs 7 + a trained checkpoint. Tasks 1–6 are unit-testable immediately (tiny models, CPU) and do not wait on any GPU run.
- **Controller-launched:** Task 7's real training run and Task 8's gate run (subagent-launched long/background jobs die at teardown). Subagents write + smoke the scripts; the controller launches.
- **Dependency on the existing campaign:** Task 7 consumes the D0 per-T banks (Task 1b of the kernel-portfolio plan produces the supercooled bank). Tasks 1–6 have no such dependency — start them now.
- **Exactness first:** every exactness/stationarity test failure is a real bug (never-refute-bug-hypothesis) — do not loosen tolerances. The two-state `sample==transition(y←x)` identity (Task 4) and the mutation-verified stationarity (Task 6) are the load-bearing gates.
- **Fallback:** if G-pca-gap or G-mismatch-rate shows instability (spec §6), design-B (canonical-lift ordering) is the documented control — a follow-up plan, not an inline scramble.
