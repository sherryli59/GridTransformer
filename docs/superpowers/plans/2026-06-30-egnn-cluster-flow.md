# EGNN Cluster Flow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `EGNNClusterFlow` — a conditional EGNN continuous normalizing flow that jointly places the k=7 KA cluster particles (all-see-all, no coupling blind spot) — to pass the g(r) gate proposal-A/B failed.

**Architecture:** Frame-free conditional CNF. The k cluster particles move under the periodic traceable-EGNN central-force velocity; the cage (nearest n_cage particles) is fixed context. Full-Gaussian base at the cage centroid (position pinned, no zero-COM). Exact cluster-restricted per-particle divergence. Trained by species-aware per-particle OT flow-matching; sampled by fixed-step RK4.

**Tech Stack:** PyTorch, scipy (Hungarian OT). Reuse `egnn_traceable.EGNN_dynamics`, `ka_cluster.py` (slot_order/cluster_slots), `ka_cluster_flow.gate_measure`.

## Global Constraints

- Exactness: the CNF log_q must be exact for the learned velocity. **The gate decides GO/NO-GO, never the training loss.** Standing directive: never conclude "no bug"; concrete checks only.
- **NOT translation/rotation-invariant for the cluster alone:** NO zero-COM subspace, base is a full 2k-dim Gaussian at the cage centroid, log-det over all 2k dims, cage is the fixed reference (never center on the moving cluster COM). The periodic EGNN branch (L set) already omits COM centering.
- Species are PER-BATCH `[B,N]` (slot-order permutes per config). Cluster/cage species from `s[:,idx]`.
- EGNN import: `from liquid_coupling_flow.ipl44.learndiffeq.learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics`.
- `EGNN_dynamics(n_particles=P, n_dimension=2, cutoff=r_c, max_neighbors, L=<box>, n_species=2, hidden_nf, n_layers)`. Periodic branch (L set) velocity = `(diffij*pot).sum(dim=2)` (pure central force). `forward_and_divergence` returns `(vel[B,P,2], divergence[B])` with internal `divergence_term[B,P]` per-particle (total = `-divergence_term.sum(-1)`).
- Fixed cloud size P = k + n_cage (EGNN requires fixed n_particles). n_cage=48, r_c=3.0, k=7 → P=55.
- Commit only the specific files each task names; **never `git add -A`** (repo is dirty).
- ALWAYS print the full clickable path to any plot (CLAUDE.md).
- Device default `"cuda" if torch.cuda.is_available() else "cpu"`. `ART = liquid_coupling_flow/artifacts`. Reference `ka_reference_N100.pt` (keys `x[n,N,2]`, `s[N]`), N=100, ρ=1.2.

---

### Task 1: Cloud construction + cage-centroid base

**Files:**
- Create: `liquid_coupling_flow/ka_cluster_egnn.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_egnn.py`

**Interfaces:**
- Consumes: `ka_cluster` (`cluster_slots`), `ka_cluster_flow` (`slot_order`, `_scaffold`), `ka_gridformer._wrap_pm`.
- Produces:
  - `build_cloud(pos[B,N,2], s[B,N], cluster_idx[k], sc, L, n_cage) -> (cloud[B,P,2], cloud_sp[B,P], c[B,2])` — cloud = cluster (first k) then the nearest n_cage non-cluster particles (by min-image dist to the cluster centroid); `c` = cage centroid.
  - `cage_centroid(cage[B,n_cage,2], L) -> [B,2]` (min-image mean anchored on cage[:,0]).
  - `base_logp(x0[B,k,2], c[B,2], sigma_b) -> [B]`; `sample_base(c[B,2], k, sigma_b, gen=None) -> [B,k,2]`.
  - `compute_sigma_b(data, s, geo, sc, L, k, n_cage) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ka_cluster_egnn.py
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
    # (cheap proxy: the nearest cage particle is closer to c than the farthest, i.e. sorted by distance)
    from liquid_coupling_flow.ka_gridformer import _wrap_pm
    dc = _wrap_pm(cloud[:, 7:] - c[:, None], L).norm(dim=-1)
    assert (dc[:, 1:] >= dc[:, :-1] - 1e-4).all()          # cage sorted nearest-first

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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k "cloud or base_logp or sigma" -v`
Expected: FAIL (module not found).

- [ ] **Step 3: Write minimal implementation**

```python
# ka_cluster_egnn.py
"""EGNN cluster flow: conditional continuous normalizing flow placing the k cluster particles under the periodic
traceable-EGNN central-force velocity (all-see-all, no coupling blind spot), cage fixed context, frame-free.
See docs/superpowers/specs/2026-06-30-egnn-cluster-flow-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold
from liquid_coupling_flow.ka_gridformer import _wrap_pm

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def cage_centroid(cage, L):                                  # cage [B,m,2] -> [B,2] min-image mean on cage[:,0]
    a = cage[:, :1]
    return torch.remainder(a[:, 0] + _wrap_pm(cage - a, L).mean(1), L)


def build_cloud(pos, s, cluster_idx, sc, L, n_cage):
    """cloud = [cluster (k); nearest n_cage non-cluster particles]. Returns (cloud[B,P,2], sp[B,P], c[B,2])."""
    B, N, _ = pos.shape; dev = pos.device; k = cluster_idx.shape[0]
    clu = pos[:, cluster_idx]                                                     # [B,k,2]
    ccen = torch.remainder(clu[:, :1, 0].squeeze(1)[:, None]  # placeholder, replaced below
                           , L) if False else None
    # cluster centroid (min-image on the seed)
    seed = clu[:, :1]                                                             # [B,1,2]
    ccen = torch.remainder(seed[:, 0] + _wrap_pm(clu - seed, L).mean(1), L)       # [B,2]
    d = _wrap_pm(pos - ccen[:, None], L).norm(dim=-1)                             # [B,N] dist to cluster centroid
    d[:, cluster_idx] = float("inf")                                             # exclude the cluster
    cage_idx = d.topk(n_cage, largest=False).indices                             # [B,n_cage]
    cage = torch.gather(pos, 1, cage_idx[..., None].expand(-1, -1, 2))            # [B,n_cage,2]
    cage_sp = torch.gather(s, 1, cage_idx)                                        # [B,n_cage]
    cloud = torch.cat([clu, cage], 1)                                            # [B,P,2]
    sp = torch.cat([s[:, cluster_idx], cage_sp], 1)                              # [B,P]
    return cloud, sp, cage_centroid(cage, L)


def base_logp(x0, c, sigma_b):                              # x0 [B,k,2], c [B,2] -> [B]
    d2 = ((x0 - c[:, None]) ** 2).sum(-1)
    return (-d2 / (2 * sigma_b ** 2) - math.log(2 * math.pi * sigma_b ** 2)).sum(-1)


def sample_base(c, k, sigma_b, gen=None):                  # -> [B,k,2] full 2k-dim Gaussian at c (no zero-COM)
    return c[:, None] + sigma_b * torch.randn(c.shape[0], k, 2, generator=gen, device=c.device, dtype=c.dtype)


@torch.no_grad()
def compute_sigma_b(data, s, geo, sc, L, k, n_cage):
    """Isotropic std of the true cluster particles' min-image displacement from the cluster centroid."""
    N = data.shape[1]; pos, s_ord = slot_order(data, s, geo, N); disp = []
    for seed in range(0, N, 3):
        cl = KC.cluster_slots(seed, sc, k, L); clu = pos[:, cl]
        ccen = torch.remainder(clu[:, :1, 0][:, None] * 0 + clu[:, :1] + 0, L)  # dummy to keep shape clarity
        sd = clu[:, :1]
        ccen = torch.remainder(sd[:, 0] + _wrap_pm(clu - sd, L).mean(1), L)
        disp.append(_wrap_pm(clu - ccen[:, None], L).reshape(-1, 2))
    return float(torch.cat(disp, 0).std().item())
```

*(Note: remove the two dead `ccen = ... if False`/dummy placeholder lines during implementation — they are shown only to mark where the min-image centroid is computed; the real centroid is the `torch.remainder(seed + wrap_pm(...).mean(1), L)` line.)*

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k "cloud or base_logp or sigma" -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_cluster_egnn.py liquid_coupling_flow/tests/test_ka_cluster_egnn.py
git commit -m "feat(egnn-flow): cloud construction + cage-centroid Gaussian base + sigma_b"
```

---

### Task 2: Conditional velocity + cluster-restricted per-particle divergence

**Files:**
- Modify: `liquid_coupling_flow/ipl44/learndiffeq/learndiffeq/particles/velocities/egnn_traceable.py` (add ONE additive method)
- Modify: `liquid_coupling_flow/ka_cluster_egnn.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_egnn.py`

**Interfaces:**
- Consumes: `EGNN_dynamics`, Task 1.
- Produces:
  - `EGNN_dynamics.forward_and_perparticle_divergence(xs, t, a=None, differentiable=True) -> (vel[B,P,2], divpp[B,P])` where `divpp[:,i] = ∂v_i/∂x_i` (trace, per particle); `divpp.sum(-1)` equals the existing `forward_and_divergence`'s scalar divergence.
  - `ConditionalEGNN(nn.Module)` wrapping an `EGNN_dynamics`; `vel_div(cloud[B,P,2], t, sp[B,P], k) -> (v_cluster[B,k,2], div_cluster[B])`, `div_cluster = divpp[:, :k].sum(-1)`.

- [ ] **Step 1: Write the failing test (divergence vs brute-force — the definitive exactness check)**

```python
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
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k divergence -v`
Expected: FAIL (`ConditionalEGNN` not defined).

- [ ] **Step 3a: Add the additive method to egnn_traceable**

Insert this method into `EGNN_dynamics` (right after `forward_and_divergence`, ~line 259). It duplicates that method's body but returns the per-particle term instead of summing:

```python
    def forward_and_perparticle_divergence(self, xs, t, a=None, differentiable=True):
        """Same as forward_and_divergence but returns per-particle divergence [B,P] (sum(-1) == the scalar div).
        Periodic (L set) branch only — the KA cluster flow uses L."""
        common = self._compute_common_terms(xs, t, a)
        diffij = common['diffij']; h_final = common['h_final']; B = common['B']; P = common['P']
        n_neighbors = diffij.shape[2]
        t_pot = common['t_neighbors'].reshape(-1, 1); h_flat = h_final.reshape(-1, h_final.shape[-1])
        sp = self._sp_feat(common, n_neighbors)
        with torch.enable_grad():
            rij = rearrange(diffij.norm(dim=-1), 'b p m -> (b p m) 1').requires_grad_(True)
            parts = (rij, t_pot, h_flat) if sp is None else (rij, t_pot, h_flat, sp)
            pot = self.pot_model(torch.cat(parts, dim=-1)).reshape(B, P, n_neighbors, 1)
            vel = (diffij * pot).sum(dim=2)
            dpotdr = torch.autograd.grad(pot.sum(), rij, create_graph=differentiable, retain_graph=differentiable)[0]
            dpotdr = dpotdr.reshape(B, P, n_neighbors, 1); rij_r = rij.reshape(B, P, n_neighbors, 1)
            divergence_term = (pot + (diffij * dpotdr * diffij / (rij_r + 1e-8))).sum((-1, -2))   # [B,P]
        return vel, -divergence_term                                                             # [B,P,2],[B,P]
```

- [ ] **Step 3b: Add `ConditionalEGNN` to ka_cluster_egnn.py**

```python
# append to ka_cluster_egnn.py
from liquid_coupling_flow.ipl44.learndiffeq.learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics


class ConditionalEGNN(nn.Module):
    """Wraps a periodic EGNN_dynamics over the fixed cloud (k cluster + n_cage cage). vel_div returns ONLY the
    cluster velocities and the cluster-restricted divergence (cage is fixed context -> not in the log-det)."""
    def __init__(self, n_cage=48, k=7, r_c=3.0, L=None, hidden_nf=64, n_layers=4, n_species=2):
        super().__init__()
        P = k + n_cage
        self.egnn = EGNN_dynamics(n_particles=P, n_dimension=2, cutoff=r_c, max_neighbors=P - 1,
                                  L=L, n_species=n_species, hidden_nf=hidden_nf, n_layers=n_layers)

    def vel_div(self, cloud, t, sp, k):
        vel, divpp = self.egnn.forward_and_perparticle_divergence(cloud, t, sp)   # [B,P,2],[B,P]
        return vel[:, :k], divpp[:, :k].sum(-1)                                    # cluster velocities + div
```

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k divergence -v`
Expected: PASS (analytic == brute-force to 1e-4).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ipl44/learndiffeq/learndiffeq/particles/velocities/egnn_traceable.py liquid_coupling_flow/ka_cluster_egnn.py liquid_coupling_flow/tests/test_ka_cluster_egnn.py
git commit -m "feat(egnn-flow): per-particle divergence method + ConditionalEGNN (cluster-restricted, exact)"
```

---

### Task 3: RK4 conditional integrator (sample/log_q) + sampler==scorer

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_egnn.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_egnn.py`

**Interfaces:**
- Consumes: Task 1 (base, build_cloud), Task 2 (`ConditionalEGNN`).
- Produces: `EGNNClusterFlow(nn.Module)` with attrs `sigma_b, n_cage, k, n_steps, ce`; and
  - `sample(pos,s,cluster_idx,sc,L) -> (xC_lab[B,k,2], logq[B])` (@no_grad)
  - `log_q(pos,s,cluster_idx,xC_query,sc,L) -> logq[B]`
  - `_integrate(cloud, sp, k, L, reverse) -> (cluster_final[B,k,2], logdet[B])` (RK4, cage fixed, divergence accumulated).

- [ ] **Step 1: Write the failing test**

```python
def _flow(n_cage=48, L=None):
    return E.EGNNClusterFlow(sigma_b=1.1, n_cage=n_cage, r_c=3.0, L=L, hidden_nf=32, n_layers=2, n_steps=12)

def test_sampler_equals_scorer():
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    flow = _flow(n_cage, L).to(DEV).eval()
    xC, logq = flow.sample(pos, s, cl, sc, L)
    logq2 = flow.log_q(pos, s, cl, xC, sc, L)
    assert xC.shape == (4, 7, 2)
    assert torch.allclose(logq, logq2, atol=2e-3), (logq - logq2).abs().max()

def test_not_translation_invariant():
    """Translating the cluster ALONE (cage fixed) must change logq (position is pinned by the cage)."""
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    flow = _flow(n_cage, L).to(DEV).eval()
    xC, _ = flow.sample(pos, s, cl, sc, L)
    lq0 = flow.log_q(pos, s, cl, xC, sc, L)
    lq1 = flow.log_q(pos, s, cl, torch.remainder(xC + 0.5, L), sc, L)   # move cluster only
    assert (lq0 - lq1).abs().max() > 1e-2
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k "sampler_equals or translation" -v`
Expected: FAIL (`EGNNClusterFlow` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to ka_cluster_egnn.py
class EGNNClusterFlow(nn.Module):
    def __init__(self, sigma_b, n_cage=48, k=7, r_c=3.0, L=None, hidden_nf=64, n_layers=4, n_steps=16, n_species=2):
        super().__init__()
        self.sigma_b = float(sigma_b); self.n_cage = n_cage; self.k = k; self.n_steps = n_steps
        self.ce = ConditionalEGNN(n_cage=n_cage, k=k, r_c=r_c, L=L, hidden_nf=hidden_nf, n_layers=n_layers,
                                  n_species=n_species)

    def _integrate(self, cloud, sp, k, L, reverse):
        """RK4 integrate the cluster (first k) over t; cage (rest) fixed. reverse=False: t 0->1 (base->data),
        accumulate -div; reverse=True: t 1->0 (data->base), accumulate +div. Returns (cluster[B,k,2], logdet)."""
        dev = cloud.device; B = cloud.shape[0]; dt = 1.0 / self.n_steps
        logdet = torch.zeros(B, device=dev, dtype=cloud.dtype)
        cl = cloud[:, :k]; cage = cloud[:, k:]
        def f(clpos, t):
            c2 = torch.cat([clpos, cage], 1)
            v, d = self.ce.vel_div(c2, torch.tensor(t, device=dev, dtype=cloud.dtype), sp, k)
            return torch.remainder(v, L) - v + v, d      # v unchanged; (kept explicit for clarity)
        steps = range(self.n_steps)
        for i in steps:
            t = 1.0 - i * dt if reverse else i * dt
            h = -dt if reverse else dt
            # RK4 on positions; divergence accumulated with the k1 stage (midpoint-consistent enough at n_steps>=12)
            v1, d1 = self.ce.vel_div(torch.cat([cl, cage], 1), _t(t, dev, cloud), sp, k)
            v2, _ = self.ce.vel_div(torch.cat([torch.remainder(cl + 0.5 * h * v1, L), cage], 1), _t(t + 0.5 * h, dev, cloud), sp, k)
            v3, _ = self.ce.vel_div(torch.cat([torch.remainder(cl + 0.5 * h * v2, L), cage], 1), _t(t + 0.5 * h, dev, cloud), sp, k)
            v4, _ = self.ce.vel_div(torch.cat([torch.remainder(cl + h * v3, L), cage], 1), _t(t + h, dev, cloud), sp, k)
            cl = torch.remainder(cl + (h / 6.0) * (v1 + 2 * v2 + 2 * v3 + v4), L)
            logdet = logdet + (h * d1)                    # ∫ div dt (sign folded into h)
        return cl, logdet

    @torch.no_grad()
    def sample(self, pos, s, cluster_idx, sc, L):
        cloud, sp, c = build_cloud(pos, s, cluster_idx, sc, L, self.n_cage)
        z = sample_base(c, self.k, self.sigma_b)
        cloud = torch.cat([z, cloud[:, self.k:]], 1)
        cl, logdet = self._integrate(cloud, sp, self.k, L, reverse=False)
        logq = base_logp(z, c, self.sigma_b) - logdet     # d log p = -div dt over the forward pass
        return cl, logq

    def log_q(self, pos, s, cluster_idx, xC_query, sc, L):
        cloud, sp, c = build_cloud(pos, s, cluster_idx, sc, L, self.n_cage)
        cloud = torch.cat([xC_query, cloud[:, self.k:]], 1)
        z, logdet = self._integrate(cloud, sp, self.k, L, reverse=True)
        return base_logp(z, c, self.sigma_b) + logdet
```

Add the tiny helper near the top of the file:
```python
def _t(t, dev, ref):
    return torch.tensor(float(t), device=dev, dtype=ref.dtype)
```
*(During implementation, delete the dead `f(...)` inner function in `_integrate` — it was a scratch placeholder; the RK4 stages below it are the real code.)*

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k "sampler_equals or translation" -v`
Expected: PASS. If `sampler_equals` is borderline, raise `n_steps` (16→24) — do NOT loosen the 2e-3 tolerance to hide a real integrator inconsistency.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_cluster_egnn.py liquid_coupling_flow/tests/test_ka_cluster_egnn.py
git commit -m "feat(egnn-flow): RK4 conditional integrator (sample/log_q) + sampler==scorer + position-pinned"
```

---

### Task 4: Species-aware per-particle OT + OT flow-matching training

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_egnn.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_egnn.py`

**Interfaces:**
- Consumes: Task 1–3, `scipy.optimize.linear_sum_assignment`, `ka_gridformer_train.augment`.
- Produces:
  - `ot_assign(z[B,k,2], x[B,k,2], sp[B,k], L) -> perm[B,k]` — species-aware per-particle OT (Hungarian on the min-image squared cost, cross-species = +inf).
  - `train(steps, k=7, n_cage=48, ...) -> ck`; `load_flow(ck, device) -> EGNNClusterFlow`.

- [ ] **Step 1: Write the failing test**

```python
def test_ot_species_and_cost():
    torch.manual_seed(0)
    z = torch.randn(3, 7, 2, device=DEV); x = torch.randn(3, 7, 2, device=DEV)
    sp = torch.tensor([0,0,0,0,1,1,1], device=DEV)[None].expand(3, 7)
    perm = E.ot_assign(z, x, sp, L=9.13)
    assert perm.shape == (3, 7)
    for b in range(3):
        assert sorted(perm[b].tolist()) == list(range(7))          # a valid permutation
        assert (sp[b] == sp[b][perm[b]]).all()                     # species preserved by the matching
    # OT cost <= identity cost
    def cost(p):
        idx = p[:, :, None].expand(-1, -1, 2)
        return ((z - torch.gather(x, 1, idx)) ** 2).sum((-1, -2))
    ident = torch.arange(7, device=DEV)[None].expand(3, 7)
    assert (cost(perm) <= cost(ident) + 1e-5).all()

def test_train_smoke_and_load():
    ck = E.train(steps=60, n_cage=48, hidden_nf=32, n_layers=2, save=False)
    assert 0.7 < ck["sigma_b"] < 1.6 and ck["loss_last"] < ck["loss_first"]
    flow = E.load_flow(ck, DEV)
    sc, L, geo, pos, s, cl, n_cage = _cfg(B=4)
    xC, logq = flow.sample(pos, s, cl, sc, L)
    assert torch.isfinite(logq).all() and xC.shape == (4, 7, 2)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k "ot_species or train" -v`
Expected: FAIL (`ot_assign`/`train` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to ka_cluster_egnn.py
from scipy.optimize import linear_sum_assignment


def ot_assign(z, x, sp, L):
    """Species-aware per-particle OT: Hungarian on the min-image squared cost, cross-species forbidden. -> perm[B,k]."""
    B, k, _ = z.shape; dev = z.device
    d = _wrap_pm(z[:, :, None, :] - x[:, None, :, :], L)                          # [B,k,k,2]  z_i vs x_j
    cost = (d ** 2).sum(-1)                                                       # [B,k,k]
    forbid = (sp[:, :, None] != sp[:, None, :])                                   # [B,k,k] species mismatch
    cost = cost.masked_fill(forbid, 1e6)
    perms = []
    cc = cost.detach().cpu().numpy()
    for b in range(B):
        _, col = linear_sum_assignment(cc[b]); perms.append(torch.as_tensor(col, device=dev))
    return torch.stack(perms, 0)                                                  # [B,k]


_ARCH = ("n_cage", "k", "r_c", "hidden_nf", "n_layers", "n_steps", "n_species")


def load_flow(ck, device):
    arch = {kk: ck[kk] for kk in _ARCH if kk in ck}
    P = EGNNClusterFlow(sigma_b=ck["sigma_b"], L=ck["L"], **arch).to(device).eval()
    P.load_state_dict(ck["state_dict"]); return P


def train(steps=15000, k=7, n_cage=48, r_c=3.0, hidden_nf=64, n_layers=4, n_steps=16, lr=3e-4, train_N=100,
          save=True, device="cuda" if torch.cuda.is_available() else "cpu"):
    """OT conditional flow matching: regress the EGNN velocity onto the OT-straightened base->data field."""
    import time
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, s = ref["x"].to(device), ref["s"].to(device).long(); N = data.shape[1]
    sc, L, geo = _scaffold(N, device); sigma_b = compute_sigma_b(data, s, geo, sc, L, k, n_cage)
    flow = EGNNClusterFlow(sigma_b=sigma_b, n_cage=n_cage, k=k, r_c=r_c, L=L, hidden_nf=hidden_nf,
                           n_layers=n_layers, n_steps=n_steps).to(device).train()
    opt = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-4); Bsz = 128
    print(f"EGNN-FLOW train N={train_N} steps={steps} k={k} n_cage={n_cage} sigma_b={sigma_b:.3f} "
          f"params {sum(p.numel() for p in flow.parameters())/1e6:.2f}M", flush=True)
    loss_first = None; t0 = time.time(); ckpt = os.path.join(ART, f"ka_cluster_egnn_N{train_N}.pt")
    arch = dict(n_cage=n_cage, k=k, r_c=r_c, hidden_nf=hidden_nf, n_layers=n_layers, n_steps=n_steps, n_species=2)
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (Bsz,), device=device)
        pos, s_ord = slot_order(augment(data[idx], L), s, geo, N)
        seed = int(torch.randint(0, N, (1,)).item()); cl = KC.cluster_slots(seed, sc, k, L)
        cloud, sp, c = build_cloud(pos, s_ord, cl, sc, L, n_cage); x1 = cloud[:, :k]
        z = sample_base(c, k, sigma_b)                                            # base
        perm = ot_assign(z, x1, sp[:, :k], L)                                     # species-aware per-particle OT
        x1p = torch.gather(x1, 1, perm[..., None].expand(-1, -1, 2))              # OT-matched data
        target = _wrap_pm(x1p - z, L)                                             # straight displacement (min-image)
        t = torch.rand(Bsz, device=device)
        xt = torch.remainder(z + t[:, None, None] * target, L)                    # interpolant
        cloud_t = torch.cat([xt, cloud[:, k:]], 1)
        v_pred, _ = flow.ce.vel_div(cloud_t, t, sp, k)                            # EGNN velocity at (xt,t)
        loss = ((v_pred - target) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(flow.parameters(), 5.0); opt.step()
        if loss_first is None: loss_first = loss.item()
        if step % 1000 == 0:
            print(f"  step {step:5d} fm-loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if save and (step + 1) % 2000 == 0:
            torch.save({"state_dict": flow.state_dict(), "sigma_b": sigma_b, "L": L, "step": step + 1,
                        "loss_first": loss_first, "loss_last": loss.item(), **arch}, ckpt)
    ck = {"state_dict": flow.state_dict(), "sigma_b": sigma_b, "L": L, "step": steps,
          "loss_first": loss_first, "loss_last": loss.item(), **arch}
    if save:
        torch.save(ck, ckpt); print(f"saved ka_cluster_egnn_N{train_N}.pt", flush=True)
    return ck
```

Note: `flow.ce.vel_div` needs `t` as a per-batch tensor here; confirm `EGNN_dynamics._expand_t` accepts `t` shape `[B]` (it does — see its ndim==1 branch). Training is fp32 (no autocast; the divergence path is precision-sensitive, though training only uses the velocity).

- [ ] **Step 4: Run to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_egnn.py -k "ot_species or train" -v`
Expected: PASS (~1–2 min for the 60-step smoke).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_cluster_egnn.py liquid_coupling_flow/tests/test_ka_cluster_egnn.py
git commit -m "feat(egnn-flow): species-aware per-particle OT + OT flow-matching training + loader"
```

---

### Task 5: gate() + real train + g(r) gate (GO/NO-GO)

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_egnn.py`

**Interfaces:**
- Consumes: `ka_cluster_flow.gate_measure`, `load_flow`, `slot_order`, `_scaffold`.
- Produces: `gate(train_N=100, ...)`; `__main__` modes `train`/`gate`.

- [ ] **Step 1: Add the gate entry**

```python
# append to ka_cluster_egnn.py
@torch.no_grad()
def gate(train_N=100, k=7, device="cuda" if torch.cuda.is_available() else "cpu", Bsz=128):
    from liquid_coupling_flow.ka_cluster_flow import gate_measure
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    s = ref["s"].to(device).long(); N = ref["x"].shape[1]; sc, L, geo = _scaffold(N, device)
    ck = torch.load(os.path.join(ART, f"ka_cluster_egnn_N{train_N}.pt"), map_location=device, weights_only=False)
    P = load_flow(ck, device); pos0, sso = slot_order(ref["x"][:Bsz].to(device), s, geo, N)
    out = os.path.join(ART, f"ka_cluster_egnn_gate_N{train_N}.png")
    res = gate_measure(P, pos0, sso, sc, L, N, k, out_png=out)
    print("GATE(egnn-flow):", res, flush=True); print("saved", out, flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        gate()
    else:
        train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 15000)
```

- [ ] **Step 2: Verify the gate wiring on the smoke checkpoint (no long run yet)**

```bash
python -c "from liquid_coupling_flow import ka_cluster_egnn as E; ck=E.train(steps=40, n_cage=48, hidden_nf=32, n_layers=2, save=True); print('ok', ck['step'])"
python -m liquid_coupling_flow.ka_cluster_egnn gate
```
Expected: prints `GATE(egnn-flow): {'clash_pct':..., 'gbb_peak':..., 'seed_spread': nan}` and saves the figure (untrained numbers are meaningless — this only confirms the gate runs end-to-end on `EGNNClusterFlow`). **Print the full clickable figure path.**

- [ ] **Step 3: Commit the code**

```bash
git add liquid_coupling_flow/ka_cluster_egnn.py
git commit -m "feat(egnn-flow): gate() entry + __main__ (train/gate)"
```

- [ ] **Step 4 (CONTROLLER, not the implementer): real training + the decisive gate**

Detached, teardown-resilient, at full size:
```bash
setsid bash -c "python -u -m liquid_coupling_flow.ka_cluster_egnn 15000 > <scratch>/egnn_train.log 2>&1" < /dev/null &
# after it finishes (checkpoints every 2000):
python -m liquid_coupling_flow.ka_cluster_egnn gate
```
**GATE DECISION:** GO iff clash → ~0 and g_BB peak → ~2.0 (clearly beating proposal-B's 55% / 1.24). Then re-run the intra/context + parity split (`scratchpad/parity_clash.py` adapted to `EGNNClusterFlow`) — the same-parity signature MUST be gone (there is no parity partition). On NO-GO with the same-parity signature gone and intra-clash still high → that is the real expressiveness/precision wall → SMC corrector.

---

## Self-Review

**Spec coverage:** conditional EGNN CNF frame-free (Task 3) ✓; cluster-restricted exact divergence + brute-force check (Task 2) ✓; full-Gaussian cage-centroid base, no zero-COM, position pinned + the not-translation-invariant test (Tasks 1,3) ✓; physical-cutoff fixed cloud n_cage=48/r_c=3.0 (Task 1–2) ✓; species-aware per-particle OT flow matching (Task 4) ✓; RK4 sampler==scorer (Task 3) ✓; gate reuse + GO/NO-GO + parity re-check (Task 5) ✓; sigma_b from data (Task 1) ✓; interface mirrors ClusterProposal (Task 3) ✓.

**Placeholder scan:** the two scratch/dead lines (Task 1 `if False` centroid marker; Task 3 inner `f`) are explicitly called out to delete during implementation; no TBD/vague requirements remain.

**Type consistency:** `vel_div(cloud,t,sp,k)->(v[B,k,2],div[B])` used identically in Tasks 2/3/4; `build_cloud->(cloud,sp,c)`, `sample_base(c,k,sigma_b)`, `base_logp(x0,c,sigma_b)` consistent across Tasks 1/3/4; `sample`/`log_q` signatures match `ClusterProposal` + the gate; `load_flow`/`_ARCH` match the checkpoint keys written by `train`.
