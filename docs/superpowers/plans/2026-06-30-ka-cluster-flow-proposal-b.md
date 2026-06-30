# Proposal-B ClusterFlow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `ClusterFlow` — an E(2)-equivariant, exact-likelihood coupling flow that jointly places the k=7 KA cluster particles in the x_R-only frame — to beat proposal-A's diffuseness wall on the committed g(r) gate.

**Architecture:** From-scratch in-frame RQ-spline coupling flow. Anchor-centered Gaussian base (isotropic σ_b=0.747, fixed). Per-block geometric distance-attention conditioner (reusing `GeomEdgeBias`/`GeomAttnLayer`) over {active cluster, frozen cluster, x_R cage} with species + pair-type bias. Equivariance from the frame ⇒ total Jacobian = flow-log-det × frame-Jacobian(=1). Interface mirrors `ClusterProposal` so `diag`/gate/MH are drop-in.

**Tech Stack:** PyTorch. Reuse `transforms_spline.RQSplineElementwise` (already exists, non-circular linear-tail RQ spline), `ka_flow_coupling.GeomEdgeBias`/`GeomAttnLayer`, `ka_cluster.py` (frame + slot-order), `ka_cluster_flow.py` (`slot_order`, `_scaffold`, gate measurement).

## Global Constraints

- Exactness is non-negotiable; **the gate decides GO/NO-GO, never the loss** (a flow can be diffuse too). Standing directive: never conclude "no bug"; concrete checks only.
- Commit only the specific files each task names; **never `git add -A`** (repo is dirty).
- Species are **per-batch** in the cluster setting: `slot_order` permutes particles into slots per config, so `s_ord[b,j]` varies with `b`. Every species tensor here is `[B, ·]`, not `[·]`.
- In-frame coordinates are **non-periodic** (local box); use plain differences, not `_wrap`. (The frame's `to_frame` already min-images lab→frame.)
- ALWAYS print the full clickable path to any plot generated (CLAUDE.md).
- Device default `"cuda" if torch.cuda.is_available() else "cpu"`. Artifacts dir = `liquid_coupling_flow/artifacts` (`ART`).
- Reference: `artifacts/ka_reference_N100.pt` (keys `x [n,N,2]`, `s [N]`); N=100, k=7, ρ=1.2.

---

### Task 1: Anchor-Gaussian base + σ_b helper

**Files:**
- Create: `liquid_coupling_flow/ka_cluster_flow_b.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_flow_b.py`

**Interfaces:**
- Consumes: `ka_cluster` (`cluster_slots`, `frame_ctx_slots`, `cluster_scaffold_center`, `frame_from_positions`, `to_frame`), `ka_cluster_flow` (`slot_order`, `_scaffold`).
- Produces:
  - `anchor_base_logp(z, q_scaf, sigma_b) -> logp[B]` — z,q_scaf `[B,k,2]`; 2D isotropic Gaussian per particle, summed over k.
  - `sample_base(q_scaf, sigma_b, gen=None) -> z[B,k,2]`.
  - `compute_sigma_b(data, s, geo, sc, L, k, n_ctx=16) -> float` — min-imaged in-frame displacement std (expected ≈0.747).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ka_cluster_flow_b.py
import math, torch
from liquid_coupling_flow import ka_cluster_flow_b as B
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def test_anchor_base_logp_matches_closed_form():
    torch.manual_seed(0)
    q = torch.randn(4, 7, 2, device=DEV); z = q + 0.3 * torch.randn(4, 7, 2, device=DEV); sb = 0.747
    got = B.anchor_base_logp(z, q, sb)
    d2 = ((z - q) ** 2).sum(-1)                                   # [B,k]
    want = (-d2 / (2 * sb ** 2) - math.log(2 * math.pi * sb ** 2)).sum(-1)
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()

def test_sample_base_stats():
    q = torch.zeros(2000, 7, 2, device=DEV); sb = 0.747
    z = B.sample_base(q, sb)
    assert z.std().item() == __import__("pytest").approx(sb, rel=0.05)
    assert z.mean().abs().item() < 0.05
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k base -v`
Expected: FAIL (module `ka_cluster_flow_b` not found).

- [ ] **Step 3: Write minimal implementation**

```python
# ka_cluster_flow_b.py
"""Proposal-B: ClusterFlow — exact-likelihood, E(2)-equivariant coupling flow over the k cluster particles'
in-frame coordinates. Anchor-centered Gaussian base + per-block geometric distance-attention conditioner.
See docs/superpowers/specs/2026-06-30-ka-cluster-flow-proposal-b-design.md. Interface mirrors ClusterProposal."""
from __future__ import annotations
import os, math, torch, torch.nn as nn, torch.nn.functional as F
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def anchor_base_logp(z, q_scaf, sigma_b):                     # z,q_scaf [B,k,2] -> [B]
    d2 = ((z - q_scaf) ** 2).sum(-1)                          # [B,k]
    return (-d2 / (2 * sigma_b ** 2) - math.log(2 * math.pi * sigma_b ** 2)).sum(-1)


def sample_base(q_scaf, sigma_b, gen=None):                  # [B,k,2]
    return q_scaf + sigma_b * torch.randn(q_scaf.shape, generator=gen, device=q_scaf.device, dtype=q_scaf.dtype)


@torch.no_grad()
def compute_sigma_b(data, s, geo, sc, L, k, n_ctx=16):
    """Isotropic std of the min-imaged in-frame displacement (true cluster pos - scaffold anchor) over the
    reference + a stride of seeds. This is the FIXED base width."""
    from liquid_coupling_flow.ka_gridformer import _wrap_pm
    N = data.shape[1]; pos0, _ = slot_order(data, s, geo, N); disp = []
    for seed in range(0, N, 3):
        cl = KC.cluster_slots(seed, sc, k, L)
        slots = KC.frame_ctx_slots(cl, sc, L, n_ctx); sc_c = KC.cluster_scaffold_center(cl, sc, L).to(pos0.dtype)
        o, R = KC.frame_from_positions(pos0[:, slots, :], sc_c, L)
        u_true = KC.to_frame(pos0[:, cl], o, R, L)
        q_scaf = KC.to_frame(sc[cl].to(pos0.dtype)[None].expand(pos0.shape[0], -1, -1), o, R, L)
        disp.append(_wrap_pm(u_true - q_scaf, L).reshape(-1, 2))
    return float(torch.cat(disp, 0).std().item())
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k base -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_cluster_flow_b.py liquid_coupling_flow/tests/test_ka_cluster_flow_b.py
git commit -m "feat(cluster-flow): anchor-Gaussian base + sigma_b helper"
```

---

### Task 2: Per-block geometric conditioner (with invertibility masking)

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_flow_b.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_flow_b.py`

**Interfaces:**
- Consumes: `ka_flow_coupling.GeomAttnLayer`; `transforms_spline.RQSplineElementwise`.
- Produces: `ClusterConditioner(nn.Module)` with
  `params(b, u, u_ctx, sp_cl, sp_ctx, c, par) -> [B, Na, P]` where `u [B,k,2]` (cluster in-frame), `u_ctx [B,n_ctx,2]` (x_R in-frame), `sp_cl [B,k]`, `sp_ctx [B,n_ctx]`, `c∈{0,1}` transformed coord, `par∈{0,1}` active parity; `Na` = #active cluster particles; `P = 3*num_bins-1`. Plus `n_blocks` and `meta[(par,c)]` schedule. **Invariant:** output does not depend on the active cluster particles' coordinate `c` (masked) — guarantees coupling invertibility.

- [ ] **Step 1: Write the failing test**

```python
def test_conditioner_masks_active_c():
    """Params for active particles must NOT depend on their own transformed coord c (coupling invertibility)."""
    torch.manual_seed(0)
    cond = B.ClusterConditioner(num_bins=8, n_species=2).to(DEV).eval()
    u = torch.randn(3, 7, 2, device=DEV); uc = torch.randn(3, 16, 2, device=DEV)
    sp = torch.randint(0, 2, (3, 7), device=DEV); spc = torch.randint(0, 2, (3, 16), device=DEV)
    b = 0; par, c = cond.meta[b]
    p0 = cond.params(b, u, uc, sp, spc, c, par)
    u2 = u.clone(); amask = (torch.arange(7, device=DEV) % 2 == par)
    u2[:, amask, c] += 5.0                                        # perturb ONLY the active, transformed coord
    p1 = cond.params(b, u2, uc, sp, spc, c, par)
    assert torch.allclose(p0, p1, atol=1e-5), (p0 - p1).abs().max()   # params unchanged
    # control: perturbing the OTHER coord DOES change params
    u3 = u.clone(); u3[:, amask, 1 - c] += 0.5
    assert (cond.params(b, u3, uc, sp, spc, c, par) - p0).abs().max() > 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k conditioner -v`
Expected: FAIL (`ClusterConditioner` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to ka_cluster_flow_b.py
from liquid_coupling_flow.ka_flow_coupling import GeomAttnLayer


class _GeomEdgeBiasB(nn.Module):
    """GeomEdgeBias with PER-BATCH species (s [B,M]). Bias from loc-coord distance + pair-type embedding."""
    def __init__(self, n_head, n_species, n_rbf=16, cutoff=2.4):
        super().__init__()
        self.n_species = n_species
        self.register_buffer("centers", torch.linspace(0.0, cutoff, n_rbf))
        self.gamma = (n_rbf / cutoff) ** 2
        self.pt_emb = nn.Embedding(n_species * n_species, n_rbf)
        self.mlp = nn.Sequential(nn.Linear(n_rbf, n_rbf), nn.SiLU(), nn.Linear(n_rbf, n_head))

    def forward(self, d, s):                                      # d [B,M,M], s [B,M] long
        rbf = torch.exp(-self.gamma * (d[..., None] - self.centers) ** 2)         # [B,M,M,R]
        pt = (s[:, :, None] * self.n_species + s[:, None, :])                     # [B,M,M]
        feat = rbf + self.pt_emb(pt)                                              # [B,M,M,R]
        return self.mlp(feat).permute(0, 3, 1, 2)                                 # [B,nH,M,M]


class ClusterConditioner(nn.Module):
    """Per coupling block, builds one token per {cluster ∪ x_R} particle and runs geometric attention to emit
    spline params for the ACTIVE cluster particles. Active cluster coord c is MASKED -> params independent of it."""
    def __init__(self, num_bins=8, n_cycles=4, n_species=2, d_model=192, n_head=6, n_layer=4, box=4.0,
                 n_rbf=16, cutoff=2.4):
        super().__init__()
        self.P = 3 * num_bins - 1; self.box = box
        self.meta = [(par, c) for _ in range(n_cycles) for par in (0, 1) for c in (0, 1)]
        self.enc = nn.Linear(4, d_model)                         # featpos(pos) = [p, sin(pi p/box)] per coord -> 4
        self.sp_emb = nn.Embedding(n_species, d_model)
        self.role_emb = nn.Embedding(3, d_model)                 # 0=x_R, 1=frozen cluster, 2=active cluster
        self.block_emb = nn.Embedding(len(self.meta), d_model)
        self.cmask = nn.Parameter(torch.randn(1) * 0.02)         # stands in for the masked active c
        self.edge_bias = _GeomEdgeBiasB(n_head, n_species, n_rbf, cutoff)
        self.layers = nn.ModuleList([GeomAttnLayer(d_model, n_head) for _ in range(n_layer)])
        self.heads = nn.ModuleList([nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, self.P))
                                    for _ in self.meta])

    def _featpos(self, p):                                       # [...,2] -> [...,4]
        return torch.cat([p, torch.sin(math.pi * p / self.box)], -1)

    def params(self, b, u, u_ctx, sp_cl, sp_ctx, c, par):
        Bsz, k, _ = u.shape; nctx = u_ctx.shape[1]; dev = u.device
        amask_cl = (torch.arange(k, device=dev) % 2 == par)                      # [k] active cluster
        # mask the active cluster's transformed coord c
        u_m = u.clone()
        u_m[:, amask_cl, c] = self.cmask
        # token features for cluster then x_R
        feat_cl = self._featpos(u_m); feat_ctx = self._featpos(u_ctx)
        tok_cl = self.enc(feat_cl) + self.sp_emb(sp_cl)
        tok_ctx = self.enc(feat_ctx) + self.sp_emb(sp_ctx) + self.role_emb.weight[0]
        role_cl = torch.where(amask_cl, 2, 1)                                     # [k]
        tok_cl = tok_cl + self.role_emb(role_cl)[None]
        tok = torch.cat([tok_cl, tok_ctx], 1) + self.block_emb.weight[b]          # [B,M,d], M=k+nctx
        # loc-coord (1-c) distance among all tokens (known for all; non-periodic)
        loc = 1 - c
        loc_all = torch.cat([u[:, :, loc], u_ctx[:, :, loc]], 1)                  # [B,M]
        d = (loc_all[:, :, None] - loc_all[:, None, :]).abs()                     # [B,M,M]
        s_all = torch.cat([sp_cl, sp_ctx], 1)                                     # [B,M]
        bias = self.edge_bias(d, s_all)
        h = tok
        for layer in self.layers:
            h = layer(h, bias)
        return self.heads[b](h[:, :k][:, amask_cl])                              # [B,Na,P]
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k conditioner -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_cluster_flow_b.py liquid_coupling_flow/tests/test_ka_cluster_flow_b.py
git commit -m "feat(cluster-flow): per-block geometric conditioner with invertibility masking"
```

---

### Task 3: Assemble `ClusterFlow` (sample/log_q) + exactness guards

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_flow_b.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_flow_b.py`

**Interfaces:**
- Consumes: Task 1 (`anchor_base_logp`, `sample_base`), Task 2 (`ClusterConditioner`), `transforms_spline.RQSplineElementwise`, `ka_cluster` (`cluster_frame`/`frame_ctx_slots`/`to_frame`/`from_frame`), `ka_cluster_flow._scaffold`/`slot_order`.
- Produces: `ClusterFlow(nn.Module)` with
  - `sample(pos[B,N,2], s[B,N], cluster_idx[k], sc[N,2], L) -> (xC_lab[B,k,2], logq[B])` (@no_grad)
  - `log_q(pos[B,N,2], s[B,N], cluster_idx[k], xC_query[B,k,2], sc[N,2], L) -> logq[B]` (grad-capable)
  - attributes: `sigma_b`, `n_ctx`, `box`, `cond` (the `ClusterConditioner`), `spline`.

- [ ] **Step 1: Write the failing test**

```python
from liquid_coupling_flow import ka_cluster as KC

def _setup(Bsz=4):
    sc, L, geo = B._scaffold(100, DEV)
    ref = torch.load(f"{B.ART}/ka_reference_N100.pt", map_location=DEV, weights_only=False)
    pos, s = B.slot_order(ref["x"][:Bsz].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    cl = KC.cluster_slots(5, sc, 7, L)
    return sc, L, pos, s, cl

def test_sampler_equals_scorer():
    """PRIMARY exactness guard: the density the sampler reports == log_q at the sampled point (untrained ok)."""
    sc, L, pos, s, cl = _setup()
    flow = B.ClusterFlow(sigma_b=0.747).to(DEV).eval()
    xC, logq = flow.sample(pos, s, cl, sc, L)
    logq2 = flow.log_q(pos, s, cl, xC, sc, L)
    assert xC.shape == (4, 7, 2)
    assert torch.allclose(logq, logq2, atol=1e-3), (logq - logq2).abs().max()

def test_logdet_matches_autograd():
    """Analytic flow log-det == autograd jacobian log|det| of xC_query -> log_q's z-mapping, small batch."""
    sc, L, pos, s, cl = _setup(Bsz=1)
    flow = B.ClusterFlow(sigma_b=0.747).to(DEV).eval()
    xC, _ = flow.sample(pos, s, cl, sc, L)
    x0 = xC[0].reshape(-1).double().requires_grad_(True)        # [2k]
    def to_z(xflat):                                            # map query -> base z (the flow inverse), in-frame
        return flow._x_to_z(pos[:1].double(), s[:1], cl, xflat.reshape(1, 7, 2), sc, L)[0].reshape(-1)
    J = torch.autograd.functional.jacobian(to_z, x0)           # [2k,2k]
    logdet_auto = torch.linalg.slogdet(J)[1]
    logdet_anal = flow._x_to_z(pos[:1].double(), s[:1], cl, xC[:1].double(), sc, L)[1][0]   # returns (z, sum_logdet)
    assert abs(logdet_auto.item() - logdet_anal.item()) < 1e-3
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k "sampler or logdet" -v`
Expected: FAIL (`ClusterFlow` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to ka_cluster_flow_b.py
from liquid_coupling_flow.transforms_spline import RQSplineElementwise


class ClusterFlow(nn.Module):
    def __init__(self, sigma_b, num_bins=8, n_cycles=4, n_ctx=32, box=4.0, tail_bound=4.0,
                 d_model=192, n_head=6, n_layer=4, n_species=2):
        super().__init__()
        self.sigma_b = float(sigma_b); self.n_ctx = n_ctx; self.box = box
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.cond = ClusterConditioner(num_bins=num_bins, n_cycles=n_cycles, n_species=n_species,
                                       d_model=d_model, n_head=n_head, n_layer=n_layer, box=box)

    def _frame_ctx(self, pos, s, cluster_idx, sc, L):
        """Frame (from the cluster_frame's own 16 slots) + the conditioner's n_ctx x_R tokens + q_scaf anchors."""
        origin, R = KC.cluster_frame(pos, cluster_idx, sc, L)                     # x_R-only frame (16 slots)
        slots = KC.frame_ctx_slots(cluster_idx, sc, L, self.n_ctx)
        u_ctx = KC.to_frame(pos[:, slots, :], origin, R, L)                       # [B,n_ctx,2]
        sp_ctx = s[:, slots]
        q_scaf = KC.to_frame(sc[cluster_idx].to(pos.dtype)[None].expand(pos.shape[0], -1, -1), origin, R, L)
        return origin, R, u_ctx, sp_ctx, q_scaf

    def _amask(self, k, par, dev):
        return (torch.arange(k, device=dev) % 2 == par)

    def _z_to_u(self, z, u_ctx, sp_cl, sp_ctx):
        """Base z [B,k,2] -> cluster in-frame u, accumulating forward log-det (sum over active coords/blocks)."""
        u = z.clone(); ld = torch.zeros(u.shape[0], device=u.device)
        for b in range(len(self.cond.meta)):
            par, c = self.cond.meta[b]; am = self._amask(u.shape[1], par, u.device)
            p = self.cond.params(b, u, u_ctx, sp_cl, sp_ctx, c, par)              # [B,Na,P]
            y, d = self.spline.forward(u[:, am, c], p)
            u = u.clone(); u[:, am, c] = y; ld = ld + d.sum(-1)
        return u, ld

    def _x_to_z(self, pos, s, cluster_idx, xC_query, sc, L):
        """Cluster lab xC_query -> base z (flow inverse), accumulating inverse log-det. Returns (z, sum_logdet)."""
        origin, R, u_ctx, sp_ctx, _ = self._frame_ctx(pos, s, cluster_idx, sc, L)
        sp_cl = s[:, cluster_idx]
        u = KC.to_frame(xC_query, origin, R, L); ld = torch.zeros(u.shape[0], device=u.device)
        for b in reversed(range(len(self.cond.meta))):
            par, c = self.cond.meta[b]; am = self._amask(u.shape[1], par, u.device)
            p = self.cond.params(b, u, u_ctx, sp_cl, sp_ctx, c, par)
            z, d = self.spline.inverse(u[:, am, c], p)
            u = u.clone(); u[:, am, c] = z; ld = ld + d.sum(-1)
        return u, ld

    @torch.no_grad()
    def sample(self, pos, s, cluster_idx, sc, L):
        origin, R, u_ctx, sp_ctx, q_scaf = self._frame_ctx(pos, s, cluster_idx, sc, L)
        sp_cl = s[:, cluster_idx]
        z = sample_base(q_scaf, self.sigma_b)
        u, ld = self._z_to_u(z, u_ctx, sp_cl, sp_ctx)
        xC_lab = KC.from_frame(u, origin, R, L)
        logq = anchor_base_logp(z, q_scaf, self.sigma_b) - ld
        return xC_lab, logq

    def log_q(self, pos, s, cluster_idx, xC_query, sc, L):
        _, _, _, _, q_scaf = self._frame_ctx(pos, s, cluster_idx, sc, L)
        z, ld = self._x_to_z(pos, s, cluster_idx, xC_query, sc, L)
        return anchor_base_logp(z, q_scaf, self.sigma_b) + ld
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k "sampler or logdet" -v`
Expected: PASS. (If `logdet` is borderline in fp32, the test already casts inputs to double.)

- [ ] **Step 5: Add the inherited frame keystone to this suite + commit**

```python
def test_frame_independent_of_xC_bit_exact():
    """Jacobian-1 exactness: moving the whole cluster leaves the frame identical to the bit."""
    sc, L, pos, s, cl = _setup(Bsz=1)
    o0, R0 = KC.cluster_frame(pos[0], cl, sc, L)
    p2 = pos[0].clone(); p2[cl] = torch.remainder(p2[cl] + torch.tensor([1.7, -1.1], device=DEV), L)
    o1, R1 = KC.cluster_frame(p2, cl, sc, L)
    assert (o1 - o0).abs().max().item() == 0.0 and (R1 - R0).abs().max().item() == 0.0
```

```bash
git add liquid_coupling_flow/ka_cluster_flow_b.py liquid_coupling_flow/tests/test_ka_cluster_flow_b.py
git commit -m "feat(cluster-flow): assemble ClusterFlow (sample/log_q) + exactness guards"
```

---

### Task 4: Training (conditional MLE, σ_b from data, checkpoint + loader)

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_flow_b.py`
- Test: `liquid_coupling_flow/tests/test_ka_cluster_flow_b.py`

**Interfaces:**
- Consumes: Task 3 (`ClusterFlow`), Task 1 (`compute_sigma_b`), `ka_cluster_flow.slot_order`/`_scaffold`, `ka_gridformer_train.augment`, `ka_cluster.cluster_slots`.
- Produces: `train(steps, k=7, ...)` saving `artifacts/ka_cluster_flow_b_N{N}.pt` (state_dict + arch + `sigma_b` + `k`); `load_flow(ck, device) -> ClusterFlow`.

- [ ] **Step 1: Write the failing test (short overfit smoke + loader round-trip)**

```python
def test_train_smoke_and_load(tmp_path):
    ck = B.train(steps=80, train_N=100, save=False)              # returns the checkpoint dict, no disk write
    assert ck["sigma_b"] == __import__("pytest").approx(0.747, abs=0.05)
    flow = B.load_flow(ck, DEV)
    sc, L, pos, s, cl = _setup()
    xC, logq = flow.sample(pos, s, cl, sc, L)
    assert torch.isfinite(logq).all() and xC.shape == (4, 7, 2)
    assert ck["loss_last"] < ck["loss_first"]                    # learned something in 80 steps
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k train -v`
Expected: FAIL (`train`/`load_flow` not defined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to ka_cluster_flow_b.py
_ARCH_KEYS = ("num_bins", "n_cycles", "n_ctx", "box", "tail_bound", "d_model", "n_head", "n_layer")


def load_flow(ck, device):
    arch = {kk: ck[kk] for kk in _ARCH_KEYS if kk in ck}
    P = ClusterFlow(sigma_b=ck["sigma_b"], **arch).to(device).eval()
    P.load_state_dict(ck["state_dict"]); return P


def train(steps=20000, k=7, train_N=100, num_bins=8, n_cycles=4, n_ctx=32, box=4.0, tail_bound=4.0,
          d_model=192, n_head=6, n_layer=4, lr=3e-4, save=True,
          device="cuda" if torch.cuda.is_available() else "cpu"):
    import time
    from liquid_coupling_flow.ka_gridformer_train import augment
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    data, s = ref["x"].to(device), ref["s"].to(device).long(); N = data.shape[1]
    sc, L, geo = _scaffold(N, device)
    sigma_b = compute_sigma_b(data, s, geo, sc, L, k)
    arch = dict(num_bins=num_bins, n_cycles=n_cycles, n_ctx=n_ctx, box=box, tail_bound=tail_bound,
                d_model=d_model, n_head=n_head, n_layer=n_layer)
    P = ClusterFlow(sigma_b=sigma_b, **arch).to(device).train()
    opt = torch.optim.AdamW(P.parameters(), lr=lr, weight_decay=1e-4); Bsz = 128
    warm = 400
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda st: min((st + 1) / warm,
        0.5 + 0.5 * math.cos(math.pi * max(0, st - warm) / max(1, steps - warm))))
    print(f"CLUSTERFLOW train N={train_N} steps={steps} k={k} sigma_b={sigma_b:.3f} {arch} "
          f"params {sum(p.numel() for p in P.parameters())/1e6:.2f}M", flush=True)
    loss_first = None; t0 = time.time()
    for step in range(steps):
        idx = torch.randint(0, data.shape[0], (Bsz,), device=device)
        pos, s_ord = slot_order(augment(data[idx], L), s, geo, N)
        seed = int(torch.randint(0, N, (1,)).item()); cl = KC.cluster_slots(seed, sc, k, L)
        loss = -(P.log_q(pos, s_ord, cl, pos[:, cl], sc, L) / k).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(P.parameters(), 5.0); opt.step(); sched.step()
        if loss_first is None: loss_first = loss.item()
        if step % 1000 == 0:
            print(f"  step {step:5d} -logq/k {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    ck = {"state_dict": P.state_dict(), "k": k, "sigma_b": sigma_b, "step": steps,
          "loss_first": loss_first, "loss_last": loss.item(), **arch}
    if save:
        torch.save(ck, os.path.join(ART, f"ka_cluster_flow_b_N{train_N}.pt"))
        print(f"saved ka_cluster_flow_b_N{train_N}.pt", flush=True)
    return ck
```

(Note: train uses fp32 — no bf16 autocast — because the spline/log-det is precision-sensitive; the conditional flow is small.)

- [ ] **Step 4: Run to verify it passes**

Run: `pytest liquid_coupling_flow/tests/test_ka_cluster_flow_b.py -k train -v`
Expected: PASS (~1 min).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_cluster_flow_b.py liquid_coupling_flow/tests/test_ka_cluster_flow_b.py
git commit -m "feat(cluster-flow): conditional-MLE training + data-derived sigma_b + loader"
```

---

### Task 5: g(r) gate (reuse the committed measurement) → GO/NO-GO

**Files:**
- Modify: `liquid_coupling_flow/ka_cluster_flow.py` (extract the `diag` measurement into a reusable function)
- Modify: `liquid_coupling_flow/ka_cluster_flow_b.py` (add `gate()` + `__main__`)

**Interfaces:**
- Consumes: `ClusterFlow`/`load_flow`; the extracted `ka_cluster_flow.gate_measure`.
- Produces: `ka_cluster_flow.gate_measure(P, pos0, sso, sc, L, N, k, out_png) -> dict` (clash%, g_BB peak, seed spread; saves figure); `ka_cluster_flow_b.gate(train_N=100)`; `__main__` modes `train` / `gate`.

- [ ] **Step 1: Refactor — extract `gate_measure` from `diag`**

In `ka_cluster_flow.py`, move the body of `diag` (single-cluster clash, seed spread/P(true bin), 3.3b g_BB, the 2-panel figure) into a standalone `gate_measure(P, pos0, sso, sc, L, N, k, out_png)` that takes an already-loaded model `P` exposing `.sample(pos,s,cl,sc,L)`. Re-point `diag` to call it (`P = _load(ck, device); gate_measure(P, pos0, sso, sc, L, N, k, out_png=...)`). Behavior must be unchanged.

- [ ] **Step 2: Run the existing proposal-A diag to confirm the refactor is behavior-preserving**

Run: `python -m liquid_coupling_flow.ka_cluster_flow diag`
Expected: same numbers as before the refactor (clash ~44%, g_BB ~1.35) and the figure still saved. (No automated test; this is the regression check — eyeball the printed numbers.)

- [ ] **Step 3: Add `ClusterFlow` gate entry**

```python
# append to ka_cluster_flow_b.py
@torch.no_grad()
def gate(train_N=100, k=7, device="cuda" if torch.cuda.is_available() else "cpu", Bsz=128):
    from liquid_coupling_flow.ka_cluster_flow import gate_measure
    ref = torch.load(os.path.join(ART, f"ka_reference_N{train_N}.pt"), map_location=device, weights_only=False)
    s = ref["s"].to(device).long(); N = ref["x"].shape[1]; sc, L, geo = _scaffold(N, device)
    ck = torch.load(os.path.join(ART, f"ka_cluster_flow_b_N{train_N}.pt"), map_location=device, weights_only=False)
    P = load_flow(ck, device)
    pos0, sso = slot_order(ref["x"][:Bsz].to(device), s, geo, N)
    out = os.path.join(ART, f"ka_cluster_flow_b_gate_N{train_N}.png")
    res = gate_measure(P, pos0, sso, sc, L, N, k, out_png=out)
    print("GATE(flow):", res, flush=True); print("saved", out, flush=True)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "gate":
        gate()
    else:
        train(steps=int(sys.argv[1]) if len(sys.argv) > 1 else 20000)
```

- [ ] **Step 4: Train for real, then run the gate**

```bash
python -u -m liquid_coupling_flow.ka_cluster_flow_b 20000        # background; ~depends on size
python -m liquid_coupling_flow.ka_cluster_flow_b gate
```
Expected: prints clash% / g_BB peak / seed spread and saves `artifacts/ka_cluster_flow_b_gate_N100.png` (**print the full clickable path**). **GATE DECISION:** GO iff clash → ~0 and g_BB peak → ~2.0 (clearly beating proposal-A's 44% / 1.35). On NO-GO (clash ~40% / g_BB ~1.4, like sharpening), STOP — do not sharpen the flow; the local cage underdetermines the position → SMC corrector.

- [ ] **Step 5: Commit (code only; not the checkpoint/figure)**

```bash
git add liquid_coupling_flow/ka_cluster_flow.py liquid_coupling_flow/ka_cluster_flow_b.py
git commit -m "feat(cluster-flow): reusable gate_measure + ClusterFlow g(r) gate entry"
```

---

## Self-Review

**Spec coverage:** base (Task 1) ✓; transform = existing `RQSplineElementwise` used in Task 3 ✓; per-layer geometric conditioner + species two-ways + masking (Task 2) ✓; in-frame assembly + equivariance + exactness guards (Task 3) ✓; data-derived fixed σ_b + checkpoint (Tasks 1,4) ✓; interface mirrors `ClusterProposal` (Task 3) ✓; the g(r) gate + kill criteria (Task 5) ✓. Slot-order non-locality tail is a recorded design note (handled by the spline's linear tails) — no task needed.

**Placeholder scan:** none — every code/test step has full code.

**Type consistency:** `params(b,u,u_ctx,sp_cl,sp_ctx,c,par)->[B,Na,P]`, `P=3*num_bins-1`, used identically in Tasks 2 and 3; `sample`/`log_q` signatures match `ClusterProposal` and the gate; `sigma_b` float through Tasks 1→4→loader; `meta` schedule shared via `self.cond.meta`.
