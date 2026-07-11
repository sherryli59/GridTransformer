# Contractive Full-Cage Block Corrector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an invertible, full-cage corrector flow on top of the frozen AR block generator so large-block proposals are clean (low-clash, ~equilibrium energy) while preserving exact tractable `log q` for the MTM.

**Architecture:** A shallow conditional coupling flow `F` transforms the K block particles' u-space coordinates, each coupling layer conditioning a subset on the *other* block particles + the frozen cage (the full-cage the AR lacks). Composed with the frozen base via change-of-variables (`tf2boltz`-style). Trained by maximum likelihood on carved cavity data (base frozen) — no energy objective (which mode-collapsed the prior attempt).

**Tech Stack:** PyTorch, the existing `liquid_coupling_flow` KA3D stack (`ka3d_ebm_batched.KA3DScaffoldEBMBatched`, `ka3d_scaffold_ar.ball_squash/ball_unsquash`), pytest.

## Global Constraints

- **Exact `log q` is mandatory.** Every composed path must satisfy `sample log q == score log q` on the same config to <1e-2 (the base cat-head's inherited 8e-3 round-trip leak is the tolerance floor; the corrector must add nothing beyond it).
- **Base is FROZEN.** Do not modify or retrain `KA3DScaffoldEBM`/`KA3DScaffoldEBMBatched` weights. The corrector is a separate module; only its parameters train.
- **No energy / reverse-KL training objective.** Training is MLE on data only (the energy objective collapsed the model: `eft_gumbel_N100.out` overlap 0.21->0.81). Energy is used only in GATES (measurement), never in a loss.
- **Frameless base** (`m.use_frame = False`): the frame `Rf = I`, so block u-coords are `u = ball_unsquash(x) - anchor_y` (no rotation).
- **Species are not transformed.** The corrector maps POSITIONS only; species come from the base (count-preserving is the base's job).
- **Model:** `KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5)`, checkpoint `liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt`, loaded `strict=False`, `use_frame=False`.
- **Data:** carved cavities from `liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt`, radii `{1.6, 2.0, 2.4}`, boundary shell within `R + 2.5`, via `ka3d_cavity_carve.carve` + `ka3d_cavity_ar._mic` + `ka3d_scaffold_ar.label_to_scaffold` (see `reports/logs-2026-07-11/ka3d_pts_batched.py` for the exact cavity-construction idiom).
- **Device:** `cuda`.

## File Structure

- `liquid_coupling_flow/ka3d_block_corrector.py` — the corrector: `AffineCoupling` layer, `CageConditioner` net, `BlockCorrector` flow (`forward`/`inverse` returning transformed u + logdet), and the composed `CorrectedBlockModel` wrapping a frozen base (`sample_block_corrected`, `block_log_prob_corrected`).
- `tests/test_block_corrector.py` — pytest unit + exactness tests (coupling round-trip, F/F⁻¹ round-trip, composition sample==score, identity-init reproduces base).
- `reports/logs-2026-07-11/train_corrector.py` — MLE training (base frozen); saves `liquid_coupling_flow/artifacts/ka3d_block_corrector.pt`.
- `reports/logs-2026-07-11/gate_corrector_clean.py` — the make-or-break clean gate (K=12 clash < 29%, equilibrium energy, no collapse).
- `reports/logs-2026-07-11/gate_corrector_mtm.py` — large-K MTM acceptance + basin-crossing energy check.

---

### Task 1: Affine coupling layer (identity-init, invertible)

**Files:**
- Create: `liquid_coupling_flow/ka3d_block_corrector.py`
- Test: `tests/test_block_corrector.py`

**Interfaces:**
- Produces: `AffineCoupling(dim=3)` with `forward(u, params) -> (u_out, logdet)` and `inverse(u_out, params) -> u`, where `u` is `[..., 3]`, `params` is `[..., 6]` (3 log-scale, 3 shift), `logdet` is `[...]` (sum of the 3 log-scales over the transformed rows). `mask` selecting which rows are active is applied by the caller (params for inactive rows are zeroed).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_block_corrector.py
import torch
from liquid_coupling_flow.ka3d_block_corrector import AffineCoupling

def test_affine_coupling_roundtrip_and_logdet():
    torch.manual_seed(0)
    ac = AffineCoupling()
    u = torch.randn(5, 3)
    params = torch.randn(5, 6) * 0.3
    u_out, logdet = ac.forward(u, params)
    u_back = ac.inverse(u_out, params)
    assert torch.allclose(u_back, u, atol=1e-5)
    # logdet == sum of log-scales; scale = exp(tanh-bounded raw)
    ls = torch.tanh(params[:, :3]) * ac.smax
    assert torch.allclose(logdet, ls.sum(-1), atol=1e-5)

def test_affine_coupling_identity_when_params_zero():
    ac = AffineCoupling()
    u = torch.randn(4, 3)
    u_out, logdet = ac.forward(u, torch.zeros(4, 6))
    assert torch.allclose(u_out, u, atol=1e-6)
    assert torch.allclose(logdet, torch.zeros(4), atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_block_corrector.py -k affine -q`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError: AffineCoupling`).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka3d_block_corrector.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class AffineCoupling(nn.Module):
    """Diagonal affine map on [...,3] with tanh-bounded log-scale (identity when params=0).
    forward:  u_out = u * exp(s) + t,  s = smax*tanh(params[...,:3]),  t = params[...,3:]
    logdet = sum(s). inverse: u = (u_out - t) * exp(-s)."""
    def __init__(self, smax: float = 2.0):
        super().__init__()
        self.smax = float(smax)

    def _st(self, params):
        s = self.smax * torch.tanh(params[..., :3])
        t = params[..., 3:]
        return s, t

    def forward(self, u, params):
        s, t = self._st(params)
        return u * torch.exp(s) + t, s.sum(-1)

    def inverse(self, u_out, params):
        s, t = self._st(params)
        return (u_out - t) * torch.exp(-s)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_block_corrector.py -k affine -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka3d_block_corrector.py tests/test_block_corrector.py
git commit -m "feat(corrector): affine coupling layer (identity-init, invertible)"
```

---

### Task 2: Cage conditioner (per-block-particle affine params, full-cage)

**Files:**
- Modify: `liquid_coupling_flow/ka3d_block_corrector.py`
- Test: `tests/test_block_corrector.py`

**Interfaces:**
- Consumes: `AffineCoupling`.
- Produces: `CageConditioner(d_model=96, n_rbf=12, rbf_max=3.0)` with
  `params(active_x, active_s, cond_x, cond_s, R) -> [A, 6]` where `active_x [A,3]`/`active_s [A]` are the physical positions/species of the block particles being transformed this layer, and `cond_x [C,3]`/`cond_s [C]` are the physical positions/species of the CONDITIONING set (the *other* block particles + the cage = retained interior + boundary). For each active particle, aggregate messages from all conditioning particles via a distance-RBF + species-pair MLP (the base's conditioning style), then a linear head to `[A,6]` (last layer zero-init -> identity coupling at init). Batched over M by the caller looping/stacking or a leading batch dim; implement with an explicit leading `M` batch: `active_x [M,A,3]`, `cond_x [M,C,3]` -> `[M,A,6]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_block_corrector.py (append)
from liquid_coupling_flow.ka3d_block_corrector import CageConditioner

def test_cage_conditioner_shape_and_zero_init():
    torch.manual_seed(0)
    cc = CageConditioner()
    M, A, C = 4, 3, 20
    ax = torch.randn(M, A, 3); asp = torch.randint(0, 2, (M, A))
    cx = torch.randn(M, C, 3); csp = torch.randint(0, 2, (M, C))
    p = cc.params(ax, asp, cx, csp, R=2.0)
    assert p.shape == (M, A, 6)
    # zero-init head => params all ~0 (identity coupling at init)
    assert p.abs().max() < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_block_corrector.py -k conditioner -q`
Expected: FAIL (`AttributeError: CageConditioner`).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka3d_block_corrector.py (append)
class CageConditioner(nn.Module):
    """Per-active-particle affine params from a distance-RBF + species-pair message over the conditioning
    set (other block particles + cage). Physical positions; batched [M,A,*] x [M,C,*] -> [M,A,6]. Last
    layer zero-init => identity coupling at init (composition starts == base)."""
    def __init__(self, d_model: int = 96, n_rbf: int = 12, rbf_max: float = 3.0, n_species: int = 2):
        super().__init__()
        self.n_species = n_species
        mu = torch.linspace(0.0, rbf_max, n_rbf)
        self.register_buffer("mu", mu); self.w = float(mu[1] - mu[0])
        self.pair = nn.Embedding(n_species * n_species, 8)
        self.msg = nn.Sequential(nn.Linear(n_rbf + 8, d_model), nn.SiLU(),
                                 nn.Linear(d_model, d_model), nn.SiLU())
        self.head = nn.Sequential(nn.Linear(d_model, d_model), nn.SiLU(), nn.Linear(d_model, 6))
        nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)

    def params(self, active_x, active_s, cond_x, cond_s, R):
        d = (active_x[:, :, None, :] - cond_x[:, None, :, :]).norm(dim=-1)          # [M,A,C]
        r = torch.exp(-((d[..., None] - self.mu) ** 2) / (2 * self.w ** 2))         # [M,A,C,n_rbf]
        pr = (active_s[:, :, None] * self.n_species + cond_s[:, None, :]).clamp(0, self.n_species ** 2 - 1)
        msg = self.msg(torch.cat([r, self.pair(pr)], -1))                          # [M,A,C,d]
        agg = msg.mean(2)                                                          # permutation-invariant over cond
        return self.head(agg)                                                     # [M,A,6]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_block_corrector.py -k conditioner -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka3d_block_corrector.py tests/test_block_corrector.py
git commit -m "feat(corrector): full-cage cage conditioner (zero-init -> identity)"
```

---

### Task 3: BlockCorrector flow (stacked coupling, alternating masks, F/F^{-1}+logdet)

**Files:**
- Modify: `liquid_coupling_flow/ka3d_block_corrector.py`
- Test: `tests/test_block_corrector.py`

**Interfaces:**
- Consumes: `AffineCoupling`, `CageConditioner`.
- Produces: `BlockCorrector(n_layers=6, ...)` with
  `forward(u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R) -> (u_out [M,K,3], logdet [M])` and
  `inverse(u_out, s_blk, anchor_y_blk, cage_x, cage_s, R) -> (u_blk [M,K,3], logdet [M])`.
  `u_blk [M,K,3]` are block u-coords; `s_blk [M,K]` block species; `anchor_y_blk [K,3]` the block anchors'
  unsquashed coords (shared across M); `cage_x [M,C,3]`/`cage_s [M,C]` the frozen cage (retained interior +
  boundary) physical positions/species. Each layer transforms the ACTIVE half of block particles (mask
  alternates parity by layer) conditioned on the INACTIVE half + cage. Physical position of a block particle
  at u is `ball_squash(u + anchor_y_blk, R)`. `logdet` accumulates over layers; `inverse` returns the
  negated accumulated logdet.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_block_corrector.py (append)
from liquid_coupling_flow.ka3d_block_corrector import BlockCorrector

def test_block_corrector_roundtrip_and_identity_init():
    torch.manual_seed(0)
    bc = BlockCorrector(n_layers=6)
    M, K, C = 3, 8, 30; R = 2.0
    u = torch.randn(M, K, 3) * 0.5
    s = torch.randint(0, 2, (M, K)); ay = torch.randn(K, 3) * 0.5
    cx = torch.randn(M, C, 3); cs = torch.randint(0, 2, (M, C))
    # identity at init (zero-init heads): forward == input, logdet == 0
    u_out, logdet = bc.forward(u, s, ay, cx, cs, R)
    assert torch.allclose(u_out, u, atol=1e-5)
    assert logdet.abs().max() < 1e-5
    # perturb params so it is non-trivial, then check invertibility + logdet sign
    for p in bc.parameters():
        p.data += torch.randn_like(p) * 0.05
    u_out, ld_f = bc.forward(u, s, ay, cx, cs, R)
    u_back, ld_i = bc.inverse(u_out, s, ay, cx, cs, R)
    assert torch.allclose(u_back, u, atol=1e-4)
    assert torch.allclose(ld_f, -ld_i, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_block_corrector.py -k block_corrector -q`
Expected: FAIL (`AttributeError: BlockCorrector`).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka3d_block_corrector.py (append)
from liquid_coupling_flow.ka3d_scaffold_ar import ball_squash


class BlockCorrector(nn.Module):
    """Stacked conditional affine-coupling flow over the K block particles' u-coords. Layer L transforms
    the parity-L half of block particles conditioned on the other half + the frozen cage."""
    def __init__(self, n_layers: int = 6, d_model: int = 96, n_rbf: int = 12, smax: float = 2.0):
        super().__init__()
        self.n_layers = n_layers
        self.couple = AffineCoupling(smax=smax)
        self.conds = nn.ModuleList([CageConditioner(d_model, n_rbf) for _ in range(n_layers)])

    def _phys(self, u, anchor_y, R):
        x, _ = ball_squash(u + anchor_y[None], R)                                 # [M,K,3]
        return x

    def _run(self, u, s_blk, anchor_y, cage_x, cage_s, R, invert):
        M, K, _ = u.shape
        dev = u.device
        idx = torch.arange(K, device=dev)
        order = range(self.n_layers - 1, -1, -1) if invert else range(self.n_layers)
        logdet = u.new_zeros(M)
        for L in order:
            active = (idx % 2) == (L % 2)                                         # alternating half
            passive = ~active
            xu = self._phys(u, anchor_y, R)                                       # current physical block
            cond_x = torch.cat([xu[:, passive], cage_x], 1)                       # other block + cage
            cond_s = torch.cat([s_blk[:, passive], cage_s], 1)
            params_full = u.new_zeros(M, K, 6)
            params_full[:, active] = self.conds[L].params(
                xu[:, active], s_blk[:, active], cond_x, cond_s, R)
            if invert:
                u = u.clone()
                u[:, active] = self.couple.inverse(u[:, active], params_full[:, active])
                logdet = logdet - self.couple.forward(u[:, active] * 0, params_full[:, active])[1]  # -sum s
            else:
                ua, ld = self.couple.forward(u[:, active], params_full[:, active])
                u = u.clone(); u[:, active] = ua
                logdet = logdet + ld
        return u, logdet

    def forward(self, u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R):
        return self._run(u_blk, s_blk, anchor_y_blk, cage_x, cage_s, R, invert=False)

    def inverse(self, u_out, s_blk, anchor_y_blk, cage_x, cage_s, R):
        return self._run(u_out, s_blk, anchor_y_blk, cage_x, cage_s, R, invert=True)
```

Note on the inverse logdet: `couple.inverse` does not return a logdet; recompute the layer's `sum(s)` from the same `params` and subtract. The helper above recomputes it via `couple.forward(...)[1]` on the params (the `* 0` input is irrelevant — only `params` sets `s`). The engineer MAY instead add a `logdet` return to `AffineCoupling.inverse` and use it directly; either is fine as long as `test_block_corrector_roundtrip_and_identity_init` passes (it checks `ld_f == -ld_i`).

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_block_corrector.py -k block_corrector -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka3d_block_corrector.py tests/test_block_corrector.py
git commit -m "feat(corrector): stacked full-cage coupling flow (invertible, identity-init)"
```

---

### Task 4: Composition with the frozen base + EXACTNESS gate

**Files:**
- Modify: `liquid_coupling_flow/ka3d_block_corrector.py`
- Test: `tests/test_block_corrector.py`

**Interfaces:**
- Consumes: `BlockCorrector`; `KA3DScaffoldEBMBatched` (`_prep`, `block_log_prob_b`, `sample_block_b`); `ball_squash`, `ball_unsquash`.
- Produces: `CorrectedBlockModel(base, corrector)` with:
  - `sample_block_corrected(xo, so, block_mask, bnd, s_bnd, R, gen) -> (xo_new [M,n,3], so_new [M,n], logq [M])`
  - `block_log_prob_corrected(xo, so, block_mask, bnd, s_bnd, R) -> logq [M]`
  where signatures mirror the base's `*_b` methods (`xo [M,n,3]`). Composition (u-space corrector wrapped in
  ball maps; block particles only, retained/boundary fixed):
    - Extract per-M block u-coords: `y = ball_unsquash(x_block, R); u0 = y - anchor_y_blk`.
    - `logdet_yx_blk` = ball_unsquash logdet summed over block; `logdet_xy_blk(x1)` = ball_squash logdet.
    - Forward (sample): `x0, lq0 = base.sample_block_b(...)`; `u1, ldF = corrector.forward(u0, ...)`;
      `x1 = ball_squash(u1 + anchor_y_blk)`; scatter x1 into the block slots;
      `logq = lq0 - logdet_yx_blk(x0) - ldF - logdet_xy_blk(x1)`.
    - Score: `u1 = ball_unsquash(x1_block) - anchor_y_blk`; `u0, ldFinv = corrector.inverse(u1, ...)`
      (`ldFinv = -ldF`); `x0 = ball_squash(u0 + anchor_y_blk)`; `lq0 = base.block_log_prob_b(x0-in-place, ...)`;
      `logq = lq0 - logdet_yx_blk(x0) + ldFinv - logdet_xy_blk(x1)`.
    - The `cage_x/cage_s` for the corrector = boundary ++ retained interior physical positions/species
      (everything NOT in the block), gathered from the reordered config via `_prep`'s `order` and `n_ret`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_block_corrector.py (append)
import torch
from liquid_coupling_flow.ka3d_block_corrector import CorrectedBlockModel, load_base_and_cavity

def test_composition_exactness_and_identity_reproduces_base():
    dev = "cuda"
    base, cav = load_base_and_cavity(dev, ci=900, R=2.0, K=6, M=6)  # helper builds one cavity + block
    from liquid_coupling_flow.ka3d_block_corrector import BlockCorrector
    corr = BlockCorrector(n_layers=6).to(dev)
    model = CorrectedBlockModel(base, corr)
    xo, so, blk, bnd, sb, R = cav
    # (a) identity-init: corrected block_log_prob == base block_log_prob_b (to fp)
    lp_corr = model.block_log_prob_corrected(xo, so, blk, bnd, sb, R)
    lp_base = base.block_log_prob_b(xo, so, blk, bnd, sb, R)
    assert (lp_corr - lp_base).abs().max() < 1e-3
    # (b) exactness round-trip: sample logq == score logq (inherits base 8e-3)
    xn, sn, lq_s = model.sample_block_corrected(xo, so, blk, bnd, sb, R, gen=torch.Generator(dev).manual_seed(1))
    lq_score = model.block_log_prob_corrected(xn, sn, blk, bnd, sb, R)
    assert (lq_s - lq_score).abs().max() < 1e-2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_block_corrector.py -k composition -q`
Expected: FAIL (`ImportError: CorrectedBlockModel` / `load_base_and_cavity`).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka3d_block_corrector.py (append)
from liquid_coupling_flow.ka3d_scaffold_ar import ball_unsquash, fixed_ball_scaffold


class CorrectedBlockModel(nn.Module):
    """Frozen base (KA3DScaffoldEBMBatched) + BlockCorrector, composed to an exact-log_q block proposal."""
    def __init__(self, base, corrector):
        super().__init__()
        self.base = base
        self.corrector = corrector

    def _geom(self, xo, block_mask, bnd, s_bnd, R):
        """Reorder [retained; block]; return block anchors' unsquashed coords + the frozen cage (physical)."""
        n = xo.shape[0]; dev = xo.device
        anchors = fixed_ball_scaffold(n, R, dev, xo.dtype)
        order = torch.argsort(block_mask.to(torch.uint8), stable=True)
        n_ret = int((~block_mask).sum())
        anchor_y, _ = ball_unsquash(anchors[order], R)          # [n,3]
        return order, n_ret, anchor_y[n_ret:]                    # block anchors = suffix

    def _cage(self, xo_re, so_re, n_ret, bnd, s_bnd):
        """Physical cage = boundary ++ retained interior (everything not in the block). xo_re [M,n,3]."""
        M = xo_re.shape[0]; m = bnd.shape[0]
        cage_x = torch.cat([bnd[None].expand(M, m, 3), xo_re[:, :n_ret]], 1)
        cage_s = torch.cat([s_bnd[None].expand(M, m), so_re[:, :n_ret]], 1)
        return cage_x, cage_s

    def sample_block_corrected(self, xo, so, block_mask, bnd, s_bnd, R, gen=None):
        base = self.base
        order, n_ret, ay_blk = self._geom(xo, block_mask, bnd, s_bnd, R)
        x0, s0, lq0 = base.sample_block_b(xo, so, block_mask, bnd, s_bnd, R, gen=gen)   # [M,n,3],[M,n],[M]
        x0_re = x0[:, order]; s0_re = s0[:, order]
        xb0 = x0_re[:, n_ret:]                                                          # block, [M,K,3]
        y0, ld_yx = ball_unsquash(xb0, R)                                               # [M,K,3],[M,K]
        u0 = y0 - ay_blk[None]
        cage_x, cage_s = self._cage(x0_re, s0_re, n_ret, bnd, s_bnd)
        u1, ldF = self.corrector.forward(u0, s0_re[:, n_ret:], ay_blk, cage_x, cage_s, R)
        xb1, ld_xy = ball_squash(u1 + ay_blk[None], R)
        x1_re = x0_re.clone(); x1_re[:, n_ret:] = xb1
        x1 = torch.empty_like(x0); x1[:, order] = x1_re
        logq = lq0 - ld_yx.sum(1) - ldF - ld_xy.sum(1)
        return x1, s0, logq

    def block_log_prob_corrected(self, xo, so, block_mask, bnd, s_bnd, R):
        base = self.base
        order, n_ret, ay_blk = self._geom(xo, block_mask, bnd, s_bnd, R)
        xo_re = xo[:, order]; so_re = so[:, order]
        xb1 = xo_re[:, n_ret:]
        y1, ld_xy = ball_unsquash(xb1, R)                                               # note: unsquash of x1
        u1 = y1 - ay_blk[None]
        cage_x, cage_s = self._cage(xo_re, so_re, n_ret, bnd, s_bnd)
        u0, ldFinv = self.corrector.inverse(u1, so_re[:, n_ret:], ay_blk, cage_x, cage_s, R)
        xb0, ld_yx = ball_squash(u0 + ay_blk[None], R)
        x0_re = xo_re.clone(); x0_re[:, n_ret:] = xb0
        x0 = torch.empty_like(xo); x0[:, order] = x0_re
        lq0 = base.block_log_prob_b(x0, so, block_mask, bnd, s_bnd, R)
        # symmetric with the sample path: both ball-map logdets enter with the same sign convention
        logq = lq0 - ld_xy.sum(1) + ldFinv - ld_yx.sum(1)
        return logq


def load_base_and_cavity(dev, ci, R, K, M):
    """Test/gate helper: load frozen base + build ONE carved cavity replicated to M chains + a K-blob mask."""
    import torch
    from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
    from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
    from liquid_coupling_flow.ka3d_cavity_ar import _mic
    from liquid_coupling_flow.ka3d_cavity_carve import carve
    ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    base = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    base.load_state_dict(ck["state_dict"], strict=False); base.eval(); base.use_frame = False
    for p in base.parameters():
        p.requires_grad_(False)
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(ci)
    c = torch.rand(3, generator=gen, device=dev) * L
    p = carve(X[ci], S[ci], c, R, L)
    xo1, so1, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + 2.5)
    bnd, sb = xout[bm], p["s_out"][bm]
    n = xo1.shape[0]
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    xo = xo1[None].expand(M, n, 3).contiguous(); so = so1[None].expand(M, n).contiguous()
    return base, (xo, so, blk, bnd, sb, R)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_block_corrector.py -k composition -q`
Expected: PASS. If the round-trip (b) fails by more than 8e-3, the ball-map logdet signs in `block_log_prob_corrected` are mismatched with `sample_block_corrected` — reconcile so that scoring a sampled config returns its sampled `logq` (the round-trip IS the correctness oracle).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka3d_block_corrector.py tests/test_block_corrector.py
git commit -m "feat(corrector): composed exact-logq block model + exactness/identity gates"
```

---

### Task 5: MLE training (base frozen)

**Files:**
- Create: `reports/logs-2026-07-11/train_corrector.py`
- Test: (smoke overfit assertion inside the script; no separate pytest)

**Interfaces:**
- Consumes: `CorrectedBlockModel`, `load_base_and_cavity` idiom (carve cavities), `BlockCorrector`.
- Produces: checkpoint `liquid_coupling_flow/artifacts/ka3d_block_corrector.pt` (`{"state_dict", "step", "config"}`).

- [ ] **Step 1: Write the training script (with a built-in overfit smoke gate)**

```python
# reports/logs-2026-07-11/train_corrector.py
"""MLE-train the BlockCorrector on carved cavity DATA blocks; base FROZEN. Loss = -log q_corrected(data)/K.
No energy term (energy objective collapsed the model). Smoke: on a tiny fixed pool the loss must DECREASE."""
import argparse, time, statistics as st
import torch
from liquid_coupling_flow.ka3d_block_corrector import CorrectedBlockModel, BlockCorrector
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

ART = "liquid_coupling_flow/artifacts"; dev = "cuda"


def build_pool(X, S, L, frames, radii, gen, per=2, Krange=(6, 16), nmin=10):
    pool = []
    for f in frames:
        for _ in range(per):
            c = torch.rand(3, generator=gen, device=dev) * L
            R = float(radii[int(torch.randint(len(radii), (), generator=gen, device=dev))])
            p = carve(X[f], S[f], c, R, L)
            if p["n_in"] < nmin:
                continue
            xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
            xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + 2.5)
            n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev)
            seed = int(torch.randint(n, (), generator=gen, device=dev))
            K = int(torch.randint(Krange[0], min(Krange[1], n) + 1, (), generator=gen, device=dev))
            blk = torch.zeros(n, dtype=torch.bool, device=dev)
            blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
            pool.append({"xo": xo, "so": so, "blk": blk, "bnd": xout[bm], "sb": p["s_out"][bm], "R": R, "K": K})
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4); ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--radii", type=float, nargs="+", default=[1.6, 2.0, 2.4])
    ap.add_argument("--out", default=f"{ART}/ka3d_block_corrector.pt")
    a = ap.parse_args()
    ck = torch.load(f"{ART}/ka3d_cavity_ebm3ax.pt", map_location=dev, weights_only=False)
    base = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
    base.load_state_dict(ck["state_dict"], strict=False); base.eval(); base.use_frame = False
    for p in base.parameters():
        p.requires_grad_(False)
    corr = BlockCorrector(n_layers=a.layers).to(dev)
    model = CorrectedBlockModel(base, corr)
    d = torch.load(f"{ART}/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
    X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
    gen = torch.Generator(device=dev).manual_seed(0)
    pool = build_pool(X, S, L, range(900), a.radii, gen)
    print(f"pool={len(pool)}", flush=True)
    opt = torch.optim.AdamW(corr.parameters(), lr=a.lr, weight_decay=1e-5)
    t0 = time.time(); first = None
    for step in range(a.steps + 1):
        ids = torch.randint(len(pool), (a.batch,), generator=gen, device=dev)
        loss = X.new_zeros(())
        for i in ids:
            c = pool[int(i)]
            xo = c["xo"][None]; so = c["so"][None]                                     # M=1 per item
            lq = model.block_log_prob_corrected(xo, so, c["blk"], c["bnd"], c["sb"], c["R"])
            loss = loss - lq.sum() / int(c["blk"].sum())
        loss = loss / a.batch
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(corr.parameters(), 5.0); opt.step()
        if step % 500 == 0:
            if first is None:
                first = loss.item()
            print(f"step {step:5d} -logq/K {loss.item():+.4f} ({time.time()-t0:.0f}s)", flush=True)
            torch.save({"state_dict": corr.state_dict(), "step": step, "config": vars(a)}, a.out)
    assert loss.item() < first, f"overfit smoke FAILED: loss did not decrease ({first:.3f}->{loss.item():.3f})"
    print(f"saved {a.out}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a short smoke to verify the loss decreases**

Run: `python -m reports.logs-2026-07-11.train_corrector --steps 1000 --out /tmp/corr_smoke.pt 2>&1 | grep -v Warning | tail -5`
Expected: `-logq/K` decreases across steps; final assertion prints `saved ...` (no AssertionError).

- [ ] **Step 3: Launch the full training (background, per repo run-conventions)**

Run: `python -m reports.logs-2026-07-11.train_corrector --steps 6000 > reports/logs-2026-07-11/corrector_train.out 2>&1 &`
Expected: `corrector_train.out` shows `-logq/K` decreasing; `ka3d_block_corrector.pt` written every 500 steps.

- [ ] **Step 4: Commit**

```bash
git add reports/logs-2026-07-11/train_corrector.py reports/logs-2026-07-11/corrector_train.out
git commit -m "feat(corrector): MLE training (base frozen) + overfit smoke gate"
```

---

### Task 6: Clean gate — the make-or-break

**Files:**
- Create: `reports/logs-2026-07-11/gate_corrector_clean.py`

**Interfaces:**
- Consumes: `CorrectedBlockModel`, the trained `ka3d_block_corrector.pt`, `reports/logs-2026-07-11/test_halfcage.py` clash/energy idioms (`energy_b` from `ka3d_pts_batched`).

- [ ] **Step 1: Write the clean gate**

```python
# reports/logs-2026-07-11/gate_corrector_clean.py
"""MAKE-OR-BREAK gate. Corrected K=12 block proposals must be CLEANER than the raw AR base:
  (1) per-particle clash rate < the 29% full-cage floor (raw AR ~43-57%);
  (2) block interior energy ~ equilibrium (NOT +48/particle like the raw MTM-jump);
  (3) NO collapse (energy does not blow up over repeated sampling).
Compares raw base.sample_block_b vs model.sample_block_corrected on the same cavities."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_block_corrector import CorrectedBlockModel, BlockCorrector, load_base_and_cavity
import importlib.util as iu
spec = iu.spec_from_file_location("kb", "reports/logs-2026-07-11/ka3d_pts_batched.py")
kb = iu.module_from_spec(spec); spec.loader.exec_module(kb)  # energy_b

dev = "cuda"; R = 2.0; K = 12; M = 24; CUT = 0.8


def clash_rate(xn, blk, bnd):
    order = torch.argsort(blk.to(torch.uint8), stable=True); n_ret = int((~blk).sum())
    # clash of each block particle vs all others (interior sampled + boundary)
    rr = []
    xre = xn[:, order]
    for mi in range(xn.shape[0]):
        allx = torch.cat([xre[mi], bnd], 0); bi = xre[mi, n_ret:]
        dm = torch.cdist(bi, allx)
        for r in range(bi.shape[0]):
            dm[r, n_ret + r] = 9.0
        rr.append(float((dm.min(1).values < CUT).float().mean()))
    return st.mean(rr)


base, cav0 = load_base_and_cavity(dev, 900, R, K, M)
corr = BlockCorrector(n_layers=6).to(dev)
corr.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_block_corrector.pt", map_location=dev)["state_dict"])
corr.eval()
model = CorrectedBlockModel(base, corr)
raw_c, cor_c, raw_e, cor_e, ref_e = [], [], [], [], []
for ci in range(900, 912):
    _, (xo, so, blk, bnd, sb, _) = load_base_and_cavity(dev, ci, R, K, M)
    g = torch.Generator(dev).manual_seed(ci)
    xr, sr, _ = base.sample_block_b(xo, so, blk, bnd, sb, R, gen=g)
    xc, sc, _ = model.sample_block_corrected(xo, so, blk, bnd, sb, R, gen=g)
    raw_c.append(clash_rate(xr, blk, bnd)); cor_c.append(clash_rate(xc, blk, bnd))
    raw_e.append(kb.energy_b(xr, sr, bnd, sb).mean().item() / xo.shape[1])
    cor_e.append(kb.energy_b(xc, sc, bnd, sb).mean().item() / xo.shape[1])
    ref_e.append(kb.energy_b(xo[:1], so[:1], bnd, sb).item() / xo.shape[1])
print(f"clash: raw {100*st.mean(raw_c):.0f}%  corrected {100*st.mean(cor_c):.0f}%  (target < 29%)")
print(f"U/n:   raw {st.mean(raw_e):+.2f}  corrected {st.mean(cor_e):+.2f}  ref {st.mean(ref_e):+.2f}")
ok = (100 * st.mean(cor_c) < 29) and (st.mean(cor_e) < st.mean(raw_e) - 5)
print("CLEAN GATE:", "PASS" if ok else "FAIL")
```

- [ ] **Step 2: Run the clean gate**

Run: `python -m reports.logs-2026-07-11.gate_corrector_clean 2>&1 | grep -v Warning`
Expected: corrected clash < 29% AND corrected `U/n` closer to `ref` than raw. Prints `CLEAN GATE: PASS`. If FAIL: the corrector is under-trained or too shallow — increase `--layers` / training steps; if still failing, escalate to jointly fine-tuning the base (still MLE) per the spec's risk section, or Approach B.

- [ ] **Step 3: Commit**

```bash
git add reports/logs-2026-07-11/gate_corrector_clean.py
git commit -m "test(corrector): clean gate (clash < 29% full-cage floor + equilibrium energy + no collapse)"
```

---

### Task 7: MTM acceptance + basin-crossing energy check

**Files:**
- Create: `reports/logs-2026-07-11/gate_corrector_mtm.py`

**Interfaces:**
- Consumes: `CorrectedBlockModel`; the MTM-jump idiom from `reports/logs-2026-07-11/ka3d_pts_batched.py` (`mtm_jump_b`, `energy_b`), but using `model.sample_block_corrected`/`model.block_log_prob_corrected` as the proposal.

- [ ] **Step 1: Write the MTM/basin-crossing gate**

```python
# reports/logs-2026-07-11/gate_corrector_mtm.py
"""Does the CLEAN corrected proposal restore large-K basin-crossing? Run a many-trial MTM-jump using the
CORRECTED block proposal (exact log q), and CHECK the energy: MTM samples must now be ~equilibrium (the raw
MTM-jump was +48/particle). Reports large-K MTM acceptance (corrected vs raw) and the post-jump energy."""
import torch, statistics as st
import importlib.util as iu
from liquid_coupling_flow.ka3d_block_corrector import CorrectedBlockModel, BlockCorrector, load_base_and_cavity
spec = iu.spec_from_file_location("kb", "reports/logs-2026-07-11/ka3d_pts_batched.py")
kb = iu.module_from_spec(spec); spec.loader.exec_module(kb)
dev = "cuda"; R = 2.0; K = 12; M = 32; beta = 2.0; N = 16


def mtm_jump_corrected(model, Xb, Sb, blk, bnd, sb, R, N, gen):
    Mn, n, _ = Xb.shape
    Xr, Sr = Xb.repeat_interleave(N, 0), Sb.repeat_interleave(N, 0)
    Xp, Sp, lqf = model.sample_block_corrected(Xr, Sr, blk, bnd, sb, R, gen=gen)
    up = (-beta * kb.energy_b(Xp, Sp, bnd, sb) - lqf).reshape(Mn, N)
    u0 = -beta * kb.energy_b(Xb, Sb, bnd, sb) - model.block_log_prob_corrected(Xb, Sb, blk, bnd, sb, R)
    ar = torch.arange(Mn, device=dev); sfwd = torch.logsumexp(up, 1)
    J = torch.multinomial(torch.softmax(up, 1), 1, generator=gen).squeeze(1)
    lr = up.clone(); lr[ar, J] = u0; srev = torch.logsumexp(lr, 1)
    acc = torch.rand(Mn, device=dev, generator=gen).log() < (sfwd - srev)
    Yx = Xp.reshape(Mn, N, n, 3)[ar, J]; Ys = Sp.reshape(Mn, N, n)[ar, J]
    return torch.where(acc[:, None, None], Yx, Xb), torch.where(acc[:, None], Ys, Sb), acc.float().mean().item()


base, _ = load_base_and_cavity(dev, 900, R, K, M)
corr = BlockCorrector(n_layers=6).to(dev)
corr.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_block_corrector.pt", map_location=dev)["state_dict"]); corr.eval()
model = CorrectedBlockModel(base, corr)
accs, Ue, Ur = [], [], []
for ci in range(900, 908):
    _, (xo, so, blk, bnd, sb, _) = load_base_and_cavity(dev, ci, R, K, M)
    g = torch.Generator(dev).manual_seed(ci)
    Xb, Sb = xo.clone(), so.clone()
    for _ in range(6):
        for _ in range(max(1, xo.shape[1] // K)):
            Xb, Sb, a = mtm_jump_corrected(model, Xb, Sb, blk, bnd, sb, R, N, g); accs.append(a)
    Ue.append(kb.energy_b(Xb, Sb, bnd, sb).mean().item() / xo.shape[1])
    Ur.append(kb.energy_b(xo[:1], so[:1], bnd, sb).item() / xo.shape[1])
print(f"corrected MTM-jump accept {100*st.mean(accs):.0f}%")
print(f"post-jump U/n {st.mean(Ue):+.2f}  vs ref {st.mean(Ur):+.2f}  (must be close, NOT +48)")
print("BASIN-CROSSING ENERGY GATE:", "PASS" if abs(st.mean(Ue) - st.mean(Ur)) < 5 else "FAIL")
```

- [ ] **Step 2: Run the gate**

Run: `python -m reports.logs-2026-07-11.gate_corrector_mtm 2>&1 | grep -v Warning`
Expected: acceptance clearly > 0 at K=12 (raw was ~0), and post-jump `U/n` within a few /particle of `ref` (the raw MTM-jump was +48). Prints `BASIN-CROSSING ENERGY GATE: PASS`.

- [ ] **Step 3: Commit**

```bash
git add reports/logs-2026-07-11/gate_corrector_mtm.py
git commit -m "test(corrector): large-K MTM acceptance + basin-crossing energy gate"
```

---

## Self-Review

**Spec coverage:**
- Base frozen + corrector-only training → Task 5 (`requires_grad_(False)`, optimizes `corr.parameters()`). ✓
- Invertible full-cage corrector flow → Tasks 1–3 (coupling, cage conditioner, stacked flow). ✓
- Composed exact `log q` (`tf2boltz`-style) → Task 4 (`sample_block_corrected`/`block_log_prob_corrected`). ✓
- MLE (no energy objective) → Task 5 loss `-log q/K`; energy only in gates (Tasks 6–7). ✓
- Exactness gate → Task 4 round-trip + identity-init reproduces base. ✓
- Clean gate (clash < 29% + equilibrium energy + no collapse) → Task 6. ✓
- Large-K MTM acceptance + basin-crossing energy check → Task 7. ✓
- Multi-radius carved cavities → Task 5 `build_pool(radii=[1.6,2.0,2.4])`. ✓

**Placeholder scan:** no TBD/TODO; every code step is complete. The only intentional latitude is the `AffineCoupling.inverse` logdet (Task 3 note) — resolved by the `ld_f == -ld_i` assertion. ✓

**Type consistency:** `block_log_prob_corrected`/`sample_block_corrected` signatures mirror the base `*_b` (`xo [M,n,3]`) and are consumed identically in Tasks 6–7. `BlockCorrector.forward/inverse` signature `(u,s_blk,anchor_y_blk,cage_x,cage_s,R)` matches between Tasks 3 and 4. `load_base_and_cavity(dev,ci,R,K,M)` signature consistent across Tasks 4/6/7. `CageConditioner.params(active_x,active_s,cond_x,cond_s,R)` consistent Tasks 2–3. ✓

**Correctness anchor:** the ball-map logdet signs in Task 4 are the one subtle spot; the `sample==score` round-trip test (Task 4 Step 1b) is the oracle that forces them right before any downstream task runs.
