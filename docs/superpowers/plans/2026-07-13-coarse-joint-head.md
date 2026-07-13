# Coarse-Joint-Then-Fine Position Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the KA3D cavity AR's sequential per-axis position head with a coarse 3D-joint categorical (16³ cells, tilt evaluated at true 3D cell centers, exact 3D min-sep maskable) followed by per-axis fine refinement — eliminating the two measured clash mechanisms: hedged early-axis commitment and the slice-blind a/b tilts.

**Architecture:** New `CoarseFineHead` module (16³ coarse cells × 8³ fine bins = identical 128/axis final resolution) + `KA3DScaffoldEBMCoarse(KA3DScaffoldEBMBatched)` overriding exactly three methods: `_tilted_lp_u` (unbatched scorer → training via inherited `log_prob_pair`), `_tilted_lp_u_b` (batched scorer → `block_log_prob_b`), `sample_block_b` (batched sampler). Trunk/species/pair-potential nets warm-start from `ka3d_cavity_ebm3ax_rho115_best.pt` (strict=False); only the new head params train from scratch initially.

**Tech Stack:** PyTorch; existing `_phi_pair` pair-potential nets (reused for the coarse 3D tilt); repo test convention = standalone scripts in `reports/logs-2026-07-13/` + pytest in `liquid_coupling_flow/tests/`.

## Global Constraints

- Exactness contract: `sample_block_b(...args) == block_log_prob_b(...same args)` round-trip median < 1e-3, 100% of rows < 1e-3 at pos_temp ∈ {1.0, 0.4} (the standard from `test_tempered_score.py`).
- Cavity models are FRAMELESS: assert `use_frame == False` in every override (frame=identity; u = y − anchor_y).
- Final resolution unchanged: coarse 16 cells × fine 8 bins per axis over [−2.5, 2.5] → bin width 0.0390625 == current `fl.bw`; `logq` units directly comparable to baseline.
- min-sep mask must be CONSERVATIVE (forbid only cells fully inside the clash zone: `d_center < cut·σ − half_diag`) so data blocks always score finite; default `min_sep=None` = no-op.
- GPU is shared: training launches use `run_in_background`, redirect `> log 2>&1`, checkpoint every eval (both `last` and `_best`), and never overwrite `ka3d_cavity_ebm3ax_rho115*.pt`.
- Every run's log + artifacts go to `reports/logs-2026-07-13/` and get committed when the run finishes.

---

### Task 1: CoarseFineHead geometry module

**Files:**
- Create: `liquid_coupling_flow/ka3d_coarse_head.py`
- Test: `liquid_coupling_flow/tests/test_coarse_head.py`

**Interfaces:**
- Produces: `CoarseFineHead(d_model, u_range=2.5, n_coarse=16, n_fine=8)` with:
  `cell_index(u[...,3]) -> long[...]` in [0, 4096); `cell_center(idx) -> [...,3]`;
  `fine_bin(u, idx) -> long[...,3]` in [0,8) per axis; `fine_ctr(idx, fb) -> [...,3]` (absolute u of fine-bin center);
  attributes `n_cells=4096`, `cw` (cell width 0.3125), `bwf` (fine width 0.0390625), `half_diag`,
  `_log_bwf3 = 3*log(bwf)`; modules `head_coarse: Linear(d,4096)`, `cell_emb: Embedding(4096,d)`,
  `head_fa/head_fb/head_fc: Linear(d,8)`, `femb_a/femb_b: Embedding(8,d)`.

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_coarse_head.py
import torch
from liquid_coupling_flow.ka3d_coarse_head import CoarseFineHead


def test_roundtrip_u_to_bins_to_u():
    h = CoarseFineHead(d_model=32)
    u = (torch.rand(500, 3) - 0.5) * 2 * 2.5 * 0.999
    ci = h.cell_index(u)
    fb = h.fine_bin(u, ci)
    ctr = h.fine_ctr(ci, fb)
    assert ci.min() >= 0 and ci.max() < 4096
    assert fb.min() >= 0 and fb.max() < 8
    # every point lies within half a fine bin of its reconstructed center
    assert (u - ctr).abs().max() <= h.bwf / 2 + 1e-6


def test_grid_constants():
    h = CoarseFineHead(d_model=32)
    assert abs(h.cw - 5.0 / 16) < 1e-9
    assert abs(h.bwf - 5.0 / 128) < 1e-9        # == current fl.bw -> same final resolution
    assert abs(h.half_diag - h.cw * 3 ** 0.5 / 2) < 1e-9


def test_cell_center_inverse():
    h = CoarseFineHead(d_model=32)
    idx = torch.arange(4096)
    assert torch.equal(h.cell_index(h.cell_center(idx)), idx)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_coarse_head.py -q`
Expected: FAIL with `ModuleNotFoundError: ka3d_coarse_head`

- [ ] **Step 3: Write the implementation**

```python
# liquid_coupling_flow/ka3d_coarse_head.py
"""Coarse-joint-then-fine categorical position head for the KA3D cavity AR.

Motivation (measured 2026-07-13): the sequential per-axis head commits the a-axis through a hedged
marginal with a slice-blind tilt (Vax evaluated at (a,0,0)), so clashes are locked in the a/b plane
where the c-only hard mask cannot act (clash 31.6->31.9%). Here the FIRST commitment is a joint 16^3
coarse cell whose tilt is the true 3D pair energy at the cell center, and the exact hard min-sep mask
acts at cell granularity BEFORE any commitment. Fine stage: 8 bins/axis within the cell (slice error
bounded by cw/2=0.156). 16*8 = 128/axis == the old resolution; logq comparable.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn


class CoarseFineHead(nn.Module):
    def __init__(self, d_model, u_range=2.5, n_coarse=16, n_fine=8):
        super().__init__()
        self.rng, self.nc, self.nf = float(u_range), int(n_coarse), int(n_fine)
        self.n_cells = self.nc ** 3
        self.cw = 2 * self.rng / self.nc
        self.bwf = self.cw / self.nf
        self.half_diag = self.cw * math.sqrt(3.0) / 2
        self._log_bwf3 = 3.0 * math.log(self.bwf)
        self.head_coarse = nn.Linear(d_model, self.n_cells)
        self.cell_emb = nn.Embedding(self.n_cells, d_model)
        self.head_fa = nn.Linear(d_model, self.nf)
        self.head_fb = nn.Linear(d_model, self.nf)
        self.head_fc = nn.Linear(d_model, self.nf)
        self.femb_a = nn.Embedding(self.nf, d_model)
        self.femb_b = nn.Embedding(self.nf, d_model)
        # cached [4096,3] cell centers (buffer -> moves with .to(device))
        ax = (torch.arange(self.nc) + 0.5) * self.cw - self.rng
        gx, gy, gz = torch.meshgrid(ax, ax, ax, indexing="ij")
        self.register_buffer("centers", torch.stack([gx, gy, gz], -1).reshape(-1, 3), persistent=False)

    def _axcell(self, u1):
        return ((u1 + self.rng) / self.cw).long().clamp(0, self.nc - 1)

    def cell_index(self, u):
        a, b, c = self._axcell(u[..., 0]), self._axcell(u[..., 1]), self._axcell(u[..., 2])
        return (a * self.nc + b) * self.nc + c

    def cell_center(self, idx):
        return self.centers[idx]

    def fine_bin(self, u, idx):
        lo = self.cell_center(idx) - self.cw / 2
        return ((u - lo) / self.bwf).long().clamp(0, self.nf - 1)

    def fine_ctr(self, idx, fb):
        lo = self.cell_center(idx) - self.cw / 2
        return lo + (fb.float() + 0.5) * self.bwf
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_coarse_head.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka3d_coarse_head.py liquid_coupling_flow/tests/test_coarse_head.py
git commit -m "feat(ka3d): CoarseFineHead geometry (16^3 coarse x 8^3 fine, == 128/axis resolution)"
```

---

### Task 2: Coarse 3D tilt + conservative min-sep cell mask

**Files:**
- Modify: `liquid_coupling_flow/ka3d_coarse_head.py` (append)
- Test: `liquid_coupling_flow/tests/test_coarse_head.py` (append)

**Interfaces:**
- Consumes: `model._phi_pair(dist, sj, cage_s, net, emb)` (existing; maps pair distances+species → per-pair energy, summed by caller), `CoarseFineHead.centers`.
- Produces: `coarse_tilt_V(model, anchor_y[S,3], cage_x[S,Kc,3], cage_s[S,Kc], cage_v[S,Kc], sj[S], R, head) -> V[S,4096]`
  (true 3D pair energy at each cell center vs the causal kNN cage; positions mapped through `ball_squash` exactly as `_V_axis` does — copy its position-construction convention verbatim from `ka3d_scaffold_ebm.py:70-84` when implementing);
  `cells_allowed(head, anchor_y[S,3], sj[S], cage_x, cage_s, cage_v, R, cut) -> bool[S,4096]` forbidding only cells with `d(center_pos, neighbor) < cut*sigma_ij − half_diag_pos` for some valid neighbor, with the fallback `allowed |= ~allowed.any(-1,keepdim=True)`. `half_diag_pos` must be computed in POST-squash space: use the max over the cell's 8 corners of |squash(corner)−squash(center)| (upper bound of the squash image of the cell), computed once per call.

- [ ] **Step 1: Write the failing test** — brute-force agreement of the tilt on 5 random slots (compare `coarse_tilt_V` row against explicitly looping cells and calling `_phi_pair` per cell), and the mask's conservative property: for 200 random data-like points u, the cell containing u is NEVER forbidden when the point itself has min gap ≥ cut (assert via constructing a cage at known distances).
- [ ] **Step 2: Run, verify FAIL** (`NameError: coarse_tilt_V`).
- [ ] **Step 3: Implement** `coarse_tilt_V` (chunk cells in blocks of 512 to bound memory: S×512×Kc pair distances per chunk) and `cells_allowed`.
- [ ] **Step 4: Run, verify PASS.**
- [ ] **Step 5: Commit** `feat(ka3d): coarse 3D tilt + conservative cell min-sep mask`.

---

### Task 3: KA3DScaffoldEBMCoarse — unbatched scorer (training path)

**Files:**
- Modify: `liquid_coupling_flow/ka3d_coarse_head.py` (append class)
- Test: `liquid_coupling_flow/tests/test_coarse_head.py` (append)

**Interfaces:**
- Consumes: `KA3DScaffoldEBMBatched` (all context machinery inherited); `CoarseFineHead`; `coarse_tilt_V`.
- Produces: `KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5)` whose `_tilted_lp_u(h_e, u, frame, anchor_y, cage_x, cage_s, cage_v, sj, R, n_chunk=16)` returns `[S]` log-densities via
  `log p(cell) + log p(fa|cell) + log p(fb|cell,fa) + log p(fc|cell,fa,fb) − 3 log bwf`, where
  `logits_cell = coarse.head_coarse(h_e) − coarse_tilt_V(...)`,
  fine logits = `coarse.head_f{a,b,c}(h_e + coarse.cell_emb(ci) [+ femb_a(fa)] [+ femb_b(fb)])` minus per-axis tilts evaluated ON THE 8-point fine grid with off-axis coordinates at the current partial position (cell center, then chosen fine values) — reuse `_phi_pair` via a small `_V_fine_axis(...)` helper with an explicit grid argument.
  The inherited `log_prob_pair` then trains the coarse head with zero further changes.
  Constructor sets `self.coarse = CoarseFineHead(self.d_model)` and asserts `not self.use_frame` in the override.

- [ ] **Step 1: Failing test** — construct `KA3DScaffoldEBMCoarse` random-init on CPU, call `log_prob_pair` on one real carved cavity (load the N4096 dataset, carve with the correct `L` — copy the frame construction from `reports/logs-2026-07-12/test_tempered_score.py:20-28` verbatim); assert finite, and assert `loss.backward()` puts nonzero grads on `head_coarse.weight` and on `phi` (the reused pair net).
- [ ] **Step 2: FAIL** (`AttributeError: _tilted_lp_u` not overridden / class missing).
- [ ] **Step 3: Implement the class + override.**
- [ ] **Step 4: PASS.**
- [ ] **Step 5: Commit** `feat(ka3d): KA3DScaffoldEBMCoarse unbatched scorer -> log_prob_pair trains coarse head`.

---

### Task 4: Batched sampler + batched scorer + THE exactness gate

**Files:**
- Modify: `liquid_coupling_flow/ka3d_coarse_head.py` (append overrides)
- Test: `reports/logs-2026-07-13/test_coarse_exact.py` (standalone, mirrors `test_tempered_score.py`)

**Interfaces:**
- Consumes: `KA3DScaffoldEBMBatched.sample_block_b` / `_tilted_lp_u_b` / `block_log_prob_b` structure (copy `sample_block_b` from `ka3d_ebm_batched.py:172-236` and replace ONLY the position-axis section lines ~204-215 with: coarse logits → tempered softmax (pos_temp divides logits, same convention) → optional `cells_allowed` mask → sample cell → three fine axes → dither within fine bin → position; logq accumulates coarse+fine −3 log bwf −logdet_xy).
- Produces: exact round-trip `sample_block_b(pos_temp, min_sep) == block_log_prob_b(pos_temp, min_sep)`; `min_sep` at COARSE granularity only (drop the fine-c mask — the coarse mask is the effective one and one mask keeps the fp-boundary surface small).

- [ ] **Step 1: Write the standalone exactness test** (copy `reports/logs-2026-07-12/test_tempered_score.py`, swap the class for `KA3DScaffoldEBMCoarse` random-init — exactness needs no trained weights — and add `min_sep in (None, 0.85)` × `pos_temp in (1.0, 0.4)` grid; assert median < 1e-3 and 100% match, plus the T-mismatch control > 0.1).
- [ ] **Step 2: FAIL** (`NotImplementedError` from unimplemented overrides).
- [ ] **Step 3: Implement `sample_block_b` + `_tilted_lp_u_b` overrides.**
- [ ] **Step 4: Run — PASS on all 4 grid points.** Expected line format: `T=0.4 min_sep=0.85: median|score-sample| ~1e-5 match 100%`.
- [ ] **Step 5: Commit** `feat(ka3d): coarse-head batched sampler/scorer, exact round-trip incl. cell min-sep`.

---

### Task 5: Trainer wiring + smoke

**Files:**
- Modify: `reports/logs-2026-07-12/train_ebm3d_bigbox.py` (add `--coarse` flag: line 111 becomes a branch constructing `KA3DScaffoldEBMCoarse`; default `--out` for the coarse run = `{ART}/ka3d_cavity_coarse.pt`; `--warm` default unchanged — strict=False reports the new-param count)
- Test: smoke run (no pytest)

- [ ] **Step 1: Add the flag + branch** (import inside the branch to keep the default path untouched).
- [ ] **Step 2: Smoke** — `timeout 900 python reports/logs-2026-07-12/train_ebm3d_bigbox.py --coarse --steps 200 --eval-every 100 --train-frames 8 --per-frame 10 --out liquid_coupling_flow/artifacts/ka3d_cavity_coarse_smoke.pt > reports/logs-2026-07-13/train_coarse_smoke.out 2>&1` — expected: warm-start line reporting ~10 new tensors, loss decreasing over 200 steps, no NaN; `rm` the smoke ckpt after.
- [ ] **Step 3: Commit** `feat(ka3d): --coarse trainer branch + smoke`.

---

### Task 6: Full fine-tune + evaluation gates (run task)

**Files:**
- Create: `reports/logs-2026-07-13/train_coarse.out` (log), `reports/logs-2026-07-13/eval_coarse_gates.py`

- [ ] **Step 1: Launch** (GPU must be free of the rep_prior retrain first):
  `python reports/logs-2026-07-12/train_ebm3d_bigbox.py --coarse --steps 40000 --out liquid_coupling_flow/artifacts/ka3d_cavity_coarse.pt > reports/logs-2026-07-13/train_coarse.out 2>&1` (background). Early-stop patience is inherited (8 evals).
- [ ] **Step 2: Gates** (`eval_coarse_gates.py`, mirrors `probe_cavity_dependence.py` frames):
  - G-NLL: held NLL ≤ **−2.636** (the ebm3ax_rho115 baseline; the joint head removes hedging entropy, so expect better).
  - G-CLASH: sampled-block clash%<0.9σ at pos_temp 0.4 < **15%** (baseline 20.4%; the 3D-informed coarse commitment is the mechanism under test).
  - G-DEFICIT: median MTM `sf−sr` on 8 held cavities (K=8, N=16) improved from **−63** (report the distribution; any systematic shift is signal).
  - G-MASK: with `min_sep=0.85`, clash% strictly below the unmasked coarse number AND round-trip still exact on GPU samples.
- [ ] **Step 3: Commit** logs + gate results; update `reports/` with a short results note; decision point: if G-CLASH/G-DEFICIT pass, rebuild `fm_bank` on the coarse base and retrain the flow corrector against it.

## Self-Review

- Spec coverage: hedged-commitment fix (Tasks 1,3,4), slice-blind tilt fix (Task 2 coarse tilt; fine tilts bounded by cw/2), exact 3D min-sep (Tasks 2,4, gate G-MASK), training path (Tasks 3,5,6), exactness contract (Task 4, Global Constraints). ✓
- Placeholders: Task 2/3/4 steps reference exact existing code lines to copy conventions from rather than inlining 100-line methods — the referenced files/lines are stated precisely; acceptable for this repo where the implementer is expected to read the two named methods. ✓
- Type consistency: `CoarseFineHead` attribute names (`cw`, `bwf`, `half_diag`, `centers`, `head_coarse`, `cell_emb`, `head_fa/fb/fc`, `femb_a/femb_b`) used identically in Tasks 2–4. `KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5)` constructor signature consistent in Tasks 3–5. ✓
