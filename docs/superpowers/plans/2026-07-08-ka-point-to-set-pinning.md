# Point-to-Set Static Length via Random Pinning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the point-to-set (PTS) amorphous-order length ξ_pin(T) of the 2D KA glass by the random-pinning protocol, using the N=100-trained EGNN geometry table as an exact-guarded (Metropolis) species proposal inside a masked constrained sampler, with all learned parts applied zero-shot at N=256/576.

**Architecture:** A new `ka_pin.py` provides masked (frozen-subset) versions of the existing exact kernels — tamed-MALA, paired-reeval swap, conditional-Bernoulli block-relabel — so pinned particles never move while contributing to the full energy. An occupancy-overlap module computes Q(t) on a cell grid; an extractor fits the stretched-exponential plateau Q_∞ and the threshold length ξ with vertical→horizontal error propagation; a gates module enforces correctness, convergence (two-arm + run-length), co-sticking, and drift checks. A campaign driver runs the T×c ladder and aggregates ξ_pin(T).

**Tech Stack:** Python, PyTorch (CUDA), pytest. Reuses `liquid_coupling_flow/ka_energy.py` (`ka_energy`, `ka_forces`), `liquid_coupling_flow/ipl44/ipl_swap_smc.py` (`_tame`, `_pick`, `_swap_log_ratio`, `_cb_sample`, `_cb_logprob`, `uniform_weight_fn`), and `liquid_coupling_flow/ipl44/joint_flow.py` (`JointSpeciesFlow`, `random_22_labeling`).

## Global Constraints

- Cell grid: square cells side **a_c = 0.3·σ_AA = 0.3** (σ_AA=1.0); occupancy n_i ∈ {0,1}; cells containing a pinned particle are excluded from all sums.
- Overlap: `Q(t) = Σ_i⟨n_i(t)n_i(0)⟩ / Σ_i⟨n_i(0)⟩` over included cells; occupancy-based (species-exchange invariant); t = MCMC iterations.
- Random baseline **computed, not fit**: `Q_rand = ρ₀·a_c² = 1.2·0.09 = 0.108`.
- Q_∞ = long-time plateau via stretched-exponential fit `Q(t)=Q_∞+A·exp[-(t/τ)^β]` on the reference-init (decaying) arm; cross-checked vs scrambled-init (rising) arm.
- Pin spacing `ℓ_c = (c·ρ₀)^(-1/2)`; box constraint `L/ℓ_c ≥ 4`.
- ξ_pin(T) = ℓ_c where `Q_∞ - Q_rand` crosses threshold; primary **0.2**, robustness **0.1, 0.3**; growth ξ(0.5)>ξ(0.65)>ξ(0.8) must hold at all three.
- **ξ resolvability**: `δξ_tol = ΔQ/|dQ_∞/dℓ_c|` with ΔQ=0.02; total ξ error = δξ_tol ⊕ bootstrap; growth claimed only where δξ_total < claimed Δξ between adjacent T.
- Learned model: `jf_ka100tt_best.pt`, two-time, **knn=32**, geometry query `t_pos=1, t_spec=0`; applied zero-shot at N=256/576, behind exact Metropolis only.
- Physics: β=1/T, ρ=1.2, `L=(N/1.2)**0.5`; σ/ε = KA matrices in `ka_energy`; report x_B (mobile-set B fraction) alongside every U/N.
- Durability (CLAUDE.md): logs → `reports/logs-<date>/`; per-(T,c) trajectories/masks/fits → `liquid_coupling_flow/artifacts/pts/` incrementally the moment each unit finishes; print full path of every plot; never send stderr to /dev/null.

---

## File Structure

- `liquid_coupling_flow/ka_pin.py` — masked kernels (`masked_mala`, `masked_swap`, `masked_block_relabel`), `pin_mask`, `geometry_table`, `constrained_run` driver.
- `liquid_coupling_flow/ka_pin_overlap.py` — `cell_occupancy`, `overlap_Q`, `q_rand`.
- `liquid_coupling_flow/ka_pin_extract.py` — `stretched_exp_fit`, `xi_threshold`.
- `liquid_coupling_flow/ka_pin_gates.py` — `g_corr`, `g_conv`, `g_stick`, `drift_check`.
- `liquid_coupling_flow/ka_pin_refs.py` — reference-config generation/loading per T.
- `liquid_coupling_flow/ka_pin_campaign.py` — T×c ladder driver + aggregation + figures.
- `liquid_coupling_flow/tests/test_ka_pin.py` — all unit tests for the above.

---

## Task 1: Masked tamed-MALA + pin masks

**Files:**
- Create: `liquid_coupling_flow/ka_pin.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: `ka_energy(x,s,L)`→[B], `ka_forces(x,s,L)`→[B,N,2], `_tame(F,fmax)` from `ipl_swap_smc`.
- Produces: `pin_mask(B,N,c,device,generator=None)`→`mobile` bool[B,N] (True=mobile); `masked_mala(x,s,U,mobile,beta,L,dt,energy_fn,force_fn,fmax=60.0)`→(x,U,acc_rate float).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_ka_pin.py
import torch
from liquid_coupling_flow.ka_pin import pin_mask, masked_mala
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces

def _setup(B=4, N=64, dev="cpu", seed=0):
    torch.manual_seed(seed)
    L = (N / 1.2) ** 0.5
    x = torch.rand(B, N, 2, device=dev) * L
    s = (torch.rand(B, N, device=dev) < 0.37).long()
    return x, s, L

def test_pin_mask_count_and_frozen_fixed():
    x, s, L = _setup()
    mobile = pin_mask(4, 64, c=0.25, device="cpu", generator=torch.Generator().manual_seed(1))
    assert mobile.dtype == torch.bool and mobile.shape == (4, 64)
    # exactly ceil(c*N)=16 pinned per row
    assert int((~mobile[0]).sum()) == 16
    U = ka_energy(x, s, L)
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
    x2, U2, acc = masked_mala(x, s, U, mobile, beta=2.0, L=L, dt=0.01, energy_fn=efn, force_fn=ffn)
    # frozen particles are bit-for-bit unchanged
    assert torch.equal(x2[~mobile], x[~mobile])
    # energy recomputed from x2 equals the tracked U2 (accept bookkeeping is exact)
    assert torch.allclose(ka_energy(x2, s, L), U2, atol=1e-4)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_pin_mask_count_and_frozen_fixed -q`
Expected: FAIL (ModuleNotFoundError: `ka_pin`).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_pin.py
"""Random-pinning point-to-set: masked (frozen-subset) exact kernels + constrained-run driver.
Pinned particles never move but contribute to the full KA energy; every learned component is behind
exact Metropolis. See docs/superpowers/specs/2026-07-08-ka-point-to-set-pinning-design.md."""
import math, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import _tame, _pick, _swap_log_ratio, _cb_sample, _cb_logprob
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces


def pin_mask(B, N, c, device, generator=None):
    """Random pinning: freeze ceil(c*N) particles per config. Returns mobile bool[B,N] (True = mobile)."""
    n_pin = math.ceil(c * N)
    mobile = torch.ones(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        perm = torch.randperm(N, generator=generator, device=device if generator is None else generator.device)
        mobile[b, perm[:n_pin].to(device)] = False
    return mobile


def masked_mala(x, s, U, mobile, beta, L, dt, energy_fn, force_fn, fmax=60.0):
    """Tamed-MALA restricted to mobile particles: drift & noise are masked to zero on frozen sites, so their
    proposal residuals vanish and the MH ratio involves only mobile DOFs. Frozen contribute to energy/forces.
    Returns (x, U, acc_rate)."""
    m = mobile[..., None].to(x.dtype)                                  # [B,N,1]
    F = _tame(force_fn(x, s), fmax) * m
    mu = x + 0.5 * dt * dt * beta * F
    xp = torch.remainder(mu + dt * torch.randn_like(x) * m, L)         # frozen: mu=x, noise=0 -> xp=x
    Up = energy_fn(xp, s)
    Fp = _tame(force_fn(xp, s), fmax) * m
    mup = xp + 0.5 * dt * dt * beta * Fp
    d_f = xp - mu; d_f = d_f - L * torch.round(d_f / L)                # frozen residual = 0
    d_r = x - mup; d_r = d_r - L * torch.round(d_r / L)
    logq = (-(d_r ** 2).sum((1, 2)) + (d_f ** 2).sum((1, 2))) / (2 * dt * dt)
    log_ratio = -beta * (Up - U) + logq
    acc = torch.log(torch.rand(x.shape[0], device=x.device)) < log_ratio
    x = torch.where(acc[:, None, None], xp, x); U = torch.where(acc, Up, U)
    return x, U, float(acc.float().mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_pin_mask_count_and_frozen_fixed -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): masked tamed-MALA + random pin masks"
```

---

## Task 2: Masked exact species moves (swap + block-relabel) + geometry table

**Files:**
- Modify: `liquid_coupling_flow/ka_pin.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: `_pick`, `_swap_log_ratio`, `_cb_sample`, `_cb_logprob`, `uniform_weight_fn` from `ipl_swap_smc`; `JointSpeciesFlow`, `random_22_labeling` from `ipl44.joint_flow`.
- Produces: `masked_swap(x,s,U,mobile,beta,energy_fn,weight_fn)`→(s,U,acc bool[B]); `masked_block_relabel(x,s,U,mobile,beta,energy_fn,table_fn,k)`→(s,U,acc bool[B]); `load_geometry_table(N,L,nB,ckpt_path,device)`→`table_fn(x)->[B,N]` (P(B), s-independent).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
from liquid_coupling_flow.ka_pin import masked_swap, masked_block_relabel
from liquid_coupling_flow.ipl44.ipl_swap_smc import uniform_weight_fn

def test_masked_species_moves_keep_frozen_and_composition():
    x, s, L = _setup(seed=2)
    mobile = pin_mask(4, 64, 0.25, "cpu", generator=torch.Generator().manual_seed(3))
    efn = lambda a, b: ka_energy(a, b, L)
    U = ka_energy(x, s, L)
    s0 = s.clone()
    s1, U1, acc = masked_swap(x, s, U, mobile, 2.0, efn, uniform_weight_fn)
    assert torch.equal(s1[~mobile], s0[~mobile])                 # frozen species untouched
    assert torch.equal(s1.sum(1), s0.sum(1))                     # swap conserves count
    assert torch.allclose(ka_energy(x, s1, L), U1, atol=1e-4)
    # block-relabel with a constant table (uncertain everywhere) -> exact, frozen fixed, count preserved
    table_fn = lambda xx: torch.full((xx.shape[0], xx.shape[1]), 0.4)
    s2, U2, acc2 = masked_block_relabel(x, s0, U, mobile, 2.0, efn, table_fn, k=4)
    assert torch.equal(s2[~mobile], s0[~mobile])
    assert torch.equal(s2.sum(1), s0.sum(1))
    assert torch.allclose(ka_energy(x, s2, L), U2, atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_masked_species_moves_keep_frozen_and_composition -q`
Expected: FAIL (ImportError: `masked_swap`).

- [ ] **Step 3: Write minimal implementation** (append to `ka_pin.py`)

```python
def masked_swap(x, s, U, mobile, beta, energy_fn, weight_fn):
    """Paired-reeval MH swap of one mobile-A with one mobile-B per config (frozen never picked). Exact."""
    B = x.shape[0]; ar = torch.arange(B, device=x.device)
    pB_f = weight_fn(x, s); dt = pB_f.dtype
    isA = ((s == 0) & mobile).to(dt); isB = ((s == 1) & mobile).to(dt)
    i = _pick(pB_f, isA); j = _pick(1.0 - pB_f, isB)
    s_prop = s.clone(); s_prop[ar, i] = 1; s_prop[ar, j] = 0
    U_prop = energy_fn(x, s_prop)
    pB_r = weight_fn(x, s_prop)
    log_ratio = -beta * (U_prop - U) + _swap_log_ratio(pB_f, pB_r, s, s_prop, i, j)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    return torch.where(acc[:, None], s_prop, s), torch.where(acc, U_prop, U), acc


def masked_block_relabel(x, s, U, mobile, beta, energy_fn, table_fn, k):
    """Gumbel-top-k block relabel over MOBILE sites only (score=-inf on frozen), conditional-Bernoulli redraw
    preserving the block count. x-only randomized selection cancels in the MH ratio -> exact."""
    B, N = s.shape
    W = table_fn(x).clamp(1e-6, 1 - 1e-6)
    score = -((W - 0.5).abs() + 1e-3).log()
    score = score.masked_fill(~mobile, -1e30)                        # frozen never selected
    gumbel = -torch.log(-torch.log(torch.rand_like(score) + 1e-12) + 1e-12)
    blk = torch.argsort(score + gumbel, dim=1, descending=True)[:, :k]
    wblk = W.gather(1, blk); sblk = s.gather(1, blk)
    m = sblk.sum(1).long()
    sblk_new = _cb_sample(wblk, m)
    log_ratio = -beta * (energy_fn(x, s.scatter(1, blk, sblk_new)) - U) \
        + _cb_logprob(wblk, sblk) - _cb_logprob(wblk, sblk_new)
    s_prop = s.scatter(1, blk, sblk_new)
    U_prop = energy_fn(x, s_prop)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    return torch.where(acc[:, None], s_prop, s), torch.where(acc, U_prop, U), acc


def load_geometry_table(N, L, nB, ckpt_path, device):
    """Load the N=100-trained two-time joint flow (knn=32) and return table_fn(x)->P(species=B) [B,N], which
    is s-INDEPENDENT (queried at t_pos=1, t_spec=0 with the canonical labelling) -> valid block-relabel table."""
    from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow
    ck = torch.load(ckpt_path, map_location=device, weights_only=False); cfg = ck["cfg"]
    jf = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"],
                          two_time=True).to(device)
    jf.load_state_dict(ck["state_dict"]); jf.eval(); jf.knn = 32
    s_canon = torch.zeros(1, N, dtype=torch.long, device=device); s_canon[:, :nB] = 1

    def table_fn(x):
        with torch.no_grad():
            Bx = x.shape[0]
            _, lg = jf(torch.ones(Bx, 1, 1, device=device), x, s_canon.expand(Bx, -1),
                       t_spec=torch.zeros(Bx, 1, 1, device=device))
            return torch.softmax(lg, -1)[..., 1]
    return table_fn
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_masked_species_moves_keep_frozen_and_composition -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): masked exact swap + block-relabel + geometry-table loader"
```

---

## Task 3: Occupancy overlap Q(t) on a cell grid

**Files:**
- Create: `liquid_coupling_flow/ka_pin_overlap.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Produces: `cell_occupancy(x,L,a_c=0.3)`→occ bool[B,ncell] (ncell = round(L/a_c)², row-major cell index); `q_rand(a_c=0.3,rho=1.2)`→float; `overlap_Q(occ_t, occ_ref, excluded)`→Q float, where `excluded` bool[B,ncell] marks cells to drop (cells holding a pinned particle in the reference).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, q_rand, overlap_Q

def test_overlap_self_is_one_and_qrand():
    assert abs(q_rand(0.3, 1.2) - 0.108) < 1e-9
    x, s, L = _setup(seed=5)
    occ = cell_occupancy(x, L, a_c=0.3)
    excluded = torch.zeros_like(occ)
    # Q(ref, ref) over all cells = 1 exactly
    assert abs(overlap_Q(occ, occ, excluded) - 1.0) < 1e-9
    # occupancy is species-blind: relabelling s must not change occ
    occ2 = cell_occupancy(x, L, a_c=0.3)
    assert torch.equal(occ, occ2)
    # a fully-decorrelated config has Q near q_rand (loose bound, statistical)
    torch.manual_seed(9); xr = torch.rand_like(x) * L
    q = overlap_Q(cell_occupancy(xr, L), occ, excluded)
    assert 0.0 <= q <= 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_overlap_self_is_one_and_qrand -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_pin_overlap.py
"""Occupancy overlap Q(t) on a square cell grid for point-to-set. n_i in {0,1}; occupancy-based so Q is
invariant under species exchange (Berthier-Kob definition). Cells holding a pinned particle are excluded."""
import torch


def _ncells(L, a_c):
    return max(1, int(round(L / a_c)))


def cell_occupancy(x, L, a_c=0.3):
    """x [B,N,2] in [0,L)^2 -> occ bool [B, g*g] (g=round(L/a_c)); True if >=1 particle in the cell."""
    B, N, _ = x.shape
    g = _ncells(L, a_c)
    xi = torch.remainder(x, L)
    ci = torch.clamp((xi / L * g).long(), 0, g - 1)                    # [B,N,2] cell coords
    idx = ci[..., 0] * g + ci[..., 1]                                  # [B,N] flat cell index
    occ = torch.zeros(B, g * g, dtype=torch.bool, device=x.device)
    occ.scatter_(1, idx, torch.ones_like(idx, dtype=torch.bool))
    return occ


def pinned_cells(x, mobile, L, a_c=0.3):
    """Cells containing >=1 pinned particle in config x -> excluded bool [B, g*g]."""
    g = _ncells(L, a_c)
    xi = torch.remainder(x, L)
    ci = torch.clamp((xi / L * g).long(), 0, g - 1)
    idx = (ci[..., 0] * g + ci[..., 1])
    excl = torch.zeros(x.shape[0], g * g, dtype=torch.bool, device=x.device)
    frozen = ~mobile
    for b in range(x.shape[0]):
        excl[b].scatter_(0, idx[b][frozen[b]], torch.ones(int(frozen[b].sum()), dtype=torch.bool, device=x.device))
    return excl


def q_rand(a_c=0.3, rho=1.2):
    """Uncorrelated-occupancy baseline Q_rand = rho * a_c^2."""
    return rho * a_c * a_c


def overlap_Q(occ_t, occ_ref, excluded):
    """Q = sum_i n_i(t) n_i(0) / sum_i n_i(0) over NON-excluded cells, averaged over the batch."""
    keep = ~excluded
    num = (occ_t & occ_ref & keep).sum(1).float()
    den = (occ_ref & keep).sum(1).float().clamp_min(1.0)
    return float((num / den).mean())
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_overlap_self_is_one_and_qrand -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin_overlap.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): occupancy overlap Q(t) on cell grid + pinned-cell exclusion"
```

---

## Task 4: Constrained-run driver (two-arm init, record Q(t), incremental save)

**Files:**
- Modify: `liquid_coupling_flow/ka_pin.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: `masked_mala`, `masked_swap`, `masked_block_relabel`, `pin_mask` (Tasks 1-2); `cell_occupancy`, `pinned_cells`, `overlap_Q` (Task 3); `random_22_labeling` from `ipl44.joint_flow`.
- Produces: `constrained_run(x_ref, s_ref, mobile, T, L, n_iter, table_fn=None, dt=0.01, record_every=5, arm="ref", scramble_x=None)`→dict with keys `t`[list], `Q`[list], `U`[list], `x_final`, `s_final`, `arm`, plus `x_ref`,`mobile` echoed. `arm="ref"` starts mobile at reference positions; `arm="scramble"` starts them at `scramble_x`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
from liquid_coupling_flow.ka_pin import constrained_run

def test_constrained_run_two_arms_frozen_fixed():
    x, s, L = _setup(B=8, N=100, seed=7)
    mobile = pin_mask(8, 100, 0.2, "cpu", generator=torch.Generator().manual_seed(8))
    torch.manual_seed(1); scr = torch.rand_like(x) * L
    ref = constrained_run(x, s, mobile, T=0.8, L=L, n_iter=30, dt=0.01, record_every=5, arm="ref")
    scb = constrained_run(x, s, mobile, T=0.8, L=L, n_iter=30, dt=0.01, record_every=5,
                          arm="scramble", scramble_x=scr)
    # frozen positions never move in either arm
    assert torch.equal(ref["x_final"][~mobile], x[~mobile])
    assert torch.equal(scb["x_final"][~mobile], x[~mobile])
    # Q(t) recorded; ref arm starts at 1 (mobile at reference), scramble arm starts below 1
    assert ref["Q"][0] > 0.95 and scb["Q"][0] < ref["Q"][0]
    assert len(ref["t"]) == len(ref["Q"]) == len(ref["U"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_constrained_run_two_arms_frozen_fixed -q`
Expected: FAIL (ImportError).

- [ ] **Step 3: Write minimal implementation** (append to `ka_pin.py`)

```python
def constrained_run(x_ref, s_ref, mobile, T, L, n_iter, table_fn=None, dt=0.01,
                    record_every=5, arm="ref", scramble_x=None, n_swap=4, k_block=8, fmax=60.0):
    """Re-equilibrate mobile particles among frozen pins; record occupancy overlap Q(t) vs x_ref.
    arm='ref': mobile start at reference positions (Q decays from 1); arm='scramble': mobile start at
    scramble_x (Q rises). Species channel: masked_swap (+ masked_block_relabel if table_fn given), both exact.
    Learned part (table_fn) enters block-relabel only, behind MH."""
    from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q
    beta = 1.0 / T
    x = x_ref.clone(); s = s_ref.clone()
    if arm == "scramble":
        assert scramble_x is not None
        m3 = mobile[..., None]
        x = torch.where(m3, torch.remainder(scramble_x, L), x)          # only mobile scrambled
    efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
    U = efn(x, s)
    occ_ref = cell_occupancy(x_ref, L); excl = pinned_cells(x_ref, mobile, L)
    rec = {"t": [], "Q": [], "U": [], "arm": arm}
    for it in range(n_iter + 1):
        if it % record_every == 0:
            rec["t"].append(it)
            rec["Q"].append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            rec["U"].append(float((U / x.shape[1]).median()))
        if it == n_iter:
            break
        x, U, _ = masked_mala(x, s, U, mobile, beta, L, dt, efn, ffn, fmax)
        for _ in range(n_swap):
            s, U, _ = masked_swap(x, s, U, mobile, beta, efn, uniform_weight_fn)
        if table_fn is not None:
            s, U, _ = masked_block_relabel(x, s, U, mobile, beta, efn, table_fn, k_block)
    rec["x_final"] = x; rec["s_final"] = s; rec["x_ref"] = x_ref; rec["mobile"] = mobile
    return rec
```

Add the missing import at the top of `ka_pin.py` (with the other `ipl_swap_smc` imports): `uniform_weight_fn`.

```python
from liquid_coupling_flow.ipl44.ipl_swap_smc import (
    _tame, _pick, _swap_log_ratio, _cb_sample, _cb_logprob, uniform_weight_fn)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_constrained_run_two_arms_frozen_fixed -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): constrained-run driver with two-arm init and Q(t) recording"
```

---

## Task 5: Extractors — stretched-exp Q_∞ and threshold ξ with error propagation

**Files:**
- Create: `liquid_coupling_flow/ka_pin_extract.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: `scipy.optimize.curve_fit`, numpy.
- Produces: `stretched_exp_fit(t,Q)`→dict `{Qinf,A,tau,beta,ok}` (fit `Q=Qinf+A*exp[-(t/tau)**beta]`); `xi_threshold(lvals,Qinf,Qinf_err,thr,Qrand=0.108,dQ_tol=0.02)`→dict `{xi,dxi,kind}` where `kind∈{"point","range","none"}`, `dxi` combines the vertical→horizontal tolerance `dQ_tol/|slope|` with bootstrap-able `Qinf_err/|slope|`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
import numpy as np
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit, xi_threshold

def test_stretched_exp_recovers_plateau():
    t = np.arange(0, 200, 5.0)
    Q = 0.35 + 0.6 * np.exp(-(t / 30.0) ** 0.7)
    fit = stretched_exp_fit(t, Q)
    assert fit["ok"] and abs(fit["Qinf"] - 0.35) < 0.02

def test_xi_threshold_and_error_propagation():
    lvals = np.array([1.5, 2.0, 2.5, 3.0, 3.5])
    excess = np.array([0.50, 0.35, 0.22, 0.12, 0.05])           # Qinf - Qrand, monotone decreasing
    Qinf = excess + 0.108
    out = xi_threshold(lvals, Qinf, Qinf_err=np.full(5, 0.01), thr=0.2, Qrand=0.108, dQ_tol=0.02)
    assert out["kind"] == "point" and 2.0 < out["xi"] < 2.5
    # near-flat curve inflates dxi (loose vertical tol -> large horizontal error)
    flat = np.array([0.30, 0.27, 0.24, 0.22, 0.205]) + 0.108
    o2 = xi_threshold(lvals, flat, np.full(5, 0.01), thr=0.2, Qrand=0.108, dQ_tol=0.02)
    assert o2["dxi"] > out["dxi"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py -k "stretched or xi_threshold" -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_pin_extract.py
"""Extractors for point-to-set: stretched-exponential plateau Q_inf and threshold length xi with
vertical->horizontal error propagation (dxi = dQ / |dQ_inf/dl|)."""
import numpy as np
from scipy.optimize import curve_fit


def _sexp(t, Qinf, A, tau, beta):
    return Qinf + A * np.exp(-((t / np.maximum(tau, 1e-9)) ** beta))


def stretched_exp_fit(t, Q):
    t = np.asarray(t, float); Q = np.asarray(Q, float)
    p0 = [Q[-1], max(Q[0] - Q[-1], 1e-3), max(t[-1] / 3, 1.0), 0.7]
    bounds = ([0.0, 0.0, 1e-3, 0.2], [1.0, 1.5, 1e6, 1.5])
    try:
        p, _ = curve_fit(_sexp, t, Q, p0=p0, bounds=bounds, maxfev=20000)
        return {"Qinf": float(p[0]), "A": float(p[1]), "tau": float(p[2]), "beta": float(p[3]), "ok": True}
    except Exception:
        return {"Qinf": float(Q[-1]), "A": 0.0, "tau": float(t[-1]), "beta": 1.0, "ok": False}


def xi_threshold(lvals, Qinf, Qinf_err, thr, Qrand=0.108, dQ_tol=0.02):
    """xi = l where (Qinf - Qrand) crosses thr, by linear interpolation of the excess vs l.
    dxi = (dQ_tol (+) Qinf_err_local) / |local slope|. kind='range' if multiple crossings."""
    l = np.asarray(lvals, float); e = np.asarray(Qinf, float) - Qrand
    err = np.asarray(Qinf_err, float)
    crossings = []
    for i in range(len(l) - 1):
        a, b = e[i] - thr, e[i + 1] - thr
        if a == 0.0:
            crossings.append((l[i], i))
        if a * b < 0:                                            # sign change between i and i+1
            frac = a / (a - b)
            xi = l[i] + frac * (l[i + 1] - l[i]); crossings.append((xi, i))
    if not crossings:
        return {"xi": None, "dxi": None, "kind": "none"}
    def dxi_at(i):
        slope = abs((e[i + 1] - e[i]) / (l[i + 1] - l[i])) if i + 1 < len(l) else 1e-9
        slope = max(slope, 1e-9)
        eloc = 0.5 * (err[i] + err[min(i + 1, len(l) - 1)])
        return float(np.hypot(dQ_tol, eloc) / slope)
    if len(crossings) == 1:
        xi, i = crossings[0]
        return {"xi": float(xi), "dxi": dxi_at(i), "kind": "point"}
    xis = [c[0] for c in crossings]
    return {"xi": (float(min(xis)), float(max(xis))), "dxi": max(dxi_at(c[1]) for c in crossings),
            "kind": "range"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py -k "stretched or xi_threshold" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin_extract.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): stretched-exp Qinf + threshold xi with error propagation"
```

---

## Task 6: Gates — G-corr, G-conv (two-arm + run-length), G-stick, drift

**Files:**
- Create: `liquid_coupling_flow/ka_pin_gates.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: `stretched_exp_fit` (Task 5).
- Produces: `g_conv(ref_run, scr_run, tol=0.02)`→dict `{passed,q_ref,q_scr,gap,tau,run_ok}` (run_ok requires `t_max ≥ 3*tau` on both arms); `g_corr(learned_run, brute_run, tol=0.02)`→dict `{passed,gap}`; `g_stick(ref_run, third_run, ext_run, tol=0.02)`→dict `{passed,gap_third,gap_ext}`; `drift_check(occ_registered_Q, occ_std_Q, tol=0.02)`→dict `{flagged,disparity}`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
from liquid_coupling_flow.ka_pin_gates import g_conv, g_corr, g_stick

def _run(qinf, tau, tmax, n=40, arm="ref"):
    t = np.linspace(0, tmax, n)
    sign = 1.0 if arm == "ref" else -1.0
    Q = qinf + sign * 0.4 * np.exp(-(t / tau) ** 0.8)
    return {"t": list(t), "Q": list(Q), "arm": arm}

def test_gconv_needs_agreement_and_run_length():
    good_ref = _run(0.30, 20, 200, arm="ref"); good_scr = _run(0.30, 20, 200, arm="scramble")
    r = g_conv(good_ref, good_scr, tol=0.02)
    assert r["passed"] and r["run_ok"]
    # same plateaus but run too short (tmax < 3 tau) -> run_ok False -> gate fails
    short_ref = _run(0.30, 100, 150, arm="ref"); short_scr = _run(0.30, 100, 150, arm="scramble")
    assert not g_conv(short_ref, short_scr, tol=0.02)["passed"]
    # disagreeing plateaus -> fail
    assert not g_conv(_run(0.30, 20, 200, "ref"), _run(0.36, 20, 200, "scramble"), tol=0.02)["passed"]

def test_gstick_and_gcorr():
    a = _run(0.30, 20, 200, "ref"); b = _run(0.30, 20, 200, "scramble"); c = _run(0.30, 20, 400, "ref")
    assert g_stick(a, b, c, tol=0.02)["passed"]
    assert g_corr(_run(0.30, 20, 200), _run(0.31, 20, 200), tol=0.02)["passed"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py -k "gconv or gstick" -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_pin_gates.py
"""Blocking gates for point-to-set (in order): G-corr (correctness vs brute-force at an easy point),
G-conv (two-arm agreement AND run-length >= 3 tau), G-stick (third-init + 2x extension at the binding point),
drift_check (long-wavelength contamination). Failure of G-conv => report the point as a BOUND, never fit."""
import numpy as np
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit


def _qinf_tau(run):
    f = stretched_exp_fit(run["t"], run["Q"]); return f["Qinf"], f["tau"]


def g_conv(ref_run, scr_run, tol=0.02):
    q_ref, tau_r = _qinf_tau(ref_run); q_scr, tau_s = _qinf_tau(scr_run)
    tmax = min(max(ref_run["t"]), max(scr_run["t"]))
    run_ok = (tmax >= 3.0 * tau_r) and (tmax >= 3.0 * tau_s)         # Qinf<->tau degeneracy guard
    gap = abs(q_ref - q_scr)
    return {"passed": bool(run_ok and gap <= tol), "q_ref": q_ref, "q_scr": q_scr,
            "gap": float(gap), "tau": float(max(tau_r, tau_s)), "run_ok": bool(run_ok)}


def g_corr(learned_run, brute_run, tol=0.02):
    ql, _ = _qinf_tau(learned_run); qb, _ = _qinf_tau(brute_run)
    return {"passed": bool(abs(ql - qb) <= tol), "gap": float(abs(ql - qb))}


def g_stick(ref_run, third_run, ext_run, tol=0.02):
    """Binding-point co-sticking: a THIRD init (different equilibrium reference) and a 2x-extended run must
    reach the same plateau as the reference arm; two arms can co-stick at a wrong plateau in a glass."""
    q_ref, _ = _qinf_tau(ref_run); q_third, _ = _qinf_tau(third_run); q_ext, _ = _qinf_tau(ext_run)
    g3, ge = abs(q_ref - q_third), abs(q_ref - q_ext)
    return {"passed": bool(g3 <= tol and ge <= tol), "gap_third": float(g3), "gap_ext": float(ge)}


def drift_check(occ_registered_Q, occ_std_Q, tol=0.02):
    """Long-wavelength spot-check at largest l_c: disparity between locally-registered and standard Q_inf."""
    d = abs(occ_registered_Q - occ_std_Q)
    return {"flagged": bool(d > tol), "disparity": float(d)}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py -k "gconv or gstick" -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin_gates.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): blocking gates G-corr/G-conv(run-length)/G-stick/drift"
```

---

## Task 7: Reference configurations per temperature

**Files:**
- Create: `liquid_coupling_flow/ka_pin_refs.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: `swap_mcmc_fast` from `liquid_coupling_flow.ka_mcmc_fast`; `make_species` from `liquid_coupling_flow.ka_mcmc`; `ka_energy`; existing PT2 dataset `liquid_coupling_flow/artifacts/pt_ladder_hb_N256.pt` (T=0.5).
- Produces: `get_references(T, N, n_configs, device, out_dir)`→dict `{x:[n,N,2], s:[N], L, T, xB}`; for T=0.5 loads PT2, else generates by swap MC; saves to `out_dir/refs_T{T}_N{N}.pt`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
import os
from liquid_coupling_flow.ka_pin_refs import get_references

def test_get_references_highT_smoke(tmp_path):
    r = get_references(T=0.8, N=100, n_configs=8, device="cpu", out_dir=str(tmp_path))
    assert r["x"].shape == (8, 100, 2) and r["s"].shape == (100,)
    assert 0.30 < r["xB"] < 0.45                                    # ~0.37 nominal
    assert os.path.exists(os.path.join(str(tmp_path), "refs_T0.8_N100.pt"))
    # equilibrated high-T energy is well below the ideal-gas 0 and finite
    from liquid_coupling_flow.ka_energy import ka_energy
    u = float((ka_energy(r["x"], r["s"], r["L"]) / 100).median()); assert -4.0 < u < -1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_get_references_highT_smoke -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_pin_refs.py
"""Equilibrium reference configs per temperature for point-to-set. T=0.5 loaded from the PT2 dataset;
higher T generated by swap MC (cheap and well-equilibrated above the supercooled regime). x_B recorded."""
import os, torch
from liquid_coupling_flow.ka_mcmc import make_species
from liquid_coupling_flow.ka_mcmc_fast import swap_mcmc_fast
from liquid_coupling_flow.ka_energy import ka_energy

PT2_N256 = "liquid_coupling_flow/artifacts/pt_ladder_hb_N256.pt"


def get_references(T, N, n_configs, device, out_dir, fracB=0.35):
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"refs_T{T}_N{N}.pt")
    L = (N / 1.2) ** 0.5
    if abs(T - 0.5) < 1e-9 and N == 256 and os.path.exists(PT2_N256):
        d = torch.load(PT2_N256, map_location="cpu", weights_only=False)
        x = d["x"][:n_configs].float(); s = d["s"].long()
    else:
        s = make_species(N, fracB).to(device)
        x, s = swap_mcmc_fast(N, L, T, s, device=device, n_chains=n_configs,
                              n_equil=4000, n_collect=1, every=1, step=0.05)
        x = x.cpu(); s = s.cpu()
    res = {"x": x, "s": s, "L": L, "T": T, "N": N, "xB": float((s == 1).float().mean())}
    torch.save(res, out)
    return res
```

Note: confirm `swap_mcmc_fast`'s return shape during implementation — the test asserts `x.shape==(8,100,2)`; if `swap_mcmc_fast` returns extra collected frames, slice `x[:n_configs]` before saving. Adjust the call's `n_collect/every` so it yields exactly `n_configs` chains, matching the observed signature.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_get_references_highT_smoke -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka_pin_refs.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): per-temperature equilibrium reference configs"
```

---

## Task 8: Campaign driver — T×c ladder, aggregation, ξ_pin(T) figure

**Files:**
- Create: `liquid_coupling_flow/ka_pin_campaign.py`
- Test: `liquid_coupling_flow/tests/test_ka_pin.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_cell(T, c, refs, n_iter, table_fn, device, out_dir, n_real=16)`→dict `{T,c,lc,Qinf,Qinf_err,gconv,xB,arms}` (runs ref+scramble arms batched over `n_real` realizations, applies `g_conv`, extracts Qinf via `stretched_exp_fit` on the ref arm, saves per-cell `.pt` incrementally); `aggregate(cells, thresholds=(0.1,0.2,0.3))`→per-threshold `{T:xi_result}` using `xi_threshold`; `main()` CLI running the spec's ladder and writing `artifacts/pts/xi_of_T.png` + `.pt`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_ka_pin.py
from liquid_coupling_flow.ka_pin_campaign import run_cell, aggregate

def test_run_cell_and_aggregate_smoke(tmp_path):
    from liquid_coupling_flow.ka_pin_refs import get_references
    refs = get_references(0.8, 100, 8, "cpu", str(tmp_path))
    cell = run_cell(T=0.8, c=0.16, refs=refs, n_iter=20, table_fn=None, device="cpu",
                    out_dir=str(tmp_path), n_real=8)
    assert set(["T", "c", "lc", "Qinf", "gconv", "xB"]).issubset(cell.keys())
    assert cell["lc"] > 0
    # aggregate a synthetic monotone set of cells into a xi at threshold 0.2
    cells = [{"T": 0.8, "lc": lc, "Qinf": q, "Qinf_err": 0.01}
             for lc, q in zip([1.5, 2.0, 2.5, 3.0], [0.55, 0.38, 0.22, 0.12])]
    agg = aggregate(cells, thresholds=(0.2,))
    assert 0.8 in {round(t, 3) for t in agg[0.2].keys()}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_run_cell_and_aggregate_smoke -q`
Expected: FAIL (ModuleNotFoundError).

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ka_pin_campaign.py
"""Point-to-set campaign: run the T x c ladder (random pinning), gate each cell, extract xi_pin(T).
All learned parts N=100-trained, zero-shot at N>=256, behind exact Metropolis. Saves per-cell trajectories
incrementally and the final xi(T) figure."""
import os, math, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_pin import pin_mask, constrained_run
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit, xi_threshold
from liquid_coupling_flow.ka_pin_gates import g_conv
from liquid_coupling_flow.ka_pin_overlap import q_rand

RHO = 1.2


def run_cell(T, c, refs, n_iter, table_fn, device, out_dir, n_real=16, dt=0.01, record_every=5):
    os.makedirs(out_dir, exist_ok=True)
    x = refs["x"][:n_real].to(device); s = refs["s"].to(device)
    if s.dim() == 1:
        s = s[None].expand(n_real, -1).contiguous()
    N = x.shape[1]; L = refs["L"]
    gen = torch.Generator(device=device); gen.manual_seed(hash((round(T, 3), round(c, 3))) % (2**31))
    mobile = pin_mask(n_real, N, c, device, generator=gen)
    torch.manual_seed(12345); scr = torch.rand_like(x) * L
    ref_run = constrained_run(x, s, mobile, T, L, n_iter, table_fn, dt, record_every, arm="ref")
    scr_run = constrained_run(x, s, mobile, T, L, n_iter, table_fn, dt, record_every,
                              arm="scramble", scramble_x=scr)
    gc = g_conv(ref_run, scr_run)
    fit = stretched_exp_fit(ref_run["t"], ref_run["Q"])
    lc = (c * RHO) ** -0.5
    cell = {"T": T, "c": c, "lc": lc, "Qinf": fit["Qinf"], "Qinf_err": 0.01,
            "gconv": gc, "xB": float((s[mobile] == 1).float().mean()),
            "arms": {"ref": ref_run, "scramble": scr_run}}
    torch.save(cell, os.path.join(out_dir, f"cell_T{T}_c{c}.pt"))       # incremental save
    print(f"[pts] T={T} c={c} lc={lc:.2f} Qinf={fit['Qinf']:.3f} "
          f"gconv={'PASS' if gc['passed'] else 'BOUND'} -> {os.path.join(out_dir, f'cell_T{T}_c{c}.pt')}",
          flush=True)
    return cell


def aggregate(cells, thresholds=(0.1, 0.2, 0.3), Qrand=None):
    Qrand = q_rand() if Qrand is None else Qrand
    by_T = {}
    for cl in cells:
        by_T.setdefault(cl["T"], []).append(cl)
    out = {}
    for thr in thresholds:
        out[thr] = {}
        for T, cs in by_T.items():
            cs = sorted(cs, key=lambda z: z["lc"])
            lv = np.array([z["lc"] for z in cs]); qi = np.array([z["Qinf"] for z in cs])
            qe = np.array([z.get("Qinf_err", 0.01) for z in cs])
            out[thr][T] = xi_threshold(lv, qi, qe, thr, Qrand=Qrand)
    return out


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = "liquid_coupling_flow/artifacts/pts"; os.makedirs(out_dir, exist_ok=True)
    from liquid_coupling_flow.ka_pin import load_geometry_table
    from liquid_coupling_flow.ka_pin_refs import get_references
    ladder = {256: [0.24, 0.16, 0.12, 0.08], 576: [0.06, 0.04]}
    temps = [0.8, 0.65, 0.5]
    cells = []
    for T in temps:
        for N, cs in ladder.items():
            if N == 576 and T == 0.5:
                continue                                             # stretch goal (see spec §6)
            refs = get_references(T, N, 16, dev, out_dir)
            nB = int((refs["s"] == 1).sum()) if refs["s"].dim() == 1 else int(refs["s"][0].sum())
            table_fn = load_geometry_table(N, refs["L"], nB,
                                           "liquid_coupling_flow/ipl44/data/jf_ka100tt_best.pt", dev)
            for c in cs:
                cells.append(run_cell(T, c, refs, n_iter=3000, table_fn=table_fn, device=dev, out_dir=out_dir))
    agg = aggregate(cells)
    torch.save({"cells": cells, "agg": agg}, os.path.join(out_dir, "pts_summary.pt"))
    fig, ax = plt.subplots(figsize=(6, 4.4))
    for thr, mk in zip((0.1, 0.2, 0.3), ("o", "s", "^")):
        Ts = sorted(agg[thr].keys())
        xis = [agg[thr][T]["xi"] if isinstance(agg[thr][T]["xi"], float) else np.nan for T in Ts]
        dxis = [agg[thr][T]["dxi"] or 0 for T in Ts]
        ax.errorbar(Ts, xis, yerr=dxis, marker=mk, label=f"thr={thr}", capsize=3)
    ax.set_xlabel("T"); ax.set_ylabel(r"$\xi_{pin}$"); ax.legend(); ax.set_title("Point-to-set length vs T")
    p = os.path.join(out_dir, "xi_of_T.png"); fig.tight_layout(); fig.savefig(p, dpi=130)
    print(f"[pts] SAVED {os.path.abspath(p)}", flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py::test_run_cell_and_aggregate_smoke -q`
Expected: PASS.

- [ ] **Step 5: Full-suite check + commit**

Run: `cd /mnt/ssd/GridTransformer && python -m pytest liquid_coupling_flow/tests/test_ka_pin.py -q`
Expected: all PASS.

```bash
git add liquid_coupling_flow/ka_pin_campaign.py liquid_coupling_flow/tests/test_ka_pin.py
git commit -m "feat(pts): campaign driver, aggregation, xi_pin(T) figure"
```

---

## Deferred (out of scope for v1; see spec §7)

- **Learned position accelerator (heat-bath):** slot the β-conditioned full-cage heat-bath into `constrained_run` behind the same masked-MH interface as the learned *position* proposal; measure interior-mixing speedup vs masked-MALA. This is the documented v2 trigger and the thing that would make the learned kernel accelerate Q (occupancy overlap is species-invariant, so the v1 geometry-table move does not speed Q — expected mixing ≈ 1×).
- **Cavity geometry (v1.5):** add `cavity_mask(center,R)` to `ka_pin.py` (same `mobile` interface) and an exponential-decay ξ extractor; the frozen-mask harness and all gates are reused unchanged.
- **G-stick / drift wiring in `main()`:** run `g_stick` at the smallest passing c at T=0.5 (third reference + 2× extension) and `drift_check` at the largest ℓ_c; both gate functions exist (Task 6) — wire them into a follow-up campaign pass once the base ladder is measured.

---

## Self-Review

**1. Spec coverage:**
- Definitions (a_c, Q_rand, occupancy Q, stretched-exp Q_∞, ℓ_c, thresholds) → Tasks 3, 5, Global Constraints. ✓
- ξ error propagation / resolvability → Task 5 `xi_threshold` (`dxi`) + aggregation. ✓ (resolvability *comparison* across T is applied in analysis; the per-point `dxi` is computed.)
- Non-monotone → range → Task 5 `kind="range"`. ✓
- G-corr / G-conv (two-arm + run-length ≥3τ) / G-stick (third-arm+2×) / drift → Task 6. ✓
- MH-vs-G-corr division of guarantees → correctness rests on the exact kernels (Tasks 1-2 tests assert energy-bookkeeping exactness); G-corr validates implementation at an easy point (Task 6). ✓
- N-transfer as extended R via larger boxes + L/ℓ_c≥4 → Task 8 ladder (c=0.06/0.04 at N=576). ✓
- Occupancy species-invariance → Task 3 test. ✓
- Composition recording (x_B) → Tasks 7, 8. ✓
- Reuse EGNN knn=32, behind MH; AR not in loop → Task 2 `load_geometry_table`, block-relabel only. ✓
- Incremental saves / plot paths / logs → Task 8. ✓

**2. Placeholder scan:** No "TBD"/"handle edge cases"/uncoded steps. Task 7 flags a signature-confirmation for `swap_mcmc_fast` (a real check against existing code, not a placeholder) — the implementer verifies the return shape and slices to `n_configs`.

**3. Type consistency:** `mobile` is bool[B,N] (True=mobile) everywhere (Tasks 1,2,4,8). `table_fn(x)->[B,N]` P(B) (Tasks 2,4,8). `constrained_run` returns the dict consumed by `g_conv`/`stretched_exp_fit` (keys `t`,`Q` — Tasks 4,5,6,8). `xi_threshold` signature identical in Task 5 and Task 8 `aggregate`. `Qrand=0.108` from `q_rand()` (Tasks 3,5,8). Consistent.

**Deferred-vs-spec note:** v1 omits the heat-bath position accelerator, so the "learned kernel accelerates interior mixing" leg is *measured, not assumed* (spec §8) and expected ≈1× on occupancy Q — the honest v1 claim is correctness + transfer of the measurement pipeline, with the learned geometry table as the transferable exact-guarded species component and the accelerator as the v2 trigger.
