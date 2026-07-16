# Identity-Bridge PT (per-particle size/identity annealing for binary-KA cavity PTS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and gate a (T, u) *identity-interpolation* replica ladder for binary-KA cavities — a per-species continuous-size bridge that makes species-identity redistribution a fast channel (free at u→0, dead at u=1) — and measure, in a matched-compute A/B against the verbatim BCY (T, λ) shrinkage ladder, whether it fixes the measured round-trip starvation and closes the dual-init certificate at R=2.0.

**Architecture:** A numba CPU engine (the 52×-faster path measured 2026-07-15) whose energy kernels take *per-rung interaction tables*, so ONE engine runs both arms: the BCY λ-arm (tables = λ̃-scaled σ) and the new u-arm (tables = entrywise interpolation of the KA σ/ε matrices toward a common value, making species progressively indistinguishable). Identity-swap moves (A↔B label exchange, count-preserving) are exact MH at every rung and become free as u→0. Exactness anchors: ka_energy agreement at u=1, label-permutation invariance at u=0, and the validated q_c(R=2.0)=0.64±0.05 truth. **No neural network anywhere in this plan** — learned proposals are a separate, later plan gated on this one's outcome (Task 5).

**Tech Stack:** Python 3.11, numba (installed 2026-07-15, v0.66), torch (CPU, only for data loading + the validated `bcy_qc` estimator), pytest.

## Global Constraints

- Data: `liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt` (ρ=1.2, T=0.5, 1024 configs, L=7.5282882310482275). Held-out configs = indices ≥ 921 (the 90% split used by train_cavity_ersi.py).
- Cavity radius ceiling: **R ≤ 2.4** (periodic validity needs 2R + r_cut < L; 2·2.4+2.5 = 7.3 < 7.528). R ≥ 2.6 requires a bigger box — OUT OF SCOPE here.
- KA matrices (from `liquid_coupling_flow/ka_energy.py`): SIGMA = [[1.0, 0.8], [0.8, 0.88]], EPS = [[1.0, 1.5], [1.5, 0.5]]; shifted LJ, r_cut = 2.5·σ_pair; composition 80:20.
- BCY conventions carried over verbatim (verified correct 2026-07-15, memory `pt-shrinkage-infeasible`): λ̃ = λ mobile–mobile, (1+λ)/2 mobile–pinned; T(λ) = T_BOT + (T_DEC−T_BOT)(1−λ)/(1−λ_DEC) with Table-IV R=2.0 row T_DEC=1.0, λ_DEC=0.8; boundary shell = exterior particles within R+2.5 of center.
- All MC energies float64. Quench-style Adam is forbidden (diverges on r⁻¹²) — not needed in this plan.
- Every run saves incrementally per completed unit (per cavity, per arm) to a `.pt` — never end-only saves (memory `checkpoint-incrementally`).
- Save full traces (U(t), q_c(t), TRIPS, per-rung acceptance, label-decorrelation), never just summary scalars (memory `record-simulation-data`).
- Long runs: launch with `run_in_background`, redirect `> log.out 2>&1`, kill by exact PID only (never `pkill -f`/`pgrep -f` with self-matching patterns — bit us 3× this week).
- Validation anchor (measured 3 independent ways, 2026-07-15): converged bottom-rung q_c(A,B) at R=2.0 must land in **[0.55, 0.75]**.
- Baseline to beat (measured 2026-07-15, `instrument_pt.out`): λ-arm at 15k–40k sweeps gives **TRIPS ≈ 1–2, dual-init q_c gap stuck ≈ 0.4–0.45**.

---

### Task 1: Package scaffold + per-rung interaction tables

**Files:**
- Create: `liquid_coupling_flow/ptu/__init__.py` (empty)
- Create: `liquid_coupling_flow/ptu/tables.py`
- Test: `liquid_coupling_flow/tests/test_ptu_tables.py`

**Interfaces:**
- Produces: `build_tables_u(u, sig_bar=0.95, eps_bar=1.0) -> (sig_mm[2,2], eps_mm[2,2], sig_mp[2,2], eps_mp[2,2])` — float64 numpy arrays; mm = mobile–mobile pair tables at interpolation u, mp = mobile–pinned (pinned stay physical; interpolation strength (1+u)/2, mirroring BCY's λ̃ convention).
- Produces: `build_tables_lam(lam) -> (sig_mm, eps_mm, sig_mp, eps_mp)` — the BCY λ-arm: sig_mm = lam·SIGMA, sig_mp = ((1+lam)/2)·SIGMA, eps unchanged.
- Produces: `t_of(x, x_bot, x_top, T_bot, T_top) -> float` — linear temperature schedule along the ladder coordinate.

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ptu_tables.py
import numpy as np
from liquid_coupling_flow.ptu.tables import build_tables_u, build_tables_lam
from liquid_coupling_flow.ka_energy import SIGMA, EPS

KA_SIG = np.array(SIGMA, dtype=np.float64)
KA_EPS = np.array(EPS, dtype=np.float64)


def test_u1_is_physical_ka():
    sig_mm, eps_mm, sig_mp, eps_mp = build_tables_u(1.0)
    assert np.allclose(sig_mm, KA_SIG) and np.allclose(eps_mm, KA_EPS)
    assert np.allclose(sig_mp, KA_SIG) and np.allclose(eps_mp, KA_EPS)


def test_u0_is_species_blind():
    sig_mm, eps_mm, _, _ = build_tables_u(0.0, sig_bar=0.95, eps_bar=1.0)
    # all entries identical -> species labels carry zero energy information
    assert np.allclose(sig_mm, 0.95) and np.allclose(eps_mm, 1.0)


def test_u0_mp_is_halfway():
    # pinned partner keeps physical character: interpolation strength (1+u)/2 -> 0.5 at u=0
    _, _, sig_mp, eps_mp = build_tables_u(0.0, sig_bar=0.95, eps_bar=1.0)
    assert np.allclose(sig_mp, 0.5 * KA_SIG + 0.5 * 0.95)
    assert np.allclose(eps_mp, 0.5 * KA_EPS + 0.5 * 1.0)


def test_lam_arm_matches_bcy_convention():
    lam = 0.9
    sig_mm, eps_mm, sig_mp, eps_mp = build_tables_lam(lam)
    assert np.allclose(sig_mm, lam * KA_SIG)
    assert np.allclose(sig_mp, 0.5 * (1 + lam) * KA_SIG)
    assert np.allclose(eps_mm, KA_EPS) and np.allclose(eps_mp, KA_EPS)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_tables.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.ptu'`

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ptu/tables.py
"""Per-rung interaction tables for the identity-bridge (u) and BCY-shrinkage (lam) ladders.

u-arm: entrywise interpolation of the KA sigma/eps matrices toward common values (sig_bar, eps_bar).
At u=0 mobile-mobile pairs are species-blind (identity swaps are FREE); at u=1 physical KA.
Mobile-pinned uses interpolation strength (1+u)/2 (pinned particles keep physical character),
mirroring BCY's lam~=(1+lam)/2 convention for mobile-pinned pairs.
lam-arm: BCY Eq.2 verbatim (sig scaled, eps untouched) — verified correct 2026-07-15."""
import numpy as np
from liquid_coupling_flow.ka_energy import SIGMA, EPS

KA_SIG = np.array(SIGMA, dtype=np.float64)
KA_EPS = np.array(EPS, dtype=np.float64)


def build_tables_u(u, sig_bar=0.95, eps_bar=1.0):
    u = float(u)
    w_mm = u
    w_mp = 0.5 * (1.0 + u)
    sig_mm = w_mm * KA_SIG + (1 - w_mm) * sig_bar
    eps_mm = w_mm * KA_EPS + (1 - w_mm) * eps_bar
    sig_mp = w_mp * KA_SIG + (1 - w_mp) * sig_bar
    eps_mp = w_mp * KA_EPS + (1 - w_mp) * eps_bar
    return sig_mm, eps_mm, sig_mp, eps_mp


def build_tables_lam(lam):
    lam = float(lam)
    sig_mm = lam * KA_SIG
    sig_mp = 0.5 * (1.0 + lam) * KA_SIG
    return sig_mm, KA_EPS.copy(), sig_mp, KA_EPS.copy()


def t_of(x, x_bot, x_top, T_bot, T_top):
    """Linear T along the ladder coordinate (x = u or lam); x_bot -> T_bot, x_top -> T_top."""
    if x_top == x_bot:
        return T_bot
    f = (x - x_bot) / (x_top - x_bot)
    return T_bot + (T_top - T_bot) * f
```

Also create the empty `liquid_coupling_flow/ptu/__init__.py`.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_tables.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ptu/__init__.py liquid_coupling_flow/ptu/tables.py liquid_coupling_flow/tests/test_ptu_tables.py
git commit -m "feat(ptu): per-rung interaction tables for identity-bridge and BCY-shrinkage ladders"
```

---

### Task 2: numba energy kernels with table arguments + exactness gates

**Files:**
- Create: `liquid_coupling_flow/ptu/kernels.py`
- Test: `liquid_coupling_flow/tests/test_ptu_kernels.py`

**Interfaces:**
- Consumes: `build_tables_u`, `build_tables_lam` from Task 1.
- Produces (all `@njit(cache=True)`, float64):
  - `row_e(allx, alls, i, x0, x1, x2, n, n_tot, sig_mm, eps_mm, sig_mp, eps_mp) -> float` — pair-row energy of mobile particle i at (x0,x1,x2); partners j<n use mm tables, j≥n (pinned) use mp tables; shifted LJ, r_cut=2.5·σ_pair, self excluded.
  - `mobile_U(allx, alls, n, n_tot, sig_mm, eps_mm, sig_mp, eps_mp) -> float` — total mobile energy (mm counted once, mp fully).
  - `disp_sweep(allx, alls, n, n_tot, beta, R, step, sig_mm, eps_mm, sig_mp, eps_mp) -> int` — one displacement sweep, BCY move (l·n̂, l~U[0,0.3], hard wall |x|<R), returns accepts.

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ptu_kernels.py
import numpy as np
import torch
import pytest
from liquid_coupling_flow.ptu.tables import build_tables_u, build_tables_lam
from liquid_coupling_flow.ptu.kernels import row_e, mobile_U, disp_sweep
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic

torch.set_grad_enabled(False)
BIGL = 100.0


@pytest.fixture(scope="module")
def cavity():
    D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt",
                   map_location="cpu", weights_only=False)
    X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
    g = torch.Generator().manual_seed(3)
    ci = 950  # held-out config
    c = torch.rand(3, generator=g) * L
    p = carve(X[ci], S[ci], c, 2.0, L)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L)
    bm = xout.norm(dim=-1) < (2.0 + 2.5)
    bnd, sb = xout[bm], p["s_out"][bm]
    allx = np.concatenate([xin.double().numpy(), bnd.double().numpy()])
    alls = np.concatenate([p["s_in"].numpy(), sb.numpy()]).astype(np.int64)
    n = xin.shape[0]
    return allx, alls, n, allx.shape[0], xin, p["s_in"], bnd, sb


def test_u1_matches_ka_energy(cavity):
    """At u=1 the kernel must reproduce ka_energy's mobile-involved energy to 1e-8/particle.
    ka_energy(all) - ka_energy(boundary only) = mobile-mobile + mobile-pinned."""
    allx, alls, n, n_tot, xin, sin, bnd, sb = cavity
    tabs = build_tables_u(1.0)
    u_kernel = mobile_U(allx, alls, n, n_tot, *tabs)
    x_all = torch.tensor(allx)[None]
    s_all = torch.tensor(alls)[None]
    u_full = float(ka_energy(x_all, s_all, BIGL)[0])
    u_bb = float(ka_energy(torch.tensor(allx[n:])[None], torch.tensor(alls[n:])[None], BIGL)[0])
    assert abs(u_kernel - (u_full - u_bb)) / n < 1e-8


def test_u0_label_permutation_invariance(cavity):
    """At u=0 mobile-mobile is species-blind: permuting MOBILE labels changes mobile_U only
    through the mobile-pinned term; with a label-permutation among mobiles ONLY, and mp tables
    at half-interpolation, the mm part must be exactly invariant. Test the mm-only invariance
    by using mp tables == mm tables (fully species-blind everywhere)."""
    allx, alls, n, n_tot, *_ = cavity
    sig_mm, eps_mm, _, _ = build_tables_u(0.0)
    tabs_blind = (sig_mm, eps_mm, sig_mm, eps_mm)
    u0 = mobile_U(allx, alls, n, n_tot, *tabs_blind)
    rng = np.random.default_rng(0)
    alls_perm = alls.copy()
    alls_perm[:n] = rng.permutation(alls[:n])
    u1 = mobile_U(allx, alls_perm, n, n_tot, *tabs_blind)
    assert abs(u0 - u1) < 1e-10


def test_lam_arm_matches_prior_implementation(cavity):
    """lam-arm kernel at lam=0.9 must agree with the (verified-correct) torch Cavity.full_U
    convention: deformed sigma inside LJ, fixed rc=2.5*deformed sigma, lam-independent shift.
    Regression pin: compute once with the torch reference formula inline."""
    allx, alls, n, n_tot, *_ = cavity
    lam = 0.9
    tabs = build_tables_lam(lam)
    u_kernel = mobile_U(allx, alls, n, n_tot, *tabs)
    # inline torch reference (same math as bcy_gpts_v2.Cavity.full_U, f64)
    import itertools
    from liquid_coupling_flow.ka_energy import SIGMA, EPS
    x = torch.tensor(allx); s = torch.tensor(alls)
    d2 = torch.cdist(x[None], x[None])[0] ** 2
    d2.fill_diagonal_(1e12)
    sig0 = torch.tensor(SIGMA, dtype=torch.float64)[s[:, None], s[None, :]]
    eps0 = torch.tensor(EPS, dtype=torch.float64)[s[:, None], s[None, :]]
    lam_col = torch.full((n_tot,), 0.5 * (1 + lam), dtype=torch.float64)
    lam_col[:n] = lam
    sig = sig0 * lam_col[None, :]
    inv6 = (sig ** 2 / d2) ** 3
    e = 4 * eps0 * (inv6 ** 2 - inv6)
    s6 = (1.0 / 2.5) ** 6
    e = torch.where(d2 < (2.5 * sig) ** 2, e - 4 * eps0 * (s6 ** 2 - s6), torch.zeros_like(e))
    u_ref = float(e[:n, :n].sum() * 0.5 + e[:n, n:].sum())
    assert abs(u_kernel - u_ref) / n < 1e-8


def test_disp_sweep_runs_and_accepts(cavity):
    allx, alls, n, n_tot, *_ = cavity
    tabs = build_tables_u(1.0)
    a = allx.copy()
    np.random.seed(0)
    acc = disp_sweep(a, alls, n, n_tot, 2.0, 2.0, 0.3, *tabs)
    assert 0 < acc < n                     # some but not all moves accepted at T=0.5
    assert np.all(np.abs(a[n:] - allx[n:]) == 0.0)   # pinned particles never move
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_kernels.py -v`
Expected: FAIL with `ModuleNotFoundError` (kernels.py absent)

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ptu/kernels.py
"""numba energy/MC kernels with per-rung interaction tables. Partner class decides the table:
j < n -> mobile-mobile (mm), j >= n -> mobile-pinned (mp). Shifted LJ, rc = 2.5*sigma_pair.
All float64. Displacement move = BCY convention: dx = l*nhat, l ~ U[0, step], hard wall |x|<R."""
import numpy as np
from numba import njit


@njit(cache=True, fastmath=True)
def row_e(allx, alls, i, x0, x1, x2, n, n_tot, sig_mm, eps_mm, sig_mp, eps_mp):
    e = 0.0
    si = alls[i]
    for j in range(n_tot):
        if j == i:
            continue
        dx = allx[j, 0] - x0; dy = allx[j, 1] - x1; dz = allx[j, 2] - x2
        r2 = dx * dx + dy * dy + dz * dz
        if j < n:
            s = sig_mm[si, alls[j]]; ep = eps_mm[si, alls[j]]
        else:
            s = sig_mp[si, alls[j]]; ep = eps_mp[si, alls[j]]
        rc = 2.5 * s
        if r2 >= rc * rc:
            continue
        sr6 = (s * s / r2) ** 3
        sc6 = (1.0 / 2.5) ** 6
        e += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
    return e


@njit(cache=True, fastmath=True)
def mobile_U(allx, alls, n, n_tot, sig_mm, eps_mm, sig_mp, eps_mp):
    u = 0.0
    for i in range(n):
        si = alls[i]
        for j in range(i + 1, n):                       # mm once
            dx = allx[j, 0] - allx[i, 0]; dy = allx[j, 1] - allx[i, 1]; dz = allx[j, 2] - allx[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s = sig_mm[si, alls[j]]; ep = eps_mm[si, alls[j]]
            rc = 2.5 * s
            if r2 < rc * rc:
                sr6 = (s * s / r2) ** 3
                sc6 = (1.0 / 2.5) ** 6
                u += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
        for j in range(n, n_tot):                       # mp fully
            dx = allx[j, 0] - allx[i, 0]; dy = allx[j, 1] - allx[i, 1]; dz = allx[j, 2] - allx[i, 2]
            r2 = dx * dx + dy * dy + dz * dz
            s = sig_mp[si, alls[j]]; ep = eps_mp[si, alls[j]]
            rc = 2.5 * s
            if r2 < rc * rc:
                sr6 = (s * s / r2) ** 3
                sc6 = (1.0 / 2.5) ** 6
                u += 4.0 * ep * ((sr6 * sr6 - sr6) - (sc6 * sc6 - sc6))
    return u


@njit(cache=True, fastmath=True)
def disp_sweep(allx, alls, n, n_tot, beta, R, step, sig_mm, eps_mm, sig_mp, eps_mp):
    acc = 0
    for i in range(n):
        l = step * np.random.random()
        v0 = np.random.randn(); v1 = np.random.randn(); v2 = np.random.randn()
        vn = (v0 * v0 + v1 * v1 + v2 * v2) ** 0.5
        xn0 = allx[i, 0] + l * v0 / vn
        xn1 = allx[i, 1] + l * v1 / vn
        xn2 = allx[i, 2] + l * v2 / vn
        if xn0 * xn0 + xn1 * xn1 + xn2 * xn2 >= R * R:
            continue
        e0 = row_e(allx, alls, i, allx[i, 0], allx[i, 1], allx[i, 2], n, n_tot,
                   sig_mm, eps_mm, sig_mp, eps_mp)
        e1 = row_e(allx, alls, i, xn0, xn1, xn2, n, n_tot,
                   sig_mm, eps_mm, sig_mp, eps_mp)
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            allx[i, 0] = xn0; allx[i, 1] = xn1; allx[i, 2] = xn2
            acc += 1
    return acc
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_kernels.py -v`
Expected: 4 passed (first run slow: numba JIT compile ~30 s)

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ptu/kernels.py liquid_coupling_flow/tests/test_ptu_kernels.py
git commit -m "feat(ptu): numba table-driven cavity MC kernels, exactness-gated vs ka_energy"
```

---

### Task 3: Identity-swap move + acceptance-vs-u probe (fixes u_min and T_top)

**Files:**
- Create: `liquid_coupling_flow/ptu/identity.py`
- Create: `reports/logs-2026-07-16/ptu_probe_u.py`
- Test: `liquid_coupling_flow/tests/test_ptu_identity.py`

**Interfaces:**
- Consumes: `row_e`, `disp_sweep`, `build_tables_u` from Tasks 1–2.
- Produces: `identity_sweep(allx, alls, n, n_tot, beta, n_try, sig_mm, eps_mm, sig_mp, eps_mp) -> (int, int)` — n_try attempted A↔B label exchanges between random unlike-species mobile pairs, exact MH on ΔU under the rung's tables (labels swap, positions fixed, counts preserved); returns (accepts, attempts).
- Produces (probe script): the measured curve `acc_identity(u)` and top-rung fluidity check; its output fixes the ladder constants `U_MIN`, `T_TOP`, `N_RUNGS` consumed by Task 4.

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ptu_identity.py
import numpy as np
import pytest
from liquid_coupling_flow.ptu.tables import build_tables_u
from liquid_coupling_flow.ptu.kernels import mobile_U
from liquid_coupling_flow.ptu.identity import identity_sweep
from liquid_coupling_flow.tests.test_ptu_kernels import cavity  # reuse fixture


def test_identity_swap_free_when_species_blind(cavity):
    """With fully species-blind tables (mm==mp==blind), dU==0 for every label swap -> acc == att."""
    allx, alls, n, n_tot, *_ = cavity
    sig_mm, eps_mm, _, _ = build_tables_u(0.0)
    tabs = (sig_mm, eps_mm, sig_mm, eps_mm)
    np.random.seed(0)
    a = alls.copy()
    acc, att = identity_sweep(allx, a, n, n_tot, 2.0, 200, *tabs)
    assert att > 0 and acc == att


def test_identity_swap_dead_at_physical(cavity):
    """At u=1 (physical KA) plain label swaps are known exact-dead (measured: ~0 acceptance)."""
    allx, alls, n, n_tot, *_ = cavity
    tabs = build_tables_u(1.0)
    np.random.seed(0)
    a = alls.copy()
    acc, att = identity_sweep(allx, a, n, n_tot, 2.0, 500, *tabs)
    assert att > 0 and acc / att < 0.02


def test_identity_swap_preserves_counts(cavity):
    allx, alls, n, n_tot, *_ = cavity
    tabs = build_tables_u(0.5)
    np.random.seed(1)
    a = alls.copy()
    nb_before = int((a[:n] == 1).sum())
    identity_sweep(allx, a, n, n_tot, 2.0, 300, *tabs)
    assert int((a[:n] == 1).sum()) == nb_before
    assert np.all(a[n:] == alls[n:])            # pinned labels untouched
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_identity.py -v`
Expected: FAIL with `ModuleNotFoundError` (identity.py absent)

- [ ] **Step 3: Write minimal implementation**

```python
# liquid_coupling_flow/ptu/identity.py
"""Count-preserving identity-swap move: exchange the species LABELS of one mobile A and one
mobile B (positions fixed), exact MH on dU under the rung's tables. At u=0 (species-blind
tables) dU==0 -> free; at u=1 this is the known-dead plain swap. Pinned labels never touched."""
import numpy as np
from numba import njit
from liquid_coupling_flow.ptu.kernels import row_e


@njit(cache=True, fastmath=True)
def identity_sweep(allx, alls, n, n_tot, beta, n_try, sig_mm, eps_mm, sig_mp, eps_mp):
    acc = 0
    att = 0
    for _ in range(n_try):
        ia = np.random.randint(n)
        ib = np.random.randint(n)
        if alls[ia] == alls[ib]:
            continue
        att += 1
        e0 = (row_e(allx, alls, ia, allx[ia, 0], allx[ia, 1], allx[ia, 2], n, n_tot,
                    sig_mm, eps_mm, sig_mp, eps_mp)
              + row_e(allx, alls, ib, allx[ib, 0], allx[ib, 1], allx[ib, 2], n, n_tot,
                      sig_mm, eps_mm, sig_mp, eps_mp))
        sa = alls[ia]
        alls[ia] = alls[ib]; alls[ib] = sa
        e1 = (row_e(allx, alls, ia, allx[ia, 0], allx[ia, 1], allx[ia, 2], n, n_tot,
                    sig_mm, eps_mm, sig_mp, eps_mp)
              + row_e(allx, alls, ib, allx[ib, 0], allx[ib, 1], allx[ib, 2], n, n_tot,
                      sig_mm, eps_mm, sig_mp, eps_mp))
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            acc += 1
        else:
            sb = alls[ia]
            alls[ia] = alls[ib]; alls[ib] = sb          # revert
    return acc, att
```

NOTE (for the implementer): the e0/e1 pair double-counts the ia–ib direct interaction identically in both states only when the pair's σ/ε entry is symmetric under the label exchange — for the KA matrices σ_AB=σ_BA so the A↔B exchange leaves the ia–ib pair term itself invariant; ΔU from the double-counted pair cancels in e1−e0. No correction needed.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_identity.py -v`
Expected: 3 passed

- [ ] **Step 5: Write the probe script (measured constants for Task 4)**

```python
# reports/logs-2026-07-16/ptu_probe_u.py
"""PROBE: identity-swap acceptance and displacement fluidity vs u — fixes U_MIN, T_TOP, N_RUNGS.
For u in a grid, equilibrate one R=2.0 cavity at (u, T(u)) with displacement sweeps, then measure
identity-swap acceptance. Also check top-rung fluidity (disp-acc in [0.15, 0.6]; U/N stationary --
guards against monodisperse freezing/crystallization at low u). Full curves saved."""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ptu.tables import build_tables_u, t_of
from liquid_coupling_flow.ptu.kernels import disp_sweep, mobile_U
from liquid_coupling_flow.ptu.identity import identity_sweep
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
torch.set_grad_enabled(False)

R = 2.0; T_BOT = 0.5
T_TOP_GRID = [0.7, 0.9]
U_GRID = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.35, 0.2, 0.0]
EQ_SW = 3000; MEAS_SW = 1000
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
g = torch.Generator().manual_seed(3)
ci = 950
c = torch.rand(3, generator=g) * L
p = carve(X[ci], S[ci], c, R, L)
xin = _mic(p["x_in"], c, L); xout = _mic(p["x_out"], c, L)
bm = xout.norm(dim=-1) < (R + 2.5)
allx0 = np.concatenate([xin.double().numpy(), xout[bm].double().numpy()])
alls0 = np.concatenate([p["s_in"].numpy(), p["s_out"][bm].numpy()]).astype(np.int64)
n = xin.shape[0]; n_tot = allx0.shape[0]
print(f"probe: cav {ci} n={n} n_tot={n_tot} | u grid {U_GRID} x T_top {T_TOP_GRID}", flush=True)
out = {}
for T_TOP in T_TOP_GRID:
    for u in U_GRID:
        T = t_of(u, 1.0, 0.0, T_BOT, T_TOP)
        beta = 1.0 / T
        tabs = build_tables_u(u)
        allx = allx0.copy(); alls = alls0.copy()
        np.random.seed(7)
        t0 = time.time()
        dacc = 0
        us = []
        for sw in range(EQ_SW):
            dacc += disp_sweep(allx, alls, n, n_tot, beta, R, 0.3, *tabs)
            if sw % 200 == 0:
                us.append(mobile_U(allx, alls, n, n_tot, *tabs) / n)
        iacc = iatt = 0
        for sw in range(MEAS_SW):
            disp_sweep(allx, alls, n, n_tot, beta, R, 0.3, *tabs)
            a, t = identity_sweep(allx, alls, n, n_tot, beta, 10, *tabs)
            iacc += a; iatt += t
        drift = abs(us[-1] - us[len(us) // 2])
        out[(T_TOP, u)] = {"id_acc": iacc / max(iatt, 1), "disp_acc": dacc / (EQ_SW * n),
                           "U_trace": us, "drift_half": drift}
        torch.save(out, "reports/logs-2026-07-16/ptu_probe_u.pt")
        print(f"  T_top={T_TOP} u={u:.2f} T={T:.3f}: identity-acc {iacc/max(iatt,1):.3f} "
              f"disp-acc {dacc/(EQ_SW*n):.2f} U/n {us[-1]:+.3f} drift {drift:.3f} "
              f"({time.time()-t0:.0f}s)", flush=True)
print("PROBE DONE", flush=True)
```

- [ ] **Step 6: Run the probe**

Run: `cd /mnt/ssd/GridTransformer && nohup python reports/logs-2026-07-16/ptu_probe_u.py > reports/logs-2026-07-16/ptu_probe_u.out 2>&1 &` (background; ~30–60 min)
Expected output shape: identity-acc ≈ 0.00–0.02 at u=1.0 rising monotonically to ≈ 1.0 at u=0.0; disp-acc within [0.15, 0.6] at every rung; U/n drift < 0.1 at every rung (no freezing).
**Decision rule (record in the log):** `U_MIN` = the largest u with identity-acc ≥ 0.25; `T_TOP` = the smaller grid value that satisfies the fluidity checks; `N_RUNGS` = 12 linear in u over [U_MIN, 1.0] as the starting ladder (adjust only if Task 4's per-rung exchange acceptance min < 0.1).

- [ ] **Step 7: Commit**

```bash
git add liquid_coupling_flow/ptu/identity.py liquid_coupling_flow/tests/test_ptu_identity.py reports/logs-2026-07-16/ptu_probe_u.py reports/logs-2026-07-16/ptu_probe_u.out reports/logs-2026-07-16/ptu_probe_u.pt
git commit -m "feat(ptu): count-preserving identity-swap kernel + acceptance-vs-u probe (fixes U_MIN, T_TOP)"
```

---

### Task 4: Replica ladder with exchange, TRIPS/flow metrics, dual-init q_c

**Files:**
- Create: `liquid_coupling_flow/ptu/ladder.py`
- Create: `liquid_coupling_flow/ptu/qc.py`
- Test: `liquid_coupling_flow/tests/test_ptu_ladder.py`

**Interfaces:**
- Consumes: Tasks 1–3 kernels; probe constants (U_MIN, T_TOP, N_RUNGS) — pass as arguments, do not hard-code.
- Produces: `class Ladder` with:
  - `__init__(self, allx0, alls0, n, n_tot, mode, coords, temps, nch=2, exch_mean=10, step=0.3, R=2.0, seed=0)` — `mode` ∈ {"u", "lam"}; `coords` = ladder values (u or λ per rung, index 0 = physical bottom); builds per-rung tables once; state = 2 stacks (A=reference-init, B=randomized-at-top-init) × NR rungs × nch chains.
  - `randomize_stack_B(self, n_sweeps)` — top-rung (coords[-1], temps[-1]) displacement+identity sweeps applied to stack-B walkers only.
  - `run(self, n_sweeps, rec_every, qc_fn) -> dict` — main loop: per sweep, every rung does `disp_sweep` + (u-mode only) `identity_sweep(n_try=4)`; Poisson exchange between adjacent rungs (prob 1/exch_mean per pair per sweep) with the cross-table acceptance `dlog = -beta_a*(U_a(x_b)-U_a(x_a)) - beta_b*(U_b(x_a)-U_b(x_b))`; maintains TRIPS (bottom↔top lineage round trips) and Katzgraber flow f(r) exactly as in `reports/logs-2026-07-15/pt_decorrelation_metrics.py`; every `rec_every` sweeps records bottom-rung `qc_fn(x_bottom, x_ref)` for both stacks and label-overlap `mean(alls_bottom[:n] == alls_ref[:n])`. Returns full traces dict.
- Produces: `qc.py` with `bcy_qc(X, Y, gen) -> float` — verbatim copy of the calibrated estimator from `reports/logs-2026-07-15/bcy_gpts_v2.py` (b=0.2, rc_core=0.5, P_MC=1500, K_NN=6; validated self→1.000, random→0.009).

- [ ] **Step 1: Write the failing test**

```python
# liquid_coupling_flow/tests/test_ptu_ladder.py
import numpy as np
import torch
from liquid_coupling_flow.ptu.tables import build_tables_u
from liquid_coupling_flow.ptu.ladder import Ladder
from liquid_coupling_flow.ptu.qc import bcy_qc
from liquid_coupling_flow.tests.test_ptu_kernels import cavity  # fixture


def test_qc_calibration(cavity):
    *_, xin, sin, bnd, sb = cavity
    g = torch.Generator().manual_seed(7)
    x = xin.numpy()
    assert bcy_qc(x, x, g) > 0.98                       # self ~ 1
    rng = np.random.default_rng(0)
    u = rng.standard_normal((x.shape[0], 3)); u /= np.linalg.norm(u, axis=1, keepdims=True)
    xr = u * 2.0 * rng.random((x.shape[0], 1)) ** (1 / 3)
    assert bcy_qc(x, xr, g) < 0.05                      # random ~ 0


def test_ladder_smoke_and_trips_counter(cavity):
    allx, alls, n, n_tot, xin, *_ = cavity
    coords = [1.0, 0.6, 0.2]                            # 3-rung toy, u-mode
    temps = [0.5, 0.7, 0.9]
    lad = Ladder(allx, alls, n, n_tot, "u", coords, temps, nch=2, exch_mean=1, seed=0)
    lad.randomize_stack_B(50)
    g = torch.Generator().manual_seed(7)
    res = lad.run(300, rec_every=100, qc_fn=lambda a, b: bcy_qc(a, b, g))
    # with exch_mean=1 on a soft 3-rung toy ladder, exchanges fire and trips accumulate
    assert res["trips"] >= 1
    assert len(res["qcA"]) == 3 and len(res["qcB"]) == 3
    assert res["exch_att"].min() > 0
    # stack A starts at reference: its first recorded qc must be high
    assert res["qcA"][0] > 0.4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_ladder.py -v`
Expected: FAIL with `ModuleNotFoundError` (ladder.py, qc.py absent)

- [ ] **Step 3: Implement `qc.py`** — copy the `bcy_qc` function body verbatim from `reports/logs-2026-07-15/bcy_gpts_v2.py` lines 58–72 (constants B_OV=0.2, RC_CORE=0.5, P_MC=1500, K_NN=6 inlined at module top), accepting numpy or tensor inputs via `torch.as_tensor(..., dtype=torch.float32)`.

- [ ] **Step 4: Implement `ladder.py`**

```python
# liquid_coupling_flow/ptu/ladder.py
"""(T, coord) replica ladder over cavity states with exact Metropolis exchange.
mode='u': identity-bridge (tables from build_tables_u; per-rung identity_sweep enabled).
mode='lam': BCY shrinkage (tables from build_tables_lam; displacement-only, BCY verbatim).
Metrics: TRIPS (lineage bottom->top->bottom), Katzgraber flow f(r), per-rung exchange acc,
dual-init bottom-rung q_c + label-overlap traces. Ported from
reports/logs-2026-07-15/pt_decorrelation_metrics.py (flow/trips bookkeeping) and
bcy_gpts_v2.py (exchange acceptance)."""
import numpy as np
from liquid_coupling_flow.ptu.tables import build_tables_u, build_tables_lam
from liquid_coupling_flow.ptu.kernels import disp_sweep, mobile_U
from liquid_coupling_flow.ptu.identity import identity_sweep


class Ladder:
    def __init__(self, allx0, alls0, n, n_tot, mode, coords, temps, nch=2,
                 exch_mean=10, step=0.3, R=2.0, seed=0):
        assert mode in ("u", "lam")
        self.n, self.n_tot, self.mode = n, n_tot, mode
        self.coords = list(coords); self.temps = list(temps)
        self.NR = len(coords); self.nch = nch
        self.exch_mean = exch_mean; self.step = step; self.R = R
        build = build_tables_u if mode == "u" else build_tables_lam
        self.tabs = [build(c) for c in coords]
        self.betas = [1.0 / t for t in temps]
        # state[stack][rung][chain] -> (allx, alls) copies
        self.X = [[[allx0.copy() for _ in range(nch)] for _ in range(self.NR)] for _ in range(2)]
        self.S = [[[alls0.copy() for _ in range(nch)] for _ in range(self.NR)] for _ in range(2)]
        self.x_ref = allx0[:n].copy(); self.s_ref = alls0[:n].copy()
        self.rng = np.random.default_rng(seed)
        np.random.seed(seed)                                # numba kernels use np.random global
        # trips/flow bookkeeping per (stack, rung, chain)
        self.seen_top = np.zeros((2, self.NR, nch), dtype=np.bool_)
        self.flow = np.full((2, self.NR, nch), 0.5)
        self.flow_sum = np.zeros(self.NR); self.flow_cnt = np.zeros(self.NR)
        self.trips = 0
        self.exch_acc = np.zeros(self.NR - 1); self.exch_att = np.zeros(self.NR - 1)

    def _sweep_all(self):
        for st in range(2):
            for r in range(self.NR):
                beta = self.betas[r]; tabs = self.tabs[r]
                for ch in range(self.nch):
                    disp_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n, self.n_tot,
                               beta, self.R, self.step, *tabs)
                    if self.mode == "u":
                        identity_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n,
                                       self.n_tot, beta, 4, *tabs)

    def _exchange(self):
        for r in range(self.NR - 1):
            if self.rng.random() >= 1.0 / self.exch_mean:
                continue
            for st in range(2):
                for ch in range(self.nch):
                    xa, sa = self.X[st][r][ch], self.S[st][r][ch]
                    xb, sb = self.X[st][r + 1][ch], self.S[st][r + 1][ch]
                    Uaa = mobile_U(xa, sa, self.n, self.n_tot, *self.tabs[r])
                    Uab = mobile_U(xb, sb, self.n, self.n_tot, *self.tabs[r])
                    Uba = mobile_U(xa, sa, self.n, self.n_tot, *self.tabs[r + 1])
                    Ubb = mobile_U(xb, sb, self.n, self.n_tot, *self.tabs[r + 1])
                    dlog = -self.betas[r] * (Uab - Uaa) - self.betas[r + 1] * (Uba - Ubb)
                    self.exch_att[r] += 1
                    if np.log(self.rng.random()) < dlog:
                        self.exch_acc[r] += 1
                        self.X[st][r][ch], self.X[st][r + 1][ch] = xb, xa
                        self.S[st][r][ch], self.S[st][r + 1][ch] = sb, sa
                        for arr in (self.seen_top, self.flow):
                            tmp = arr[st, r, ch].copy()
                            arr[st, r, ch] = arr[st, r + 1, ch]
                            arr[st, r + 1, ch] = tmp
        self.seen_top[:, self.NR - 1, :] = True
        self.flow[:, self.NR - 1, :] = 0.0
        self.trips += int(self.seen_top[:, 0, :].sum())
        self.seen_top[:, 0, :] = False
        self.flow[:, 0, :] = 1.0
        det = self.flow != 0.5
        self.flow_sum += (self.flow * det).sum(axis=(0, 2))
        self.flow_cnt += det.sum(axis=(0, 2))

    def randomize_stack_B(self, n_sweeps):
        beta = self.betas[-1]; tabs = self.tabs[-1]
        for st_r_ch in [(1, r, ch) for r in range(self.NR) for ch in range(self.nch)]:
            st, r, ch = st_r_ch
            for _ in range(n_sweeps):
                disp_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n, self.n_tot,
                           beta, self.R, self.step, *tabs)
                if self.mode == "u":
                    identity_sweep(self.X[st][r][ch], self.S[st][r][ch], self.n,
                                   self.n_tot, beta, 4, *tabs)

    def run(self, n_sweeps, rec_every, qc_fn):
        qcA, qcB, labA, labB, rec_sw = [], [], [], [], []
        for sw in range(1, n_sweeps + 1):
            self._sweep_all()
            self._exchange()
            if sw % rec_every == 0:
                a = np.mean([qc_fn(self.X[0][0][ch][:self.n], self.x_ref) for ch in range(self.nch)])
                b = np.mean([qc_fn(self.X[1][0][ch][:self.n], self.x_ref) for ch in range(self.nch)])
                la = np.mean([np.mean(self.S[0][0][ch][:self.n] == self.s_ref) for ch in range(self.nch)])
                lb = np.mean([np.mean(self.S[1][0][ch][:self.n] == self.s_ref) for ch in range(self.nch)])
                qcA.append(a); qcB.append(b); labA.append(la); labB.append(lb); rec_sw.append(sw)
        fr = self.flow_sum / np.maximum(self.flow_cnt, 1)
        return {"qcA": qcA, "qcB": qcB, "labA": labA, "labB": labB, "rec_sw": rec_sw,
                "trips": self.trips, "flow": fr,
                "exch_acc": self.exch_acc, "exch_att": self.exch_att}
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest liquid_coupling_flow/tests/test_ptu_ladder.py -v`
Expected: 2 passed

- [ ] **Step 6: Commit**

```bash
git add liquid_coupling_flow/ptu/ladder.py liquid_coupling_flow/ptu/qc.py liquid_coupling_flow/tests/test_ptu_ladder.py
git commit -m "feat(ptu): dual-init replica ladder (u/lam modes) with TRIPS, flow, exchange, qc traces"
```

---

### Task 5: THE GATE — matched-compute A/B: (T,λ) BCY arm vs (T,u) identity arm at R=2.0

**Files:**
- Create: `reports/logs-2026-07-16/ptu_ab_R20.py`
- Output: `reports/logs-2026-07-16/ptu_ab_R20.out`, `reports/logs-2026-07-16/ptu_ab_R20.pt`

**Interfaces:**
- Consumes: `Ladder`, `bcy_qc`, probe constants from Task 3's recorded decision (read them from the probe log; passed as CLI args `--u_min --t_top`).
- Produces: the pass/kill verdict for the identity-bridge premise + the traces for the write-up.

- [ ] **Step 1: Write the A/B runner**

```python
# reports/logs-2026-07-16/ptu_ab_R20.py
"""A/B GATE: BCY (T,lam) shrinkage ladder vs (T,u) identity-bridge ladder, matched compute,
R=2.0, NCAV held-out cavities. Metrics per arm: TRIPS, Katzgraber flow linearity, per-rung
exchange acc, dual-init q_c gap trajectory + closure sweep, bottom-rung label-overlap decay
(the identity channel's direct signature). Matched compute = same (rungs x chains x sweeps)
budget; lam-arm uses the verbatim Table-IV R=2.0 ladder midpoint-densified to 21 rungs
(the measured-best lam configuration); u-arm uses N_RUNGS linear rungs over [U_MIN, 1.0].
Usage: ptu_ab_R20.py --u_min 0.35 --t_top 0.9 --sw 60000 --ncav 2"""
import sys, time, argparse
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ptu.tables import t_of
from liquid_coupling_flow.ptu.ladder import Ladder
from liquid_coupling_flow.ptu.qc import bcy_qc
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
torch.set_grad_enabled(False)

p = argparse.ArgumentParser()
p.add_argument("--u_min", type=float, required=True)
p.add_argument("--t_top", type=float, required=True)
p.add_argument("--sw", type=int, default=60000)
p.add_argument("--ncav", type=int, default=2)
p.add_argument("--rand_sw", type=int, default=4000)
a = p.parse_args()
R = 2.0; T_BOT = 0.5; Q_TOL = 0.1; REC = 500
_BASE = [1.0000, 0.9825, 0.9640, 0.9450, 0.9250, 0.9050, 0.8850, 0.8640, 0.8423, 0.8200, 0.7960]
LAM = []
for i, v in enumerate(_BASE):
    LAM.append(v)
    if i + 1 < len(_BASE):
        LAM.append(0.5 * (v + _BASE[i + 1]))
T_LAM = [T_BOT + (1.0 - T_BOT) * (1.0 - l) / (1.0 - 0.8) for l in LAM]
N_RUNGS = 12
US = list(np.linspace(1.0, a.u_min, N_RUNGS))
T_US = [t_of(u, 1.0, a.u_min, T_BOT, a.t_top) for u in US]
# matched compute: sweeps_u = sw * (21 * nch) / (N_RUNGS * nch)
SW_LAM = a.sw
SW_U = int(a.sw * len(LAM) / N_RUNGS)
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
X, S, L = D["x"].float(), D["s"].long(), float(D["L"])
gsel = torch.Generator().manual_seed(1620)
gq = torch.Generator().manual_seed(7)
results = {"args": vars(a)}
print(f"A/B R={R}: lam-arm 21 rungs x {SW_LAM} sw | u-arm {N_RUNGS} rungs x {SW_U} sw "
      f"(matched compute) | u in [{a.u_min},1.0], T_top {a.t_top}", flush=True)
ncav = 0
for _ in range(60):
    ci = int(torch.randint(921, 1024, (1,), generator=gsel))
    c = torch.rand(3, generator=gsel) * L
    pr = carve(X[ci], S[ci], c, R, L)
    if pr["n_in"] < 14:
        continue
    xin = _mic(pr["x_in"], c, L); xout = _mic(pr["x_out"], c, L)
    bm = xout.norm(dim=-1) < (R + 2.5)
    allx0 = np.concatenate([xin.double().numpy(), xout[bm].double().numpy()])
    alls0 = np.concatenate([pr["s_in"].numpy(), pr["s_out"][bm].numpy()]).astype(np.int64)
    n = xin.shape[0]; n_tot = allx0.shape[0]
    print(f"=== cav {ci} (n={n}) ===", flush=True)
    for mode, coords, temps, sw_budget in (("lam", LAM, T_LAM, SW_LAM), ("u", US, T_US, SW_U)):
        t0 = time.time()
        lad = Ladder(allx0, alls0, n, n_tot, mode, coords, temps,
                     nch=2, exch_mean=10, R=R, seed=100 + ci)
        lad.randomize_stack_B(a.rand_sw)
        res = lad.run(sw_budget, REC, qc_fn=lambda x, y: bcy_qc(x, y, gq))
        # closure sweep: first record where running-mean gap < Q_TOL
        qa, qb = np.array(res["qcA"]), np.array(res["qcB"])
        closure = -1
        for k in range(4, len(qa)):
            if abs(qa[k - 4:k].mean() - qb[k - 4:k].mean()) < Q_TOL:
                closure = res["rec_sw"][k]
                break
        ea = res["exch_acc"] / np.maximum(res["exch_att"], 1)
        lin = np.linspace(1, 0, len(res["flow"]))
        res.update({"closure": closure, "mode": mode, "n": n})
        results[(ci, mode)] = res
        torch.save(results, "reports/logs-2026-07-16/ptu_ab_R20.pt")
        print(f"  {mode:>3}-arm: TRIPS={res['trips']:>4} | closure sweep {closure} "
              f"| final qcA {qa[-1]:+.3f} qcB {qb[-1]:+.3f} gap {abs(qa[-1]-qb[-1]):.3f} "
              f"| labB(end) {res['labB'][-1]:.2f} | exch min/med {ea.min():.2f}/{np.median(ea):.2f} "
              f"| flow-dev {np.abs(res['flow']-lin).max():.2f} ({time.time()-t0:.0f}s)", flush=True)
    ncav += 1
    if ncav >= a.ncav:
        break
print("A/B DONE", flush=True)
```

- [ ] **Step 2: Syntax-check and smoke-run (tiny budget)**

Run: `python -c "import ast; ast.parse(open('reports/logs-2026-07-16/ptu_ab_R20.py').read())" && python reports/logs-2026-07-16/ptu_ab_R20.py --u_min 0.35 --t_top 0.9 --sw 1500 --ncav 1 --rand_sw 200`
Expected: both arm rows print with TRIPS ≥ 0 and no NaN in q_c columns; runtime < 20 min.

- [ ] **Step 3: Launch the full A/B (background)**

Run: `nohup python reports/logs-2026-07-16/ptu_ab_R20.py --u_min <FROM_PROBE> --t_top <FROM_PROBE> --sw 60000 --ncav 2 > reports/logs-2026-07-16/ptu_ab_R20.out 2>&1 &` (~overnight at numba speed; incremental saves per arm per cavity)

- [ ] **Step 4: Evaluate the gate (record verdict in the log and in memory)**

Expected λ-arm baseline (from 2026-07-15 measurements): TRIPS ≈ 0–3 at this budget, closure = −1 (never), gap stuck ≈ 0.4.
**PASS** if, on ≥ half the cavities, the u-arm achieves BOTH: (i) TRIPS ≥ 10, AND (ii) dual-init closure < 60000 sweeps with converged q_c(A,B) mean ∈ [0.55, 0.75] — while the λ-arm fails both at matched compute. → proceed to Task 6/7 and update memory (`identity-bridge works`).
**KILL** if the u-arm's TRIPS/closure are within 2× of the λ-arm's. → the identity channel is not rate-limiting; write the negative to memory (extend `pt-shrinkage-infeasible`), commit, STOP (do not build Task 6/7); the classical G_PTS production falls back to long λ-PT.
**AMBIGUOUS** (u-arm better but no closure): double SW once (120k) on one cavity before deciding.

- [ ] **Step 5: Commit results + verdict**

```bash
git add reports/logs-2026-07-16/ptu_ab_R20.py reports/logs-2026-07-16/ptu_ab_R20.out reports/logs-2026-07-16/ptu_ab_R20.pt
git commit -m "measure(ptu): identity-bridge vs BCY-shrinkage A/B at R=2.0 -- <PASS/KILL verdict one-liner>"
```

---

### Task 6 (GATED on Task 5 PASS): per-particle σ moves (semi-grand facilitation channel)

**Files:**
- Create: `liquid_coupling_flow/ptu/sigma_moves.py`
- Modify: `liquid_coupling_flow/ptu/ladder.py` (add optional `sigma_move=True` path)
- Test: `liquid_coupling_flow/tests/test_ptu_sigma.py`

**Interfaces:**
- Consumes: Ladder from Task 4.
- Produces: `sigma_sweep(allx, alls, sig_i, n, n_tot, beta, dsig, kappa_pin, sig_targets, ...) -> int` — per-particle diameter Gaussian steps σᵢ → σᵢ+N(0,dsig), MH on ΔU + Δ[κ_pin·(σᵢ−σ_target(αᵢ))²]; the pinning strength κ_pin is the rung coordinate (∞ ≡ physical deltas ↔ small ≡ broad). This requires row energies with PER-PARTICLE σ (combination rule σ_ij = (σᵢ+σⱼ)/2·nonadditivity(αᵢ,αⱼ)) — a kernels.py variant `row_e_sigma`.

Detailed steps mirror Tasks 2–3 exactly (failing test: σ-marginal at broad rung matches the Boltzmann of the pinning potential on an ideal 2-particle system, computed analytically; physical rung with κ_pin=1e6 reproduces Task-2 energies to 1e-6). **Design note:** this task is only written in outline because its parameters (κ ladder, dsig) should be probed the same way Task 3 probed u — copy the Task 3 probe pattern verbatim. A full step-level expansion must be written before execution, as an addendum to this plan, if and only if Task 5 passes.

---

### Task 7 (GATED on Task 5 PASS): production — certified q_c at R=2.0 and R=2.3 + write-up

**Files:**
- Create: `reports/logs-2026-07-16/ptu_production.py` (winner arm, NCAV=6 per R, R ∈ {2.0, 2.3}, SW from measured closure ×3 safety)
- Create: `reports/2026-07-16-identity-bridge-verdict.md`
- Modify: memory `pt-shrinkage-infeasible.md` (append the bridge verdict) + `MEMORY.md` index line

**Steps:** (1) run production with per-cavity incremental saves; (2) check every converged cavity's q_c(A,B) ∈ [0.55, 0.75] at R=2.0 (the 3-way-validated anchor) — any violation is a bug, stop and bisect; (3) report G_PTS(2.0), G_PTS(2.3), χ_T at both R (variance of per-snapshot q_c — BCY's susceptibility, our secondary observable), with full P(q_c) histograms saved; (4) write the verdict doc: sweeps-to-certificate vs λ-arm, the mechanism traces (label-overlap decay = the identity channel signature), honest caveats; (5) commit everything; (6) update memory.

---

## Explicitly OUT OF SCOPE (future plans, gated on this one)

- **Any neural network.** Learned identity/redistribution proposals (the 40× denoiser-swap port) come only after Task 5 passes AND the classical identity channel shows proposal-quality (not channel-existence) as its bottleneck.
- Joint (x, σ) flows; the polydisperse-bulk benchmark; CRAFT rung flows.
- R ≥ 2.6 (needs the N=4096 ρ=1.2 equilibrated box — separate work item).
- The retrieval/boundary-chaos probe (separate plan; independent of this one).

## Self-Review

- Spec coverage: identity-bridge concept → Tasks 1–5; facilitation (per-particle σ) → Task 6 (gated, outline-level by design); production G_PTS → Task 7; matched-compute A/B with kill gate → Task 5; classical-only constraint → header + out-of-scope. ✓
- Placeholder scan: Task 6 is intentionally outline-level with an explicit instruction to expand before execution (gated) — declared, not hidden. All other tasks carry complete code. ✓
- Type consistency: `row_e/mobile_U/disp_sweep/identity_sweep` signatures consistent across Tasks 2–4; `build_tables_*` return 4-tuples everywhere; `Ladder.run` return keys match Task 5's consumption (`qcA/qcB/labB/rec_sw/trips/flow/exch_acc/exch_att`). ✓
- Known-number gates wired in: u=1 swap-dead (<0.02), λ-arm TRIPS 0–3 baseline, q_c anchor [0.55, 0.75], flow-dev vs linear. ✓
