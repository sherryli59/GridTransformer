# Joint (σ,x) Block-Proposal Gate (below-swap-arrest project, Gate G1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure the MH acceptance of learned joint (σ-permutation + flow-relaxed-position) block proposals as a function of block size k, at temperatures bracketing the swap-arrest of the standard polydisperse glass-former — the kill/go gate for the "learned kernels below the swap arrest" project.

**Architecture:** A numba CPU substrate for the Ninarello–Berthier–Coslovich (NBC 2017) continuously-polydisperse soft-sphere model (canonical-in-σ ensemble, pair-swap moves — no semi-grand μ calibration needed), swap-equilibrated banks down a descending T-ladder to locate the operational swap arrest, then a σ-conditioned block flow (reusing the exact-divergence `CavityBlockFlow` EGNN) trained on deployment-matched within-block re-equilibration pairs. The gate measurement is acceptance-vs-k against three baselines, plus one bottom-line τ-with-kernel run.

**Tech Stack:** Python 3.11, numba 0.66 (all MC), torch CPU/GPU (reference energies, flow training), pytest. Reuses `liquid_coupling_flow/ka3d_cavity_egnn.py` (CavityBlockFlow) and the ptu testing conventions.

## Global Constraints

- **Model (NBC 2017, exact values):** 3D, N=300, number density ρ=1.0 (L=N^{1/3}), pair potential v(r) = (σ_ij/r)^12 + c0 + c2·(r/σ_ij)^2 + c4·(r/σ_ij)^4 for r < x_c·σ_ij, else 0; x_c = 1.25; c0 = −28/x_c^12, c2 = 48/x_c^14, c4 = −21/x_c^16 (these make v, v′, v″ → 0 at the cutoff — Task 1 tests this analytically). Nonadditive σ_ij = 0.5(σ_i+σ_j)(1 − 0.2|σ_i−σ_j|). Diameters i.i.d. from P(σ) = A·σ^{−3} on [0.725, 1.61] (A normalizes; this gives ⟨σ⟩ = 1.000 — Task 1 tests it). Periodic box, minimum image. All energies float64.
- **Ensemble:** canonical in σ (fixed diameter multiset). σ moves = PAIR SWAPS (σ_i ↔ σ_j), the literature-standard swap MC. No semi-grand chemical potential anywhere.
- **τ_α definition:** self-overlap Q(t) = (1/N)Σ_i θ(0.3 − |r_i(t)−r_i(0)|) with minimum-image displacement; τ_α = first t with Q(t) < e^{−1}, measured from log-spaced origins after equilibration.
- **Operational swap arrest T\*:** the lowest T on the ladder where swap-MC τ_α exceeds the budget 10^7 sweeps (measured, not assumed; literature anchors for sanity: T_MCT ≈ 0.104, onset ≈ 0.2, swap reported effective to ≈ 0.05–0.06 — flag if grossly inconsistent).
- **numba conventions (from the ptu package, binding):** `seed_numba(seed)` for any jitted-RNG reproducibility (plain np.random.seed does NOT reach jitted code); exactness gates vs an independent torch f64 implementation to <1e-8/particle; kill/check processes by exact PID; long runs save incrementally per completed unit; full distributions saved, never just summary scalars.
- **Flow deployment mode (binding, from the campaign's measured negatives):** learned proposals ONLY as block-local MH kernels with exact log q (acceptance depends on block size k, never on N); NO one-shot/global generation; NO importance-weight transport. Training transport must be SHORT (re-equilibration after σ-permutation, not noise→packing).
- **Ciarella-style diagnostics (binding):** report acceptance-vs-k curves (the paper's own headline metric), and check sample diversity of the trained flow (per-block variance of generated positions vs data — the mode-collapse tell).
- **Scheduling:** the 12-core box is saturated by the KA production fleet until it completes; execute Tasks 2+ AFTER the fleet finishes or on the cluster. Tasks 0–1 (code+tests) can run anytime.
- Held-out discipline: bank configs split 90/10 by INDEPENDENT RUN (not by frame); the gate evaluates on held-out configs only.

---

### Task 1: Polydisperse model kernels (numba) + torch reference + exactness gates

**Files:**
- Create: `liquid_coupling_flow/poly/__init__.py` (empty)
- Create: `liquid_coupling_flow/poly/model.py`
- Create: `liquid_coupling_flow/poly/torch_ref.py`
- Test: `liquid_coupling_flow/tests/test_poly_model.py`

**Interfaces:**
- Produces (`model.py`, all float64; positions wrapped in [0,L)):
  - `XC = 1.25; C0 = -28.0/XC**12; C2 = 48.0/XC**14; C4 = -21.0/XC**16; EPS_NA = 0.2`
  - `draw_sigmas(n, seed) -> np.ndarray[n]` — i.i.d. from P(σ)∝σ^{−3} on [0.725, 1.61] via inverse-CDF.
  - `@njit pair_v(r2, sij) -> float` — the smoothed IPL pair energy (0 beyond cutoff).
  - `@njit sigma_ij(si, sj) -> float` — nonadditive combination.
  - `@njit row_e(x, sig, i, xi0, xi1, xi2, L) -> float` — energy of particle i at trial position vs all j≠i, min-image.
  - `@njit total_U(x, sig, L) -> float`.
  - `@njit disp_sweep(x, sig, L, beta, step) -> int` — N single-particle Gaussian-ball displacement attempts, Metropolis; returns accepts.
  - `@njit swap_sweep(x, sig, L, beta, n_try) -> (int, int)` — n_try random-pair σ-exchanges, Metropolis on ΔU; returns (acc, att).
  - `@njit seed_numba(seed)` — jitted np.random.seed.
- Produces (`torch_ref.py`): `total_U_torch(x, sig, L) -> float` — independent f64 implementation (torch.cdist-based, min-image) for the exactness gate. Must NOT import model.py.

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_poly_model.py
import numpy as np
import pytest
from liquid_coupling_flow.poly.model import (XC, C0, C2, C4, draw_sigmas, pair_v, sigma_ij,
                                             row_e, total_U, disp_sweep, swap_sweep, seed_numba)
from liquid_coupling_flow.poly.torch_ref import total_U_torch


def test_cutoff_smoothness_analytic():
    """c0,c2,c4 must make v, v', v'' vanish at r = XC*sij (the NBC C2-smooth cutoff)."""
    sij = 1.0
    rc = XC * sij
    h = 1e-6
    def v(r):
        return pair_v(r * r, sij)
    assert abs(v(rc - 1e-9)) < 1e-9
    d1 = (v(rc - h) - v(rc - 2 * h)) / h
    d2 = (v(rc - h) - 2 * v(rc - 2 * h) + v(rc - 3 * h)) / h ** 2
    assert abs(d1) < 1e-4 and abs(d2) < 1e-1


def test_sigma_distribution_mean_one():
    s = draw_sigmas(200_000, seed=1)
    assert s.min() >= 0.725 - 1e-12 and s.max() <= 1.61 + 1e-12
    assert abs(s.mean() - 1.0) < 2e-3


def test_nonadditivity():
    assert abs(sigma_ij(1.0, 1.0) - 1.0) < 1e-12
    si, sj = 0.8, 1.4
    assert abs(sigma_ij(si, sj) - 0.5 * (si + sj) * (1 - 0.2 * abs(si - sj))) < 1e-12


def test_total_U_matches_torch_reference():
    seed_numba(0)
    rng = np.random.default_rng(0)
    n = 128
    L = n ** (1.0 / 3.0) / 1.0                       # rho = 1
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=2)
    u_nb = total_U(x, sig, L)
    u_th = total_U_torch(x, sig, L)
    assert abs(u_nb - u_th) / n < 1e-8


def test_row_e_consistent_with_total_U():
    seed_numba(0)
    rng = np.random.default_rng(3)
    n = 64
    L = n ** (1.0 / 3.0)
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=4)
    i = 7
    u0 = total_U(x, sig, L)
    xi_new = (x[i] + 0.1) % L
    du_row = (row_e(x, sig, i, xi_new[0], xi_new[1], xi_new[2], L)
              - row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L))
    x2 = x.copy(); x2[i] = xi_new
    du_full = total_U(x2, sig, L) - u0
    assert abs(du_row - du_full) < 1e-9


def test_sweeps_run_and_preserve_multiset():
    seed_numba(0)
    rng = np.random.default_rng(5)
    n = 64
    L = n ** (1.0 / 3.0)
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=6)
    ms = np.sort(sig.copy())
    acc = disp_sweep(x, sig, L, 1.0 / 0.3, 0.15)
    assert 0 < acc < n
    a, t = swap_sweep(x, sig, L, 1.0 / 0.3, 100)
    assert t == 100 and 0 <= a <= t
    assert np.allclose(np.sort(sig), ms)             # pair swaps preserve the diameter multiset
    assert np.all((x >= 0) & (x < L))
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest liquid_coupling_flow/tests/test_poly_model.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'liquid_coupling_flow.poly'`

- [ ] **Step 3: Implement `model.py`**

```python
# liquid_coupling_flow/poly/model.py
"""NBC-2017 continuously polydisperse soft spheres (numba, float64, canonical-in-sigma).
v(r) = (sij/r)^12 + C0 + C2 (r/sij)^2 + C4 (r/sij)^4 for r < XC*sij (C2-smooth at cutoff);
nonadditive sij = 0.5(si+sj)(1 - 0.2|si-sj|); P(sigma) ~ sigma^-3 on [0.725, 1.61] (<sigma>=1)."""
import numpy as np
from numba import njit

XC = 1.25
C0 = -28.0 / XC ** 12
C2 = 48.0 / XC ** 14
C4 = -21.0 / XC ** 16
EPS_NA = 0.2
SIG_MIN, SIG_MAX = 0.725, 1.61


def draw_sigmas(n, seed):
    rng = np.random.default_rng(seed)
    u = rng.random(n)
    a, b = SIG_MIN ** -2, SIG_MAX ** -2                 # inverse CDF of A s^-3
    return 1.0 / np.sqrt(a - u * (a - b))


@njit(cache=True, fastmath=True)
def seed_numba(seed):
    np.random.seed(seed)


@njit(cache=True, fastmath=True)
def sigma_ij(si, sj):
    return 0.5 * (si + sj) * (1.0 - EPS_NA * abs(si - sj))


@njit(cache=True, fastmath=True)
def pair_v(r2, sij):
    rc = XC * sij
    if r2 >= rc * rc:
        return 0.0
    inv = sij * sij / r2
    x2 = r2 / (sij * sij)
    return inv ** 6 + C0 + C2 * x2 + C4 * x2 * x2


@njit(cache=True, fastmath=True)
def row_e(x, sig, i, xi0, xi1, xi2, L):
    e = 0.0
    for j in range(x.shape[0]):
        if j == i:
            continue
        dx = x[j, 0] - xi0; dy = x[j, 1] - xi1; dz = x[j, 2] - xi2
        dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
        e += pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(sig[i], sig[j]))
    return e


@njit(cache=True, fastmath=True)
def total_U(x, sig, L):
    u = 0.0
    for i in range(x.shape[0]):
        for j in range(i + 1, x.shape[0]):
            dx = x[j, 0] - x[i, 0]; dy = x[j, 1] - x[i, 1]; dz = x[j, 2] - x[i, 2]
            dx -= L * np.round(dx / L); dy -= L * np.round(dy / L); dz -= L * np.round(dz / L)
            u += pair_v(dx * dx + dy * dy + dz * dz, sigma_ij(sig[i], sig[j]))
    return u


@njit(cache=True, fastmath=True)
def disp_sweep(x, sig, L, beta, step):
    n = x.shape[0]
    acc = 0
    for _ in range(n):
        i = np.random.randint(n)
        xn0 = (x[i, 0] + step * np.random.randn()) % L
        xn1 = (x[i, 1] + step * np.random.randn()) % L
        xn2 = (x[i, 2] + step * np.random.randn()) % L
        e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
        e1 = row_e(x, sig, i, xn0, xn1, xn2, L)
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            x[i, 0] = xn0; x[i, 1] = xn1; x[i, 2] = xn2
            acc += 1
    return acc


@njit(cache=True, fastmath=True)
def swap_sweep(x, sig, L, beta, n_try):
    n = x.shape[0]
    acc = 0
    for _ in range(n_try):
        i = np.random.randint(n)
        j = np.random.randint(n)
        if i == j:
            j = (j + 1) % n
        e0 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
              + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L))
        si = sig[i]
        sig[i] = sig[j]; sig[j] = si
        e1 = (row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
              + row_e(x, sig, j, x[j, 0], x[j, 1], x[j, 2], L))
        if np.random.random() < np.exp(-beta * (e1 - e0)):
            acc += 1
        else:
            sj = sig[i]
            sig[i] = sig[j]; sig[j] = sj
    return acc, n_try
```

NOTE for the implementer (same double-count argument as the ptu identity kernel): in `swap_sweep`, e0/e1 each count the i–j direct pair twice, but `sigma_ij` is symmetric under the σ exchange, so the doubled term cancels exactly in e1−e0.

- [ ] **Step 4: Implement `torch_ref.py`**

```python
# liquid_coupling_flow/poly/torch_ref.py
"""Independent f64 torch reference for the exactness gate. Must NOT import poly.model."""
import torch

XC = 1.25
C0 = -28.0 / XC ** 12
C2 = 48.0 / XC ** 14
C4 = -21.0 / XC ** 16


def total_U_torch(x, sig, L):
    x = torch.as_tensor(x, dtype=torch.float64)
    s = torch.as_tensor(sig, dtype=torch.float64)
    d = x[:, None, :] - x[None, :, :]
    d -= L * torch.round(d / L)
    r2 = (d ** 2).sum(-1)
    sij = 0.5 * (s[:, None] + s[None, :]) * (1.0 - 0.2 * (s[:, None] - s[None, :]).abs())
    n = x.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    r2u = r2[iu[0], iu[1]]
    su = sij[iu[0], iu[1]]
    inside = r2u < (XC * su) ** 2
    inv6 = (su ** 2 / r2u.clamp_min(1e-18)) ** 6
    x2 = r2u / su ** 2
    v = inv6 + C0 + C2 * x2 + C4 * x2 * x2
    return float(v[inside].sum())
```

- [ ] **Step 5: Run tests to verify pass**

Run: `python -m pytest liquid_coupling_flow/tests/test_poly_model.py -v`
Expected: 6 passed (first run ~60 s numba JIT)

- [ ] **Step 6: Commit**

```bash
git add liquid_coupling_flow/poly/__init__.py liquid_coupling_flow/poly/model.py liquid_coupling_flow/poly/torch_ref.py liquid_coupling_flow/tests/test_poly_model.py
git commit -m "feat(poly): NBC polydisperse soft-sphere numba kernels, exactness-gated vs independent torch ref"
```

---

### Task 2: Swap-equilibrated T-ladder bank + τ_α(T) curves + arrest location

**Files:**
- Create: `reports/logs-2026-07-17/poly_bank.py`
- Output: `reports/logs-2026-07-17/poly_bank_T{T}.pt` per rung, `poly_tau_curves.pt`, `poly_bank.log`

**Interfaces:**
- Consumes: everything from Task 1.
- Produces: per-T banks `{"x": [n_frames, N, 3], "sig": [N], "L": float, "T": float, "tau_local": float, "tau_swap": float}`, the τ(T) table, and the measured `T_STAR` (operational arrest) recorded in the log and in `poly_tau_curves.pt`.

- [ ] **Step 1: Write the bank script**

```python
# reports/logs-2026-07-17/poly_bank.py
"""Descending-T swap-equilibrated banks for the NBC polydisperse model + tau_alpha(T) for
local-only vs local+swap dynamics. Locates the OPERATIONAL SWAP ARREST T* = lowest rung where
tau_swap exceeds BUDGET sweeps. Each rung hands its final configs to the next (swap-equilibrated
cooling). Incremental save per rung. Usage: poly_bank.py [N] [BUDGET]"""
import sys, time
import numpy as np
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import (draw_sigmas, disp_sweep, swap_sweep, total_U,
                                             seed_numba)

N = int(sys.argv[1]) if len(sys.argv) > 1 else 300
BUDGET = int(sys.argv[2]) if len(sys.argv) > 2 else 10_000_000
LADDER = [0.30, 0.20, 0.15, 0.12, 0.10, 0.085, 0.075, 0.065, 0.058, 0.052, 0.047]
N_RUNS = 4                       # independent runs (90/10 held-out split is BY RUN)
EQ_SW = 200_000                  # per-rung equilibration before tau measurement (adaptive: 20*tau_prev)
STEP = 0.12
A_OV = 0.3
L = N ** (1.0 / 3.0)
seed_numba(11)


def q_self(x, x0, L):
    d = x - x0
    d -= L * np.round(d / L)
    return float((np.sqrt((d ** 2).sum(1)) < A_OV).mean())


def tau_alpha(x, sig, L, beta, use_swap, budget):
    """Return first t (sweeps) with Q(t) < 1/e, or -1 if not reached within budget. Logs Q(t)."""
    x0 = x.copy()
    t = 0
    ts, qs = [], []
    next_meas = 1
    while t < budget:
        disp_sweep(x, sig, L, beta, STEP)
        if use_swap:
            swap_sweep(x, sig, L, beta, N)
        t += 1
        if t >= next_meas:
            q = q_self(x, x0, L)
            ts.append(t); qs.append(q)
            if q < np.exp(-1.0):
                return t, ts, qs
            next_meas = int(next_meas * 1.3) + 1
    return -1, ts, qs


out = {"LADDER": LADDER, "N": N, "BUDGET": BUDGET}
for run in range(N_RUNS):
    rng = np.random.default_rng(run)
    x = rng.random((N, 3)) * L
    sig = draw_sigmas(N, seed=100 + run)
    for T in LADDER:
        beta = 1.0 / T
        t0 = time.time()
        for _ in range(EQ_SW):
            disp_sweep(x, sig, L, beta, STEP)
            swap_sweep(x, sig, L, beta, N)
        frames = []
        tau_s, ts_s, qs_s = tau_alpha(x.copy(), sig.copy(), L, beta, True, BUDGET)
        tau_l, ts_l, qs_l = tau_alpha(x.copy(), sig.copy(), L, beta, False, min(BUDGET, 2_000_000))
        for _ in range(16):                              # bank frames, swap-decorrelated
            for _ in range(max(1000, 3 * max(tau_s, 1))):
                disp_sweep(x, sig, L, beta, STEP)
                swap_sweep(x, sig, L, beta, N)
            frames.append(x.copy())
        out[(run, T)] = {"x": np.stack(frames), "sig": sig.copy(), "L": L, "T": T,
                         "tau_swap": tau_s, "tau_local": tau_l,
                         "Q_swap": (ts_s, qs_s), "Q_local": (ts_l, qs_l),
                         "U_N": total_U(x, sig, L) / N}
        torch.save(out, "reports/logs-2026-07-17/poly_tau_curves.pt")
        print(f"run {run} T={T}: tau_swap={tau_s} tau_local={tau_l} U/N={out[(run,T)]['U_N']:+.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
        if tau_s < 0:
            print(f"run {run}: SWAP ARREST at T={T} (tau_swap > {BUDGET})", flush=True)
            break
print("BANK DONE", flush=True)
```

- [ ] **Step 2: Syntax-check + micro-smoke**

Run: `python -c "import ast; ast.parse(open('reports/logs-2026-07-17/poly_bank.py').read())"` then a smoke with `N=64 BUDGET=200000` and LADDER truncated via env or a quick edit-revert: `python reports/logs-2026-07-17/poly_bank.py 64 200000` interrupted after the first two rungs print.
Expected: τ_swap ≪ τ_local at T=0.15–0.12 (swap speedup visible); U/N decreasing with T.

- [ ] **Step 3: Launch full bank in background (AFTER the KA fleet frees cores)**

Run: `nohup python reports/logs-2026-07-17/poly_bank.py 300 10000000 > reports/logs-2026-07-17/poly_bank.log 2>&1 &`
Expected wall: hours–1 day. **Decision recorded in log:** T_STAR = lowest rung with tau_swap = −1 (per run; use the median over runs). Sanity anchors: swap speedup ≥ 100× somewhere below T=0.12; T_STAR in [0.045, 0.07] (flag if wildly off literature).

- [ ] **Step 4: Commit script + (when done) curves**

```bash
git add reports/logs-2026-07-17/poly_bank.py reports/logs-2026-07-17/poly_bank.log reports/logs-2026-07-17/poly_tau_curves.pt
git commit -m "feat(poly): swap T-ladder bank + tau curves; operational swap arrest located"
```

---

### Task 3: Deployment-matched block-pair training data

**Files:**
- Create: `reports/logs-2026-07-17/poly_block_data.py`
- Test: `liquid_coupling_flow/tests/test_poly_blockdata.py`
- Output: `reports/logs-2026-07-17/poly_blockpairs_T{T}.pt`

**Interfaces:**
- Consumes: Task 1 kernels; Task 2 banks.
- Produces: `make_block_pair(x, sig, L, beta, k, seed) -> dict` with keys `idx[k]` (block particle indices = k-NN of a random seed particle), `env_x[N-k,3]`, `env_sig[N-k]`, `x_old[k,3]`, `sig_old[k]`, `perm[k]` (the internal σ-permutation applied), `sig_new[k]`, `x_new[k,3]` (block positions re-equilibrated under sig_new with env frozen, by `RELAX_SW` sweeps of block-local displacement MC), `L`, `beta`. Positions are min-imaged about the block centroid (the flow works in centered coordinates, like the cavity code).
- The dataset file: a list of such dicts, ≥ 20,000 pairs per (T, k) from TRAIN-run banks only.

- [ ] **Step 1: Failing test**

```python
# liquid_coupling_flow/tests/test_poly_blockdata.py
import numpy as np
import torch
import sys
sys.path.insert(0, "/mnt/ssd/GridTransformer/reports/logs-2026-07-17")
from poly_block_data import make_block_pair
from liquid_coupling_flow.poly.model import draw_sigmas, seed_numba


def test_block_pair_structure_and_invariants():
    seed_numba(0)
    rng = np.random.default_rng(0)
    n = 128
    L = n ** (1.0 / 3.0)
    x = rng.random((n, 3)) * L
    sig = draw_sigmas(n, seed=1)
    p = make_block_pair(x, sig, L, beta=1.0 / 0.12, k=8, seed=3)
    assert p["x_old"].shape == (8, 3) and p["x_new"].shape == (8, 3)
    assert np.allclose(np.sort(p["sig_new"]), np.sort(p["sig_old"]))     # internal perm only
    assert not np.allclose(p["sig_new"], p["sig_old"])                   # a real permutation
    assert p["env_x"].shape[0] == n - 8
    # env untouched, block moved but bounded (short transport)
    disp = np.abs(p["x_new"] - p["x_old"]).max()
    assert 0 < disp < 1.5
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest liquid_coupling_flow/tests/test_poly_blockdata.py -v` → ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# reports/logs-2026-07-17/poly_block_data.py
"""Deployment-matched block pairs: freeze env, permute the block's own sigmas (2 random
transpositions), re-equilibrate block positions by block-local displacement MC (RELAX_SW sweeps).
Pair = (old positions + old assignment) -> (relaxed positions + new assignment). Short transport,
same-particle correspondence (no OT). Usage as module (make_block_pair) or script (build dataset)."""
import sys
import numpy as np
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.poly.model import row_e, seed_numba
from numba import njit

RELAX_SW = 400


@njit(cache=True, fastmath=True)
def _block_relax(x, sig, L, beta, idx, n_sw, step):
    for _ in range(n_sw):
        for t in range(idx.shape[0]):
            i = idx[np.random.randint(idx.shape[0])]
            xn0 = (x[i, 0] + step * np.random.randn()) % L
            xn1 = (x[i, 1] + step * np.random.randn()) % L
            xn2 = (x[i, 2] + step * np.random.randn()) % L
            e0 = row_e(x, sig, i, x[i, 0], x[i, 1], x[i, 2], L)
            e1 = row_e(x, sig, i, xn0, xn1, xn2, L)
            if np.random.random() < np.exp(-beta * (e1 - e0)):
                x[i, 0] = xn0; x[i, 1] = xn1; x[i, 2] = xn2


def make_block_pair(x, sig, L, beta, k, seed):
    rng = np.random.default_rng(seed)
    seed_numba(seed)
    n = x.shape[0]
    s0 = rng.integers(n)
    d = x - x[s0]
    d -= L * np.round(d / L)
    idx = np.argsort((d ** 2).sum(1))[:k].astype(np.int64)
    mask = np.zeros(n, bool); mask[idx] = True
    xw, sw = x.copy(), sig.copy()
    x_old = xw[idx].copy(); sig_old = sw[idx].copy()
    perm = rng.permutation(k)
    while np.all(perm == np.arange(k)):
        perm = rng.permutation(k)
    sw[idx] = sig_old[perm]
    _block_relax(xw, sw, L, beta, idx, RELAX_SW, 0.1)
    cen = x_old.mean(0)
    def cent(a):
        b = a - cen
        return b - L * np.round(b / L)
    return {"idx": idx, "env_x": cent(xw[~mask]), "env_sig": sw[~mask],
            "x_old": cent(x_old), "sig_old": sig_old, "perm": perm,
            "sig_new": sw[idx].copy(), "x_new": cent(xw[idx]), "L": L, "beta": beta}


if __name__ == "__main__":
    import torch, time
    T = float(sys.argv[1]); K = int(sys.argv[2]); NPAIRS = int(sys.argv[3])
    bank = torch.load("reports/logs-2026-07-17/poly_tau_curves.pt", weights_only=False)
    train_runs = [0, 1, 2]                                     # run 3 held out
    pairs = []
    t0 = time.time()
    for m in range(NPAIRS):
        run = train_runs[m % len(train_runs)]
        rec = bank[(run, T)]
        fr = rec["x"][m % rec["x"].shape[0]]
        pairs.append(make_block_pair(fr, rec["sig"], rec["L"], 1.0 / T, K, seed=9000 + m))
        if (m + 1) % 2000 == 0:
            torch.save(pairs, f"reports/logs-2026-07-17/poly_blockpairs_T{T}_k{K}.pt")
            print(f"{m+1}/{NPAIRS} ({time.time()-t0:.0f}s)", flush=True)
    torch.save(pairs, f"reports/logs-2026-07-17/poly_blockpairs_T{T}_k{K}.pt")
    print("PAIRS DONE", flush=True)
```

- [ ] **Step 4: Tests pass** — `python -m pytest liquid_coupling_flow/tests/test_poly_blockdata.py -v` → 1 passed.
- [ ] **Step 5: Commit** — `git add reports/logs-2026-07-17/poly_block_data.py liquid_coupling_flow/tests/test_poly_blockdata.py && git commit -m "feat(poly): deployment-matched (sigma-perm -> relaxed-x) block-pair generator"`

---

### Task 4: σ-conditioned block flow (reuse CavityBlockFlow) + FM training

**Files:**
- Create: `liquid_coupling_flow/poly/block_flow.py`
- Create: `reports/logs-2026-07-17/poly_train_blockflow.py`
- Test: `liquid_coupling_flow/tests/test_poly_blockflow.py`

**Interfaces:**
- Consumes: `CavityBlockFlow` from `liquid_coupling_flow/ka3d_cavity_egnn.py` (contract: `flow(x0_block[B,k,3], cage_x[B,m,3], sp_block[B,k], sp_cage[B,m]) -> (x1, logdet)`; species enter as integer labels). σ is CONTINUOUS — discretize σ into `NSIG=8` quantile bins of P(σ) and feed the bin index as the species label (n_species=8). The flow moves positions only; σ values enter through the labels of movers and env.
- Produces (`block_flow.py`): `class PolyBlockFlow` wrapping CavityBlockFlow with `n_species=8`, `sigma_to_bin(sig) -> int64 labels`, fixed sizes `K_MAX` (movers, dummy-padded exactly like train_cavity_ersi.py: spread dummies at (50+5j,0,0)) and `M_ENV=64` nearest env slots; methods:
  - `propose(x_old[k,3], sig_new_labels[k], env_x, env_labels, gen) -> (x_new[k,3], logq_fwd)` — base = x_old + Gaussian(width `BASE_W=0.35`), flow-transport, exact logq (base logp − logdet along the SAME fixed-step RK4 used for sampling; RK4_STEPS=24; logdet integrated with the same fixed grid so sample and density are self-consistent by construction).
  - `logq_of(x_target[k,3], x_center[k,3], sig_labels[k], env_x, env_labels) -> float` — density of producing x_target when base is centered at x_center (reverse leg of MH).
- Produces (training script): FM on Task-3 pairs; target v = x_new − x_t interpolant with base = x_old + noise (short transport by construction); prints FM loss AND the v=0 baseline E|x_new−x_old|² (the plan's measured-lesson: judge loss only relative to that floor); saves `liquid_coupling_flow/artifacts/poly_blockflow_T{T}_k{K}_best.pt`.
- Mode-collapse diagnostic (Ciarella-binding): after training, generate 64 proposals for one held-out block and report per-particle position variance vs the variance across 64 *data* re-relaxations of the same block — ratio in [0.5, 2] required to proceed.

- [ ] **Step 1: Failing test** — shape/logq-consistency test: `propose` then `logq_of` on the produced sample equals logq_fwd to 1e-6 (self-consistency of forward density); dummies inert (moving a dummy's env has no effect on real-particle proposals within 1e-8).

```python
# liquid_coupling_flow/tests/test_poly_blockflow.py
import numpy as np
import torch
from liquid_coupling_flow.poly.block_flow import PolyBlockFlow, sigma_to_bin


def test_propose_logq_selfconsistent():
    torch.manual_seed(0)
    f = PolyBlockFlow(k_max=16, m_env=32, hidden_nf=32, n_layers=2)      # tiny untrained net is fine
    k = 8
    rng = np.random.default_rng(0)
    x_old = rng.standard_normal((k, 3)) * 0.5
    sig = rng.uniform(0.725, 1.61, k)
    env_x = rng.standard_normal((40, 3)) * 2 + 3.0
    env_sig = rng.uniform(0.725, 1.61, 40)
    gen = torch.Generator().manual_seed(1)
    x_new, logq = f.propose(x_old, sigma_to_bin(sig), env_x, sigma_to_bin(env_sig), gen)
    assert x_new.shape == (k, 3) and np.isfinite(logq)
    logq2 = f.logq_of(x_new, x_old, sigma_to_bin(sig), env_x, sigma_to_bin(env_sig))
    assert abs(logq - logq2) < 1e-4
```

- [ ] **Step 2: verify failure** → ModuleNotFoundError.
- [ ] **Step 3: implement `block_flow.py`** (wrap CavityBlockFlow exactly as `gr_cavity_ersi.py`'s `sample_rk4` does for sampling, plus a parallel fixed-grid divergence integration for logdet; base logp = isotropic Gaussian at BASE_W; centered coordinates; dummy padding identical to train_cavity_ersi.py `dummies()`).
- [ ] **Step 4: implement + run the trainer** on T just above T_STAR, k ∈ {8, 16}: `python reports/logs-2026-07-17/poly_train_blockflow.py --T <T_above_star> --k 8 --steps 20000`. Gate: FM loss < 0.5 × (v=0 floor) AND diversity ratio in [0.5, 2].
- [ ] **Step 5: tests pass, commit** — `git commit -m "feat(poly): sigma-conditioned block flow (exact fixed-grid logq) + FM trainer"`

---

### Task 5: THE GATE — acceptance vs k, bracketing the arrest

**Files:**
- Create: `reports/logs-2026-07-17/poly_gate_acceptance.py`
- Output: `poly_gate_acceptance.pt`, `poly_gate_acceptance.png`

**Interfaces:**
- Consumes: Tasks 1–4. Held-out run (run 3) banks at T ∈ {1.5·T_STAR, ≈1.1·T_STAR, T_STAR}.
- The proposal to be measured (exact MH, per attempt): pick seed particle → block = k-NN; draw internal σ-permutation π uniformly (2 transpositions); propose x'_block = flow.propose(x_old, labels(σ∘π), env); accept with
  `A = min(1, exp(−βΔU_total) · q_rev / q_fwd)` where `ΔU_total` = energy change of the block move (positions AND σ assignment, computed with row_e sums over block particles, double-count-corrected), `q_fwd = flow density of x' given (x_old-centered base, σ∘π labels)`, `q_rev = flow.logq_of(x_old, x'_centered base, σ labels)`. The uniform-permutation factors cancel by symmetry.
- Baselines measured in the same run: (a) plain pair-swap acceptance (k=2 classical anchor); (b) x-only flow proposal (identity permutation — isolates the σ-channel's contribution); (c) σ-permutation + RELAX_SW block-local MC as an MTM-style non-learned comparator (acceptance of its end state under the same exact-MH bookkeeping is not well-defined — instead report its raw ΔU distribution as context only, clearly labeled).
- Grid: k ∈ {2, 4, 8, 16, 24} × 3 temperatures × ≥ 2000 attempts each, from held-out configs; save EVERY attempt's (ΔU, logq_fwd, logq_rev, accepted) — full distributions, not means.

- [ ] **Step 1: Write the gate script** (structure identical to Task 2's script conventions: argparse T/k lists, incremental save per (T,k) cell, per-cell print `T={} k={} acc={:.4f} n={} dU_med={:+.2f} logq_gap_med={:+.2f}`).
- [ ] **Step 2: Smoke** on 50 attempts at the easiest cell (highest T, k=4): acceptance must be > 0 and finite logq everywhere.
- [ ] **Step 3: Full run + plot** acceptance-vs-k, one curve per T, baselines overlaid.
- [ ] **Step 4: Evaluate the gate and record the verdict in the log, the plan ledger, and memory:**

| Outcome | Verdict |
|---|---|
| acc(k=8) ≥ 1% at T_STAR AND joint ≥ 3× the x-only baseline at the same (T,k) | **PASS** → proceed to the kernel-in-production plan (τ-with-kernel campaign, separate plan) |
| acc decays to <0.1% by k=8 at all T (the 2D-KA selection-wall pattern) | **KILL** → record: the σ-augmentation does not evade the collective-move selection wall; write the negative + Ciarella-extension note |
| acc(k=8) ≥ 1% but joint ≈ x-only | **KILL the σ-hypothesis specifically** (positions alone carry the acceptance; the swap-analog channel is not the lever) — record which |

- [ ] **Step 5: Commit everything** — `git commit -m "measure(poly): GATE joint (sigma,x) block-proposal acceptance vs k bracketing swap arrest -- <verdict>"`

---

## Explicitly OUT OF SCOPE (separate plans, gated on Task 5 PASS)

- The production below-arrest campaign (kernel mixed into swap-MC, τ(T) frontier extension, CRAFT/T4 variants).
- Semi-grand ensemble variants; N-scaling studies; T-conditioning of the flow.
- Any one-shot/global generative deployment (permanently out, per the campaign's five refutations and Ciarella's global-MCMC result).

## Self-Review

- Coverage: gate question (acceptance vs k, bracketing arrest) → Tasks 2 (locates arrest) + 5 (measures acceptance); learned proposal with exact density → Task 4; deployment-matched training → Task 3; substrate exactness → Task 1; Ciarella diagnostics → Task 4 diversity gate + Task 5 full-distribution logging. ✓
- Placeholders: Task 4 steps 3–4 specify exact contracts, constants (NSIG=8, BASE_W=0.35, RK4_STEPS=24, M_ENV=64), and the reuse source (`sample_rk4` pattern from gr_cavity_ersi.py); implementation is derivative of committed code by explicit reference. Task 5 step 1 pins the exact acceptance formula and logging schema in the Interfaces block. ✓
- Type consistency: `make_block_pair` keys consumed by the trainer and gate match Task 3's produced dict; `propose/logq_of` signatures consistent between Tasks 4 and 5; `sigma_to_bin` defined in Task 4, used in Task 5. ✓
- Known-number anchors wired in: ⟨σ⟩=1.000 test, C2-smooth cutoff analytic test, swap-speedup ≥100× sanity, T_STAR ∈ [0.045, 0.07] flag, v=0 FM floor, diversity ratio ∈ [0.5, 2]. ✓
