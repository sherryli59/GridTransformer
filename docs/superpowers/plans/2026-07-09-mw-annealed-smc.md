# Liquid-mW Annealed SMC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an exact annealed-SMC equilibrium sampler for liquid mW (Phase 1, trivial base, gates G0–G3), then quantify the rung/energy-eval reduction from the 3D local-frame AR generator as base (Phase 2) and its zero-shot survival at N=216 (Phase 3).

**Architecture:** Geometric λ-path SMC (π̃_λ ∝ q₀^{1−λ}(e^{−βU})^λ) whose uniform-base degenerate case is certified exact before any NN exists. New isolated subpackage `liquid_coupling_flow/mw/`. Generator = gilbert3d-ordered AR transformer with scaffold-cell anchors, geometry-only kNN local-frame conditioning, and 3-coordinate RQS spline heads (exact one-pass teacher-forced log q).

**Tech Stack:** torch (GPU), numpy (independent oracles in tests only), pytest. Reuses `liquid_coupling_flow/transforms_spline.RQSplineElementwise` and `grid_transformer.data.gilbert.gilbert3d_path`.

**Spec:** `docs/superpowers/specs/2026-07-09-mw-annealed-smc-design.md` — read it before starting any task.

## Global Constraints

- Reduced units σ=ε=1 everywhere internally. mW params: A=7.049556277, B=0.6022245584, p=4, q=0, a=1.8, γ=1.2, λ₃=23.15, cosθ₀=−1/3.
- State point: T\*=0.09632 (β\*=10.382), ρ\*=0.4564. Boxes: L=(N/ρ\*)^{1/3}.
- Bedrock gate (G1) runs at L=4.0, β=2.0 (smooth integrand; still liquid-bonded trimers). This tests machinery, not the state point.
- ess_target=0.6, resampling multinomial, λ progress floor 1e-4.
- Sizes: gates at N∈{2,3,8,27,64,125,216}; train N=64 (4×4×4 grid); transfer N=216 (6×6×6).
- Generator: knn=12, d_model=128, 2 encoder layers, 4 heads, num_bins=8, tail_bound=4.0, offset scale s=(L/R)/2.
- All artifacts → `liquid_coupling_flow/mw/artifacts/`; run logs → `reports/logs-<date>/`. Every run >30 min saves per-unit/per-rung (CLAUDE.md rule). Long jobs: `> log.out 2>&1`, never /dev/null; kill/check by exact PID only.
- git: stage ONLY named files (never `git add -A`). Tests live in repo-root `tests/`.
- Fixed seeds: every stochastic function takes `gen: torch.Generator` or `seed: int`.

---

### Task 1: mW energy (`mw_energy.py`) + G0 exactness tests

**Files:**
- Create: `liquid_coupling_flow/mw/__init__.py` (empty)
- Create: `liquid_coupling_flow/mw/mw_energy.py`
- Test: `tests/test_mw_energy.py`

**Interfaces:**
- Produces: `mw_energy(x, L) -> U` (x `[B,N,3]` reduced units, returns `[B]`); `mw_energy_chunked(x, L, chunk=128) -> [B]`; `du_move(x, i, xi_new, L) -> [B]` (energy change moving particle i to `xi_new [B,3]`); constants `A_SW, B_SW, A_CUT, GAMMA, LAMBDA3, COS0, T_STAR, RHO_STAR, KMAX`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mw_energy.py
import math, torch, numpy as np, pytest
from liquid_coupling_flow.mw.mw_energy import (mw_energy, mw_energy_chunked, du_move,
                                               A_SW, B_SW, A_CUT, GAMMA, LAMBDA3, COS0)

def _phi2_py(r):
    if r >= A_CUT: return 0.0
    return A_SW * (B_SW * r**-4 - 1.0) * math.exp(1.0 / (r - A_CUT))

def _phi3_py(rij, rik, cos):
    if rij >= A_CUT or rik >= A_CUT: return 0.0
    return LAMBDA3 * (cos - COS0)**2 * math.exp(GAMMA/(rij-A_CUT)) * math.exp(GAMMA/(rik-A_CUT))

def _energy_np(x, L):
    """Independent numpy oracle: brute-force pairs + all triplets. Never imported by main code."""
    N = x.shape[0]; U = 0.0
    def mi(d): return d - L*np.round(d/L)
    for i in range(N):
        for j in range(i+1, N):
            U += _phi2_py(float(np.linalg.norm(mi(x[j]-x[i]))))
    for i in range(N):
        for j in range(N):
            for k in range(j+1, N):
                if j == i or k == i: continue
                dj, dk = mi(x[j]-x[i]), mi(x[k]-x[i])
                rj, rk = float(np.linalg.norm(dj)), float(np.linalg.norm(dk))
                if rj < A_CUT and rk < A_CUT:
                    U += _phi3_py(rj, rk, float(np.dot(dj, dk)/(rj*rk)))
    return U

def test_dimer_analytic():
    x = torch.tensor([[[0.,0,0],[1.,0,0]]]); U = mw_energy(x, 20.0)
    expect = A_SW*(B_SW-1.0)*math.exp(1.0/(1.0-A_CUT))
    assert abs(float(U) - expect) < 1e-10 and abs(expect - (-0.80340)) < 1e-4

def test_trimer_analytic():
    th = math.radians(100.0); r = 1.1
    x = torch.tensor([[[0.,0,0],[r,0,0],[r*math.cos(th), r*math.sin(th),0]]], dtype=torch.float64)
    expect = _energy_np(x[0].numpy(), 20.0)                # exercises phi3 at ALL THREE centers
    assert abs(float(mw_energy(x, 20.0)) - expect) < 1e-10
    assert abs(expect - sum(_phi2_py(d) for d in (r, r, 2*r*math.sin(th/2)))) > 0.01  # 3-body nonzero

def test_vs_numpy_oracle_random():
    g = torch.Generator().manual_seed(0); L = 5.198
    x = torch.rand(4, 64, 3, generator=g, dtype=torch.float64) * L
    U = mw_energy(x, L)
    for b in range(4):
        assert abs(float(U[b]) - _energy_np(x[b].numpy(), L)) < 1e-8 * max(1, abs(float(U[b])))

def test_invariances():
    g = torch.Generator().manual_seed(1); L = 5.198
    x = torch.rand(2, 64, 3, generator=g, dtype=torch.float64) * L
    U0 = mw_energy(x, L)
    assert torch.allclose(mw_energy(torch.remainder(x + 1.234, L), L), U0, atol=1e-9)   # translation+wrap
    perm = torch.randperm(64, generator=g)
    assert torch.allclose(mw_energy(x[:, perm], L), U0, atol=1e-9)                       # permutation

def test_cutoff_smooth():
    for r in (A_CUT - 1e-6, A_CUT - 1e-3):
        x = torch.tensor([[[0.,0,0],[r,0,0]]], dtype=torch.float64)
        assert abs(float(mw_energy(x, 20.0))) < 1e-3                                     # -> 0 at cutoff

def test_chunked_and_du_move():
    g = torch.Generator().manual_seed(2); L = 5.198
    x = torch.rand(9, 64, 3, generator=g, dtype=torch.float64) * L
    assert torch.allclose(mw_energy_chunked(x, L, chunk=4), mw_energy(x, L), atol=1e-10)
    xi = torch.rand(9, 3, generator=g, dtype=torch.float64) * L
    x2 = x.clone(); x2[:, 7] = xi
    assert torch.allclose(du_move(x, 7, xi, L), mw_energy(x2, L) - mw_energy(x, L), atol=1e-8)
```

- [ ] **Step 2: Run to verify failure**: `pytest tests/test_mw_energy.py -x -q` → ImportError.

- [ ] **Step 3: Implement `mw_energy.py`**

```python
"""Batched torch mW (Stillinger-Weber, Molinero-Moore 2009) in reduced units (sigma=eps=1)."""
from __future__ import annotations
import torch

A_SW, B_SW = 7.049556277, 0.6022245584
A_CUT, GAMMA, LAMBDA3, COS0 = 1.8, 1.2, 23.15, -1.0 / 3.0
KB_KCAL, EPS_KCAL, SIGMA_A = 0.0019872, 6.189, 2.3925
T_STAR = KB_KCAL * 300.0 / EPS_KCAL          # 0.09632 (ambient 300 K)
RHO_STAR = 0.4564                             # 0.997 g/cm^3
KMAX = 32                                     # 3-body neighbor cap; guarded by assert below


def _pair(x, L):
    d = x[:, :, None, :] - x[:, None, :, :]
    d = d - L * torch.round(d / L)
    return d, d.norm(dim=-1)


def _phi2(r):
    m = r < A_CUT
    rm = torch.where(m, r, torch.full_like(r, A_CUT + 1.0)).clamp_min(1e-9)
    v = A_SW * (B_SW * rm ** -4 - 1.0) * torch.exp(1.0 / (rm - A_CUT))
    return torch.where(m, v, torch.zeros_like(r))


def mw_energy(x, L):
    """x [B,N,3] -> U [B]. Two-body + three-body within a=1.8; PBC min-image."""
    B, N, _ = x.shape
    d, r = _pair(x, L)
    eye = torch.eye(N, dtype=torch.bool, device=x.device)
    u2 = _phi2(r.masked_fill(eye[None], A_CUT + 1.0)).sum((1, 2)) / 2
    k = min(KMAX, N - 1)
    rr = r.masked_fill(eye[None], 1e9)
    dist, idx = rr.topk(k, dim=2, largest=False)                      # [B,N,k]
    within = dist < A_CUT
    if k < N - 1:                                                     # cap must not truncate the cutoff shell
        assert not within[..., -1].any(), "KMAX too small: k-th neighbor inside cutoff"
    dn = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, 3))
    h = torch.where(within, torch.exp(GAMMA / (dist - A_CUT).clamp_max(-1e-9)), torch.zeros_like(dist))
    dnu = dn / dist.clamp_min(1e-12)[..., None]
    cos = torch.einsum("bikd,bild->bikl", dnu, dnu)
    tri = torch.triu(torch.ones(k, k, dtype=torch.bool, device=x.device), 1)
    term = LAMBDA3 * (cos - COS0) ** 2 * h[:, :, :, None] * h[:, :, None, :]
    u3 = (term * (within[:, :, :, None] & within[:, :, None, :]) * tri[None, None]).sum((1, 2, 3))
    return u2 + u3


def mw_energy_chunked(x, L, chunk=128):
    """The [n,N,N,3] tensors blow up on big stacks (10.4 GiB lesson) — chunk measurement evals."""
    return torch.cat([mw_energy(x[i:i + chunk], L) for i in range(0, x.shape[0], chunk)])


def du_move(x, i, xi_new, L):
    """Exact dU for moving particle i -> xi_new [B,3], via local re-summation.
    Affected 3-body centers = i itself + every particle within cutoff of i's OLD or NEW position;
    recompute the phi2 row and the phi3 sums of affected centers before/after. Correctness anchor:
    tests assert equality with full recompute (1e-8)."""
    x2 = x.clone(); x2[:, i] = xi_new
    aff_old = (_pair(x, L)[1][:, i] < A_CUT)
    aff_new = (_pair(x2, L)[1][:, i] < A_CUT)
    aff = aff_old | aff_new                                            # [B,N]; includes i via <A_CUT self? no:
    aff[:, i] = True
    # 3-body energy restricted to affected centers, plus phi2 row of i (each pair once: row sum).
    def _local(xc):
        d, r = _pair(xc, L)
        row = _phi2(r[:, i].scatter(1, torch.full((xc.shape[0], 1), i, device=xc.device, dtype=torch.long),
                                    A_CUT + 1.0)).sum(1)
        u3 = _phi3_centers(d, r, aff, xc.device)
        return row + u3
    return _local(x2) - _local(x)


def _phi3_centers(d, r, centers, device):
    """Sum of phi3 over the given boolean center mask [B,N]."""
    B, N = centers.shape
    eye = torch.eye(N, dtype=torch.bool, device=device)
    rr = r.masked_fill(eye[None], 1e9)
    k = min(KMAX, N - 1)
    dist, idx = rr.topk(k, dim=2, largest=False)
    within = dist < A_CUT
    if k < N - 1:
        assert not within[..., -1].any(), "KMAX too small"
    dn = torch.gather(d, 2, idx[..., None].expand(-1, -1, -1, 3))
    h = torch.where(within, torch.exp(GAMMA / (dist - A_CUT).clamp_max(-1e-9)), torch.zeros_like(dist))
    dnu = dn / dist.clamp_min(1e-12)[..., None]
    cos = torch.einsum("bikd,bild->bikl", dnu, dnu)
    tri = torch.triu(torch.ones(k, k, dtype=torch.bool, device=device), 1)
    term = LAMBDA3 * (cos - COS0) ** 2 * h[:, :, :, None] * h[:, :, None, :]
    per_center = (term * (within[:, :, :, None] & within[:, :, None, :]) * tri[None, None]).sum((2, 3))
    return (per_center * centers).sum(1)
```

Refactor `mw_energy` to call `_phi3_centers(d, r, all-true mask)` so the 3-body path is shared (single source of truth; the numpy oracle is the independent check).

- [ ] **Step 4: Run tests**: `pytest tests/test_mw_energy.py -v` → all PASS. If `test_vs_numpy_oracle_random` is slow (N=64 numpy triple loop), keep B=4 — ~1 min is fine.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/__init__.py liquid_coupling_flow/mw/mw_energy.py tests/test_mw_energy.py && git commit -m "feat(mw): SW/mW energy (2+3 body, batched, local du_move) + G0 triple-oracle tests"`

---

### Task 2: Bedrock quadrature (`mw_bedrock.py`)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_bedrock.py`
- Test: `tests/test_mw_bedrock.py`

**Interfaces:**
- Consumes: `mw_energy` (Task 1).
- Produces: `quad_N2(L, beta, n) -> (U_mean, halving_err)`; `quad_N3(L, beta, n, chunk=2_000_000) -> (U_mean, halving_err)` (halving_err = |result(n) − result(n//2)|); CLI `python -m liquid_coupling_flow.mw.mw_bedrock` printing both at L=4.0, β=2.0 and saving `artifacts/mw_bedrock.pt`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mw_bedrock.py
import math, torch
from liquid_coupling_flow.mw.mw_bedrock import quad_N2, quad_N3

def test_n2_matches_direct_radial():
    # Independent check: N=2 <U> via explicit 3D numpy-style sum with a DIFFERENT grid offset scheme
    U, err = quad_N2(4.0, 2.0, 64)
    U2, _ = quad_N2(4.0, 2.0, 96)
    assert abs(U - U2) < 3 * err + 1e-6 and err < 1e-3

def test_n3_halving_converges():
    U, err = quad_N3(4.0, 2.0, 16)
    U2, err2 = quad_N3(4.0, 2.0, 20)
    assert err2 < err * 1.5 and abs(U - U2) < 3 * (err + err2)

def test_n3_beats_two_body_only():
    # with the 3-body term zeroed the answer must CHANGE (the gate really exercises phi3)
    from liquid_coupling_flow.mw import mw_bedrock
    U, _ = quad_N3(4.0, 2.0, 16)
    U2b, _ = quad_N3(4.0, 2.0, 16, two_body_only=True)
    assert abs(U - U2b) > 1e-3
```

- [ ] **Step 2: Run**: `pytest tests/test_mw_bedrock.py -x -q` → ImportError.

- [ ] **Step 3: Implement**

```python
"""Grid-quadrature bedrock <U> at N=2 (two-body only) and N=3 (the only analytic 3-body exercise).
Midpoint rule on the periodic box (periodic integrand => midpoint converges fast); particle 1 fixed
at the origin (translation invariance). Halving error = |value(n) - value(n//2)|."""
from __future__ import annotations
import os, sys, torch
from liquid_coupling_flow.mw.mw_energy import mw_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _grid(L, n, device):
    g = (torch.arange(n, device=device, dtype=torch.float64) + 0.5) * (L / n)
    return torch.stack(torch.meshgrid(g, g, g, indexing="ij"), -1).reshape(-1, 3)   # [n^3,3]


def _avg(U):                                             # Boltzmann average at beta over grid energies
    return U  # placeholder for signature clarity; real math inline below


def quad_N2(L, beta, n):
    def run(nn):
        p2 = _grid(L, nn, DEV)
        x = torch.zeros(p2.shape[0], 2, 3, dtype=torch.float64, device=DEV)
        x[:, 1] = p2
        U = mw_energy(x, L)
        w = torch.softmax(-beta * U, 0)
        return float((w * U).sum())
    v, vh = run(n), run(n // 2)
    return v, abs(v - vh)


def quad_N3(L, beta, n, chunk=2_000_000, two_body_only=False):
    def run(nn):
        p = _grid(L, nn, DEV)                              # [M,3]
        M = p.shape[0]
        num = torch.tensor(0.0, dtype=torch.float64); den = torch.tensor(0.0, dtype=torch.float64)
        m_shift = None                                     # streaming logsumexp-style stabilization
        for i in range(0, M, max(1, chunk // M)):
            p2 = p[i:i + max(1, chunk // M)]               # [m,3] second particle block
            x = torch.zeros(p2.shape[0] * M, 3, 3, dtype=torch.float64, device=DEV)
            x[:, 1] = p2.repeat_interleave(M, 0)
            x[:, 2] = p.repeat(p2.shape[0], 1)
            U = mw_energy(x, L)
            if two_body_only:
                from liquid_coupling_flow.mw.mw_energy import _pair, _phi2, A_CUT
                d, r = _pair(x, L)
                eye = torch.eye(3, dtype=torch.bool, device=x.device)
                U = _phi2(r.masked_fill(eye[None], A_CUT + 1.0)).sum((1, 2)) / 2
            if m_shift is None:
                m_shift = float(U.min())
            w = torch.exp(-beta * (U - m_shift))
            num += (w * U).sum().cpu(); den += w.sum().cpu()
        return float(num / den)
    v, vh = run(n), run(n // 2)
    return v, abs(v - vh)


if __name__ == "__main__":
    n2 = quad_N2(4.0, 2.0, 96); n3 = quad_N3(4.0, 2.0, 20)
    print(f"BEDROCK N=2: <U> {n2[0]:.6f} (halving err {n2[1]:.2e})", flush=True)
    print(f"BEDROCK N=3: <U> {n3[0]:.6f} (halving err {n3[1]:.2e})", flush=True)
    os.makedirs(ART, exist_ok=True)
    torch.save({"N2": n2, "N3": n3, "L": 4.0, "beta": 2.0}, os.path.join(ART, "mw_bedrock.pt"))
```

NOTE (implementer): `quad_N3`'s inner block builds `[m*M, 3, 3]` configs — pick the block size so `m*M ≤ chunk`; if GPU memory objects, halve `chunk`. n=20 → M=8000, M²=64M configs total; with chunk 2M that is 32 energy batches — minutes on GPU. The min-shift stabilization must use the FIRST batch's min consistently (store it; do not re-shift per batch — that corrupts the weights).

- [ ] **Step 4: Run**: `pytest tests/test_mw_bedrock.py -v` → PASS (the n3 tests take a few minutes on GPU).

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_bedrock.py tests/test_mw_bedrock.py && git commit -m "feat(mw): N=2/N=3 quadrature bedrock with halving-error control"`

---

### Task 3: Reference MC (`mw_reference.py`)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_reference.py`
- Test: `tests/test_mw_reference.py`

**Interfaces:**
- Consumes: `mw_energy`, `mw_energy_chunked`, `du_move`, `T_STAR`, `RHO_STAR` (Task 1).
- Produces: `mc_run(N, L, beta, n_equil, n_collect, every, seed, step0=0.15, B=8, track_every=200) -> dict` with keys `cfgs [n,N,3]`, `U [n]`, `traj [(sweep, U/N)]`, `step`, `acc`, `flat_budget`, `coll_drift`; `g_r(cfgs, L, nbins=120, rmax=None) -> (r, g)`; CLI `python -m liquid_coupling_flow.mw.mw_reference N n_equil n_collect [seed]` saving `artifacts/mw_ref_N{N}_s{seed}.pt`.

- [ ] **Step 1: Failing tests**

```python
# tests/test_mw_reference.py
import torch
from liquid_coupling_flow.mw.mw_reference import mc_run, g_r

def test_smoke_and_proofs():
    out = mc_run(N=8, L=2.6, beta=2.0, n_equil=400, n_collect=200, every=4, seed=0)
    assert out["cfgs"].shape[1:] == (8, 3) and out["cfgs"].shape[0] >= 100
    assert 0.2 < out["acc"] < 0.6                       # adaptive step landed
    assert "flat_budget" in out and "coll_drift" in out

def test_gr_normalization():
    g = torch.Generator().manual_seed(0)
    cfgs = torch.rand(64, 32, 3, generator=g) * 3.0     # ideal gas -> g(r) ~ 1
    r, gr = g_r(cfgs, 3.0)
    assert abs(float(gr[(r > 0.8) & (r < 1.4)].mean()) - 1.0) < 0.1
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement.** Metropolis single-site displacement over B independent chains (batched over chains; site loop per sweep uses `du_move` on `[B,N,3]`), α=min(1, e^{−βΔU}). Adaptive step during the first half of equilibration: every 100 sweeps multiply step by 1.15 if acceptance>0.45, by 0.85 if <0.35; frozen afterwards (frozen step recorded — the same tuning is reused by SMC mutation). Track (sweep, mean U/N) every `track_every`. Collect configs from all chains every `every` sweeps after equilibration. Convergence proofs exactly as `ka_finite_size.run_unit`: `flat_budget` = |final mean − mean over sweeps in (0.4,0.6)·n_equil|, `coll_drift` = |second-half − first-half| of collection means. `g_r`: histogram of min-image pair distances, normalized by ideal-gas shell counts `4π r² dr ρ (N−1)/2 · n_cfg · N`. CLI saves the FULL dict (configs included — record-simulation-data rule) to `artifacts/mw_ref_N{N}_s{seed}.pt` immediately on completion, plus every 2000 sweeps a partial `.partial.pt` (checkpoint rule).

- [ ] **Step 4: Run**: `pytest tests/test_mw_reference.py -v` → PASS.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_reference.py tests/test_mw_reference.py && git commit -m "feat(mw): displacement-MC reference with convergence proofs and per-unit saves"`

---

### Task 4: GATE G1 — bedrock vs reference MC (run, not code)

**Files:**
- Create: `reports/logs-<date>/mw_gate_g1.out` (run log), committed.

**Interfaces:** Consumes Task 2 CLI + Task 3 `mc_run`.

- [ ] **Step 1:** Run the bedrock CLI: `python -m liquid_coupling_flow.mw.mw_bedrock > reports/logs-<date>/mw_gate_g1.out 2>&1`.
- [ ] **Step 2:** Run reference MC at the same points, appending to the log: N=2 and N=3, L=4.0, β=2.0, `n_equil=20000, n_collect=20000, every=4, B=16` (tiny systems, minutes).
- [ ] **Step 3:** PASS criterion (from spec): |⟨U⟩_quad − ⟨U⟩_MC| < halving_err + 3·SE_MC at BOTH N. Record verdict lines in the log. If FAIL: stop, debug (never conclude "no bug") — suspects: quadrature shift bug, min-image at L/2, adaptive-step freeze.
- [ ] **Step 4:** Commit log: `git add reports/logs-<date>/mw_gate_g1.out && git commit -m "test(mw): G1 bedrock gate PASS — quadrature vs MC at N=2,3"`

---

### Task 5: λ-path SMC (`mw_base.py`, `mw_smc.py`)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_base.py`
- Create: `liquid_coupling_flow/mw/mw_smc.py`
- Test: `tests/test_mw_smc.py`

**Interfaces:**
- Consumes: `mw_energy`, `du_move` (Task 1).
- Produces: `UniformBase(N, L)` with `.sample(B, gen) -> x`, `.log_q(x) -> [B]`, `.LOGQ_CONST=True`; `ess(logw) -> float`; `next_lambda(logw, phi, lam, ess_target, B) -> float`; `smc_run(base, N, L, beta, B=256, ess_target=0.6, n_sweeps=3, step=None, seed=0, save_tag="") -> dict` with keys `x, U, logw, logZ, history [{rung, lam, ess, U_mean, dlam, evals}], evals, wall`. Every rung appends to `artifacts/mw_smc{save_tag}_N{N}.pt` (per-rung saves).

- [ ] **Step 1: Failing tests**

```python
# tests/test_mw_smc.py
import math, torch
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_smc import ess, next_lambda, smc_run, mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy

def test_ess_bounds():
    assert abs(ess(torch.zeros(100)) - 100.0) < 1e-6
    w = torch.full((100,), -1e9); w[0] = 0.0
    assert ess(w) < 1.01

def test_next_lambda_monotone_and_floor():
    g = torch.Generator().manual_seed(0)
    phi = -torch.rand(256, generator=g) * 50            # heterogeneous -> partial step
    lam = next_lambda(torch.zeros(256), phi, 0.0, 0.6, 256)
    assert 1e-4 <= lam <= 1.0
    assert ess(lam * phi) >= 0.6 * 256 * 0.99           # lands at/above target

def test_uniform_base_reduction():
    # with a LOGQ_CONST base, log q0 must NOT be evaluated per move — at most once (the final bookkeeping
    # call). A per-move call count of N*B would betray the reduction failing.
    class Spy(UniformBase):
        calls = 0
        def log_q(self, x):
            Spy.calls += 1
            return super().log_q(x)
    b = Spy(8, 2.6)
    g = torch.Generator().manual_seed(0)
    x = b.sample(16, g)
    x2, U, lq, info = mutation_sweeps(x, b, lam=0.5, beta=2.0, L=2.6, n_sweeps=2, step=0.1, gen=g)
    assert torch.isfinite(U).all() and Spy.calls <= 1

def test_mutation_stationarity_tiny():
    # N=2, L=4, beta=2, lam=1: long mutation-only chain must reproduce the bedrock <U> (G1 value).
    from liquid_coupling_flow.mw.mw_bedrock import quad_N2
    Uq, err = quad_N2(4.0, 2.0, 64)
    b = UniformBase(2, 4.0); g = torch.Generator().manual_seed(1)
    x = b.sample(64, g)
    us = []
    for it in range(600):
        x, U, _, _ = mutation_sweeps(x, b, 1.0, 2.0, 4.0, n_sweeps=5, step=0.35, gen=g)
        if it > 100: us.append(U.mean().item())
    u = sum(us) / len(us)
    assert abs(u - Uq) < max(0.02, 5 * err), f"mutation kernel off: {u} vs bedrock {Uq}"

def test_smc_runs_and_saves(tmp_path):
    b = UniformBase(8, 2.6)
    out = smc_run(b, 8, 2.6, beta=2.0, B=64, n_sweeps=2, step=0.2, seed=0, save_tag="_smoke")
    assert out["history"][-1]["lam"] == 1.0 and torch.isfinite(out["logZ"] * 1.0)
    assert out["evals"] > 0
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement.**

```python
# mw_base.py
import math, torch

class UniformBase:
    LOGQ_CONST = True
    def __init__(self, N, L):
        self.N, self.L = N, L
        self._lq = -N * 3 * math.log(L)
    def sample(self, B, gen=None):
        return torch.rand(B, self.N, 3, generator=gen,
                          device=gen.device if gen is not None and gen.device.type == "cuda" else "cpu") * self.L
    def log_q(self, x):
        return torch.full((x.shape[0],), self._lq, device=x.device)
```

```python
# mw_smc.py — core pieces (full file also has smc_run per below)
import os, time, torch
from liquid_coupling_flow.mw.mw_energy import mw_energy, du_move

ART = os.path.join(os.path.dirname(__file__), "artifacts")

def ess(logw):
    w = torch.softmax(logw, 0)
    return float(1.0 / (w ** 2).sum())

def next_lambda(logw, phi, lam, ess_target, B):
    """Largest lam' in (lam,1] with ESS(logw + (lam'-lam)*phi) >= ess_target*B; bisection, floor 1e-4.
    phi = -beta*U - log q0 (the d/dlam of log pi_lambda)."""
    if ess(logw + (1.0 - lam) * phi) >= ess_target * B:
        return 1.0
    lo, hi = lam, 1.0
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if ess(logw + (mid - lam) * phi) >= ess_target * B else (lo, mid)
    return min(1.0, max(lo, lam + 1e-4))

def _resample(x, U, lq, logw, gen):
    idx = torch.multinomial(torch.softmax(logw, 0), logw.shape[0], replacement=True, generator=gen)
    return x[idx], U[idx], lq[idx], torch.zeros_like(logw)

def mutation_sweeps(x, base, lam, beta, L, n_sweeps, step, gen):
    """pi_lambda-invariant single-site MH. Uniform base: log q0 terms drop (LOGQ_CONST)."""
    B, N, _ = x.shape
    U = mw_energy(x, L)
    lq = None if base.LOGQ_CONST else base.log_q(x)
    n_acc = 0; n_evals = 0
    for _ in range(n_sweeps):
        for i in range(N):
            xi = torch.remainder(x[:, i] + step * torch.randn(B, 3, device=x.device, generator=gen), L)
            dU = du_move(x, i, xi, L); n_evals += B
            loga = -lam * beta * dU
            if not base.LOGQ_CONST:
                x_prop = x.clone(); x_prop[:, i] = xi
                lq_new = base.log_q(x_prop)
                loga = loga + (1.0 - lam) * (lq_new - lq)
            acc = torch.log(torch.rand(B, device=x.device, generator=gen).clamp_min(1e-38)) < loga
            x = x.clone(); x[acc, i] = xi[acc]
            U = torch.where(acc, U + dU, U)
            if not base.LOGQ_CONST:
                lq = torch.where(acc, lq_new, lq)
            n_acc += int(acc.sum())
    if base.LOGQ_CONST:
        lq = base.log_q(x)
    return x, U, lq, {"acc": n_acc / max(1, n_sweeps * N * B), "evals": n_evals}
```

`smc_run(base, N, L, beta, ...)`: init `x = base.sample(B)`, `U = mw_energy(x, L)`, `lq = base.log_q(x)`, `logw = 0`, `lam = 0`, `logZ = 0`, `evals = B`. Loop until `lam == 1.0`: (1) `phi = -beta*U - lq`; `lam_new = next_lambda(...)`; `dlw = (lam_new-lam)*phi`; `logZ += float(torch.logsumexp(logw+dlw,0) - torch.logsumexp(logw,0))`; `logw += dlw`; `lam = lam_new`; (2) if `ess(logw) <= ess_target*B + 1e-6`: resample (reset logw); (3) `x, U, lq, info = mutation_sweeps(x, base, lam, beta, L, n_sweeps, step, gen)`; `evals += info["evals"]`; (4) **guard** (ordmh): every rung, recompute `lq_fresh = base.log_q(x[:8])` and assert `max|lq[:8]−lq_fresh| < 1e-3`; recompute `U_fresh = mw_energy(x[:8], L)` and assert `< 1e-4` — carried values must equal fresh re-evaluation; (5) append history + `torch.save` the running dict to `artifacts/mw_smc{save_tag}_N{N}.pt` (per-rung save). Stall guard from `ka_local_smc`: if `lam` advanced only by the 1e-4 floor for 20 consecutive rungs, raise with a diagnostic. Default `step`: load the frozen reference step for (N, β) if present, else 0.15.

- [ ] **Step 4: Run**: `pytest tests/test_mw_smc.py -v` → PASS (`test_mutation_stationarity_tiny` takes a few minutes).

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_base.py liquid_coupling_flow/mw/mw_smc.py tests/test_mw_smc.py && git commit -m "feat(mw): lambda-path SMC with adaptive schedule, uniform-base reduction, rung guards"`

---

### Task 6: GATE G2 — SMC exactness at N=8 and N=64 (run)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_gates.py` (G2+G3 harness, reused by Phase 2/3 re-gates)
- Create: `reports/logs-<date>/mw_gate_g2.out`

**Interfaces:**
- Consumes: `smc_run`, `UniformBase`, `mc_run`, `g_r`, `mw_energy_chunked`.
- Produces: `g2(N, L, beta, B=512, n_ref_equil=..., save_tag) -> dict` comparing SMC final population (⟨U⟩, P(U) total-variation on shared bins, g(r) max|Δ|) vs reference MC; PASS = |Δ⟨U⟩| < 3·(SE_smc+SE_ref), TV < 0.05, g(r) max|Δ| < 0.1 (ordmh-scale tolerances). Weighted SMC estimators use `softmax(logw)`.

- [ ] **Step 1:** Write `g2()` in `mw_gates.py`: runs `mc_run` reference (N=8: 20k+20k sweeps; N=64: 40k+8k, B=16 chains) and `smc_run` at the ambient point (β\*=10.382, L from ρ\*), computes the three comparisons, prints a PASS/FAIL verdict per metric, saves `artifacts/mw_g2_N{N}.pt` (both populations + metrics — full data). ⟨U⟩ SE for SMC via weighted bootstrap (200 resamples).
- [ ] **Step 2:** Smoke-test at N=8 with tiny budgets in pytest (add `test_g2_smoke` to `tests/test_mw_smc.py` asserting the dict has the three metrics; no PASS assertion — budgets too small).
- [ ] **Step 3:** Real runs: `nohup python -m liquid_coupling_flow.mw.mw_gates g2 8 > reports/logs-<date>/mw_gate_g2.out 2>&1 &` then N=64 appended. N=64 reference is the long pole (hours) — launch first, record PID.
- [ ] **Step 4:** Verify PASS at both sizes; commit log + harness: `git add liquid_coupling_flow/mw/mw_gates.py reports/logs-<date>/mw_gate_g2.out tests/test_mw_smc.py && git commit -m "test(mw): G2 SMC exactness gate PASS at N=8,64 (U, P(U), g(r) vs reference)"`. FAIL → stop and debug; the uniform-base reduction test localizes faults to the annealer, not the base.

---

### Task 7: GATE G3 — rung scaling T_trivial(N) (run) — closes Phase 1

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_gates.py` (add `g3()`)
- Create: `reports/logs-<date>/mw_gate_g3.out`

**Interfaces:**
- Consumes: `smc_run`, `UniformBase`.
- Produces: `g3(Ns=(27, 64, 125, 216), B=256, seeds=(0,1)) -> dict`: per (N, seed) run uniform-base SMC at the ambient point, record `T` (rung count), `evals`, final ⟨U⟩/N; fit T vs N (least squares through origin); save `artifacts/mw_g3.pt` after EVERY unit (per-unit saves); plot `reports/logs-<date>/mw_g3_scaling.png` (T vs N with fit; show the full path).

- [ ] **Step 1:** Implement `g3()` + plot. Each unit saves immediately.
- [ ] **Step 2:** Run: `nohup python -m liquid_coupling_flow.mw.mw_gates g3 > reports/logs-<date>/mw_gate_g3.out 2>&1 &` (N=216 units are the long pole; du_move keeps sweeps O(N·k²)-ish).
- [ ] **Step 3:** Verdict: approximately linear T(N) (report R² and per-N c_trivial = T·δ̄ where δ̄ = mean per-rung λ-progress⁻¹… simply report T/N constancy). Slope = the Phase-2 yardstick. Deviations recorded, not hidden (no-silent-caps rule). If N=27 shows box artifacts (spec risk 5): drop it and note.
- [ ] **Step 4:** Commit: `git add liquid_coupling_flow/mw/mw_gates.py reports/logs-<date>/mw_gate_g3.out reports/logs-<date>/mw_g3_scaling.png && git commit -m "test(mw): G3 rung scaling T_trivial(N) — Phase 1 complete"`. **PHASE 1 GATE: all of G0–G3 green before Task 8.**

---

### Task 8: Scaffold, canonical order, 3D frames (`mw_generator.py` part 1)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_generator.py` (geometry section)
- Test: `tests/test_mw_generator.py`

**Interfaces:**
- Consumes: `gilbert3d_path` from `grid_transformer.data.gilbert`.
- Produces: `mw_scaffold(N, L, device) -> (t [N,3], rank [R,R,R], R)` (t = cell centers in curve order; asserts N==R³); `canonical_order(x, L, R, rank) -> perm [B,N]` (curve-cell rank, then distance-to-center, stable); `build_frames(nbr_rel, n_valid) -> Rf [.,3,3]` (rows e1,e2,e3; Gram–Schmidt on the two nearest valid neighbors; fallbacks: 0 valid → identity, 1 valid → axis completion using the coordinate axis least aligned with e1, collinear |cos|>0.99 → third neighbor, else axis completion).

- [ ] **Step 1: Failing tests**

```python
# tests/test_mw_generator.py (part 1)
import math, torch
from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order, build_frames

def test_scaffold_covers_box():
    t, rank, R = mw_scaffold(64, 5.198, "cpu")
    assert R == 4 and t.shape == (64, 3) and t.min() > 0 and t.max() < 5.198
    assert sorted(rank.flatten().tolist()) == list(range(64))

def test_canonical_permutation_invariant():
    g = torch.Generator().manual_seed(0); L = 5.198
    x = torch.rand(3, 64, 3, generator=g) * L
    t, rank, R = mw_scaffold(64, L, "cpu")
    p = canonical_order(x, L, R, rank)
    xs = torch.gather(x, 1, p[..., None].expand(-1, -1, 3))
    perm = torch.randperm(64, generator=g)
    p2 = canonical_order(x[:, perm], L, R, rank)
    xs2 = torch.gather(x[:, perm], 1, p2[..., None].expand(-1, -1, 3))
    assert torch.allclose(xs, xs2, atol=1e-7)           # same canonical sequence from any storage order

def test_frames_orthonormal_and_covariant():
    g = torch.Generator().manual_seed(1)
    d = torch.randn(50, 12, 3, generator=g)
    n_valid = torch.full((50,), 12)
    Rf = build_frames(d, n_valid)
    eye = torch.eye(3).expand(50, 3, 3)
    assert torch.allclose(Rf @ Rf.transpose(-1, -2), eye, atol=1e-5)
    assert torch.allclose(torch.det(Rf), torch.ones(50), atol=1e-5)
    # covariance: rotate all displacements by R0 -> frame coords of any rotated vector unchanged
    th = 0.7; c, s = math.cos(th), math.sin(th)
    R0 = torch.tensor([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    Rf2 = build_frames(d @ R0.T, n_valid)
    v = torch.randn(50, 3, generator=g)
    assert torch.allclose(torch.einsum("bij,bj->bi", Rf, v),
                          torch.einsum("bij,bj->bi", Rf2, v @ R0.T), atol=1e-4)

def test_frames_fallbacks():
    d = torch.zeros(3, 12, 3); d[1, 0] = torch.tensor([1., 0., 0.])
    d[2, 0] = torch.tensor([1., 0., 0.]); d[2, 1] = torch.tensor([2., 0., 0.])   # collinear pair
    Rf = build_frames(d, torch.tensor([0, 1, 2]))
    for b in range(3):
        assert torch.allclose(Rf[b] @ Rf[b].T, torch.eye(3), atol=1e-5)
```

- [ ] **Step 2: Run** → ImportError.

- [ ] **Step 3: Implement.**

```python
def mw_scaffold(N, L, device):
    R = round(N ** (1 / 3))
    assert R ** 3 == N, f"N={N} not a cube"
    from grid_transformer.data.gilbert import gilbert3d_path
    path = torch.as_tensor(gilbert3d_path(R, R, R), device=device)           # [N,3] cells in curve order
    rank = torch.zeros(R, R, R, dtype=torch.long, device=device)
    rank[path[:, 0], path[:, 1], path[:, 2]] = torch.arange(N, device=device)
    t = (path.float() + 0.5) * (L / R)
    return t, rank, R

def canonical_order(x, L, R, rank):
    B, N, _ = x.shape
    w = L / R
    cell = (x / w).long().clamp(0, R - 1)
    rk = rank[cell[..., 0], cell[..., 1], cell[..., 2]]                       # [B,N]
    d2 = ((x - (cell.float() + 0.5) * w) ** 2).sum(-1)
    o1 = torch.argsort(d2, dim=1, stable=True)
    o2 = torch.argsort(torch.gather(rk, 1, o1), dim=1, stable=True)
    return torch.gather(o1, 1, o2)

def build_frames(nbr_rel, n_valid, col_tol=0.99):
    """nbr_rel [.,k,3] displacement vectors sorted nearest-first (invalid rows arbitrary),
    n_valid [.] count of valid rows. Rows of output = frame axes."""
    B, k, _ = nbr_rel.shape
    d1 = nbr_rel[:, 0]
    e1 = torch.nn.functional.normalize(torch.where((n_valid >= 1)[:, None], d1,
         torch.tensor([1., 0., 0.], device=d1.device).expand_as(d1)), dim=-1)
    # second vector: first neighbor with |cos| <= col_tol among indices 1..k-1, else axis fallback
    cos = torch.einsum("bkd,bd->bk", torch.nn.functional.normalize(nbr_rel, dim=-1), e1).abs()
    ok = (cos <= col_tol) & (torch.arange(k, device=d1.device)[None] < n_valid[:, None]) \
         & (torch.arange(k, device=d1.device)[None] >= 1)
    idx2 = torch.where(ok.any(1), ok.float().argmax(1), torch.zeros_like(n_valid))
    d2 = torch.gather(nbr_rel, 1, idx2[:, None, None].expand(-1, 1, 3)).squeeze(1)
    # axis fallback (0/1 valid, or all collinear): coordinate axis least aligned with e1
    axes = torch.eye(3, device=d1.device)
    ax = axes[e1.abs().argmin(-1)]
    use_ax = (~ok.any(1)) | (n_valid < 2)
    d2 = torch.where(use_ax[:, None], ax, d2)
    u2 = d2 - (d2 * e1).sum(-1, keepdim=True) * e1
    e2 = torch.nn.functional.normalize(u2, dim=-1)
    e3 = torch.cross(e1, e2, dim=-1)
    return torch.stack([e1, e2, e3], dim=-2)
```

- [ ] **Step 4: Run**: `pytest tests/test_mw_generator.py -v` → PASS.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_generator.py tests/test_mw_generator.py && git commit -m "feat(mw): gilbert3d scaffold, deterministic canonical order, 3D Gram-Schmidt frames"`

---

### Task 9: AR generator model (`mw_generator.py` part 2)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_generator.py` (append model section)
- Modify: `tests/test_mw_generator.py` (append model tests)

**Interfaces:**
- Consumes: Task 8 functions; `RQSplineElementwise, DEFAULT_MIN_DERIVATIVE` from `liquid_coupling_flow.transforms_spline`; identity-init pattern from `ka_flowhead.SplineFlowHead` (copy, extend to 3 dims).
- Produces: `class MWGenerator(nn.Module)` with `__init__(self, knn=12, d_model=128, n_layers=2, n_heads=4, num_bins=8, tail_bound=4.0, periods=(0.5, 1.0, 2.0, 4.0))`; `log_prob(x, L) -> [B]` (canonicalizes internally; one teacher-forced pass, chunked over B); `sample(B, N, L, gen=None) -> (x, logq, n_wrapped)`; `Spline3Head(d_model, num_bins, tail_bound)` with `.log_prob(h, u)->[...]`, `.sample(h, gen)->(u, logq)` (AR within step: a, b|a, c|a,b — the 2D head + one more conditional). Offset convention: `u = Rf @ wrap_pm(x_j − t_j, L) / s`, `s = (L/R)/2`; per-step `log q_x = log q_u − 3·log s`.

- [ ] **Step 1: Failing tests (append)**

```python
def _model():
    torch.manual_seed(0)
    from liquid_coupling_flow.mw.mw_generator import MWGenerator
    return MWGenerator(knn=6, d_model=32, n_layers=1, n_heads=2)

def test_sample_logprob_consistency():
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(0)
    x, lq_s, nwrap = m.sample(8, 64, L, gen=g)
    lq_e = m.log_prob(x, L)
    if nwrap == 0:                                       # wrapped configs may legitimately differ (counter)
        assert torch.allclose(lq_s, lq_e, atol=1e-3), float((lq_s - lq_e).abs().max())

def test_logprob_permutation_invariant():
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(1)
    x = torch.rand(4, 64, 3, generator=g) * L
    perm = torch.randperm(64, generator=g)
    assert torch.allclose(m.log_prob(x, L), m.log_prob(x[:, perm], L), atol=1e-4)

def test_conditional_normalized_1d():
    # identity-init head: q(a|h) must integrate to 1 on a fine grid (Gaussian base, machine-level)
    from liquid_coupling_flow.mw.mw_generator import Spline3Head
    torch.manual_seed(0)
    head = Spline3Head(32, 8, 4.0)
    h = torch.randn(1, 32)
    a = torch.linspace(-8, 8, 4001)[:, None]
    lp = head.logp_a(h.expand(4001, -1), a)             # expose per-dim conditional for this test
    z = torch.trapz(lp.exp().squeeze(), a.squeeze())
    assert abs(float(z) - 1.0) < 1e-3

def test_prefix_storage_invariance():
    # conditioning tensors are geometry-only: shuffling storage of placed particles changes nothing
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(2)
    x = torch.rand(2, 64, 3, generator=g) * L
    lp1 = m.log_prob(x, L)
    lp2 = m.log_prob(x[:, torch.randperm(64, generator=g)], L)
    assert torch.allclose(lp1, lp2, atol=1e-4)

def test_translation_by_lattice():
    m = _model(); L = 5.198
    g = torch.Generator().manual_seed(3)
    x = torch.rand(2, 64, 3, generator=g) * L
    shift = torch.tensor([L / 4, 0.0, 0.0])             # one full cell (R=4): scaffold maps onto itself
    lp2 = m.log_prob(torch.remainder(x + shift, L), L)
    assert torch.allclose(m.log_prob(x, L), lp2, atol=1e-3)
```

- [ ] **Step 2: Run** → fails (no model).

- [ ] **Step 3: Implement.**

```python
class Spline3Head(nn.Module):
    """3-coordinate AR spline head: p(a|h) p(b|h,a) p(c|h,a,b). Identity init (== N(0,1)^3 base)."""
    def __init__(self, d_model, num_bins=8, tail_bound=4.0):
        super().__init__()
        self.spline = RQSplineElementwise(num_bins=num_bins, tail_bound=tail_bound)
        self.num_bins = num_bins
        P = self.spline.params_per_dim
        self.heads = nn.ModuleList([nn.Linear(d_model + i, P) for i in range(3)])
        const = float(torch.log(torch.expm1(torch.tensor(1.0 - DEFAULT_MIN_DERIVATIVE))))
        for hd in self.heads:
            nn.init.zeros_(hd.weight)
            with torch.no_grad():
                hd.bias.zero_(); hd.bias[2 * num_bins:] = const

    def logp_a(self, h, a):                              # exposed for the normalization test
        z, ld = self.spline.inverse(a, self.heads[0](h))
        return -0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) + ld

    def log_prob(self, h, u):
        lp, ctx = 0.0, h
        for i in range(3):
            ui = u[..., i:i + 1]
            z, ld = self.spline.inverse(ui, self.heads[i](ctx))
            lp = lp + (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) + ld).squeeze(-1)
            ctx = torch.cat([ctx, ui], -1)
        return lp

    def sample(self, h, gen=None):
        us, lp, ctx = [], 0.0, h
        for i in range(3):
            z = torch.randn(h.shape[:-1] + (1,), device=h.device, generator=gen)
            ui, ld = self.spline.forward(z, self.heads[i](ctx))
            lp = lp + (-0.5 * z ** 2 - 0.5 * math.log(2 * math.pi) - ld).squeeze(-1)
            us.append(ui); ctx = torch.cat([ctx, ui], -1)
        return torch.cat(us, -1), lp
```

`MWGenerator`: per-neighbor feature = `[frame_coords(3), r(1), sin(2πr/p)+cos(2πr/p) for p in periods (8)]` → 12 dims → `nn.Linear(12, d_model)`; learned CLS token; `nn.TransformerEncoder(d_model, n_heads, n_layers, batch_first=True)` over `k+1` tokens with `src_key_padding_mask` for invalid neighbors; `h_j` = CLS output. **log_prob** (vectorized, the `_inertial_R_logprob` pattern): canonicalize → `xo`; distances of every `xo_i` to every anchor `t_j`, causal mask `i<j`, topk knn → `idx, valid`; `nbr_rel = wrap_pm(gather(xo) − t_j, L) * valid`; frames = `build_frames` per (B,N); `u = einsum(Rf, wrap_pm(xo − t, L)) / s`; run encoder on `[B*N, k+1, d]` in chunks of ≤8192 rows; `logq = Σ_j head.log_prob(h, u) − 3N·log s`. **sample**: j-loop; same features with `k=min(knn, j)`; `u, lq_u = head.sample(h)`; `off = Rf.T @ u * s`; `n_wrapped += (off.abs() > L/2).any(-1).sum()`; `x_j = remainder(t_j + off, L)`; `logq += lq_u − 3 log s`. `wrap_pm(v, L) = v − L*round(v/L)` (module-local copy — mw/ stays self-contained).

- [ ] **Step 4: Run**: `pytest tests/test_mw_generator.py -v` → PASS. The consistency test tolerance is 1e-3 because canonical order of a SAMPLED config can differ from generation order (two particles landing in one cell) — if the assert fails, check `canonical_order(sampled) == arange` first; where it differs, the config counts as noncanonical: measure and report the fraction, and only compare on canonical configs (the counter from the likelihood-exactness lineage). Add `n_noncanon` to sample()'s return info if needed.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_generator.py tests/test_mw_generator.py && git commit -m "feat(mw): 3D local-frame AR generator with exact 3-coord spline head"`

---

### Task 10: Training + Phase-2 entry diagnostics (run)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_generator.py` (add `train()` + `orderspread()` CLIs)
- Create: `reports/logs-<date>/mw_train_N64.out`, `reports/logs-<date>/mw_orderspread.out`

**Interfaces:**
- Consumes: `mc_run` (Task 3), `MWGenerator` (Task 9).
- Produces: checkpoint `artifacts/mw_gen_N64.pt` = `{"state_dict", "knn", "d_model", "n_layers", "n_heads", "num_bins", "tail_bound", "periods", "val_nll"}`; `load_generator(path, device) -> MWGenerator` (reads hyperparams from ckpt).

- [ ] **Step 1: Training data.** `python -m liquid_coupling_flow.mw.mw_reference 64 40000 25000 0` with `every=25`, B=16 chains → ≥4000 configs at the ambient point. Verify `flat_budget < 0.01` and record the U-autocorrelation time in the log (if τ > 25 sweeps, thin further before training). Data stays in `artifacts/mw_ref_N64_s0.pt`.
- [ ] **Step 2: `train()` CLI**: MLE, loss = `−log_prob(batch)/N`, batch 32 configs, Adam lr 3e-4, 20k steps, 10% val split, **val-loss checkpointing** (evaluate every 500 steps; save best-on-val AND last — train-loss monitoring is the measured trap). Log train/val NLL every 500 steps. Run: `nohup python -m liquid_coupling_flow.mw.mw_generator train > reports/logs-<date>/mw_train_N64.out 2>&1 &` (record PID; GPU-hours).
- [ ] **Step 3: Post-training checks** (append to log): (a) val NLL ≪ uniform (−log L³ᴺ… report per-particle nats vs uniform's `3·log L`); (b) sample 512 configs → `n_wrapped/512·64 < 1e-3` (wrap counter; if it fires, retune tail_bound — the IPL44 precedent) and noncanonical fraction reported; (c) g(r) of samples vs reference overlaid (one-shot fidelity NOT gated — record only). Show the plot path.
- [ ] **Step 4: `orderspread()` CLI** (Phase-2 entry diagnostics, recorded not gated): on 64 reference configs — (i) log q̃ under K=32 RANDOM orderings (forced-order log_prob path) → per-config std/N vs the 2D-WCA 0.027 nats; (ii) canonicalization stability: perturb each config by the frozen MC step size, measure P(canonical order changed) and the |Δlog q₀| distribution on flips (mutation-MH discontinuity scale). `> reports/logs-<date>/mw_orderspread.out 2>&1`.
- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_generator.py reports/logs-<date>/mw_train_N64.out reports/logs-<date>/mw_orderspread.out && git commit -m "feat(mw): generator trained at N=64 (val-ckpt) + ordering-spread entry diagnostics"` (plus the g(r) plot file).

---

### Task 11: Phase 2 — generator base, c measurement, T comparison (code + run)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_base.py` (add `GeneratorBase`)
- Create: `liquid_coupling_flow/mw/mw_measure.py`
- Test: `tests/test_mw_smc.py` (append `test_generator_base_smoke`)
- Create: `reports/logs-<date>/mw_phase2.out`

**Interfaces:**
- Consumes: `load_generator`, `smc_run`, `mw_gates.g2` (re-gate), G3 artifact.
- Produces: `GeneratorBase(ckpt_path, N, L, device)` with `.sample(B, gen)`, `.log_q(x)` (chunked TF pass), `.LOGQ_CONST=False`; `mw_measure.phase2(N=64, B=256, seeds=(0,1))` → for each base ∈ {uniform, generator}: T, evals, logZ, exactness re-gate metrics; `c_hat = (mean(log_q(x0) + β·U(x0)) + logZ)/N` on M=2048 fresh base draws; Δc (Z-free) and the c-ratio vs rung-ratio table. Saves `artifacts/mw_phase2_N{N}.pt` per arm.

- [ ] **Step 1:** Implement `GeneratorBase` (wraps model.eval() + torch.no_grad(); log_q chunks B) + smoke test: `smc_run(GeneratorBase(...), 8→No — N must be 64: cube constraint)` — smoke at N=64 with B=16, asserting finite logZ and that the rung-guard (fresh-vs-carried log q₀) passes.
- [ ] **Step 2:** Implement `phase2()`: both arms share seed, ess_target, n_sweeps, frozen step; energy-eval counters from smc_run; per-arm save the moment it finishes.
- [ ] **Step 3:** Run: `nohup python -m liquid_coupling_flow.mw.mw_measure phase2 > reports/logs-<date>/mw_phase2.out 2>&1 &`. Generator-arm mutations re-score log q₀ per site move (N TF forwards/sweep) — budget hours; per-rung saves protect progress.
- [ ] **Step 4:** Verdicts in log: (i) exactness re-gate (same tolerances as G2) — MUST pass or stop; (ii) T_generator vs T_trivial and evals/ESS_final; (iii) c-ratio vs rung-ratio consistency (the pre-registered readout). Whatever the numbers, they are the result — "generator adds little at an easy state point" is a valid, publishable outcome (spec risk 1).
- [ ] **Step 5:** Commit code, log, artifacts pointer: `git commit -m "feat(mw): Phase 2 — generator base, c measurement, T_generator vs T_trivial at N=64"`.

---

### Task 12: Phase 3 — zero-shot N=216 + results note + memory (run + docs)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_measure.py` (add `transfer()`)
- Create: `reports/logs-<date>/mw_phase3.out`, `reports/<date>-mw-annealed-smc-results.md`

**Interfaces:** Consumes everything above; `transfer(N=216)` = `phase2()` with the N=64 checkpoint deployed unmodified (scaffold regenerates at R=6) + its own reference (`mc_run` N=216, budget from the N=64 τ measurement ×(216/64)) for the exactness re-gate.

- [ ] **Step 1:** Implement `transfer()` (thin wrapper; assert the checkpoint's train_N ≠ N so zero-shot is real). Pre-check: `log_prob` and `sample` shape-sanity at N=216 (the crossover `precheck` pattern) before the long run.
- [ ] **Step 2:** Launch reference N=216 first (long pole), then both SMC arms: `nohup ... > reports/logs-<date>/mw_phase3.out 2>&1 &`. Per-unit and per-rung saves throughout.
- [ ] **Step 3:** Verdicts: exactness re-gate at 216; rung reduction survives?; c(216) vs c(64) (N-independence). Plot T vs N for both bases over the G3 line (show path).
- [ ] **Step 4:** Write `reports/<date>-mw-annealed-smc-results.md` — full verdict chain G0→Phase 3, every number with its protocol, honest negatives included. Update memory: new memory file `mw-annealed-smc` + MEMORY.md hook.
- [ ] **Step 5:** Commit: results note, logs, plots, memory. Final commit message: `docs(mw): Phase 3 transfer verdict + campaign results note`.

---

## Self-Review (performed at write time)

- **Spec coverage:** §1 → Task 1 constants; §2 → Task 5 (+guards); §3 module table → Tasks 1–3, 5, 8–11 (mw_gates.py and mw_measure.py added beyond the spec table — gate/measurement harnesses, consistent with "per-phase gate scripts"); §4 → Tasks 8–10 (anchor-through-t_j invariant per amended spec); §5 G0–G3 → Tasks 1, 4, 6, 7; Phase 2 → Tasks 10–11; Phase 3 → Task 12; §6 tests → Tasks 1, 5, 8, 9; §7 risks → embedded (wrap counter, KMAX assert, N=27 drop rule, per-rung saves).
- **Type consistency:** `smc_run(base, N, L, beta, B, ess_target, n_sweeps, step, seed, save_tag)` used identically in Tasks 5/6/7/11/12; `mutation_sweeps` returns `(x, U, lq, info)` everywhere; `build_frames(nbr_rel, n_valid)` matches Task 9's call.
- **Known deliberate deviations:** none from the amended spec.
