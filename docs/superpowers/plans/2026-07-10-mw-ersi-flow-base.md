# mW eRSI Flow Base Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train an eRSI (equivariant Riemannian Stochastic Interpolant) flow on the mW reference bank via the vendored `learndiffeq` pipeline, and deploy it as a non-causal one-shot importance-sampling proposal for the liquid-mW sampler — the first base that can produce medium-range structure the causal AR bases could not.

**Architecture:** Hybrid reuse. TRAIN with the vendored `RiemannianFlowMatching` + OT coupling + kNN-trimmed exact-divergence `EGNN_dynamics` (3D, monatomic). DEPLOY through a thin mW-native `MWeRSIFlow` wrapper exposing `sample` + composed-likelihood `log_q`. USE via one-shot IS: `w = e^{−βU(X)}/q_flow(X)`, likelihood from the vendored `sample_and_log_prob`.

**Tech Stack:** PyTorch + PyTorch Lightning (vendored `learndiffeq`), the certified mW stack (`mw_energy`, `mw_reference`, `mw_base`, `mw_smc`, `mw_gates`).

**Spec:** `docs/superpowers/specs/2026-07-10-mw-ersi-flow-base-design.md` — read before starting any task.

## Global Constraints

- Vendored package root: `liquid_coupling_flow/ipl44/learndiffeq/` (package `learndiffeq`). Import via a module-local `sys.path.insert(0, <that abs path>)` helper `_add_learndiffeq_path()` in each new mw file that needs it — do NOT install or move the package.
- System: mW ambient, monatomic. N=64, `L=(64/RHO_STAR)**(1/3)`, `RHO_STAR=0.4564`, `T_STAR=0.09632`, `beta=1/T_STAR` (import from `liquid_coupling_flow.mw.mw_energy`).
- Velocity: `EGNN_dynamics(n_particles=N, n_dimension=3, max_neighbors=K, L=L, n_species=1, ...)`, K=12 default (cage scale). The kNN trim is `max_neighbors`.
- OT coupling ON: `RiemannianFlowMatching(..., ot_particles=True, use_linear_assignment_particles=True)`.
- Exact divergence: `EGNN_dynamics.forward_and_divergence(xs, t, a) -> (vel, divergence)`; the flow's `force_automatic_div=False` (use the analytical `compute_div`).
- Base `rho0` = uniform on the torus × constant species (monatomic).
- Positions ALWAYS wrapped to `[0,L)` with `torch.remainder` before entering the pipeline or `mw_energy`.
- Integration mode = one-shot IS. `sample_and_log_prob(shape) -> (a, X, log_prob)` gives samples AND their exact log q in one call — the core needs no separate scorer.
- New mW files live in `liquid_coupling_flow/mw/`; tests in repo-root `tests/`. Artifacts → `liquid_coupling_flow/mw/artifacts/`; logs → `reports/logs-<date>/`.
- git: stage ONLY named files (never `git add -A`). Long runs: `nohup ... > log 2>&1 &`, echo PID, never `/dev/null` stderr. GPU is shared with the user's runs — kill/launch nothing outside your task.
- Fixed seeds on every stochastic path.

---

### Task 0: Vendored-pipeline 3D smoke

Verify the vendored `EGNN_dynamics` + `RiemannianFlowMatching` instantiate and run in 3D / monatomic BEFORE any mW glue — the paper ran 2D, so this is the first de-risk (spec §Risks 4).

**Files:**
- Create: `liquid_coupling_flow/mw/mw_ersi_common.py` (the `_add_learndiffeq_path()` helper + shared constants)
- Test: `tests/test_mw_ersi_smoke.py`

**Interfaces:**
- Produces: `mw_ersi_common._add_learndiffeq_path()` (idempotent sys.path insert, returns the abs path); `mw_ersi_common.LEARNDIFFEQ_ROOT` (str).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mw_ersi_smoke.py
import torch
from liquid_coupling_flow.mw.mw_ersi_common import _add_learndiffeq_path

def test_egnn_3d_forward_and_divergence_runs():
    _add_learndiffeq_path()
    from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
    N, D, K, L, B = 8, 3, 4, 4.0, 2
    m = EGNN_dynamics(n_particles=N, n_dimension=D, hidden_nf=32, n_layers=2,
                      max_neighbors=K, L=L, n_species=1)
    xs = torch.rand(B, N, D) * L
    t = torch.zeros(B)
    a = torch.zeros(B, N, dtype=torch.long)          # monatomic: all species 0
    vel, div = m.forward_and_divergence(xs, t, a)
    assert vel.shape == (B, N, D)
    assert div.shape == (B,) and torch.isfinite(div).all()
```

- [ ] **Step 2: Run to verify failure**: `pytest tests/test_mw_ersi_smoke.py -x -q` → ImportError (no mw_ersi_common).

- [ ] **Step 3: Implement `mw_ersi_common.py`**

```python
"""Shared helpers for the mW eRSI flow base: path to the vendored learndiffeq pipeline + constants."""
from __future__ import annotations
import os, sys

LEARNDIFFEQ_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "liquid_coupling_flow", "ipl44", "learndiffeq")
# __file__ = .../liquid_coupling_flow/mw/mw_ersi_common.py -> repo root is three dirs up
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEARNDIFFEQ_ROOT = os.path.join(_REPO, "liquid_coupling_flow", "ipl44", "learndiffeq")


def _add_learndiffeq_path():
    if LEARNDIFFEQ_ROOT not in sys.path:
        sys.path.insert(0, LEARNDIFFEQ_ROOT)
    return LEARNDIFFEQ_ROOT
```

Verify `LEARNDIFFEQ_ROOT` resolves: it must contain `learndiffeq/particles/velocities/egnn_traceable.py`. If the three-dirs-up arithmetic is off, fix it by locating the real path (the implementer checks `os.path.exists(os.path.join(LEARNDIFFEQ_ROOT, "learndiffeq"))`).

- [ ] **Step 4: Run**: `pytest tests/test_mw_ersi_smoke.py -v` → PASS. If `forward_and_divergence` errors in 3D, STOP and report BLOCKED with the traceback — a 3D-unsupported vendored path is a plan-level finding, not something to patch silently.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_ersi_common.py tests/test_mw_ersi_smoke.py && git commit -m "feat(mw-ersi): learndiffeq path helper + 3D EGNN forward/divergence smoke"`

---

### Task 1 (G-a): 3D kNN-trimmed exact-divergence verification

The exactness backbone: the analytical divergence must equal a brute-force autodiff divergence in 3D. The periodic branch's two historical bugs were fixed + verified in 2D (traceable-egnn memory); this re-verifies in 3D before training spends GPU.

**Files:**
- Test: `tests/test_mw_ersi_divergence.py`
- Create: `reports/logs-<date>/mw_ersi_ga.out` (run log, committed)

**Interfaces:**
- Consumes: `EGNN_dynamics.forward_and_divergence` (Task 0).

- [ ] **Step 1: Write the failing test** (brute-force divergence via autograd trace vs the analytical value)

```python
# tests/test_mw_ersi_divergence.py
import torch, pytest
from liquid_coupling_flow.mw.mw_ersi_common import _add_learndiffeq_path

def _bruteforce_div(m, xs, t, a):
    """div v = sum_i d v_i / d x_i via autograd (exact, O(3N) backward passes)."""
    B, N, D = xs.shape
    xs = xs.clone().requires_grad_(True)
    vel, _ = m.forward_and_divergence(xs, t, a, differentiable=True)
    div = torch.zeros(B, device=xs.device, dtype=xs.dtype)
    for i in range(N):
        for d in range(D):
            g = torch.autograd.grad(vel[:, i, d].sum(), xs, retain_graph=True, create_graph=False)[0]
            div = div + g[:, i, d]
    return div

@pytest.mark.parametrize("K", [4, 6, 7])
def test_analytical_div_matches_bruteforce_3d(K):
    _add_learndiffeq_path()
    from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
    torch.manual_seed(0)
    N, D, L, B = 8, 3, 4.0, 3
    m = EGNN_dynamics(n_particles=N, n_dimension=D, hidden_nf=32, n_layers=2,
                      max_neighbors=K, L=L, n_species=1).double()
    xs = (torch.rand(B, N, D, dtype=torch.float64) * L)
    t = torch.full((B,), 0.37, dtype=torch.float64)
    a = torch.zeros(B, N, dtype=torch.long)
    _, div_analytical = m.forward_and_divergence(xs, t, a)
    div_brute = _bruteforce_div(m, xs, t, a)
    assert torch.allclose(div_analytical, div_brute, atol=1e-6), \
        f"K={K}: max|d| {float((div_analytical - div_brute).abs().max()):.2e}"
```

- [ ] **Step 2: Run to verify it MEASURES** (it may pass immediately since the method exists — the point is the assertion is real): `pytest tests/test_mw_ersi_divergence.py -v`. If it FAILS, the 3D periodic divergence is wrong — STOP, report BLOCKED with max|d| per K (this is exactly the class of bug G-a exists to catch; do not loosen the tolerance).

- [ ] **Step 3: No implementation** — this task validates the vendored method. If Step 2 passed, proceed. If it failed, the divergence code needs a fix, which is a separate escalation (report BLOCKED).

- [ ] **Step 4: Record the gate**: run `pytest tests/test_mw_ersi_divergence.py -v > reports/logs-<date>/mw_ersi_ga.out 2>&1` and confirm all K pass; append a one-line verdict.

- [ ] **Step 5: Commit**: `git add tests/test_mw_ersi_divergence.py reports/logs-<date>/mw_ersi_ga.out && git commit -m "test(mw-ersi): G-a — 3D kNN-trimmed analytical divergence == brute-force autodiff (all K)"`

---

### Task 2: mW → learndiffeq data adapter

**Files:**
- Create: `liquid_coupling_flow/mw/mw_ersi_data.py`
- Test: `tests/test_mw_ersi_data.py`

**Interfaces:**
- Consumes: `mw_reference` bank artifacts; `mw_energy.RHO_STAR`.
- Produces: `write_ersi_dataset(out_pos, out_species, bank_paths, thin_events=2, val_frac=0.1, seed=0) -> dict` (writes two `torch.save` tensors — positions `[n,N,3]` float wrapped to `[0,L)`, species `[n,N]` int zeros — and returns `{"n_train", "n_val", "N", "L"}`); `L_for_N(N) -> float`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mw_ersi_data.py
import torch
from liquid_coupling_flow.mw.mw_ersi_data import write_ersi_dataset, L_for_N

def test_dataset_format(tmp_path):
    # synthetic bank in the mc_run dict shape {"cfgs": [n,N,3]}
    N = 8; L = L_for_N(N)
    cfgs = torch.rand(320, N, 3) * L * 1.5 - 0.2 * L    # deliberately out of [0,L) to test wrap
    bank = tmp_path / "bank.pt"; torch.save({"cfgs": cfgs}, bank)
    pos_p = tmp_path / "pos.pt"; sp_p = tmp_path / "sp.pt"
    info = write_ersi_dataset(str(pos_p), str(sp_p), [str(bank)], thin_events=2, val_frac=0.1, seed=0)
    pos = torch.load(pos_p, weights_only=False); sp = torch.load(sp_p, weights_only=False)
    assert pos.shape[1:] == (N, 3) and (pos >= 0).all() and (pos < L).all()   # wrapped
    assert sp.shape == pos.shape[:2] and (sp == 0).all()                      # monatomic
    assert info["N"] == N and abs(info["L"] - L) < 1e-9 and info["n_train"] > 0
```

- [ ] **Step 2: Run**: `pytest tests/test_mw_ersi_data.py -x -q` → ImportError.

- [ ] **Step 3: Implement.** `L_for_N(N) = (N / RHO_STAR) ** (1/3)`. `write_ersi_dataset`: load each bank's `cfgs` (`[n,N,3]`), concatenate, thin by `thin_events` (stride over the FIRST axis — these are already collection events), wrap with `torch.remainder(x, L)`, split the LAST `val_frac` (by index order = later-in-time, the leakage-safe split), `torch.save` the train positions to `out_pos` and a zeros `[n,N]` int64 tensor to `out_species`. (The vendored `make_dataset` loads exactly these two files.) Return the info dict. Keep val positions/species saved to sibling `*_val.pt` paths for G-b/eval reuse.

- [ ] **Step 4: Run**: `pytest tests/test_mw_ersi_data.py -v` → PASS.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_ersi_data.py tests/test_mw_ersi_data.py && git commit -m "feat(mw-ersi): bank -> learndiffeq dataset adapter (wrap, monatomic species, time-ordered val split)"`

---

### Task 3: Training entrypoint (drives vendored RiemannianFlowMatching)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_ersi_train.py`
- Test: `tests/test_mw_ersi_train.py`

**Interfaces:**
- Consumes: `write_ersi_dataset`, `L_for_N` (Task 2); `_add_learndiffeq_path` (Task 0); vendored `RiemannianFlowMatching`, `make_dataset`, `EGNN_dynamics`.
- Produces: `train(N=64, K=12, hidden_nf=128, n_layers=4, steps=..., batch=32, lr=3e-4, ot=True, out="mw_ersi_N64.pt", bank_paths=None, seed=0, max_steps=...) -> ckpt_path`. Checkpoint stores the velocity `state_dict` + all reconstruction kwargs (`N, K, hidden_nf, n_layers, L, dim_phys=3, n_species=1, ot`).

- [ ] **Step 1: Write the failing test** (tiny 3D training smoke — loss finite + decreases)

```python
# tests/test_mw_ersi_train.py
import torch
from liquid_coupling_flow.mw.mw_ersi_train import train

def test_train_smoke(tmp_path):
    # synthetic tiny bank
    N = 8; from liquid_coupling_flow.mw.mw_ersi_data import L_for_N
    L = L_for_N(N); cfgs = torch.rand(256, N, 3) * L
    bank = tmp_path / "bank.pt"; torch.save({"cfgs": cfgs}, bank)
    out = tmp_path / "m.pt"
    info = train(N=N, K=4, hidden_nf=16, n_layers=1, steps=40, batch=16, lr=1e-3,
                 out=str(out), bank_paths=[str(bank)], seed=0)
    assert out.exists()
    ck = torch.load(out, weights_only=False)
    assert ck["N"] == N and ck["K"] == 4 and "state_dict" in ck
    assert info["loss_last"] < info["loss_first"]      # learned something
```

- [ ] **Step 2: Run**: `pytest tests/test_mw_ersi_train.py -x -q` → ImportError.

- [ ] **Step 3: Implement.** `_add_learndiffeq_path()`; build the uniform-torus `rho0` (the vendored pipeline has a torus uniform prior — reuse it; if constructing directly, `rho0` returns `(a, X)` with `a` constant zeros and `X ~ U[0,L)^{N×3}`, and `rho0.log_prob(a, X) = -N*3*log(L)`). Write the dataset via `write_ersi_dataset` to temp files, `make_dataset(...)` → `dm`. Construct `RiemannianFlowMatching(n_particles=N, dim_phys=3, L=L, rho0=rho0, lr=lr, velocity_type="egnn_traceable", velocity_kwargs=dict(hidden_nf=hidden_nf, n_layers=n_layers, max_neighbors=K, n_species=1), ot_particles=ot, use_linear_assignment_particles=ot, force_automatic_div=False)`. Train with a Lightning `Trainer(max_steps=steps, ...)` + the augmentation `DataAugmentationCallback` (translations + signed-permutations; import from the experiment module or replicate the transform — monatomic makes the species-perm trivial). Record first/last train loss. Save `{"state_dict": model.velocity.state_dict(), "N", "K", "hidden_nf", "n_layers", "L", "dim_phys":3, "n_species":1, "ot", "step"}` to `out`. **Verify `velocity_type` string** against `make_velocity`'s dispatch in the vendored code (grep `make_velocity`); if the registered name differs (e.g. `"egnn"`), use the registered one and note it.

- [ ] **Step 4: Run**: `pytest tests/test_mw_ersi_train.py -v` → PASS (a few min on CPU/GPU; if the Lightning Trainer is heavyweight, cap devices=1 and disable logging).

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_ersi_train.py tests/test_mw_ersi_train.py && git commit -m "feat(mw-ersi): 3D monatomic training entrypoint (vendored RFM + OT + kNN EGNN, augmentation)"`

---

### Task 4: `MWeRSIFlow` deploy wrapper

**Files:**
- Create: `liquid_coupling_flow/mw/mw_ersi.py`
- Test: `tests/test_mw_ersi.py`

**Interfaces:**
- Consumes: `_add_learndiffeq_path` (Task 0); the trained checkpoint from `train` (Task 3); vendored `RiemannianFlowMatching.sample`, `.sample_and_log_prob`, and reverse-time scoring (`ml/maximum_likelihood.sample(values, a, reverse_time=True, return_log_jac=True)`).
- Produces: `load_ersi(path, device) -> MWeRSIFlow`; `MWeRSIFlow` with `.sample_and_logq(B, gen) -> (X [B,N,3], logq [B])` (the one-shot IS core — X wrapped to `[0,L)`, logq = composed likelihood), `.log_q(X) -> [B]` (reverse-time scorer for arbitrary configs; the optional-tail path), `.N`, `.L`, `.LOGQ_CONST=False`. A `GeneratorBase`-shaped adapter method `.as_generator_base()` returning an object with `.sample(B, gen)` and `.log_q(x)` for the existing `smc_run`.

- [ ] **Step 1: Write the failing test** (round-trip = G-b(i): the composed logq from generation equals the reverse-time `log_q(X)` of the same X)

```python
# tests/test_mw_ersi.py
import torch
from liquid_coupling_flow.mw.mw_ersi import load_ersi
from liquid_coupling_flow.mw.mw_ersi_train import train
from liquid_coupling_flow.mw.mw_ersi_data import L_for_N

def _tiny_ckpt(tmp_path):
    N = 8; L = L_for_N(N); cfgs = torch.rand(256, N, 3) * L
    bank = tmp_path / "b.pt"; torch.save({"cfgs": cfgs}, bank)
    out = tmp_path / "m.pt"
    train(N=N, K=4, hidden_nf=16, n_layers=1, steps=20, batch=16, out=str(out), bank_paths=[str(bank)], seed=0)
    return str(out)

def test_sample_logq_roundtrip(tmp_path):
    m = load_ersi(_tiny_ckpt(tmp_path), "cpu")
    g = torch.Generator().manual_seed(0)
    X, logq_gen = m.sample_and_logq(6, g)
    assert X.shape == (6, m.N, 3) and (X >= 0).all() and (X < m.L).all()
    logq_score = m.log_q(X)                          # reverse-time scorer, same fixed solver
    assert torch.allclose(logq_gen, logq_score, atol=1e-3), \
        f"round-trip max|d| {float((logq_gen - logq_score).abs().max()):.2e}"
```

- [ ] **Step 2: Run**: `pytest tests/test_mw_ersi.py -x -q` → ImportError.

- [ ] **Step 3: Implement.** `load_ersi`: reconstruct the velocity `EGNN_dynamics(**kwargs)` + a `RiemannianFlowMatching` shell (or the minimal ODE-integration object) from the ckpt, load `state_dict`, `.eval()`, fix the solver (deterministic, fixed step count — expose `n_steps` default from a solver-convergence probe, e.g. 20 RK4 or 40 Euler; document). `sample_and_logq`: call the vendored `sample_and_log_prob((B, N, 3))` with the constant monatomic `a`, wrap X to `[0,L)`, return `(X, log_prob)`. `log_q(X)`: reverse-time integrate X→base via `reverse_time=True, return_log_jac=True`, `logq = rho0.log_prob(a0, x0) + log_jac`. Both paths use the SAME fixed solver ⇒ round-trip holds. `as_generator_base()`: thin object with `LOGQ_CONST=False`, `sample(B,gen)` (calls `sample_and_logq`, returns X), `log_q(x)` (calls `.log_q`).

- [ ] **Step 4: Run**: `pytest tests/test_mw_ersi.py -v` → PASS. If the round-trip exceeds 1e-3, increase solver steps (OT should keep it tight); if it can't be met, report DONE_WITH_CONCERNS with the achieved tolerance and the step count needed (informs G-b's integration-error budget).

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_ersi.py tests/test_mw_ersi.py && git commit -m "feat(mw-ersi): MWeRSIFlow wrapper — sample_and_logq (one-shot IS core) + reverse-time log_q + GeneratorBase shim"`

---

### Task 5 (G-b): composed-likelihood integrity (run + gate)

**Files:**
- Modify: `liquid_coupling_flow/mw/mw_ersi.py` (add `normalization_check`, `solver_convergence`)
- Test: `tests/test_mw_ersi.py` (append small-N normalization unit)
- Create: `reports/logs-<date>/mw_ersi_gb.out`

**Interfaces:**
- Consumes: `MWeRSIFlow` (Task 4).
- Produces: `normalization_check(model, n_quad=..., seed=0) -> float` (∫ q_flow over a small-N box via grid quadrature at N small enough to be tractable — build a dedicated tiny model, integrate `exp(log_q)` on a coarse grid, return the integral, target ≈ 1); `solver_convergence(model, X, steps_list) -> dict` (log_q of fixed X at increasing solver steps → the residual integration error we accept).

- [ ] **Step 1: Write the failing normalization test** (tiny N=2 or N=3, coarse grid — the bedrock that the composed density integrates to 1)

```python
def test_normalization_smallN(tmp_path):
    # For a tiny system the composed density must integrate to ~1 over the box (fix all but one particle)
    from liquid_coupling_flow.mw.mw_ersi import load_ersi, normalization_check
    m = load_ersi(_tiny_ckpt(tmp_path), "cpu")     # reuse the tiny helper
    z = normalization_check(m, n_quad=12, seed=0)  # coarse grid; conditional on a fixed prefix
    assert abs(z - 1.0) < 0.15                      # coarse-grid tolerance; G-b run uses finer
```

- [ ] **Step 2: Run**: `pytest tests/test_mw_ersi.py::test_normalization_smallN -x -q` → fail (functions absent).

- [ ] **Step 3: Implement** `normalization_check` (grid-quadrature of `exp(log_q)` over the box for the smallest tractable N — document the exact construction, e.g. integrate the full joint at N=2 on a G^(2·3) grid, or a single-particle conditional; whichever is tractable, state which) and `solver_convergence` (return `{steps: log_q_mean}` for `steps_list`).

- [ ] **Step 4: Run the gate**: `pytest tests/test_mw_ersi.py -v` (unit) then a G-b run script that (i) confirms the sample↔log_q round-trip at production N=64 on 64 configs ≤1e-3, (ii) prints `solver_convergence` over steps ∈ {10,20,40,80} (report the plateau — the accepted integration-error floor), (iii) the small-N normalization at a finer grid → `reports/logs-<date>/mw_ersi_gb.out`. PASS = round-trip ≤1e-3 AND normalization within grid error AND log_q converged by the chosen production step count.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_ersi.py tests/test_mw_ersi.py reports/logs-<date>/mw_ersi_gb.out && git commit -m "test(mw-ersi): G-b — composed-likelihood integrity (round-trip, normalization, solver convergence)"`

---

### Task 6: One-shot IS harness + G-c/G-d (code)

**Files:**
- Create: `liquid_coupling_flow/mw/mw_ersi_eval.py`
- Test: `tests/test_mw_ersi_eval.py`

**Interfaces:**
- Consumes: `MWeRSIFlow` (Task 4); `mw_energy.mw_energy_chunked, T_STAR, RHO_STAR`; `mw_reference.g_r`; `mw_base.UniformBase, ExcludedVolumeBase, GeneratorBase, load_any_generator`.
- Produces: `oneshot_is(model, B=512, seed=0) -> dict` (`X, logw, ess, u_reweighted, gr_raw, gr_reweighted`); `ladder_gr(...)` producing the {uniform, exclvol, v10-AR, eRSI} one-shot g(r) comparison; `gc_gd_report(model, ...)` computing the G-c structure metrics (raw-sample shell1@1.19, shell2@1.85 vs data 2.12/1.19) and G-d (ESS/B, reweighted ⟨U⟩ vs reference) + a committed plot; CLI `python -m liquid_coupling_flow.mw.mw_ersi_eval <ckpt>`.

- [ ] **Step 1: Write the failing test** (harness on the tiny model — finite logw/ess, shells measured)

```python
# tests/test_mw_ersi_eval.py
import torch
from liquid_coupling_flow.mw.mw_ersi_eval import oneshot_is
from liquid_coupling_flow.mw.mw_ersi import load_ersi
# reuse a tiny ckpt built as in test_mw_ersi

def test_oneshot_is_runs(tiny_ersi_ckpt):     # fixture builds a tiny ckpt
    m = load_ersi(tiny_ersi_ckpt, "cpu")
    out = oneshot_is(m, B=32, seed=0)
    assert torch.isfinite(out["logw"]).all() and 1.0 <= out["ess"] <= 32.0
    assert "gr_raw" in out and out["gr_raw"][1].shape[0] > 0
```

- [ ] **Step 2: Run**: `pytest tests/test_mw_ersi_eval.py -x -q` → ImportError.

- [ ] **Step 3: Implement.** `oneshot_is`: `X, logq = model.sample_and_logq(B, gen)`; `U = mw_energy_chunked(X, L)`; `logw = -beta*U - logq` (float64); `ess = 1/sum(softmax(logw)^2)`; `gr_raw = g_r(X)`; `gr_reweighted = g_r(multinomial-resample(X, softmax(logw)))`; `u_reweighted = sum(softmax(logw)*U)/N`. `gc_gd_report`: compute shell1/shell2 of `gr_raw` vs reference (from the ext bank), ess/B, reweighted ⟨U⟩ vs the reference −1.627; overlay plot {reference, eRSI raw, eRSI reweighted}; print the pre-registered G-c verdict (shell2 approaching 1.19?) and the G-d ladder line vs {uniform, exclvol, v10}. Save full arrays to `artifacts/mw_ersi_eval.pt`.

- [ ] **Step 4: Run**: `pytest tests/test_mw_ersi_eval.py -v` → PASS.

- [ ] **Step 5: Commit**: `git add liquid_coupling_flow/mw/mw_ersi_eval.py tests/test_mw_ersi_eval.py && git commit -m "feat(mw-ersi): one-shot IS harness + G-c/G-d report (shells, ESS, reweighted observables)"`

---

### Task 7: Real training + G-c/G-d evaluation (controller-run)

**Files:**
- Create: `reports/logs-<date>/mw_ersi_train.out`, `reports/logs-<date>/mw_ersi_gcgd.out`, `reports/<date>-mw-ersi-results.md`

**Interfaces:** Consumes Tasks 3/4/6.

- [ ] **Step 1: Launch real training** on the mW N=64 bank (thin-2 + extension), K=12, hidden_nf=128, n_layers=4, OT on, augmentation, steps sized to val convergence: `nohup python -c "from liquid_coupling_flow.mw.mw_ersi_train import train; train(out='mw_ersi_N64.pt', ...)" > reports/logs-<date>/mw_ersi_train.out 2>&1 &` — echo PID; monitor val loss.
- [ ] **Step 2: G-b re-run** on the trained model (round-trip + solver convergence at production N=64) — must pass before eval.
- [ ] **Step 3: G-c / G-d**: `python -m liquid_coupling_flow.mw.mw_ersi_eval liquid_coupling_flow/mw/artifacts/mw_ersi_N64.pt > reports/logs-<date>/mw_ersi_gcgd.out 2>&1`. Record the raw-sample shells (the headline), one-shot ESS/B, reweighted ⟨U⟩, and the energy-eval ladder vs {uniform, exclvol, v10}. Show the plot path.
- [ ] **Step 4: Verdict + results note**: write `reports/<date>-mw-ersi-results.md` — G-a→G-d chain, the G-c go/no-go call (did the non-causal flow make the second shell?), honest numbers. Update memory (`mw-ersi-flow-base` + MEMORY.md hook).
- [ ] **Step 5: Commit**: results note + logs + plots + memory.

---

## Self-Review (performed at write time)

- **Spec coverage:** §Training → Tasks 2/3; §Deploy wrapper → Task 4; §Measurement → Task 6; G-a → Task 1; G-b → Task 5; G-c/G-d → Tasks 6/7; §Risks 4 (3D support) → Task 0. All spec sections mapped.
- **Placeholder scan:** solver step count is a measured tunable (G-b, Task 5) not a placeholder; `velocity_type` string is flagged for verification against `make_velocity` in Task 3 (a real discovery step, not a guess left open). No TBD/TODO.
- **Type consistency:** `sample_and_logq -> (X, logq)` used identically in Tasks 4/6; `log_q(X) -> [B]` reverse scorer consistent; `EGNN_dynamics(max_neighbors=K)` and `forward_and_divergence(xs,t,a)->(vel,div)` match the vendored signatures read from source; checkpoint keys (`N,K,hidden_nf,n_layers,L`) consistent across Tasks 3/4.
- **Known verification steps (not gaps):** Task 0 confirms `LEARNDIFFEQ_ROOT`; Task 3 confirms the `velocity_type` registered name and the exact `rho0`/augmentation constructors against vendored source — both are cheap discovery steps a fresh implementer performs, with fallback instructions given.
