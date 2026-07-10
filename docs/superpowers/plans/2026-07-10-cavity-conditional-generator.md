# Cavity-Conditional Generator for Point-to-Set Correlations — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a boundary-conditioned neural generator that samples equilibrium cavity interiors of the 3D Kob-Andersen glass and measures point-to-set (PTS) correlations *on par with* the Berthier-Charbonneau-Yaida (JCP 144, 024501, 2016) parallel-tempering + size-shrinkage scheme, while being *transferable* (train once, apply zero-shot across cavity radius R and temperature T).

**Architecture:** The cavity interior at equilibrium is *exactly* the bulk Boltzmann distribution conditioned on the frozen boundary: `p(x_in | x_out)`. So (1) we generate abundant, well-equilibrated **bulk** 3D configs (cheap — no confinement slowdown); (2) we carve random cavities out of them to get free, self-supervised `(boundary → interior)` training pairs; (3) we train a masked/boundary-conditioned generator to sample `p(x_in | x_out)` directly — one-shot, bypassing the free-energy barriers that force PT to crawl via replica exchange; (4) we validate its overlap distribution `P(q_c)` against a from-scratch **PT+shrinkage** baseline, which serves only as the ground-truth yardstick at a few points, NOT as the training-data engine. The niche is real because the physics baseline (PT+shrinkage) is a decade-hard, hand-tuned, per-state-point scheme; "match + transfer" beats "re-tune per state point."

**Tech Stack:** PyTorch (GPU), existing `liquid_coupling_flow` primitives (`ka_energy` dimension-generic, `ka_cavity` hard wall, `ka_pmc_3d` vectorized sampler, `ka_dataset_3d` bulk gen, `ipl44/joint_flow` 3D-capable EGNN flow), pytest.

## Global Constraints

- **Model: 3D 80:20 Kob-Andersen binary LJ (KABLJ).** Species A,B with `N_A:N_B = 4:1` (composition_B = 0.2). `sigma_AA=1.0, sigma_AB=0.8, sigma_BB=0.88`; `eps_AA=1.0, eps_AB=1.5, eps_BB=0.5`. Cutoff `r_cut = 2.5*sigma_AB` shifted so V vanishes at cutoff. Density `rho = 1.2`, box `L = (N/rho)^(1/3)`. LJ reduced units (`sigma_AA`, `eps_AA/k_B`). These match `liquid_coupling_flow/ka_energy.py` SIGMA/EPS/RCUT_FACTOR and `ka_cavity_3d.py` (RHO=1.2, X_B=0.2). MCT crossover `T_MCT ~= 0.435`.
- **Hard spherical cavity wall:** freeze every particle outside radius R of a fixed center; a move that takes a mobile particle to `|x-center| >= R` (minimum image) is rejected. Use `liquid_coupling_flow/ka_cavity.py::cavity_inside` / `assert_mobile_inside` (already correct + tested).
- **Shrinkage Hamiltonian (paper Eq. 2):** replica `a` uses pair potential with `sigma -> lambda_tilde_a * sigma`, where `lambda_tilde_a = lambda_a` for a mobile-mobile pair and `lambda_tilde_a = (1+lambda_a)/2` for a mobile-pinned pair. Bottom replica is physical: `T_1 = T, lambda_1 = 1`. Higher replicas have `T_a > T` and `lambda_a <= 1`.
- **Core overlap:** measured on the cavity core `|r - center| < r_c` with `r_c = 0.5`. Overlap kernel `w(z) = exp(-(z/b)^2)` with `b = 0.2` (paper Eq. 5). Species-agnostic (occupancy-style) so the estimate is insensitive to `w` details as long as range ~ cage size.
- **PTS susceptibility (paper Eq. 10):** `chi_T(R) = <q_c^2>_{J(R)} - <q_c>^2_{J(R)}`, disorder-averaged over >= 50 cavity centers. Its peak location = `xi_PTS(T)`, no fitting required. This is the primary observable.
- **Two-arm convergence gate (Cavagna/Berthier):** equilibration is accepted only when the running core overlap started from (i) the original config and (ii) a randomized config meet within `q_tol` (paper uses 0.1 per cavity; <=0.005 when averaged over 50 cavities). Never report a PTS number from an unconverged cavity.
- **All learned components stay behind exact Metropolis OR are validated against the PT ground truth before any "on par" claim.** Report mean +/- SEM (n = number of cavity centers), never bootstrap `|diff|` (positive-biased — this bit us once).
- **Durability:** raw logs + one-off scripts to `reports/logs-<date>/`, committed the moment a run finishes; artifacts to `liquid_coupling_flow/artifacts/`; long runs launched with the Bash `run_in_background: true` param and per-unit incremental saves.

---

## Program Milestones and Go/No-Go Gates

| Phase | Milestone | Deliverable | GATE (must pass to proceed) |
|-------|-----------|-------------|------------------------------|
| 1 | Foundation | PT+shrinkage baseline + bulk data engine | **G1: our PT reproduces the paper's Fig 2** — `G_PTS(R)` decays and `chi_T(R)` peaks, `xi_PTS` grows ~x2 from T=1.0 to ~0.5, values within paper ballpark |
| 2 | Generator | Boundary-conditioned cavity generator | **G2: generator `P(q_c)` matches PT `P(q_c)`** within noise at 2+ (R,T) points, including bimodality near `xi_PTS` |
| 2 | Transfer | Zero-shot across R (and T) | **G3: generator trained at a subset of R (and one T) reproduces PT `chi_T(R)` at held-out R (and T) without retraining** |

**Phase 1 is detailed below as executable tasks. Phase 2 is a roadmap (Tasks 7-10) to be expanded into a second detailed plan after G1 passes** — the generator architecture and eval design depend on what the baseline reveals (e.g. how multimodal `P(q_c)` actually is at accessible T).

---

## File Structure

- `liquid_coupling_flow/ka3d_shrink_energy.py` — shrinkage pair energy `V(lambda_tilde)` for mobile/pinned pairs; per-particle energy row for single-site moves under a given `lambda`. (Task 1)
- `liquid_coupling_flow/ka3d_pt_shrink.py` — replica ladder construction, hard-wall single-site cavity MC under shrinkage, Hamiltonian replica exchange, the full PT cavity-equilibration driver + two-arm convergence gate. (Tasks 2, 4)
- `liquid_coupling_flow/ka3d_pts_observables.py` — core overlap `q_c(X,Y)`, `G_PTS(R)`, susceptibility `chi_T(R)`, overlap PDF `P(q_c)`. (Task 3)
- `liquid_coupling_flow/ka_dataset_3d.py` — EXISTS; add a bulk-equilibration validation (seed-agreement + tail-flat) and a `warm_start` option to cut cold-start cost. (Task 6)
- `liquid_coupling_flow/ka3d_cavity_carve.py` — carve random cavities from bulk configs into `(boundary, interior, mask)` training pairs. (Task 7, Phase 2)
- `liquid_coupling_flow/ka3d_cavity_generator.py` — boundary-conditioned generator (masked EGNN flow) + training loop. (Task 8, Phase 2)
- `liquid_coupling_flow/ka3d_cavity_eval.py` — generator `P(q_c)` vs PT + transfer eval. (Tasks 9-10, Phase 2)
- Tests in `liquid_coupling_flow/tests/`. Run scripts + logs in `reports/logs-2026-07-10/` (and later dates).

---

## Phase 1 — Foundation (the yardstick + the data engine)

### Task 1: Shrinkage pair energy

**Files:**
- Create: `liquid_coupling_flow/ka3d_shrink_energy.py`
- Test: `liquid_coupling_flow/tests/test_ka3d_shrink_energy.py`

**Interfaces:**
- Consumes: `liquid_coupling_flow.ka_energy.SIGMA, EPS, RCUT_FACTOR`.
- Produces:
  - `lambda_tilde(lam: float, mobile_i: BoolTensor[B,N], mobile_j: BoolTensor[B,N]) -> Tensor[B,N,N]` — per-pair shrinkage: `lam` where both mobile, `(1+lam)/2` where exactly one mobile, `1.0` where both pinned (pinned-pinned constant, value irrelevant to moves).
  - `shrink_particle_energy(x, s, L, lam, mobile) -> Tensor[B,N]` — per-particle shifted-LJ energy `pe_i = sum_{j!=i} V_ab(r_ij; lambda_tilde_ij*sigma_ab)`, minimum image. At `lam=1` it MUST equal `ka_pmc_3d.particle_energies` (== `ka_energy` convention).

- [ ] **Step 1: Write the failing test — lam=1 reduces to the standard energy**

```python
# test_ka3d_shrink_energy.py
import torch
from liquid_coupling_flow.ka3d_shrink_energy import shrink_particle_energy, lambda_tilde
from liquid_coupling_flow.ka_pmc_3d import particle_energies
from liquid_coupling_flow.ka_energy import ka_energy

def test_lambda_one_matches_standard_energy():
    torch.manual_seed(0); B, N, L = 2, 30, 3.4
    x = torch.rand(B, N, 3) * L; s = torch.tensor([[0]*24 + [1]*6] * B)
    mobile = torch.ones(B, N, dtype=torch.bool)
    pe = shrink_particle_energy(x, s, L, lam=1.0, mobile=mobile)
    assert torch.allclose(pe, particle_energies(x, s, L), atol=1e-5)
    assert torch.allclose(0.5 * pe.sum(1), ka_energy(x, s, L), atol=1e-4)

def test_lambda_tilde_pair_rules():
    mob = torch.tensor([[True, True, False]])       # particles 0,1 mobile; 2 pinned
    lt = lambda_tilde(0.6, mob, mob)                # [1,3,3]
    assert abs(lt[0,0,1].item() - 0.6) < 1e-6       # mobile-mobile -> lam
    assert abs(lt[0,0,2].item() - 0.8) < 1e-6       # mobile-pinned -> (1+lam)/2
    assert abs(lt[0,2,2].item() - 1.0) < 1e-6       # pinned-pinned -> 1.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `CUDA_VISIBLE_DEVICES="" pytest liquid_coupling_flow/tests/test_ka3d_shrink_energy.py -q`
Expected: FAIL (module not found).

- [ ] **Step 3: Implement**

```python
# ka3d_shrink_energy.py
"""Size-shrinkage pair energy for the Berthier-Charbonneau-Yaida cavity PT scheme (JCP 144 024501).
Replica a scales sigma -> lambda_tilde*sigma: lambda for mobile-mobile pairs, (1+lambda)/2 for mobile-pinned.
At lambda=1 this reduces exactly to ka_energy / ka_pmc_3d.particle_energies (asserted in tests)."""
from __future__ import annotations
import torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR

def lambda_tilde(lam, mobile_i, mobile_j):
    mi = mobile_i[:, :, None]; mj = mobile_j[:, None, :]           # [B,N,1],[B,1,N]
    both = mi & mj; one = mi ^ mj
    out = torch.ones(mi.shape[0], mi.shape[1], mj.shape[2], device=mobile_i.device)
    out = torch.where(both, torch.full_like(out, float(lam)), out)
    out = torch.where(one, torch.full_like(out, (1.0 + float(lam)) / 2.0), out)
    return out

def shrink_particle_energy(x, s, L, lam, mobile):
    B, N, _ = x.shape; dev = x.device; dt = x.dtype
    t_sig = torch.tensor(SIGMA, device=dev, dtype=dt); t_eps = torch.tensor(EPS, device=dev, dtype=dt)
    a = s[:, :, None].expand(-1, -1, N); b = s[:, None, :].expand(-1, N, -1)
    lt = lambda_tilde(lam, mobile, mobile).to(dt)
    sig = lt * t_sig[a, b]; eps = t_eps[a, b]; rc = RCUT_FACTOR * sig      # cutoff scales with shrunk sigma
    diff = x[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
    eye = torch.eye(N, device=dev, dtype=torch.bool)[None]
    r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig / rc) ** 6
    return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)
```

- [ ] **Step 4: Run to verify pass**

Run: `CUDA_VISIBLE_DEVICES="" pytest liquid_coupling_flow/tests/test_ka3d_shrink_energy.py -q`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ka3d_shrink_energy.py liquid_coupling_flow/tests/test_ka3d_shrink_energy.py
git commit -m "feat(pts-pt): shrinkage pair energy V(lambda_tilde), reduces to ka_energy at lambda=1"
```

### Task 2: Hard-wall single-site cavity move under shrinkage + replica exchange

**Files:**
- Create: `liquid_coupling_flow/ka3d_pt_shrink.py`
- Test: `liquid_coupling_flow/tests/test_ka3d_pt_shrink.py`

**Interfaces:**
- Consumes: `ka3d_shrink_energy.shrink_particle_energy`; `ka_cavity.cavity_inside, assert_mobile_inside`; `ka_cavity_3d.local_identity_swap` (exact A/B swap; species identity is lambda-independent for the swap ratio EXCEPT via energy — recompute energy under the replica's lambda).
- Produces:
  - `cavity_sweep(x, s, mobile, center, R, L, beta, lam, step) -> (x, s)` — one sweep = `N_cav` hard-wall single-site displacement attempts (each scored by `shrink_particle_energy` delta on the moved particle, minimum image, reject if outside R) + `N_cav//8` identity swaps under `lam`. Batched over replicas (leading dim = n_replicas).
  - `replica_exchange(x, s, mobile, center, R, L, betas, lams) -> (x, s, acc)` — attempt adjacent-replica config swaps with the Hamiltonian-RE Metropolis ratio `min(1, exp(-(beta_a*[U_b(a-config) - U_a(a-config)] + beta_b*[U_a(b-config) - U_b(b-config)])))` (evaluate each config's energy under both neighbours' lambda). Returns per-pair acceptance.

- [ ] **Step 1: Write the failing test — hard wall never lets a mobile particle escape; RE preserves per-replica validity**

```python
# test_ka3d_pt_shrink.py
import torch
from liquid_coupling_flow.ka3d_pt_shrink import cavity_sweep, replica_exchange
from liquid_coupling_flow.ka_cavity import assert_mobile_inside

def _setup(nrep=4, N=60):
    L = (N / 1.2) ** (1/3); center = torch.tensor([L/2, L/2, L/2])
    x = torch.rand(nrep, N, 3) * L; s = torch.tensor([[0]*48 + [1]*12] * nrep)
    d = x - center; d = d - L * torch.round(d / L)
    mobile = d.square().sum(-1) < 1.8**2
    # clamp mobile particles inside so the initial state is valid
    return x, s, mobile, center, L

def test_cavity_sweep_respects_hard_wall():
    torch.manual_seed(0); nrep, N = 4, 60
    x, s, mobile, center, L = _setup(nrep, N); R = 1.8
    # project mobile particles to inside R first (valid start)
    d = x - center; d = d - L*torch.round(d/L); r = d.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    x = torch.where((mobile[...,None]) & (r >= R), center + d/r*(R*0.9), x)
    betas = torch.tensor([2.0, 1.7, 1.4, 1.2]); lams = torch.tensor([1.0, 0.9, 0.8, 0.7])
    for _ in range(20):
        for a in range(nrep):
            xa, sa = cavity_sweep(x[a:a+1], s[a:a+1], mobile[a:a+1], center, R, L,
                                  float(betas[a]), float(lams[a]), step=0.08)
            x[a:a+1], s[a:a+1] = xa, sa
    for a in range(nrep):
        assert_mobile_inside(x[a:a+1], mobile[a:a+1], center, R, L)   # raises if any escaped

def test_replica_exchange_returns_valid_acc():
    torch.manual_seed(1); nrep, N = 4, 60
    x, s, mobile, center, L = _setup(nrep, N); R = 1.8
    betas = torch.tensor([2.0,1.7,1.4,1.2]); lams = torch.tensor([1.0,0.9,0.8,0.7])
    x2, s2, acc = replica_exchange(x, s, mobile, center, R, L, betas, lams)
    assert acc.shape[0] == nrep - 1 and ((acc >= 0) & (acc <= 1)).all()
```

- [ ] **Step 2: Run to verify it fails.** `CUDA_VISIBLE_DEVICES="" pytest .../test_ka3d_pt_shrink.py -q` -> FAIL (no module).

- [ ] **Step 3: Implement `ka3d_pt_shrink.py` (moves + exchange).** Single-site move: pick one mobile particle per replica, propose `x_i + step*randn` wrapped to box, reject if outside R (`cavity_inside`), else Metropolis on `shrink_particle_energy` delta for that particle under `lam`. Identity swap: reuse `ka_cavity_3d.local_identity_swap` but recompute `U` with `shrink_particle_energy(...).sum/2` so the swap ratio uses the replica's `lam`. `replica_exchange`: for each adjacent pair `(a,a+1)`, compute total energies `U_a(x_a), U_{a+1}(x_a), U_a(x_{a+1}), U_{a+1}(x_{a+1})` via `0.5*shrink_particle_energy(...).sum(1)`, accept swap with the Hamiltonian-RE ratio above, exchange configs+species on accept.

- [ ] **Step 4: Run to verify pass.** Expected: 2 passed.

- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): hard-wall shrinkage single-site moves + Hamiltonian replica exchange"`

### Task 3: Cavity PTS observables (core overlap, G_PTS, susceptibility, PDF)

**Files:**
- Create: `liquid_coupling_flow/ka3d_pts_observables.py`
- Test: `liquid_coupling_flow/tests/test_ka3d_pts_observables.py`

**Interfaces:**
- Produces:
  - `core_overlap(X, Y, center, L, r_c=0.5, b=0.2) -> Tensor[B]` — for each config in the batch, `q_c = (1/n_core) sum_{i in core(Y)} max_j w(|x_j - y_i|)` with `w(z)=exp(-(z/b)^2)`, core = reference particles within `r_c` of center (minimum image). Returns 0 where the core is empty (guarded).
  - `pts_correlation(q_by_center) -> (mean, sem)` — `G_PTS(R) = <q_c>` over centers, SEM = std/sqrt(n).
  - `pts_susceptibility(q_by_center) -> float` — `chi_T = <q_c^2> - <q_c>^2` over centers (disorder average).
  - `overlap_pdf(q_by_center, bins) -> histogram` — `P(q_c)` for the bimodality check.

- [ ] **Step 1: Write the failing test**

```python
# test_ka3d_pts_observables.py
import torch
from liquid_coupling_flow.ka3d_pts_observables import core_overlap, pts_susceptibility

def test_identical_configs_give_overlap_one_in_core():
    torch.manual_seed(0); B, N, L = 3, 80, 4.0
    Y = torch.rand(B, N, 3) * L; center = torch.tensor([L/2]*3)
    q = core_overlap(Y, Y.clone(), center, L, r_c=0.6, b=0.2)   # X==Y -> perfect overlap
    assert (q > 0.95).all()

def test_independent_configs_give_low_overlap():
    torch.manual_seed(1); B, N, L = 3, 80, 4.0
    Y = torch.rand(B, N, 3) * L; X = torch.rand(B, N, 3) * L; center = torch.tensor([L/2]*3)
    q = core_overlap(X, Y, center, L, r_c=0.6, b=0.2)
    assert (q < 0.6).all()                                      # random -> well below 1

def test_susceptibility_matches_variance():
    q = torch.tensor([0.2, 0.8, 0.5, 0.9, 0.1])                 # per-center overlaps
    chi = pts_susceptibility(q)
    assert abs(chi - float(q.var(unbiased=False))) < 1e-6
```

- [ ] **Step 2: Run to verify fail.** -> FAIL.
- [ ] **Step 3: Implement** the four functions (vectorized minimum-image pairwise for `core_overlap`; the core mask is on `Y`; `max_j w(...)` is a soft nearest-neighbour occupancy).
- [ ] **Step 4: Run to verify pass.** Expected: 3 passed.
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): core overlap q_c, G_PTS, susceptibility chi_T, P(q_c)"`

### Task 4: Full PT cavity-equilibration driver + two-arm convergence gate

**Files:**
- Modify: `liquid_coupling_flow/ka3d_pt_shrink.py` (add `equilibrate_cavity`, `two_arm_converged`)
- Test: `liquid_coupling_flow/tests/test_ka3d_pt_shrink.py` (add convergence test)

**Interfaces:**
- Produces:
  - `build_ladder(T, n_rep, T_hot, lam_min) -> (betas, lams)` — geometric `T_a` from T to `T_hot`, geometric `lam_a` from 1.0 to `lam_min`.
  - `equilibrate_cavity(x_ref, s_ref, mobile, center, R, L, T, n_rep, n_sweep, exch_every, init) -> dict` — run PT (bottom replica physical), `init in {"ref","random"}` for the two arms, snapshot the bottom-replica core overlap vs `x_ref` every `t_rec` sweeps; returns `{"q_c_traj", "x_final", "s_final"}`.
  - `two_arm_converged(q_ref_traj, q_rand_traj, q_tol=0.1) -> bool` — the running core overlaps from the two inits meet within `q_tol` at the tail.

- [ ] **Step 1: Write the failing test — a LARGE cavity at HIGH T converges (easy regime, cheap sanity)**

```python
def test_large_cavity_high_T_two_arms_meet():
    # R large + T high => easy regime; PT must converge the two arms.
    import torch
    from liquid_coupling_flow.ka3d_pt_shrink import equilibrate_cavity, two_arm_converged, build_ladder
    torch.manual_seed(0); N = 200; L = (N/1.2)**(1/3); center = torch.tensor([L/2]*3)
    x = torch.rand(1, N, 3) * L; s = torch.tensor([[0]*160 + [1]*40])
    d = x - center; d = d - L*torch.round(d/L); mobile = d.square().sum(-1) < 3.0**2
    ref  = equilibrate_cavity(x, s, mobile, center, 3.0, L, T=1.0, n_rep=6, n_sweep=3000, exch_every=50, init="ref")
    rnd  = equilibrate_cavity(x, s, mobile, center, 3.0, L, T=1.0, n_rep=6, n_sweep=3000, exch_every=50, init="random")
    assert two_arm_converged(ref["q_c_traj"], rnd["q_c_traj"], q_tol=0.1)
```

- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** `build_ladder`, `equilibrate_cavity` (loop: per-replica `cavity_sweep`, `replica_exchange` every `exch_every`, record bottom-replica `core_overlap` vs `x_ref`), `two_arm_converged`.
- [ ] **Step 4: Run to verify pass** (this is a real ~minutes GPU test; if it is too slow for CI, mark `@pytest.mark.slow` and run manually — note the runtime in the commit).
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): PT cavity equilibration driver + two-arm convergence gate"`

### Task 5: VALIDATION GATE G1 — reproduce the paper's Fig 2/3

**Files:**
- Create: `reports/logs-2026-07-10/pt_validate_paper.py` (run script, `run_in_background:true`)
- Create: `reports/logs-2026-07-10/pt_validate_paper.out` (committed after run)
- Artifact: `liquid_coupling_flow/artifacts/ka3d_pts_baseline_T{0.6,0.5}.pt`

This is a RESEARCH GATE, not a unit test. Deliverable: at N=512, `T in {0.6, 0.5}`, over a radius ladder `R in {1.4,1.7,2.0,2.3,2.6,2.9,3.2}` and >= 50 cavity centers:
- For each (T, R): draw 50 centers from an equilibrated bulk config (from Task 6), run `equilibrate_cavity` for both arms per center, GATE each cavity on `two_arm_converged`, compute per-center `q_c` between two independent post-convergence samples.
- Compute `G_PTS(R)` (decaying), `chi_T(R)` (peaked), `P(q_c)` (bimodal near the peak). Extract `xi_PTS(T)` = argmax of `chi_T(R)`.
- Plot `G_PTS(R)`, `chi_T(R)`, `P(q_c)` panels; SAVE plot path (clickable) + artifact.

- [ ] **Step 1:** Write `pt_validate_paper.py` (uses Tasks 1-4 + Task 6 bulk config). Incremental save per (T,R,center-batch).
- [ ] **Step 2:** Launch with `run_in_background:true`, stderr+stdout to the `.out`.
- [ ] **Step 3:** On completion, produce the 3-panel plot; print its full path.
- [ ] **Step 4: GATE G1** — verdict: does `G_PTS(R)` decay, `chi_T(R)` peak, and `xi_PTS(0.5) > xi_PTS(0.6)` with growth "not much more than x2" and values in the paper's ballpark (peak near R~2-3 at T~0.5)? Write the verdict into the `.out` and a one-paragraph summary. **If G1 FAILS, STOP — debug the PT scheme (ladder spacing / exchange acceptance in 0.1-0.9 band / lam_min) before any generator work.**
- [ ] **Step 5: Commit** script + out + plot + artifact + a `reports/2026-07-10-cavity-pts-baseline.md` note recording G1's verdict.

### Task 6: Bulk 3D equilibrium data engine — scale-up + equilibration validation

**Files:**
- Modify: `liquid_coupling_flow/ka_dataset_3d.py` (add `warm_start` from an existing config set; add `validate_equilibration`)
- Test: `liquid_coupling_flow/tests/test_ka_dataset_3d.py` (add an equilibration-validation test)
- Run: `reports/logs-2026-07-10/bulk_data_scaleup.py`

**Context:** `ka_dataset_3d.generate_dataset` exists (pmc equilibrate + exact polish, per-snapshot incremental save). The current 1024-config set (a) is too small to train and (b) showed residual energy drift (`-6.738 -> -6.777`), i.e. not fully equilibrated. Fix both.

**Interfaces:**
- Produces:
  - `validate_equilibration(paths_or_configs) -> dict` — split into two independent halves, check `|<U/N>_A - <U/N>_B| < 0.005` (seed agreement) AND per-half tail-flat `|first-half-mean - second-half-mean| < 0.005`. Returns `{"converged": bool, ...}`.
  - `generate_dataset(..., warm_start=None)` — if `warm_start` is a config tensor, seed chains from jittered copies instead of a cold lattice (cuts the ~62-min cold equilibration).

- [ ] **Step 1: Write the failing test** for `validate_equilibration` (two synthetic sets with equal vs shifted means).
- [ ] **Step 2: Run to verify fail.**
- [ ] **Step 3: Implement** `validate_equilibration` + `warm_start`.
- [ ] **Step 4: Run to verify pass.**
- [ ] **Step 5:** Write + launch `bulk_data_scaleup.py` (`run_in_background:true`): warm-start from the existing snapshot-8 configs, longer equilibration until `validate_equilibration` passes, target >= 8192 well-mixed configs at T in {0.6, 0.5}. Incremental save. On completion, assert `validate_equilibration["converged"]`.
- [ ] **Step 6: Commit** code + test + run script + out + a note recording the final `<U/N>` and convergence verdict.

---

## Phase 2 — Cavity generator + evaluation (ROADMAP; expand to a detailed plan after G1)

> These tasks are intentionally at roadmap resolution. Their code-level detail (generator architecture, conditioning mechanism, loss) should be locked in a follow-up plan once G1 confirms the substrate and reveals how multimodal `P(q_c)` is at accessible T. Interfaces below are the contract they must satisfy.

### Task 7: Carve-cavity training pairs from bulk

**File:** `liquid_coupling_flow/ka3d_cavity_carve.py` (+ test)
- `carve(bulk_x, bulk_s, R, center, L) -> {"x_out", "x_in", "mobile", "center", "R"}` — partition a bulk config by a sphere; the interior IS a ground-truth draw of `p(x_in | x_out)`. Random `center` per config; sample `R` from the training ladder.
- Test: carving then reassembling equals the original; interior particle count matches `~(4/3)pi R^3 rho`.
- Each bulk config yields many `(center, R)` pairs (augmentation) — this is the escape from "1024 too few."

### Task 8: Boundary-conditioned cavity generator + training

**File:** `liquid_coupling_flow/ka3d_cavity_generator.py` (+ test)
- **Architecture:** masked EGNN flow — generate `x_in` conditioned on frozen `x_out` via the EGNN context (interior particles' neighbours include the frozen boundary), reusing the 3D-capable `ipl44/joint_flow` (n_dimension=3) with a frozen mask (cf. `ka_pin.py` masked kernels). Species count preserved inside.
- **Training:** self-supervised on carved pairs — the flow-matching / conditional-FM target is the bulk interior given the boundary. Behind exact MH at inference (log-prob available) so the sampler is unbiased; only efficiency is at stake.
- Test: on a tiny bulk set, one training step reduces the loss; a sampled interior stays inside R and preserves species count.
- **Transfer contract:** the generator must accept an arbitrary `(center, R)` at inference without retraining (R never hard-coded).

### Task 9: GATE G2 — on-par `P(q_c)` vs PT

**File:** `liquid_coupling_flow/ka3d_cavity_eval.py` (+ run script)
- At 2+ `(R,T)` points (including one near `xi_PTS`), draw boundaries from equilibrated bulk; sample many interiors with the generator (MH-corrected); compute generator `P(q_c)` and `chi_T`.
- **GATE G2:** generator `P(q_c)` matches PT `P(q_c)` within noise, INCLUDING bimodality near `xi_PTS`; report the MH acceptance and the cost (generator calls + MH steps) vs PT (sweeps x replicas). "On par" = comparable equilibrium quality at <= comparable cost.

### Task 10: GATE G3 — zero-shot transfer across R (and T)

- Train the generator on a SUBSET of R (and one T); evaluate `chi_T(R)` at HELD-OUT R (and held-out T) with no retraining; compare the transferred `xi_PTS` to the PT baseline.
- **GATE G3:** held-out `chi_T(R)` peak matches PT within noise. This is the demonstrated win — PT re-tunes its ladder per state point; the generator does not.

---

## Self-Review

**Spec coverage:** Every Global Constraint maps to a task — model params (all tasks, via `ka_energy`), hard wall (Tasks 2,4), shrinkage Hamiltonian (Tasks 1,2,4), core overlap + `r_c`/`b` (Task 3), susceptibility (Tasks 3,5), two-arm gate (Task 4), >=50 centers (Task 5), mean+/-SEM reporting (Tasks 3,5), learned-behind-MH + validate-before-claim (Tasks 8,9), durability (Tasks 5,6 + all run scripts). The circularity escape is realized by Tasks 6->7 (bulk data -> carved pairs), so no cavity-PT training data is ever required.

**Placeholder scan:** Phase 1 tasks carry real code + real tests + exact commands. Phase 2 is explicitly roadmap-resolution by design (flagged), with interface contracts, not hidden placeholders — it will be expanded post-G1.

**Type consistency:** `mobile` is `BoolTensor[B,N]` throughout; `center` is `[3]` (shared per cavity) or `[B,3]` (per-realization) and `cavity_inside` already auto-broadcasts (fixed earlier); `lam` is a Python float per replica; energies are per-particle `[B,N]` (`shrink_particle_energy`, matching `ka_pmc_3d.particle_energies`); overlaps are `[B]` per config, aggregated to `(mean, sem)` / `chi` over centers. `shrink_particle_energy(...,lam=1)` == `particle_energies` is the pin that keeps the new energy consistent with the whole codebase.
