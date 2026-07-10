# Cavity-Conditional Generator for Point-to-Set Correlations — Implementation Plan (v2, train-first)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a boundary-conditioned neural generator that samples equilibrium cavity interiors of the 3D Kob-Andersen glass and measures point-to-set (PTS) overlap statistics *on par with* the Berthier-Charbonneau-Yaida (JCP 144, 024501, 2016) parallel-tempering + size-shrinkage scheme, while being *transferable* (train once, apply across cavity radius R and temperature T).

**Architecture:** The cavity interior at equilibrium is *exactly* the bulk Boltzmann distribution conditioned on the frozen boundary, `p(x_in | x_out)`. So we (1) carve random cavities out of cheap **bulk** equilibrium configs to get free, self-supervised `(boundary -> interior)` training pairs — no cavity-PT training data is ever needed; (2) train a masked/boundary-conditioned generator to sample `p(x_in | x_out)` directly (one-shot, bypassing the free-energy barriers that force PT to crawl via replica exchange); (3) validate its overlap distribution `P(q_c)` against a **minimal** PT+shrinkage baseline that serves only as a yardstick. **This plan is TRAIN-FIRST:** we build just enough PT machinery to sanity-check the physics on 4-8 boundaries, then move straight to carving + training, and only expand the PT baseline (full susceptibility, 50 centers, larger R) *after* the generator demonstrates meaningful conditioning.

**Tech Stack:** PyTorch (GPU), existing `liquid_coupling_flow` primitives (`ka_energy` dimension-generic + `ka_pair_row_scatter`, `ka_cavity` hard wall, `ka_pmc_3d`, `ka_dataset_3d` bulk gen, `ipl44/joint_flow` 3D-capable EGNN flow), pytest.

## Global Constraints

- **Model: 3D 80:20 Kob-Andersen binary LJ (KABLJ).** Species A,B, `N_A:N_B = 4:1` (composition_B = 0.2). `sigma_AA=1.0, sigma_AB=0.8, sigma_BB=0.88`; `eps_AA=1.0, eps_AB=1.5, eps_BB=0.5`. **Per-pair cutoff `r_cut_ab = 2.5*sigma_ab`, shifted so V vanishes at cutoff** (NOT a universal `2.5*sigma_AB`). Density `rho = 1.2`, box `L = (N/rho)^(1/3)`. LJ reduced units. `T_MCT ~= 0.435`. Matches `ka_energy.py` (RCUT_FACTOR=2.5 applied per-pair) and `ka_cavity_3d.py` (RHO=1.2, X_B=0.2).
- **Geometry validity (HARD assert in every cavity setup):** a cavity of radius R at N (box L) is valid only if `2*R + RCUT_FACTOR*sigma_max <= L`, with `sigma_max = 1.0` (=> `2R + 2.5 <= L`), so its periodic image does not interact. At N=512 (L=7.53) this caps R <= 2.5 (use R <= 2.4 with margin). For R up to ~3.2 use **N=864** (L=8.83). Never place a cavity that fails this assert.
- **Hard spherical cavity wall:** freeze every particle outside radius R of the center; reject any move to `|x-center| >= R` (minimum image). Use `ka_cavity.py::cavity_inside` / `assert_mobile_inside` (correct + tested).
- **Shrinkage Hamiltonian (paper Eq. 2):** replica `a` scales `sigma -> lambda_tilde_a * sigma`: `lambda_tilde = lambda_a` for a mobile-mobile pair, `(1+lambda_a)/2` for a mobile-pinned pair, `1.0` for pinned-pinned. Bottom replica physical: `T_1 = T, lambda_1 = 1`. Higher replicas `T_a > T`, `lambda_a <= 1`. The cutoff scales with the shrunk sigma.
- **Replica exchange = adjacent-REPLICA config exchange in (T_a, lambda_a) space (paper's "identity exchange").** Accept with `log A = -beta_a*[H_a(x_b) - H_a(x_a)] - beta_b*[H_b(x_a) - H_b(x_b)]`. This is the paper's PT method. **A/B particle swaps are NOT part of the reproduction** (the paper contrasts its method with local particle-swap MC, and KA is a poor swap model); if added they make an *enhanced baseline*, labeled as such.
- **Single-site moves use an O(N) shrinkage pair ROW**, never a full `[B,N]` energy recompute per particle (that is O(N^3)/sweep). Pattern: `ka_energy.ka_pair_row_scatter`, extended for shrinkage.
- **Overlap = a PROXY, explicitly not exact Eq. 5-8.** Symmetric same-species nearest-neighbour core average with kernel `w(z)=exp(-(z/b)^2)`, `b=0.2`, core `|r-center| < r_c=0.5`, symmetrized `X<->Y`. Adequate for generator-vs-baseline validation; do NOT call it a Fig-2 reproduction.
- **Susceptibility (paper Eq. 10): `chi_T(R) = [ <q_c^2>_J - <q_c>_J^2 ]`** = disorder average (over boundaries J) of the per-boundary THERMAL variance. Input is `q[n_centers, n_pairs]`: per-center variance first, THEN mean over centers. (Only used in the EXPAND phase, not the pilot.)
- **Two-arm convergence gate (Cavagna/Berthier):** accept a cavity only when the running core overlap from (i) the original and (ii) a randomized init meet within `q_tol` at the tail. Never report from an unconverged cavity.
- **Report mean +/- SEM** (n = number of centers / pairs); never bootstrap `|diff|` (positive-biased). All learned components are validated against PT ground truth before any "on par" claim; the pilot generator gate compares `P(q_c)` DIRECTLY (no MH correction claimed yet).
- **Durability:** raw logs + one-off scripts to `reports/logs-<date>/`, committed when a run finishes; artifacts to `liquid_coupling_flow/artifacts/`; long runs via Bash `run_in_background: true` with per-unit incremental saves.

---

## Milestones and Go/No-Go Gates (TRAIN-FIRST ORDER)

| Phase | Gate | Content | Pass condition |
|-------|------|---------|----------------|
| G0 | Machinery | shrinkage pair-row, correct replica exchange, hard wall, proxy overlap + correct chi_T | unit tests green: lambda=1 identity, local energy delta, **explicit replica-exchange weight ratio**, overlap limits, chi_T definition |
| G1 | Tiny PT sanity | N=512, T=0.5, R={1.6,2.2,2.4}, 4-8 boundaries + one T=1.0 smoke | two-arm convergence + high->lower overlap trend with R. NO chi_T, NO Fig-2, NO 50 centers |
| G2 | Generator | carve existing bulk -> train positions-only conditional prototype (overfit 32-64 boundaries) | generator thermal `P(q_c)` matches PT at R=1.6 AND R=2.4 within noise; evidence of real boundary conditioning |
| G3 | Expand (ROADMAP) | full chi_T campaign, 50 centers, N=864 larger R, T-transfer, MH-exact generator | only entered if G2 passes |

**The pilot reuses the EXISTING bulk set** (`artifacts/ka3d_dataset_N512_T0.5.pt`, 1024 configs) for both PT boundaries and carving — too few to train a final model, but fine to overfit 32-64 boundaries as a prototype. A larger, independent-seed, certified bulk set is a G3 task, not a blocker.

---

## File Structure

- `liquid_coupling_flow/ka3d_shrink.py` — shrinkage pair energy `V(lambda_tilde)`, O(N) shrinkage pair-row, hard-wall single-site cavity move, adjacent-replica exchange. (G0: Tasks 1-2)
- `liquid_coupling_flow/ka3d_pts_observables.py` — proxy core overlap `q_c(X,Y)`, `chi_T`, `P(q_c)`. (G0: Task 3)
- `liquid_coupling_flow/ka3d_pt_cavity.py` — replica ladder, PT cavity-equilibration driver, two-arm gate. (G1: Task 4)
- `liquid_coupling_flow/ka3d_cavity_carve.py` — carve bulk -> `(boundary, interior, mask)` pairs. (G2: Task 6)
- `liquid_coupling_flow/ka3d_cavity_generator.py` — positions-only boundary-conditioned generator + training. (G2: Task 7)
- `liquid_coupling_flow/ka3d_cavity_eval.py` — generator `P(q_c)` vs PT. (G2: Task 8)
- Tests in `liquid_coupling_flow/tests/`; run scripts + logs in `reports/logs-2026-07-10/`.

---

## G0 — Machinery (unit-tested; no physics campaign)

### Task 1: Shrinkage pair energy + O(N) pair-row

**Files:** Create `liquid_coupling_flow/ka3d_shrink.py`; Test `liquid_coupling_flow/tests/test_ka3d_shrink.py`

**Interfaces — Produces:**
- `lambda_tilde(lam, mobile_i, mobile_j) -> Tensor[B,N,N]` — `lam` both-mobile, `(1+lam)/2` one-mobile, `1.0` both-pinned.
- `shrink_particle_energy(x, s, L, lam, mobile) -> Tensor[B,N]` — per-particle shifted-LJ under shrinkage; at `lam=1` equals `ka_pmc_3d.particle_energies`.
- `shrink_pair_row(x, s, idx, xi, L, lam, mobile) -> Tensor[B]` — O(N) energy of the single particle `idx=(rows,i)` placed at `xi`, against all others, under shrinkage. `delta = shrink_pair_row(new) - shrink_pair_row(old)` is the single-move energy change.

- [ ] **Step 1: Failing tests**

```python
# test_ka3d_shrink.py
import torch
from liquid_coupling_flow.ka3d_shrink import lambda_tilde, shrink_particle_energy, shrink_pair_row
from liquid_coupling_flow.ka_pmc_3d import particle_energies
from liquid_coupling_flow.ka_energy import ka_energy

def test_lambda_one_matches_standard():
    torch.manual_seed(0); B,N,L=2,30,3.4
    x=torch.rand(B,N,3)*L; s=torch.tensor([[0]*24+[1]*6]*B); mob=torch.ones(B,N,dtype=torch.bool)
    assert torch.allclose(shrink_particle_energy(x,s,L,1.0,mob), particle_energies(x,s,L), atol=1e-5)

def test_lambda_tilde_rules():
    mob=torch.tensor([[True,True,False]]); lt=lambda_tilde(0.6,mob,mob)
    assert abs(lt[0,0,1]-0.6)<1e-6 and abs(lt[0,0,2]-0.8)<1e-6 and abs(lt[0,2,2]-1.0)<1e-6

def test_pair_row_delta_matches_full_energy_delta_at_lambda1():
    torch.manual_seed(1); B,N,L=3,24,3.2
    x=torch.rand(B,N,3)*L; s=torch.tensor([[0]*19+[1]*5]*B); mob=torch.ones(B,N,dtype=torch.bool)
    rows=torch.arange(B); i=torch.tensor([2,5,7]); xi=torch.remainder(x[rows,i]+0.1,L)
    delta_row = shrink_pair_row(x,s,(rows,i),xi,L,1.0,mob) - shrink_pair_row(x,s,(rows,i),x[rows,i],L,1.0,mob)
    xnew=x.clone(); xnew[rows,i]=xi
    delta_full = ka_energy(xnew,s,L) - ka_energy(x,s,L)
    assert torch.allclose(delta_row, delta_full, atol=1e-4)
```

- [ ] **Step 2: Run -> FAIL.** `CUDA_VISIBLE_DEVICES="" pytest liquid_coupling_flow/tests/test_ka3d_shrink.py -q`
- [ ] **Step 3: Implement** `lambda_tilde`, `shrink_particle_energy` (dense, for the exchange energies + tests), `shrink_pair_row` (O(N): only particle `i`'s row of pair distances, `lambda_tilde` row = `lam` for mobile partners, `(1+lam)/2` for pinned; cutoff scales with shrunk sigma).
- [ ] **Step 4: Run -> PASS** (3 passed).
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): shrinkage energy + O(N) shrinkage pair-row (lambda=1 == ka_energy)"`

### Task 2: Hard-wall single-site cavity move + correct adjacent-replica exchange

**Files:** Modify `liquid_coupling_flow/ka3d_shrink.py`; Test add to `test_ka3d_shrink.py`

**Interfaces — Produces:**
- `cavity_move(x, s, U, mobile, center, R, L, beta, lam, step) -> (x, U, acc)` — pick one mobile particle per row, propose wrapped `x_i+step*randn`, reject if outside R (`cavity_inside`), else Metropolis on `shrink_pair_row` delta under `lam`; O(N)/move.
- `replica_exchange(x, s, mobile, center, R, L, betas, lams) -> (x, s, log_acc)` — for each adjacent pair `(a,a+1)` compute `H` via `0.5*shrink_particle_energy(...).sum(1)` under each replica's `lam`, accept with `exp(-beta_a[H_a(x_{a+1})-H_a(x_a)] - beta_{a+1}[H_{a+1}(x_a)-H_{a+1}(x_{a+1})])`, exchange configs+species on accept.

- [ ] **Step 1: Failing tests — hard wall + EXPLICIT exchange ratio**

```python
def test_cavity_move_never_escapes():
    import torch
    from liquid_coupling_flow.ka3d_shrink import cavity_move, shrink_particle_energy
    from liquid_coupling_flow.ka_cavity import assert_mobile_inside
    torch.manual_seed(0); N=80; L=(N/1.2)**(1/3); center=torch.tensor([L/2]*3); R=1.8
    x=torch.rand(1,N,3)*L; s=torch.tensor([[0]*64+[1]*16])
    d=x-center; d=d-L*torch.round(d/L); mob=d.square().sum(-1)<R*R
    r=(x-center).norm(dim=-1,keepdim=True).clamp_min(1e-6)
    x=torch.where(mob[...,None]&(r>=R), center+(x-center)/r*(R*0.9), x)   # valid start
    U=0.5*shrink_particle_energy(x,s,L,1.0,mob).sum(1)
    for _ in range(200): x,U,_=cavity_move(x,s,U,mob,center,R,L,2.0,1.0,0.08)
    assert_mobile_inside(x,mob,center,R,L)

def test_replica_exchange_weight_ratio_explicit():
    # Two replicas, two fixed configs: the acceptance probability must equal the
    # explicit ratio of joint Boltzmann weights before vs after swapping configs.
    import torch
    from liquid_coupling_flow.ka3d_shrink import replica_exchange, shrink_particle_energy
    torch.manual_seed(3); N=40; L=(N/1.2)**(1/3); center=torch.tensor([L/2]*3); R=1.6
    x=torch.rand(2,N,3)*L; s=torch.tensor([[0]*32+[1]*8]*2); mob=torch.ones(2,N,dtype=torch.bool)
    betas=torch.tensor([2.0,1.3]); lams=torch.tensor([1.0,0.8])
    def H(cfg_x, lam): return float(0.5*shrink_particle_energy(cfg_x[None],s[:1],L,float(lam),mob[:1]).sum(1))
    Ha_xa=H(x[0],1.0); Hb_xb=H(x[1],0.8); Ha_xb=H(x[1],1.0); Hb_xa=H(x[0],0.8)
    import math
    logA=-(2.0*(Ha_xb-Ha_xa)) - (1.3*(Hb_xa-Hb_xb))
    expected=min(1.0, math.exp(logA))
    # Monte-Carlo estimate the code's acceptance frequency for THIS fixed pair:
    acc_count=0; trials=4000
    for _ in range(trials):
        _,_,la=replica_exchange(x.clone(), s.clone(), mob, center, R, L, betas, lams)
        acc_count += float(min(1.0, math.exp(float(la[0]))) )  # code returns log_acc; compare its ratio
    # code must return the SAME logA (deterministic given configs), so exp(min(...)) matches:
    assert abs(min(1.0, math.exp(float(la[0]))) - expected) < 1e-4
```

- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** `cavity_move` (O(N) pair-row) and `replica_exchange` (the exact ratio above; return `log_acc` per adjacent pair so the test can check the deterministic weight ratio). NOTE: `replica_exchange` computes `log_acc` deterministically from the configs; the accept coin is separate.
- [ ] **Step 4: Run -> PASS.**
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): hard-wall O(N) cavity move + adjacent-replica exchange (weight-ratio tested)"`

### Task 3: Proxy overlap + correct susceptibility

**Files:** Create `liquid_coupling_flow/ka3d_pts_observables.py`; Test `test_ka3d_pts_observables.py`

**Interfaces — Produces:**
- `core_overlap(X, sX, Y, sY, center, L, r_c=0.5, b=0.2) -> Tensor[B]` — symmetric same-species nearest-neighbour core average: for reference `Y` particles within `r_c` of center, `w(dist to nearest SAME-species X particle)`; symmetrize with the `X<->Y` swap; average. Guarded for empty core.
- `pts_susceptibility(q_by_center_pairs) -> float` — input `[n_centers, n_pairs]`: per-center variance over pairs (thermal), then mean over centers (disorder). = paper Eq. 10.
- `overlap_pdf(q_flat, bins) -> Tensor` — `P(q_c)` histogram for the bimodality check.

- [ ] **Step 1: Failing tests**

```python
# test_ka3d_pts_observables.py
import torch
from liquid_coupling_flow.ka3d_pts_observables import core_overlap, pts_susceptibility

def test_identical_configs_overlap_near_one():
    torch.manual_seed(0); B,N,L=3,80,4.0; Y=torch.rand(B,N,3)*L; s=torch.tensor([[0]*64+[1]*16]*B)
    assert (core_overlap(Y,s,Y.clone(),s,torch.tensor([L/2]*3),L,0.7,0.2) > 0.9).all()

def test_independent_configs_low_overlap():
    torch.manual_seed(1); B,N,L=3,80,4.0
    X=torch.rand(B,N,3)*L; Y=torch.rand(B,N,3)*L; s=torch.tensor([[0]*64+[1]*16]*B)
    assert (core_overlap(X,s,Y,s,torch.tensor([L/2]*3),L,0.7,0.2) < 0.6).all()

def test_susceptibility_is_disorder_mean_of_thermal_variance():
    q = torch.tensor([[0.2,0.8,0.5],[0.9,0.85,0.95]])          # [2 centers, 3 pairs]
    expected = float(q.var(dim=1, unbiased=False).mean())       # per-center var, then mean
    assert abs(pts_susceptibility(q) - expected) < 1e-6
```

- [ ] **Step 2-4: Run fail -> implement -> pass.**
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): proxy core overlap + Eq.10 susceptibility (per-boundary thermal variance)"`

---

## G1 — Tiny PT sanity (minimal, empirical)

### Task 4: PT cavity-equilibration driver + two-arm gate

**Files:** Create `liquid_coupling_flow/ka3d_pt_cavity.py`; Test `test_ka3d_pt_cavity.py`

**Interfaces — Produces:**
- `assert_cavity_valid(R, L, sigma_max=1.0)` — raise unless `2*R + 2.5*sigma_max <= L`.
- `build_ladder(T, n_rep, T_hot=1.5, lam_min=0.7) -> (betas, lams)` — geometric T from T to T_hot; geometric lambda 1.0 -> lam_min.
- `equilibrate_cavity(x_ref, s_ref, mobile, center, R, L, T, n_rep, n_sweep, exch_every, init) -> dict` — PT (bottom replica physical), `init in {"ref","random"}`, record bottom-replica core overlap vs `x_ref` every `t_rec`; returns `{"q_c_traj", "x_final", "s_final"}`. Calls `assert_cavity_valid`.
- `two_arm_converged(q_ref_traj, q_rand_traj, q_tol=0.1) -> bool`.

- [ ] **Step 1: Failing test — geometry guard + a VALID, EQUILIBRATED-fixture convergence smoke**

```python
# test_ka3d_pt_cavity.py
import pytest, torch
from liquid_coupling_flow.ka3d_pt_cavity import assert_cavity_valid, equilibrate_cavity, two_arm_converged

def test_geometry_guard_rejects_oversized_cavity():
    L=(512/1.2)**(1/3)                    # 7.53
    assert_cavity_valid(2.4, L)           # ok (2*2.4+2.5=7.3<=7.53)
    with pytest.raises(AssertionError):
        assert_cavity_valid(3.2, L)       # 8.9 > 7.53

@pytest.mark.slow
def test_valid_small_cavity_two_arms_meet_from_equilibrated_ref():
    # Uses an EQUILIBRATED bulk config (not random) and a geometrically valid R.
    d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
    x = d["x"][-1:].float(); s = d["s"][-1:].long(); L=float(d["L"]); N=x.shape[1]
    center = x[0, 0].clone()              # center on a real particle
    dd = x-center; dd=dd-L*torch.round(dd/L); mob=dd.square().sum(-1)<1.6**2
    ref = equilibrate_cavity(x,s,mob,center,1.6,L,T=1.0,n_rep=6,n_sweep=4000,exch_every=50,init="ref")
    rnd = equilibrate_cavity(x,s,mob,center,1.6,L,T=1.0,n_rep=6,n_sweep=4000,exch_every=50,init="random")
    assert two_arm_converged(ref["q_c_traj"], rnd["q_c_traj"], q_tol=0.1)
```

- [ ] **Step 2: Run -> FAIL.**
- [ ] **Step 3: Implement** `assert_cavity_valid`, `build_ladder`, `equilibrate_cavity` (per-replica `cavity_move` sweeps + `replica_exchange` every `exch_every`, record bottom-replica `core_overlap` vs `x_ref`), `two_arm_converged`. Mark the GPU convergence test `@pytest.mark.slow`; note runtime in the commit.
- [ ] **Step 4: Run -> PASS** (geometry test always; slow test manually).
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-pt): PT cavity driver + two-arm gate + geometry guard"`

### Task 5: G1 RUN — tiny empirical PT sanity

**Files:** Create `reports/logs-2026-07-10/pt_gate_g1.py` (+ `.out`)

Deliverable (NOT a full campaign): at N=512, T=0.5, R in {1.6, 2.2, 2.4} (all pass `assert_cavity_valid`), draw 4-8 cavity centers from the equilibrated bulk config (`ka3d_dataset_N512_T0.5.pt` last snapshot). For each center run both arms; require `two_arm_converged`; record post-convergence core overlap. Add one easy T=1.0 smoke point at R=1.6.

- [ ] **Step 1:** Write `pt_gate_g1.py` (reuse Tasks 1-4; incremental save per center).
- [ ] **Step 2:** Launch `run_in_background:true`, streams to `.out`.
- [ ] **Step 3: GATE G1** — verdict written to `.out`: (a) do the two arms converge within `q_tol` for the tested cavities? (b) does mean overlap DECREASE from R=1.6 to R=2.4 (high->lower trend)? **If either fails, STOP and debug ladder spacing / exchange acceptance (target 0.1-0.9 per adjacent pair) / lam_min / n_sweep before training.** No chi_T, no Fig-2, no 50 centers here.
- [ ] **Step 4: Commit** script + out + a one-paragraph verdict in `reports/2026-07-10-cavity-pts.md`.

---

## G2 — Carve + train + generator gate (the thesis, front-loaded)

### Task 6: Carve cavities from bulk into training pairs

**Files:** Create `liquid_coupling_flow/ka3d_cavity_carve.py`; Test `test_ka3d_cavity_carve.py`

**Interfaces — Produces:**
- `carve(x, s, center, R, L) -> dict` — partition one bulk config: `{"x_in","s_in","x_out","s_out","mobile","center","R","n_in"}`, `mobile = |x-center|<R`. The interior is a ground-truth draw of `p(x_in|x_out)`. `assert_cavity_valid(R,L)` first.
- `carve_batch(bulk_x, bulk_s, L, radii, centers_per_config, rng) -> list[dict]` — many `(center in-box, R sampled from radii)` pairs per config (augmentation).

- [ ] **Step 1: Failing tests** — reassembly equals original; `n_in ~ (4/3)pi R^3 rho` within tolerance; every carve passes `assert_cavity_valid`.
- [ ] **Step 2-4:** fail -> implement -> pass.
- [ ] **Step 5: Commit.** `git commit -m "feat(pts-gen): carve bulk configs into (boundary,interior) training pairs"`

### Task 7: Positions-only boundary-conditioned generator prototype

**Files:** Create `liquid_coupling_flow/ka3d_cavity_generator.py`; Test `test_ka3d_cavity_generator.py`

**Design (explicit — the v1 plan's "reuse JointSpeciesFlow" was false):**
- **Positions-only prototype:** interior species counts held FIXED (composition inside a carved cavity is known from the bulk config); generate interior POSITIONS given the boundary. Species generation deferred.
- **Variable interior cardinality `N_cav`:** pad interiors to `N_max` with a validity mask; the loss and any set-aggregation respect the mask.
- **Boundary context:** condition on the frozen boundary particles within a shell `[R, R + r_ctx]` (r_ctx ~ r_cut) of the center — the only boundary particles that interact with the interior — as EGNN context nodes (not generated).
- **Explicit conditioning inputs:** `R` and `T` are scalar inputs to the network (concatenated into node/edge features), so a single model spans radii and temperatures — required for the G3 transfer claim; training at one T without a T-input cannot transfer in T.
- **Architecture:** EGNN over (interior padded nodes + boundary-shell context nodes), flow-matching / conditional-FM objective on interior positions. `log_prob` for exact MH is a G3 item, NOT claimed here.

**Interfaces — Produces:**
- `CavityGenerator(n_max, hidden_nf, n_layers)` with `forward(t, x_in_pad, mask, x_ctx, s_ctx, R, T) -> v_in` and `sample(x_ctx, s_ctx, n_in, R, T) -> x_in`.
- `cavity_fm_loss(model, pair, ...) -> loss` — conditional flow-matching on carved interiors.

- [ ] **Step 1: Failing tests** — forward returns `[B, n_max, 3]` finite, respects mask (padded slots ignored); one training step on a tiny carved set reduces the loss; `sample` returns interiors inside R (project/validate) with the requested `n_in`.
- [ ] **Step 2-4:** fail -> implement -> pass.
- [ ] **Step 5:** OVERFIT test/run: overfit 32-64 boundaries (single T=0.5, R in {1.6,2.4}); the model must reproduce their interiors closely (sanity that conditioning + masking work). Commit code + tests + overfit log.

### Task 8: GATE G2 — generator P(q_c) vs PT

**Files:** Create `liquid_coupling_flow/ka3d_cavity_eval.py` + `reports/logs-2026-07-10/gen_gate_g2.py`

- At R=1.6 AND R=2.4, T=0.5: draw a handful of boundaries from equilibrated bulk; from each boundary, (i) PT: generate many independent interior samples (Task 4) -> reference `P(q_c)`; (ii) generator: sample many interiors -> `P(q_c)`. Compute per-boundary thermal overlap distributions and compare.
- **GATE G2:** generator `P(q_c)` matches PT within noise at BOTH radii, and shows *boundary-dependent* structure (not a boundary-agnostic average). Report mean+/-SEM. **Pass => expand (G3). Fail => the conditional-learning thesis is refuted at pilot scale; report why (mode collapse? boundary ignored? equilibrium mismatch?) before scaling anything.**
- [ ] Steps: write eval + run (`run_in_background:true`) -> plots (save paths) -> verdict in `reports/2026-07-10-cavity-pts.md` -> commit.

---

## G3 — Expand (ROADMAP; only if G2 passes)

To be expanded into a detailed plan after G2. Scope: (1) certified independent-seed bulk data scale-up to >=8192 configs at T in {0.6,0.5}; (2) full `chi_T(R)` campaign with >=50 centers + multiple independent thermal pairs per center, N=864 for R up to ~3.2, reproducing the paper's `G_PTS`/`chi_T`/`P(q_c)` (THE Fig-2 gate, now with a validated generator to accelerate it); (3) species generation in the cavity; (4) exact `log_prob` + independence-MH for an unbiased generator; (5) transfer gates: zero-shot held-out R, then held-out T.

---

## Self-Review

**Blockers from review, all fixed:** (1) susceptibility = per-boundary thermal variance then disorder mean, input `[n_centers,n_pairs]` (Task 3, Global Constraints); (2) replica-exchange ratio corrected + explicit weight-ratio test (Task 2); (3) geometry `assert_cavity_valid` everywhere, N=512 capped at R<=2.4, N=864 for larger, equilibrated fixtures in tests (Tasks 4,5,6, Global Constraints); (4) generator design made explicit — positions-only, padded masks, boundary-shell context, R/T inputs, log_prob deferred (Task 7). **Scientific mismatches fixed:** per-pair cutoff (Global Constraints); replica-exchange != A/B swap, swaps are an optional enhancement (Global Constraints, Task 2); overlap labeled a proxy, symmetrized same-species NN (Task 3, Global Constraints); O(N) shrinkage pair-row (Task 1); bulk data reused (not generated after) and pilot avoids the trajectory-halves fallacy — a certified independent-seed set is deferred to G3. **Structure:** train-first — G0 machinery, G1 tiny PT (no chi_T/Fig-2/50-centers), G2 carve+train+gate, G3 expand only if G2 passes.

**Type consistency:** `mobile: BoolTensor[B,N]`; `center: [3]` or `[B,3]` (`cavity_inside` auto-broadcasts); `lam: float` per replica; energies per-particle `[B,N]` (`shrink_particle_energy`) / scalar per row `[B]` (`shrink_pair_row`); overlaps `[B]` per config, aggregated as `[n_centers,n_pairs] -> chi_T` (thermal-then-disorder). `shrink_particle_energy(lam=1) == particle_energies == ka_energy` is the invariant that keeps the new energy consistent.
