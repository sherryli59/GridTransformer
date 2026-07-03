# IPL44 Lever Push Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prove the joint species-position flow's denoiser is a useful learned swap proposer (MH-exact, ≥2× fewer sweeps-to-equilibrium than random swaps) on IPL44, and run a bounded professor-forcing bet on the AR transformer.

**Architecture:** Extend `liquid_coupling_flow/ipl44/joint_flow.py` (v1 API kept); new `ipl_swap_smc.py` = batched position-Metropolis + exact swap-MH kernel (learned proposer with paired re-evaluation for the reverse move) + sweeps-to-equilibrium benchmark; new `train_joint_flow.py` = scaling grid with warmup+EMA+best/last checkpointing and the G1 denoiser gate.

**Tech Stack:** PyTorch (conda env `lightning`), vendored learndiffeq EGNN (`liquid_coupling_flow/ipl44/learndiffeq/`), pytest.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-02-ipl44-lever-push-design.md`. Gates G1/G2/G0-PF/G-PF as defined there.
- lr ≤ 1e-3 with 500-step linear warmup + EMA (decay 0.999) for all new trainings (5e-3 diverged twice this week).
- Checkpoints: best + last ONLY, to `liquid_coupling_flow/ipl44/data/` (gitignored). Training logs to the same dir.
- GPU is shared: check `nvidia-smi --query-compute-apps=...` before launching; long runs via `nohup … &` + background log-watchers; OOM-retry with halved chunk.
- Equilibrium band (G2): median U within 5% of reference median AND g_BB peak within 15% of reference peak.
- Data: `/mnt/ssd/GridTransformer/datasets/ipl44_T0.1_positions.pt` + `_species.pt` (10K, UNWRAPPED — wrap with `torch.remainder(x, L)` at load; sort each config by species so it's 22×A then 22×B).
- Tests: `conda activate lightning && python -m pytest liquid_coupling_flow/tests/test_swap_smc.py -x -q` from repo root.
- Commit after each green task; never conclude "no bug" without a concrete check.

---

### Task 1: Exact swap-MH kernel

**Files:**
- Create: `liquid_coupling_flow/ipl44/ipl_swap_smc.py` (kernel part)
- Test: `liquid_coupling_flow/tests/test_swap_smc.py`

**Interfaces:**
- Produces: `swap_attempt(x, s, U, beta, energy_fn, weight_fn) -> (s_new, U_new, acc_mask)`;
  `uniform_weight_fn(x, s) -> 0.5*ones`; `_swap_log_ratio(pB_f, pB_r, s, s_new, i, j) -> logratio [B]` (internal, tested directly).
- `weight_fn(x, s) -> pB [B,N] in (0,1)`: prob that particle SHOULD be species B. `energy_fn(x, s) -> U [B]`.
- Convention: species 0=A, 1=B; forward picks A-particle i with prob `pB[i]/S_A` over current A's, B-particle j with prob `(1-pB[j])/S_B` over current B's; reverse re-evaluates `weight_fn` at the swapped state (paired re-evaluation ⇒ textbook-exact MH with state-dependent proposal).

- [ ] **Step 1: Write the failing tests**

```python
# liquid_coupling_flow/tests/test_swap_smc.py
"""Exactness tests for the swap-MH kernel: count preservation, hand-computed proposal ratio, and the decisive
long-run stationarity check against exact enumeration on a tiny system (positions frozen, swaps only)."""
import math, itertools, torch, pytest
from liquid_coupling_flow.ipl44.ipl_swap_smc import swap_attempt, uniform_weight_fn, _swap_log_ratio

torch.manual_seed(0)


def toy_energy(x, s, L=6.0):
    """Independent soft-sphere implementation (min-image, sigma depends on species pair): U=sum (sig/r)^12."""
    B, N, _ = x.shape
    sig = torch.tensor([[1.0, 1.2], [1.2, 1.4]])
    d = x[:, :, None, :] - x[:, None, :, :]
    d = d - L * torch.round(d / L)
    r = d.norm(dim=-1).clamp_min(1e-9)
    sij = sig[s[:, :, None].expand(-1, -1, N), s[:, None, :].expand(-1, N, -1)]
    e = (sij / r) ** 12
    iu = torch.triu_indices(N, N, offset=1)
    return e[:, iu[0], iu[1]].sum(-1)


def sdep_weight_fn(x, s):
    """Nontrivial, s-DEPENDENT weights to stress the reverse-evaluation path."""
    return torch.sigmoid(x[..., 0] - 3.0 + 0.5 * s.float().mean(dim=1, keepdim=True))


def test_count_preserved_and_only_ab_swaps():
    B, N = 32, 8
    x = torch.rand(B, N, 2) * 6.0
    s = torch.zeros(B, N, dtype=torch.long); s[:, :4] = 1
    s = torch.gather(s, 1, torch.argsort(torch.rand(B, N), dim=1))
    U = toy_energy(x, s)
    for _ in range(20):
        s, U, acc = swap_attempt(x, s, U, beta=2.0, energy_fn=toy_energy, weight_fn=sdep_weight_fn)
        assert (s.sum(1) == 4).all()
    # U must track the true energy of the current state
    assert torch.allclose(U, toy_energy(x, s), atol=1e-4)


def test_swap_log_ratio_hand_case():
    """N=4, one config: s=[A,A,B,B]; swap i=0 (A), j=2 (B). Hand-compute g and g'."""
    s = torch.tensor([[0, 0, 1, 1]])
    s_new = torch.tensor([[1, 0, 0, 1]])
    pB_f = torch.tensor([[0.9, 0.1, 0.2, 0.3]])   # forward weights (at state s)
    pB_r = torch.tensor([[0.6, 0.2, 0.7, 0.4]])   # reverse weights (at state s_new)
    i = torch.tensor([0]); j = torch.tensor([2])
    # forward: S_A = pB(0)+pB(1) = 1.0 -> P(i=0)=0.9 ; S_B = (1-pB(2))+(1-pB(3)) = 0.8+0.7=1.5 -> P(j=2)=0.8/1.5
    g = (0.9 / 1.0) * (0.8 / 1.5)
    # reverse (state s_new = [B,A,A,B]): A's = {1,2}: S'_A = pB_r(1)+pB_r(2) = 0.9 -> picks particle 2 (was j): 0.7/0.9
    # B's = {0,3}: S'_B = (1-pB_r(0))+(1-pB_r(3)) = 0.4+0.6 = 1.0 -> picks particle 0 (was i): 0.4/1.0
    gp = (0.7 / 0.9) * (0.4 / 1.0)
    lr = _swap_log_ratio(pB_f, pB_r, s, s_new, i, j)
    assert torch.allclose(lr, torch.tensor([math.log(gp / g)]), atol=1e-6)


@pytest.mark.parametrize("wfn_name", ["uniform", "sdep"])
def test_stationarity_exact_enumeration(wfn_name):
    """Positions frozen, swaps only: chain's empirical distribution over the C(6,3)=20 species states must
    match the exact Boltzmann distribution. Run B=256 parallel chains x 400 attempts, use the second half."""
    torch.manual_seed(1)
    N, nB, beta, L = 6, 3, 1.0, 6.0
    x1 = torch.rand(1, N, 2) * L
    wfn = uniform_weight_fn if wfn_name == "uniform" else sdep_weight_fn
    # exact distribution over labelings
    states = [c for c in itertools.combinations(range(N), nB)]
    svecs = torch.zeros(len(states), N, dtype=torch.long)
    for k, c in enumerate(states):
        svecs[k, list(c)] = 1
    Uex = toy_energy(x1.expand(len(states), -1, -1), svecs, L)
    logp = -beta * Uex; p_exact = torch.softmax(logp, 0)
    # chain
    B = 256
    x = x1.expand(B, -1, -1).contiguous()
    s = svecs[torch.randint(0, len(states), (B,))].clone()
    U = toy_energy(x, s, L)
    key = {tuple(v.tolist()): k for k, v in enumerate(svecs)}
    counts = torch.zeros(len(states))
    for t in range(400):
        s, U, _ = swap_attempt(x, s, U, beta=beta, energy_fn=lambda a, b: toy_energy(a, b, L), weight_fn=wfn)
        if t >= 200:
            for b in range(B):
                counts[key[tuple(s[b].tolist())]] += 1
    emp = counts / counts.sum()
    tv = 0.5 * (emp - p_exact).abs().sum()
    assert tv < 0.05, f"TV(empirical, exact) = {tv:.3f} for {wfn_name} proposer"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `conda activate lightning && python -m pytest liquid_coupling_flow/tests/test_swap_smc.py -x -q`
Expected: FAIL with `ModuleNotFoundError: ... ipl_swap_smc`

- [ ] **Step 3: Implement the kernel**

```python
# liquid_coupling_flow/ipl44/ipl_swap_smc.py
"""Swap-MH kernel + position Metropolis + sweeps-to-equilibrium benchmark for IPL44 (spec
docs/superpowers/specs/2026-07-02-ipl44-lever-push-design.md). The learned swap proposer uses PAIRED
RE-EVALUATION: forward weights from weight_fn at the current state, reverse weights from weight_fn at the
swapped state -> textbook-exact MH with a state-dependent proposal. Random proposer = constant weights
(ratio contribution exactly 0 in log space)."""
from __future__ import annotations
import torch

EPS = 1e-12


def uniform_weight_fn(x, s):
    return torch.full_like(s, 0.5, dtype=x.dtype)


def _pick(weights, mask):
    """Sample one index per row from weights restricted to mask; returns (idx [B], prob [B])."""
    w = weights * mask + EPS * mask
    S = w.sum(1)
    idx = torch.multinomial(w, 1).squeeze(1)
    return idx, w[torch.arange(w.shape[0]), idx] / S


def _sel_prob(weights, mask, idx):
    """Probability that _pick(weights, mask) would select idx."""
    w = weights * mask + EPS * mask
    return w[torch.arange(w.shape[0]), idx] / w.sum(1)


def _swap_log_ratio(pB_f, pB_r, s, s_new, i, j):
    """log g'(reverse)/g(forward) for swapping A-particle i with B-particle j.
    Forward at state s: pick i among A's with weight pB_f, j among B's with weight (1-pB_f).
    Reverse at state s_new: pick j among A's with weight pB_r, i among B's with weight (1-pB_r)."""
    isA, isB = (s == 0).to(pB_f.dtype), (s == 1).to(pB_f.dtype)
    isA_n, isB_n = (s_new == 0).to(pB_f.dtype), (s_new == 1).to(pB_f.dtype)
    g_i = _sel_prob(pB_f, isA, i)
    g_j = _sel_prob(1.0 - pB_f, isB, j)
    gp_j = _sel_prob(pB_r, isA_n, j)
    gp_i = _sel_prob(1.0 - pB_r, isB_n, i)
    return (gp_j + EPS).log() + (gp_i + EPS).log() - (g_i + EPS).log() - (g_j + EPS).log()


def swap_attempt(x, s, U, beta, energy_fn, weight_fn):
    """One batched MH swap attempt. Returns (s_new, U_new, accepted [B] bool)."""
    B = x.shape[0]; ar = torch.arange(B, device=x.device)
    pB_f = weight_fn(x, s)
    isA, isB = (s == 0).to(x.dtype), (s == 1).to(x.dtype)
    i, _ = _pick(pB_f, isA)
    j, _ = _pick(1.0 - pB_f, isB)
    s_prop = s.clone(); s_prop[ar, i] = 1; s_prop[ar, j] = 0
    U_prop = energy_fn(x, s_prop)
    pB_r = weight_fn(x, s_prop)
    log_ratio = -beta * (U_prop - U) + _swap_log_ratio(pB_f, pB_r, s, s_prop, i, j)
    acc = torch.log(torch.rand(B, device=x.device)) < log_ratio
    s_new = torch.where(acc[:, None], s_prop, s)
    U_new = torch.where(acc, U_prop, U)
    return s_new, U_new, acc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `conda activate lightning && python -m pytest liquid_coupling_flow/tests/test_swap_smc.py -x -q`
Expected: 4 passed (count/U-tracking, hand ratio, stationarity ×2). The stationarity test is the decisive one.

- [ ] **Step 5: Commit**

```bash
git add liquid_coupling_flow/ipl44/ipl_swap_smc.py liquid_coupling_flow/tests/test_swap_smc.py
git commit -m "feat(ipl44): exact swap-MH kernel (paired re-evaluation) with enumeration stationarity test"
```

---

### Task 2: Position Metropolis + chain runner + equilibrium metrics

**Files:**
- Modify: `liquid_coupling_flow/ipl44/ipl_swap_smc.py` (append)
- Test: `liquid_coupling_flow/tests/test_swap_smc.py` (append)

**Interfaces:**
- Consumes: `swap_attempt`, `uniform_weight_fn` (Task 1); `ipl_energy(positions, species) -> U [B]`,
  `ipl_gr_partials(positions, species, L) -> (r, gAA, gAB, gBB)`, `ipl_box() -> (N, L)` from
  `liquid_coupling_flow.ipl44.ipl_energy`.
- Produces: `position_sweep(x, s, U, beta, L, step, energy_fn) -> (x, U, acc_rate)`;
  `run_chain(x0, s0, n_sweeps, beta, L, energy_fn, weight_fn=None, n_swap=8, step=0.08, record_every=10) -> dict`
  with keys `"sweep" [K]`, `"U_median" [K]`, `"gbb_peak" [K]`, `"swap_acc" [K]`, `"pos_acc" [K]`,
  `"x","s"` (final); `sweeps_to_band(curves, U_ref_med, gbb_ref_peak, u_tol=0.05, g_tol=0.15) -> int|None`.

- [ ] **Step 1: Write the failing tests (append to test_swap_smc.py)**

```python
def test_position_sweep_preserves_equilibrium():
    """Start FROM reference equilibrium configs at beta=10: 30 sweeps must not drift the energy median
    outside the 5% band (stationarity of the position kernel)."""
    from liquid_coupling_flow.ipl44.ipl_swap_smc import position_sweep, run_chain, sweeps_to_band
    from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_box
    import torch, os
    N, L = ipl_box()
    D = "/mnt/ssd/GridTransformer/datasets"
    x = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), float(L))[:64]
    sp = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()[:64]
    o = sp.argsort(-1); sp = torch.gather(sp, 1, o)
    x = torch.gather(x, 1, o.unsqueeze(-1).expand(-1, -1, 2))
    ref_med = ipl_energy(x, sp).median()
    U = ipl_energy(x, sp)
    for _ in range(30):
        x, U, acc = position_sweep(x, sp, U, beta=10.0, L=float(L), step=0.08, energy_fn=ipl_energy)
    assert (U.median() - ref_med).abs() / ref_med.abs() < 0.05
    assert 0.05 < acc < 0.95


def test_run_chain_and_band():
    from liquid_coupling_flow.ipl44.ipl_swap_smc import run_chain, sweeps_to_band, uniform_weight_fn
    from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_box
    import torch
    N, L = ipl_box()
    x0 = torch.rand(16, N, 2) * float(L)
    s0 = torch.zeros(16, N, dtype=torch.long); s0[:, 22:] = 1
    out = run_chain(x0, s0, n_sweeps=20, beta=10.0, L=float(L), energy_fn=ipl_energy,
                    weight_fn=uniform_weight_fn, n_swap=4, record_every=5)
    assert len(out["sweep"]) == len(out["U_median"]) == len(out["gbb_peak"])
    assert (out["s"].sum(1) == 22).all()
    # sweeps_to_band: monotone curve hitting the band -> first index's sweep
    curves = {"sweep": [0, 10, 20, 30], "U_median": [30.0, 16.0, 14.8, 14.6], "gbb_peak": [0.5, 2.0, 4.0, 4.2]}
    assert sweeps_to_band(curves, U_ref_med=14.7, gbb_ref_peak=4.3) == 20
```

- [ ] **Step 2: Run to verify failure** — `python -m pytest liquid_coupling_flow/tests/test_swap_smc.py -x -q`, expected `ImportError: position_sweep`.

- [ ] **Step 3: Implement (append to ipl_swap_smc.py)**

```python
def position_sweep(x, s, U, beta, L, step, energy_fn):
    """One sweep = N sequential batched single-particle Metropolis moves. Returns (x, U, acc_rate)."""
    B, N, _ = x.shape; ar = torch.arange(B, device=x.device); n_acc = 0
    for i in torch.randperm(N).tolist():
        xp = x.clone()
        xp[:, i] = torch.remainder(xp[:, i] + step * torch.randn(B, 2, device=x.device), L)
        Up = energy_fn(xp, s)
        acc = torch.log(torch.rand(B, device=x.device)) < (-beta * (Up - U))
        x = torch.where(acc[:, None, None], xp, x); U = torch.where(acc, Up, U)
        n_acc += acc.float().mean().item()
    return x, U, n_acc / N


def run_chain(x0, s0, n_sweeps, beta, L, energy_fn, weight_fn=None, n_swap=8, step=0.08, record_every=10):
    """Alternate position sweeps and swap attempts; record batch-ensemble metrics every record_every sweeps."""
    from liquid_coupling_flow.ipl44.ipl_energy import ipl_gr_partials
    x, s = x0.clone(), s0.clone(); U = energy_fn(x, s)
    rec = {"sweep": [], "U_median": [], "gbb_peak": [], "swap_acc": [], "pos_acc": []}
    for k in range(n_sweeps + 1):
        if k % record_every == 0:
            _, _, _, gbb = ipl_gr_partials(x.cpu(), s.cpu(), L)
            rec["sweep"].append(k); rec["U_median"].append(float(U.median()))
            rec["gbb_peak"].append(float(gbb.max()))
        if k == n_sweeps:
            break
        x, U, pacc = position_sweep(x, s, U, beta, L, step, energy_fn)
        sacc = 0.0
        if weight_fn is not None:
            for _ in range(n_swap):
                s, U, a = swap_attempt(x, s, U, beta, energy_fn, weight_fn)
                sacc += a.float().mean().item()
            sacc /= n_swap
        if (k + 1) % record_every == 0 or k == 0:
            rec["swap_acc"].append(sacc); rec["pos_acc"].append(pacc)
    rec["x"], rec["s"] = x, s
    return rec


def sweeps_to_band(curves, U_ref_med, gbb_ref_peak, u_tol=0.05, g_tol=0.15):
    """First recorded sweep where BOTH |U_med-ref|/|ref| < u_tol and |gbb-ref|/ref < g_tol; None if never."""
    for k, u, g in zip(curves["sweep"], curves["U_median"], curves["gbb_peak"]):
        if abs(u - U_ref_med) / abs(U_ref_med) < u_tol and abs(g - gbb_ref_peak) / gbb_ref_peak < g_tol:
            return k
    return None
```

- [ ] **Step 4: Run tests** — expected all pass (the reference-equilibrium test also validates step=0.08 gives sane acceptance; if acceptance <0.05 or >0.95, tune `step` and record the value).
- [ ] **Step 5: Commit** — `git commit -m "feat(ipl44): position Metropolis + chain runner + sweeps-to-band"`

---

### Task 3: joint_flow.py extensions — per-species OT option + denoiser evaluation

**Files:**
- Modify: `liquid_coupling_flow/ipl44/joint_flow.py` (append two functions)
- Test: `liquid_coupling_flow/tests/test_swap_smc.py` (append)

**Interfaces:**
- Produces: `per_species_ot(x0, x1, nA, L) -> x0_aligned` (requires species-sorted configs: first nA are A);
  `denoiser_eval(model, x1, s1, t_eval=0.9, n_rep=4) -> dict(acc=float, ece=float)`.
- Note: v1's `global_position_ot` stays the default; per-species is an **ablation config** (the 6.5× OT win
  was OT-vs-nothing on eRSI, not per-species-vs-global — do not presume it transfers to the joint flow).

- [ ] **Step 1: Write the failing tests (append)**

```python
def test_per_species_ot_blocks():
    from liquid_coupling_flow.ipl44.joint_flow import per_species_ot, global_position_ot
    import torch
    B, N, nA, L = 8, 44, 22, 9.38
    x0, x1 = torch.rand(B, N, 2) * L, torch.rand(B, N, 2) * L
    xa = per_species_ot(x0, x1, nA, L)
    # block structure: aligned A-block is a permutation of x0's A-block (set equality of rows)
    for b in range(B):
        s0 = set(map(tuple, x0[b, :nA].round(decimals=5).tolist()))
        sa = set(map(tuple, xa[b, :nA].round(decimals=5).tolist()))
        assert s0 == sa


def test_denoiser_eval_runs():
    from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, denoiser_eval
    import torch
    m = JointSpeciesFlow(n_particles=44, L=9.38, hidden_nf=16, n_layers=2)
    x1 = torch.rand(6, 44, 2) * 9.38
    s1 = torch.zeros(6, 44, dtype=torch.long); s1[:, 22:] = 1
    out = denoiser_eval(m, x1, s1, t_eval=0.9, n_rep=2)
    assert 0.0 <= out["acc"] <= 1.0 and 0.0 <= out["ece"] <= 1.0
```

- [ ] **Step 2: Verify failure**, then **Step 3: implement (append to joint_flow.py)**

```python
def per_species_ot(x0, x1, nA, L):
    """Per-species OT for species-SORTED configs (first nA particles are A): Hungarian within each block."""
    a = linear_assignment_permutation(x0[:, :nA], x1[:, :nA], L=L)[0]
    b = linear_assignment_permutation(x0[:, nA:], x1[:, nA:], L=L)[0]
    return torch.cat([a, b], dim=1)


@torch.no_grad()
def denoiser_eval(model, x1, s1, t_eval=0.9, n_rep=4):
    """G1 metric: denoiser accuracy + ECE at the swap-proposer operating point. Builds (x_t, s_t) at t=t_eval
    from random s0 + Kawasaki, x_t on the interpolant toward x1 (x0 uniform, globally OT-aligned)."""
    B, N = s1.shape; dev = x1.device; L = model.L
    accs, confs, cors = [], [], []
    for _ in range(n_rep):
        x0 = torch.rand(B, N, 2, device=dev) * L
        x0 = global_position_ot(x0, x1, L)
        t = torch.full((B,), t_eval, device=dev)
        x_t = exp_map(x1, (1.0 - t)[:, None, None] * log_map(x1, x0, L), L)
        s0 = random_22_labeling(B, N, int(s1[0].sum()), dev)
        s_t = kawasaki_interpolate(s0, s1, t)
        _, logits = model(t.view(B, 1, 1), x_t, s_t)
        p = torch.softmax(logits, -1)
        pred = p.argmax(-1)
        accs.append((pred == s1).float().mean())
        confs.append(p.max(-1).values.flatten()); cors.append((pred == s1).float().flatten())
    conf = torch.cat(confs); cor = torch.cat(cors)
    bins = torch.linspace(0.5, 1.0, 11, device=conf.device); ece = torch.tensor(0.0, device=conf.device)
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf >= lo) & (conf < hi)
        if m.any():
            ece = ece + m.float().mean() * (conf[m].mean() - cor[m].mean()).abs()
    return {"acc": float(torch.stack(accs).mean()), "ece": float(ece)}
```

- [ ] **Step 4: Run tests**, **Step 5: Commit** — `git commit -m "feat(ipl44): per-species OT option + denoiser G1 eval"`

---

### Task 4: Training script with warmup+EMA+best/last

**Files:**
- Create: `liquid_coupling_flow/ipl44/train_joint_flow.py`

**Interfaces:**
- Consumes: `JointSpeciesFlow, joint_loss, global_position_ot, per_species_ot, random_22_labeling, denoiser_eval`.
- Produces: checkpoint `liquid_coupling_flow/ipl44/data/jf_{tag}_best.pt` and `_last.pt` with
  `{"state_dict" (EMA), "raw_state_dict", "cfg": {hidden_nf, n_layers, lam, ot}, "step", "val_acc", "val_ece"}`.
- CLI: `python -m liquid_coupling_flow.ipl44.train_joint_flow TAG [steps] [hidden_nf] [n_layers] [lam] [ot=global|species]`.

- [ ] **Step 1: Write the script**

```python
# liquid_coupling_flow/ipl44/train_joint_flow.py
"""Train JointSpeciesFlow on IPL44 T=0.1 (10K Zenodo, 9000/1000 split). Warmup->1e-3 constant, EMA 0.999,
best-by-val-denoiser-acc + last checkpoints only. G1 numbers printed every eval."""
import os, sys, time, copy, torch
from liquid_coupling_flow.ipl44.joint_flow import (JointSpeciesFlow, joint_loss, global_position_ot,
                                                   per_species_ot, random_22_labeling, denoiser_eval)
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L)
tag = sys.argv[1]
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
hidden = int(sys.argv[3]) if len(sys.argv) > 3 else 64
layers = int(sys.argv[4]) if len(sys.argv) > 4 else 4
lam = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
ot = sys.argv[6] if len(sys.argv) > 6 else "global"
B, lr, warmup, ema_decay = 256, 1e-3, 500, 0.999
D = "/mnt/ssd/GridTransformer/datasets"; ART = os.path.join(os.path.dirname(__file__), "data")

x = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sp = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
o = sp.argsort(-1); sp = torch.gather(sp, 1, o); x = torch.gather(x, 1, o.unsqueeze(-1).expand(-1, -1, 2))
xtr, str_, xva, sva = x[:9000].to(dev), sp[:9000].to(dev), x[9000:].to(dev), sp[9000:].to(dev)
nA = int((str_[0] == 0).sum())

m = JointSpeciesFlow(n_particles=N, L=Lf, hidden_nf=hidden, n_layers=layers).to(dev)
ema = copy.deepcopy(m).eval()
opt = torch.optim.Adam(m.parameters(), lr=lr)
npar = sum(p.numel() for p in m.parameters())
print(f"[{tag}] {npar/1e3:.1f}k params | hidden {hidden} layers {layers} lam {lam} ot {ot} | {steps} steps", flush=True)

best_acc, t0 = -1.0, time.time()
for step in range(steps):
    for g in opt.param_groups:
        g["lr"] = lr * min(1.0, (step + 1) / warmup)
    idx = torch.randint(0, xtr.shape[0], (B,), device=dev)
    x1, s1 = xtr[idx], str_[idx]
    x0 = torch.rand(B, N, 2, device=dev) * Lf
    x0 = per_species_ot(x0, x1, nA, Lf) if ot == "species" else global_position_ot(x0, x1, Lf)
    s0 = random_22_labeling(B, N, N - nA, dev)
    loss, lp, ls = joint_loss(m, x0, x1, s0, s1, lam=lam)
    if not torch.isfinite(loss):
        print(f"NON-FINITE loss at {step}, skipping"); opt.zero_grad(); continue
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 2.0); opt.step()
    with torch.no_grad():
        for pe, p in zip(ema.parameters(), m.parameters()):
            pe.mul_(ema_decay).add_(p, alpha=1 - ema_decay)
        for be, b in zip(ema.buffers(), m.buffers()):
            be.copy_(b)
    if step % 1000 == 0 or step == steps - 1:
        ev = denoiser_eval(ema, xva[:512], sva[:512])
        ck = {"state_dict": ema.state_dict(), "raw_state_dict": m.state_dict(),
              "cfg": {"hidden_nf": hidden, "n_layers": layers, "lam": lam, "ot": ot},
              "step": step, "val_acc": ev["acc"], "val_ece": ev["ece"]}
        torch.save(ck, f"{ART}/jf_{tag}_last.pt")
        star = ""
        if ev["acc"] > best_acc:
            best_acc = ev["acc"]; torch.save(ck, f"{ART}/jf_{tag}_best.pt"); star = " *BEST*"
        print(f"[{tag}] step {step:6d} loss {float(loss):.3f} (pos {float(lp):.3f} spec {float(ls):.3f}) "
              f"| val acc {ev['acc']:.4f} ece {ev['ece']:.3f}{star} | {time.time()-t0:.0f}s", flush=True)
print(f"[{tag}] DONE best val acc {best_acc:.4f}", flush=True)
```

- [ ] **Step 2: Smoke test** — `conda activate lightning && python -m liquid_coupling_flow.ipl44.train_joint_flow smoke 200 16 2 1.0 global`
Expected: loss decreases from first print; `jf_smoke_best.pt`/`_last.pt` created; no NaN. Delete smoke ckpts after.
- [ ] **Step 3: Commit** — `git commit -m "feat(ipl44): joint-flow training script (warmup+EMA, best/last, G1 eval)"`

---

### Task 5: G1 — scaling grid runs + gate decision

**Files:** none new (runs `train_joint_flow.py`; grid results recorded in the report, Task 9).

- [ ] **Step 1: Check GPU free** (`nvidia-smi`), then launch grid sequentially via nohup (each ~30–60 min):

```bash
cd /mnt/ssd/GridTransformer && source ~/xsli/softwares/anaconda3/etc/profile.d/conda.sh && conda activate lightning
A=liquid_coupling_flow/ipl44/data
nohup bash -c '
python -m liquid_coupling_flow.ipl44.train_joint_flow v1r   20000 32  3 1.0 global
python -m liquid_coupling_flow.ipl44.train_joint_flow mid   20000 64  4 1.0 global
python -m liquid_coupling_flow.ipl44.train_joint_flow big   20000 128 5 1.0 global
python -m liquid_coupling_flow.ipl44.train_joint_flow midsp 20000 64  4 1.0 species
' > $A/g1_grid.log 2>&1 &
```

- [ ] **Step 2: Watch** with a background until-loop on `g1_grid.log` for `DONE`/`NON-FINITE`/`Traceback`.
- [ ] **Step 3: Gate G1 decision.** Compare best val acc: pass = scaled (mid/big) clearly beats v1r (historical v1 ≈ 0.83). If `midsp` beats `mid`, per-species OT wins the ot ablation. Kill = neither mid nor big beats v1r meaningfully → record in report, stop Arm 1 (skip Tasks 6), proceed to Task 7.
- [ ] **Step 4: Commit** any log/notes — `git commit -m "exp(ipl44): G1 scaling grid results"`

---

### Task 6: G2 — swap-proposer benchmark

**Files:**
- Create: `liquid_coupling_flow/ipl44/bench_swap.py`

**Interfaces:**
- Consumes: `run_chain, sweeps_to_band, uniform_weight_fn` (Tasks 1–2); best G1 checkpoint `jf_{tag}_best.pt`;
  `make_ipl_model` + `liquid_coupling_flow/ipl44/data/ipl44_curveflow.pt` for the transformer seed bank;
  reference data for the band targets.
- Produces: `data/bench_swap_curves.pt` (all curves), `data/bench_swap.png`, printed sweeps-to-band table.

- [ ] **Step 1: Write the benchmark script**

```python
# liquid_coupling_flow/ipl44/bench_swap.py
"""G2 benchmark: seeds {uniform, tfbank, flow} x proposers {random, learned@t_prop in {0.7,0.9,0.99}}.
B=128 chains, beta=10, up to 1500 sweeps (record every 10). Learned weight_fn = denoiser pB at t_prop given
the CURRENT (x, s) — paired re-evaluation in swap_attempt keeps MH exact. Wall-clock recorded per config.
Usage: python -m liquid_coupling_flow.ipl44.bench_swap JF_CKPT_TAG [n_sweeps] [B]"""
import os, sys, time, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_swap_smc import run_chain, sweeps_to_band, uniform_weight_fn
from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_gr_partials, ipl_box
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L); beta = 10.0
tag = sys.argv[1]; n_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
B = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ART = os.path.join(os.path.dirname(__file__), "data")

ck = torch.load(f"{ART}/jf_{tag}_best.pt", map_location=dev, weights_only=False)
cfg = ck["cfg"]
jf = JointSpeciesFlow(n_particles=N, L=Lf, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"]).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval()
print(f"loaded jf_{tag}_best (val acc {ck['val_acc']:.4f})", flush=True)

# reference band targets
D = "/mnt/ssd/GridTransformer/datasets"
xr = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sr = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
o = sr.argsort(-1); sr = torch.gather(sr, 1, o); xr = torch.gather(xr, 1, o.unsqueeze(-1).expand(-1, -1, 2))
U_ref = float(ipl_energy(xr[:4096].to(dev), sr[:4096].to(dev)).median())
_, _, _, gbb_r = ipl_gr_partials(xr[:4096], sr[:4096], Lf)
GBB_REF = float(gbb_r.max())
print(f"band targets: U_med {U_ref:.2f} (+-5%), g_BB {GBB_REF:.2f} (+-15%)", flush=True)


def learned_wfn(t_prop):
    def wfn(x, s):
        with torch.no_grad():
            t = torch.full((x.shape[0], 1, 1), t_prop, device=x.device)
            _, logits = jf(t, x, s)
            return torch.softmax(logits, -1)[..., 1].clamp(1e-6, 1 - 1e-6)
    return wfn


def seeds(kind):
    s0 = random_22_labeling(B, N, 22, dev)
    if kind == "uniform":
        return torch.rand(B, N, 2, device=dev) * Lf, s0
    if kind == "flow":
        x0 = torch.rand(B, N, 2, device=dev) * Lf
        return jf.sample(x0, s0, n_steps=250)
    if kind == "tfbank":
        from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model
        tck = torch.load(f"{ART}/ipl44_curveflow.pt", map_location=dev, weights_only=False)
        tm = make_ipl_model(num_bins=tck["num_bins"], tail_bound=tck["tail_bound"], knn=tck["knn"],
                            arc_range=tck["arc_range"], device=dev)
        tm.load_state_dict(tck["state_dict"]); tm.eval()
        with torch.no_grad():
            xs, ss = tm.sample(B, n_B=tck["n_B"])           # verify signature: grep "def sample" ka_curveflow.py
        o = ss.argsort(-1)
        return torch.gather(torch.remainder(xs, Lf), 1, o.unsqueeze(-1).expand(-1, -1, 2)), torch.gather(ss, 1, o)
    raise ValueError(kind)


results = {}
for seed_kind in ["uniform", "tfbank", "flow"]:
    x0, s0 = seeds(seed_kind)
    for prop_name, wfn in [("random", uniform_weight_fn)] + [(f"learned{t}", learned_wfn(t)) for t in (0.7, 0.9, 0.99)]:
        t0 = time.time()
        cur = run_chain(x0.clone(), s0.clone(), n_sweeps, beta, Lf, ipl_energy, weight_fn=wfn, n_swap=8)
        stb = sweeps_to_band(cur, U_ref, GBB_REF)
        wall = time.time() - t0
        results[(seed_kind, prop_name)] = {k: cur[k] for k in ("sweep", "U_median", "gbb_peak", "swap_acc", "pos_acc")}
        results[(seed_kind, prop_name)]["sweeps_to_band"] = stb
        results[(seed_kind, prop_name)]["wall_s"] = wall
        print(f"{seed_kind:8s} {prop_name:10s}: sweeps-to-band {stb} | wall {wall:.0f}s "
              f"| final U_med {cur['U_median'][-1]:.2f} g_BB {cur['gbb_peak'][-1]:.2f}", flush=True)
torch.save(results, f"{ART}/bench_swap_curves.pt")

fig, axes = plt.subplots(2, 3, figsize=(17, 8))
for c, seed_kind in enumerate(["uniform", "tfbank", "flow"]):
    for (sk, pn), r in results.items():
        if sk != seed_kind:
            continue
        axes[0, c].plot(r["sweep"], r["U_median"], label=pn)
        axes[1, c].plot(r["sweep"], r["gbb_peak"], label=pn)
    axes[0, c].axhline(U_ref, ls="--", c="k"); axes[0, c].set_title(f"{seed_kind}: U median"); axes[0, c].legend(fontsize=7)
    axes[1, c].axhline(GBB_REF, ls="--", c="k"); axes[1, c].set_title(f"{seed_kind}: g_BB peak"); axes[1, c].set_xlabel("sweep")
plt.tight_layout(); plt.savefig(f"{ART}/bench_swap.png", dpi=110)
print(f"saved {ART}/bench_swap.png", flush=True)
```

- [ ] **Step 2: Verify the tfbank sampler signature** — `grep -n "def sample" liquid_coupling_flow/ka_curveflow.py`; adapt the `tm.sample(...)` call to the real signature before running.
- [ ] **Step 3: Short pilot** — `python -m liquid_coupling_flow.ipl44.bench_swap <best_tag> 100 32`: all 12 cells run, curves move, swap acceptance nonzero. Fix step size / n_swap if acceptance degenerate.
- [ ] **Step 4: Full run** via nohup (`… bench_swap <best_tag> 1500 128 > data/bench_swap.log 2>&1 &`) + watcher.
- [ ] **Step 5: Gate G2 decision** — learned vs random sweeps-to-band per seed; pass = ≥2× from ≥1 seed; kill = <20% everywhere. ALWAYS show the plot path.
- [ ] **Step 6: Commit** — `git commit -m "exp(ipl44): G2 swap-proposer benchmark"`

---

### Task 7: G0-PF pre-gate — TF-vs-FR decomposition on the IPL44 transformer

**Files:**
- Create: `liquid_coupling_flow/ipl44/pf_pregate.py`

**Interfaces:**
- Consumes: `make_ipl_model` + `ipl44_curveflow.pt`; reference data; the model's AR sampling internals
  (inspect `liquid_coupling_flow/ka_curveflow.py` for the per-particle conditional sampling loop — reuse the
  pattern from the KA exposure campaign scripts `ka_exposure_*.py`).
- Produces: per-index clash curves TF vs FR (`data/pf_pregate.png`), printed drift share.

- [ ] **Step 1: Inspect** `ka_curveflow.py` sampling loop + `ka_exposure_lf.py` (the KA TF-vs-FR machinery) and
  write `pf_pregate.py` reusing that pattern on IPL44: for j in AR order, sample particle j given (a) TRUE
  reference prefix (TF) and (b) the model's own prefix (FR); clash(j) = fraction with min-image NN distance
  < 0.9·σ(pair) to the prefix. 512 configs each.
- [ ] **Step 2: Run + plot.** Drift share = (FR−TF)/FR averaged over the last quartile of AR indices.
- [ ] **Step 3: Gate decision** — pass (drift exists, PF can act): drift share > 30%; kill: TF≈FR (residual-dominated, as on KA — PF cannot help). Record either way; show plot path.
- [ ] **Step 4: Commit** — `git commit -m "exp(ipl44): PF pre-gate TF-vs-FR decomposition"`

---

### Task 8 (CONDITIONAL — only if Task 7 passes): Professor forcing, ≤5 configs

**Files:**
- Create: `liquid_coupling_flow/ipl44/ipl_pf.py`

Design (concrete; exact hidden-tap pinned by Task 7's inspection): discriminator `D = nn.GRU(hidden→64) + linear`
over the per-step final-block hidden states captured with `register_forward_hook` on the transformer's last
layer during (a) TF passes (`m.log_prob` on reference data) and (b) FR segments (rollout from TF prefixes of
random length, ≤8 free steps for cost). Loss: `L = NLL + beta_pf * BCE(D(h_FR), 1)` for the generator
(non-saturating), `BCE(D(h_TF),1)+BCE(D(h_FR),0)` for D (separate Adam, lr 1e-4). Configs (≤5): beta_pf ∈
{0.03, 0.1, 0.3} at D=GRU-64, then best beta_pf with D=GRU-128 and free-len 16. Each run: fine-tune from
`ipl44_curveflow.pt` 5000 steps, lr 1e-4 (warmup 200), EMA, best+last by val NLL.

- [ ] **Step 1:** Implement `ipl_pf.py` per the design above; smoke 100 steps (both losses finite, D acc rises above 0.5).
- [ ] **Step 2:** Run the ≤5 configs sequentially (nohup + watcher).
- [ ] **Step 3:** Gate G-PF: FR discard (U > 2·U_ref_max on 4096 samples) 0.987 → <0.9, or core mass −30%, NLL degradation <5%. Kill on budget/instability.
- [ ] **Step 4: Commit** — `git commit -m "feat(ipl44): professor-forcing arm (bounded)"`

---

### Task 9: Results report + memory

**Files:**
- Create: `reports/<date>-ipl44-lever-push-results.md`
- Modify: memory `joint-species-flow` (+ MEMORY.md line); new memory only if a gate produced a durable lesson.

- [ ] **Step 1:** Write the report: G1 grid table (params vs val acc/ece), G2 sweeps-to-band table + plot paths,
  G0-PF verdict (+ G-PF if run), gate decisions with kill/pass rationale, next-spec handoff (transfer or stop).
- [ ] **Step 2:** Update memory + MEMORY.md; commit — `git commit -m "docs(ipl44): lever-push results report"`.

## Self-Review (done at write time)

- Spec coverage: §2→Tasks 3–5, §3→Tasks 1–2+6, §4→Tasks 7–8, §5 tests→Tasks 1–3, §6 gates→Tasks 5/6/7/8, §7 constraints→Global Constraints. Covered.
- Placeholders: Task 6 Step 2 and Task 7 Step 1 are explicit *verification* steps against existing code (sampler signatures), not TBDs; Task 8 is gated-conditional by design.
- Type consistency: `weight_fn(x,s)->pB [B,N]`, `energy_fn(x,s)->U [B]` used identically in Tasks 1, 2, 6; checkpoint dict keys consistent between Tasks 4 and 6 (`state_dict`, `cfg`, `val_acc`).
