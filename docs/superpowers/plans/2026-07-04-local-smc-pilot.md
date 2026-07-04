# Local-SMC Pilot Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Spec: docs/superpowers/specs/2026-07-04-local-smc-pilot-design.md (READ FIRST — arms, corrected √N rung scaling, weight-path rules).

**Goal:** `ka_local_smc.py` — annealed SMC (β 0.8→2.0) with flow seeds (exact init weights), adaptive Δβ, resampling, the validated mutation stack, and ARM-1's exact stochastic AR-cluster transport; then gates P0–P3.

**Architecture:** one module, one entry point `smc_run(arm=...)`; ARM-0 = mutation-only; ARM-1 = +transport sweeps whose weight increments use the spline proposal's exact one-forward log_q (NO integrator in the weight path). State = (pos[B,N,2], s[B,N], logw[B]) with the canonicalization discipline from swap-and-breathe.

**Tech Stack:** reuses `KAFlowHeadModel` (seeds + exact log_prob), `swap_breathe_sweep`, `ClusterProposal.sample/log_q`, `cluster_energy`, `_u_matrix` displacement, `ka_energy`. fp32.

## Global Constraints
- NEVER `git add -A`. Long jobs `> log 2>&1`, mtime liveness, logs+scripts → reports/logs-2026-07-04/ committed, artifacts saved (FULL per-rung history + final populations).
- Weight path must be integrator-free (AR spline proposal + analytic energies only).
- Species counts invariant everywhere (asserts); displacement always on canonical species; re-canonicalize after every species-changing sweep.
- References: N=100 solid (−3.260, g_BB 2.30); N=256 soft (directional only).

---

### Task 1: `ka_local_smc.py` + P0 tests

**Files:** Create `liquid_coupling_flow/ka_local_smc.py`, `liquid_coupling_flow/tests/test_ka_local_smc.py`.

**Interfaces:**
- Consumes: `KAFlowHeadModel(rho=1.2, n_bins=ck["n_bins"], knn=ck["knn"], num_bins=ck["num_bins"], tail_bound=ck["tail_bound"])` + `load_state_dict`; `m.sample(B, N, n_B, device, return_logq=True) -> (pos, sp, logq)`; `m._Lof(N)`; `swap_breathe_sweep(P, pos, s, sc, L, geo, k=7, beta, n_moves)`; `ClusterProposal` via `_load`; `cluster_energy(xC[B,M,k,2], pos, cl, s, L)`; `_u_matrix(xa,xb,sa,sb,L,exclude_diag)`; `cluster_slots`; `_scaffold`; `geo._curve_order`.
- Produces: `smc_run(arm, N=100, B=256, beta0=0.8, beta1=2.0, ess_target=0.6, max_rungs=40, n_mut=2, n_disp=40, transport_seeds=None, seed=0, device="cuda") -> dict` and helpers `ess(logw)`, `next_beta(logw, U, beta, beta1, ess_target)`.

- [ ] **Step 1: write the module**

```python
"""Local-SMC pilot: annealed SMC over beta 0.8->2.0 for the 2D KA glass, every learned component local.
Spec: docs/superpowers/specs/2026-07-04-local-smc-pilot-design.md.
ARM-0: flow seeds (exact init weights) + adaptive-Δβ annealing + resampling + validated mutation stack
       (parallel displacement + swap-and-breathe), weights updated by -Δβ·U only.
ARM-1: ARM-0 + per-rung stochastic AR-cluster TRANSPORT sweeps with EXACT weight increments
       (SNF stochastic-kernel form; spline proposal's one-forward log_q — no integrator in the weight path)."""
from __future__ import annotations
import os, time, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import _scaffold, _load, ART
from liquid_coupling_flow.ka_cluster_mtm import cluster_energy
from liquid_coupling_flow.ka_swap_breathe import swap_breathe_sweep
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix


def ess(logw):
    """Effective sample size of log-weights [B]."""
    lw = logw - logw.max()
    w = lw.exp()
    return float((w.sum() ** 2) / (w ** 2).sum())


def next_beta(logw, U, beta, beta1, ess_target):
    """Largest beta' in (beta, beta1] such that ESS(logw - (beta'-beta)*U) >= ess_target * ESS(logw). Bisection."""
    B = logw.shape[0]
    target = ess_target * ess(logw)
    if ess(logw - (beta1 - beta) * U) >= target:
        return beta1
    lo, hi = 0.0, beta1 - beta
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess(logw - mid * U) >= target: lo = mid
        else: hi = mid
    return beta + max(lo, 1e-4)


def _canonicalize(pos, s):
    perm = torch.argsort(s, dim=1, stable=True)
    return (torch.gather(pos, 1, perm[..., None].expand(-1, -1, 2)).contiguous(),
            torch.gather(s, 1, perm).contiguous())


def _disp_sweeps(pos, s_can, L, kT, n, step=0.04):
    B, N, _ = pos.shape
    for _ in range(n):
        prop = torch.remainder(pos + step * torch.randn_like(pos), L)
        dE = (_u_matrix(prop, pos, s_can, s_can, L, True) - _u_matrix(pos, pos, s_can, s_can, L, True)).sum(-1)
        acc = torch.log(torch.rand(B, N, device=pos.device)) < (-dE / kT)
        pos = torch.where(acc[:, :, None], prop, pos)
    return pos


@torch.no_grad()
def transport_sweep(P, pos, s, logw, sc, L, geo, beta, k=7, n_seeds=None):
    """ARM-1 stochastic transport: ALWAYS-APPLY cluster resamples; the weight absorbs the mismatch EXACTLY:
    logw += -beta*(U_clu(x')-U_clu(x)) + logq(x_C|S) - logq(x'_C|S).   (positions-only; species untouched)"""
    B, N, _ = pos.shape; dev = pos.device
    n_seeds = N if n_seeds is None else n_seeds
    for seed in torch.randperm(N)[:n_seeds].tolist():
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        cl = KC.cluster_slots(seed, sc, k, L)
        xC_new, logq_fwd = P.sample(pos_o, s_o, cl, sc, L)
        logq_rev = P.log_q(pos_o, s_o, cl, pos_o[:, cl], sc, L)
        U_new = cluster_energy(xC_new.unsqueeze(1), pos_o, cl, s_o, L).squeeze(1)
        U_old = cluster_energy(pos_o[:, cl].unsqueeze(1), pos_o, cl, s_o, L).squeeze(1)
        logw = logw + (-beta * (U_new - U_old) + logq_rev - logq_fwd)
        pos_o = pos_o.clone(); pos_o[:, cl] = xC_new
        idx = order[:, cl]
        pos = pos.clone(); pos[torch.arange(B, device=dev)[:, None], idx] = pos_o[:, cl]
    return pos, logw


def _resample(pos, s, logw, gen=None):
    B = logw.shape[0]
    w = torch.softmax(logw - logw.max(), 0)
    idx = torch.multinomial(w, B, replacement=True, generator=gen)
    return pos[idx].clone(), s[idx].clone(), torch.zeros_like(logw)


@torch.no_grad()
def smc_run(arm, N=100, B=256, beta0=0.8, beta1=2.0, ess_target=0.6, max_rungs=40, n_mut=2, n_disp=40,
            transport_seeds=None, seed=0, device="cuda"):
    """arm in {'arm0','arm1'}. Returns dict(history, final pos/s/logw); saves artifacts/smc_pilot_{arm}_N{N}.pt."""
    assert arm in ("arm0", "arm1")
    torch.manual_seed(seed)
    sc, L, geo = _scaffold(N, device)
    P = _load(torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=device,
                         weights_only=False), device)
    # --- flow seeds with EXACT initial weights ---
    from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
    ck = torch.load(os.path.join(ART, "ka_flowhead_N100_k8_scratch.pt"), map_location=device, weights_only=False)
    gen_m = KAFlowHeadModel(rho=1.2, n_bins=ck["n_bins"], knn=ck["knn"], num_bins=ck["num_bins"],
                            tail_bound=ck["tail_bound"]).to(device)
    gen_m.load_state_dict(ck["state_dict"]); gen_m.eval()
    n_B = round(0.35 * N)
    pos, sp, logq_gen = gen_m.sample(B, N, n_B=n_B, device=device, return_logq=True)
    pos, s = _canonicalize(pos, sp.long())
    U = ka_energy(pos, s, L)
    logw = -beta0 * U - logq_gen                                    # exact up to a constant
    beta = beta0; hist = []; t0 = time.time()
    for rung in range(max_rungs):
        s_can = s[0]
        assert (s == s[0:1]).all()
        # --- mutation at current beta (pi_beta-invariant; exactness insurance) ---
        for _ in range(n_mut):
            pos = _disp_sweeps(pos, s_can, L, kT=1.0 / beta, n=n_disp)
            pos, s, sb_info = swap_breathe_sweep(P, pos, s, sc, L, geo, beta=beta)
            pos, s = _canonicalize(pos, s); s_can = s[0]
        # --- ARM-1: stochastic transport toward the next target (positions only) ---
        if arm == "arm1":
            pos, logw = transport_sweep(P, pos, s, logw, sc, L, geo, beta, n_seeds=transport_seeds)
        # --- anneal ---
        U = ka_energy(pos, s, L)
        new_beta = next_beta(logw, U, beta, beta1, ess_target)
        logw = logw - (new_beta - beta) * U
        beta = new_beta
        e = ess(logw)
        hist.append({"rung": rung, "beta": beta, "ess": e, "U_mean": float(U.mean()) / N,
                     "U_std": float(U.std()) / N, "sb_acc": sb_info["acceptance"], "t": time.time() - t0})
        print(f"rung {rung:3d}: beta {beta:.4f}  ESS {e:7.1f}/{B}  U/N {float(U.mean())/N:.4f}"
              f"  sb {sb_info['acceptance']*100:.2f}%  {time.time()-t0:.0f}s", flush=True)
        if e < 0.5 * B:
            pos, s, logw = _resample(pos, s, logw)
        if beta >= beta1 - 1e-9:
            break
    # final mutation at beta1 (decorrelate the resampled population)
    for _ in range(2 * n_mut):
        pos = _disp_sweeps(pos, s[0], L, kT=1.0 / beta1, n=n_disp)
        pos, s, _ = swap_breathe_sweep(P, pos, s, sc, L, geo, beta=beta1)
        pos, s = _canonicalize(pos, s)
    out = {"history": hist, "pos": pos.cpu(), "s": s.cpu(), "logw": logw.cpu(), "arm": arm, "N": N, "B": B,
           "beta0": beta0, "beta1": beta1, "wall": time.time() - t0}
    torch.save(out, os.path.join(ART, f"smc_pilot_{arm}_N{N}.pt"))
    print(f"saved smc_pilot_{arm}_N{N}.pt  wall {out['wall']:.0f}s", flush=True)
    return out
```

- [ ] **Step 2: write the P0 tests**

```python
import torch
from liquid_coupling_flow.ka_local_smc import ess, next_beta, _resample, _canonicalize

def test_ess_bounds_and_uniform():
    lw = torch.zeros(64)
    assert abs(ess(lw) - 64) < 1e-4                       # uniform weights -> ESS = B
    lw2 = torch.full((64,), -1e9); lw2[0] = 0.0
    assert ess(lw2) < 1.01                                # degenerate -> ESS ~ 1

def test_next_beta_monotone_and_bounded():
    torch.manual_seed(0)
    logw = torch.zeros(128); U = torch.randn(128) * 10 - 300
    b = next_beta(logw, U, 0.8, 2.0, ess_target=0.6)
    assert 0.8 < b <= 2.0
    b2 = next_beta(logw, U, 0.8, 2.0, ess_target=0.9)     # stricter target -> smaller step
    assert b2 <= b + 1e-9

def test_resample_preserves_counts_and_shapes():
    torch.manual_seed(0)
    pos = torch.rand(16, 100, 2); s = (torch.rand(16, 100) < 0.35).long()
    s = torch.sort(s, dim=1).values                        # canonical
    logw = torch.randn(16)
    p2, s2, lw2 = _resample(pos, s, logw)
    assert p2.shape == pos.shape and torch.equal(lw2, torch.zeros(16))
    assert torch.equal(s2.sum(1), s.sum(1)[torch.zeros(16, dtype=torch.long)] * 0 + s[0].sum())  # counts uniform

def test_smoke_two_rungs_gpu():
    """2-rung dry run at tiny B: weights finite, ESS sane, species counts invariant."""
    if not torch.cuda.is_available():
        return
    from liquid_coupling_flow.ka_local_smc import smc_run
    out = smc_run("arm0", N=100, B=16, ess_target=0.5, max_rungs=2, n_mut=1, n_disp=5, seed=0)
    assert all(torch.isfinite(torch.tensor(h["ess"])) for h in out["history"])
    assert torch.isfinite(out["logw"]).all()
    assert (out["s"].sum(1) == out["s"][0].sum()).all()
```

- [ ] **Step 3**: tests fail (module missing) → implement → `python -m pytest liquid_coupling_flow/tests/test_ka_local_smc.py -q` all pass; guards: `test_ka_swap_breathe.py` 4/4, `test_ka_cluster_mtm.py` 4/4.
- [ ] **Step 4**: commit exactly the two files: `feat(local-smc): annealed SMC pilot — flow seeds w/ exact weights, adaptive Δβ, validated mutation stack, ARM-1 exact stochastic AR-cluster transport`.

### Task 2 (controller): gates
- [ ] **P0/P1**: `smc_run("arm0", N=100, B=256)` → ESS alive every rung; end ⟨U⟩/N within 0.02 of −3.260, g_BB ≈ 2.30 (weighted final population; also post-mutation unweighted). Save log.
- [ ] **P2**: `smc_run("arm1", N=100, B=256, transport_seeds=50)` at matched wall-clock (tune transport_seeds/n_mut); compare rungs, per-rung ESS, end quality; per-rung variance breakdown (energy vs q terms) from history.
- [ ] **P3**: both arms (or arm0 + winner) at N=256, B=128, max_rungs=64 (√N scaling); directional gate vs soft reference.
- [ ] Records: report + new memory `local-smc-pilot` + ledger; commit logs/artifacts refs.

---
## LEDGER CLOSE 2026-07-04
- [x] Tasks 1+2 module+tests committed (2539283 + fixes); guards green.
- [x] P0/P1: machinery PASS (ESS alive, 23 rungs/17 min, final −3.137±0.053, g_BB 2.21); strict depth bar (−3.26±0.02) NOT met — mutation-limited, knob = cold-β mutation budget. A1's ESS=1 fixed by warm-init (exact generator-IS init weights REFUTED as seed mechanism).
- [x] P2: ARM-1 IS-transport CATASTROPHIC NEGATIVE (chains poisoned rung 0, U/N +5.6e9; ESS blind to uniformly-dead populations). Mechanistic: MH filters per-move, IS pays ×p diversity per forced move — unaffordable at ~10% survival. ARM-1 path = per-rung-trained proposals (AFT/CRAFT) only.
- [x] P3: N=256 zero-shot TRANSFER PASS — 33 rungs/74 min, ESS alive, final −3.089±0.040, g_BB 2.24; gap to true eq SIZE-INVARIANT (0.12/N both sizes). Strict ≤−3.211 bar unmet (same mutation limit). Sharp claims blocked on N=256 reference rebuild.
- [x] Records: report §2026-07-04 in reports/2026-07-02-ka-cluster-bottleneck.md; memory local-smc-pilot; logs reports/logs-2026-07-04/smc_p*.out; artifacts smc_pilot_{arm0,arm1}_N100.pt, smc_pilot_arm0_N256.pt (+v1exactinit negative kept).
