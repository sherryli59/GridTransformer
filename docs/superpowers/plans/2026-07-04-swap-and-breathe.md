# Swap-and-Breathe Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the species-transposing cluster MH move (spec: docs/superpowers/specs/2026-07-04-swap-and-breathe-design.md) and run gates G0–G2 (G3 conditional on G1 > 0).

**Architecture:** One new module mirroring the validated `ka_cluster_mtm.py` pattern: a core move on slot-ordered state (transpose one unlike pair inside the deterministic k=7 cluster, resample all k positions from the exact-log_q spline proposal under the swapped pattern, Hastings-accept) + a lab-frame sweep that slot-orders on the fly and scatters back BOTH positions and species. Species are per-row `[B,N]` throughout (the canonical-fixed-species assumption is dead once this kernel runs).

**Tech Stack:** PyTorch fp32; reuses `ClusterProposal` (spline head, `ka_cluster_flow_full_N100.pt`), `cluster_slots`, `cluster_energy` (per-row species verified), `geo._curve_order`.

## Global Constraints

- NEVER `git add -A`; stage only named files. fp32. β=2.0 (T*=0.5).
- Spline-head proposal only (full support); ckpt `liquid_coupling_flow/artifacts/ka_cluster_flow_full_N100.pt`.
- Species counts per row MUST be invariant under every move (65:35 tripwire — assert in the move).
- Long jobs: `> log 2>&1`; liveness by mtime; raw logs + harness scripts → `reports/logs-<date>/`, committed; benchmarks save artifacts not just prints (CLAUDE.md).
- Baseline being beaten: position-preserving swap acceptance = 0/102,400 at equilibrium.

---

### Task 1: `ka_swap_breathe.py` + G0 tests (one task, one review)

**Files:**
- Create: `liquid_coupling_flow/ka_swap_breathe.py`
- Create: `liquid_coupling_flow/tests/test_ka_swap_breathe.py`

**Interfaces:**
- Consumes: `ClusterProposal.sample(pos,s,cl,sc,L)->(xC,logq)` / `.log_q(pos,s,cl,xC,sc,L)->logq` (s is `[B,N]`, species-aware); `KC.cluster_slots(seed,sc,k,L)->LongTensor[k]`; `cluster_energy(xC[B,M,k,2],pos,cl,s,L)->[B,M]` from `ka_cluster_mtm`; `geo._curve_order(pos,N)->[B,N]`.
- Produces: `swap_breathe_move(P, pos_o, s_o, cl, sc, L, beta=2.0, gen=None) -> (pos_o', s_o', accepted[B] bool, info dict)` on SLOT-ORDERED state; `swap_breathe_sweep(P, pos, s, sc, L, geo, k=7, beta=2.0, n_moves=None, gen=None) -> (pos', s', info)` on lab-frame state; `info` keys: `acceptance, exchange, abort_frac, mean_dU, mean_dlogq`.

- [ ] **Step 1: Write the module**

```python
"""Swap-and-breathe: species-transposing cluster MH move.
Spec: docs/superpowers/specs/2026-07-04-swap-and-breathe-design.md. Opens the equilibrium A<->B exchange
channel (position-preserving swaps measure 0/102400 accepted at T*=0.5) by coupling a label transposition
inside the deterministic k=7 cluster with an exact-log_q positional resample of ALL k cluster positions under
the swapped pattern ("the pocket breathes"). Exact Hastings: seed uniform; cluster deterministic
(position-independent); the unlike-pair count nA*nB is transposition-invariant so selection factors cancel;
q densities exact (spline head, one forward each); rest-rest energy cancels (cluster_energy)."""
from __future__ import annotations
import torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import cluster_energy


@torch.no_grad()
def swap_breathe_move(P, pos_o, s_o, cl, sc, L, beta=2.0, gen=None):
    """One coupled move for all B chains at the given cluster. SLOT-ORDERED inputs.
    pos_o [B,N,2]; s_o [B,N] long (per-row). Returns (pos_o', s_o', accepted[B], info).
    Chains whose cluster is single-species ABORT (accepted=False; state-symmetric: the cluster is a
    deterministic function of the seed, so the abort set is identical for x and any reachable x')."""
    assert getattr(P, "head_mode", "bins") == "spline", "swap-and-breathe requires the full-support spline head"
    B, N, _ = pos_o.shape; dev = pos_o.device; k = cl.shape[0]
    s_cl = s_o[:, cl]                                                    # [B,k]
    isA = (s_cl == 0).float(); isB = (s_cl == 1).float()
    ok = (isA.sum(1) > 0) & (isB.sum(1) > 0)                             # a transposition exists
    pA = torch.where(ok[:, None], isA, torch.ones_like(isA))             # dummy rows for aborted chains
    pB = torch.where(ok[:, None], isB, torch.ones_like(isB))
    ar = torch.arange(B, device=dev)
    iA = torch.multinomial(pA, 1, generator=gen).squeeze(1)              # uniform A-member  -> uniform over
    iB = torch.multinomial(pB, 1, generator=gen).squeeze(1)              # uniform B-member     nA*nB unlike pairs
    s_new = s_cl.clone(); s_new[ar, iA] = 1; s_new[ar, iB] = 0           # transpose the pair's labels
    s_prop = s_o.clone(); s_prop[:, cl] = s_new
    xC_new, logq_fwd = P.sample(pos_o, s_prop, cl, sc, L)                # q(x' | S, s')
    logq_rev = P.log_q(pos_o, s_o, cl, pos_o[:, cl], sc, L)              # q(x  | S, s ) — frame is x_C-indep,
    U_new = cluster_energy(xC_new.unsqueeze(1), pos_o, cl, s_prop, L).squeeze(1)   # so pos_o's x_R suffices
    U_old = cluster_energy(pos_o[:, cl].unsqueeze(1), pos_o, cl, s_o, L).squeeze(1)
    dU = U_new - U_old
    log_alpha = -beta * dU + logq_rev - logq_fwd
    u = torch.rand(B, device=dev, generator=gen)
    accepted = ok & (torch.log(u) < log_alpha)
    pos_out = pos_o.clone(); s_out = s_o.clone()
    pos_out[:, cl] = torch.where(accepted[:, None, None], xC_new, pos_o[:, cl])
    s_out[:, cl] = torch.where(accepted[:, None], s_new, s_cl)
    # tripwire: per-row species counts invariant (transposition is count-preserving by construction)
    assert torch.equal(s_out.sum(1), s_o.sum(1)), "species counts changed — kernel bug"
    fin = torch.isfinite(log_alpha) & ok
    info = {"acceptance": float(accepted.float().mean()),
            "exchange": float(accepted.float().mean()),                  # every accepted move IS an exchange
            "abort_frac": float((~ok).float().mean()),
            "mean_dU": float(dU[fin].mean()) if fin.any() else float("nan"),
            "mean_dlogq": float((logq_rev - logq_fwd)[fin].mean()) if fin.any() else float("nan")}
    return pos_out, s_out, accepted, info


@torch.no_grad()
def swap_breathe_sweep(P, pos, s, sc, L, geo, k=7, beta=2.0, n_moves=None, gen=None):
    """Lab-frame state (pos [B,N,2], s [B,N] per-row). n_moves (default N) moves at uniform random seeds,
    slot-ordering on the fly and scattering back BOTH positions and species after each move."""
    B, N, _ = pos.shape; dev = pos.device
    n_moves = N if n_moves is None else n_moves
    accs, exs, abr = [], [], []
    for _ in range(n_moves):
        seed = int(torch.randint(0, N, (1,), generator=gen).item())
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        cl = KC.cluster_slots(seed, sc, k, L)
        pos_o, s_o, acc, info = swap_breathe_move(P, pos_o, s_o, cl, sc, L, beta=beta, gen=gen)
        idx = order[:, cl]
        pos = pos.clone(); s = s.clone()
        pos[torch.arange(B, device=dev)[:, None], idx] = pos_o[:, cl]
        s[torch.arange(B, device=dev)[:, None], idx] = s_o[:, cl]
        accs.append(info["acceptance"]); exs.append(info["exchange"]); abr.append(info["abort_frac"])
    return pos, s, {"acceptance": sum(accs) / len(accs), "exchange": sum(exs) / len(exs),
                    "abort_frac": sum(abr) / len(abr)}
```

- [ ] **Step 2: Write the G0 tests**

```python
import os, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_swap_breathe import swap_breathe_move, swap_breathe_sweep
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
DEV = "cuda" if torch.cuda.is_available() else "cpu"

def _env(B=8):
    sc, L, geo = _scaffold(100, DEV)
    ref = torch.load(os.path.join(ART, "ka_reference_N100.pt"), map_location=DEV, weights_only=False)
    pos, s = slot_order(ref["x"][:B].to(DEV), ref["s"].to(DEV).long(), geo, 100)
    P = _load(torch.load(os.path.join(ART, "ka_cluster_flow_full_N100.pt"), map_location=DEV,
                         weights_only=False), DEV)
    return sc, L, geo, pos, s, P

def test_counts_invariant_and_noncluster_untouched():
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env()
    cl = KC.cluster_slots(5, sc, 7, L)
    counts0 = s.sum(1).clone()
    pos2, s2, acc, info = swap_breathe_move(P, pos, s, cl, sc, L)
    assert torch.equal(s2.sum(1), counts0)                       # 65:35 preserved per row
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(pos2[:, mask], pos[:, mask])              # non-cluster positions untouched
    assert torch.equal(s2[:, mask], s[:, mask])                  # non-cluster species untouched
    assert 0.0 <= info["acceptance"] <= 1.0 and 0.0 <= info["abort_frac"] <= 1.0

def test_roundtrip_on_swapped_pattern():
    """sample then log_q under the SAME swapped pattern must agree (<1e-4): the exactness of the q-ratio."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    cl = KC.cluster_slots(5, sc, 7, L)
    s_cl = s[:, cl]
    # transpose the first unlike pair found per row (rows without one: skip by construction of the ref 65:35,
    # k=7 clusters virtually always mixed; assert we got at least one mixed row)
    s_prop = s.clone()
    swapped_any = False
    for b in range(4):
        a_idx = (s_cl[b] == 0).nonzero().squeeze(-1); b_idx = (s_cl[b] == 1).nonzero().squeeze(-1)
        if len(a_idx) and len(b_idx):
            s_prop[b, cl[a_idx[0]]] = 1; s_prop[b, cl[b_idx[0]]] = 0; swapped_any = True
    assert swapped_any
    xC, lq = P.sample(pos, s_prop, cl, sc, L)
    lq2 = P.log_q(pos, s_prop, cl, xC, sc, L)
    assert torch.allclose(lq, lq2, atol=1e-4), (lq - lq2).abs().max()

def test_abort_on_single_species_cluster():
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=2)
    cl = KC.cluster_slots(5, sc, 7, L)
    s_forced = s.clone(); s_forced[:, cl] = 0                    # all-A cluster -> no unlike pair
    pos2, s2, acc, info = swap_breathe_move(P, pos, s_forced, cl, sc, L)
    assert info["abort_frac"] == 1.0 and not acc.any()
    assert torch.equal(pos2, pos) and torch.equal(s2, s_forced)  # aborted rows unchanged

def test_sweep_scatter_back_consistency():
    """Lab-frame sweep: energies stay finite, counts invariant, state changes only via accepted moves."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    from liquid_coupling_flow.ka_energy import ka_energy
    counts0 = s.sum(1).clone()
    pos2, s2, info = swap_breathe_sweep(P, pos, s, sc, L, geo, n_moves=20)
    assert torch.equal(s2.sum(1), counts0)
    assert torch.isfinite(ka_energy(pos2, s2, L)).all()
```

- [ ] **Step 3: Run tests to verify they fail** — `python -m pytest liquid_coupling_flow/tests/test_ka_swap_breathe.py -x -q` → FAIL (module missing). Then create the module (Step 1 code), rerun → 4/4 PASS. Also run `python -m pytest liquid_coupling_flow/tests/test_ka_cluster_mtm.py -q` (4/4, untouched dependency guard).

- [ ] **Step 4: Commit**

```bash
git add liquid_coupling_flow/ka_swap_breathe.py liquid_coupling_flow/tests/test_ka_swap_breathe.py
git commit -m "feat(swap-breathe): species-transposing cluster MH move (exact Hastings; opens the dead equilibrium swap channel)"
```

---

### Task 2 (controller): Gates G1 + G2

- [ ] **G1** (acceptance at equilibrium, ≥10k moves): script `sb_g1.py` — B=128 random reference slice, 80 sweeps of `swap_breathe_sweep` (=10,240 moves), report acceptance/exchange/abort + mean_dU/mean_dlogq for accepted vs all (component breakdown: energy-driven vs q-driven rejection). Success: any stable nonzero exchange rate (baseline exactly 0). Save log to `reports/logs-2026-07-04/`.
- [ ] **G2** (stationarity + species observables): from the random slice, 50 rounds of [100 displacement sweeps (per-row-species `_u_matrix` variant or per-row loop) + 1 swap-breathe sweep]; track U/N, g_BB peak, AND species observables (mean B-B nearest-neighbor count; g_AB peak). Hold within the MTM-gate thresholds (|ΔU/N| ≤ ~0.02 band vs the plateau-drift scale, g peaks within ±0.15) — CAREFUL: the displacement kernel must use PER-ROW species now (the vectorized `_u_matrix` takes one species vector; either run it per unique row after re-canonicalizing, or simplest: re-canonicalize (argsort species, permute positions) after each swap-breathe sweep so a single species vector stays exact — physics invariant under the relabeling).
- [ ] Record results + commit logs.

### Task 3 (controller, CONDITIONAL on G1 > 0): Gate G3 — τ_s payoff

- [ ] Two arms from the random slice, matched wall-clock (~30 min each): [displacement sweeps only] vs [displacement + interleaved swap-breathe]; compute species autocorrelation C_s(Δt)=⟨s_i(t)s_i(t+Δt)⟩ (detrended, per the parallel campaign's protocol) and compare τ_s vs the 100–190 baseline. Full trajectories saved (`artifacts/sb_traj_*.pt`).
- [ ] Record verdict in report + `cluster-mtm-kernel`-adjacent memory (new memory `swap-and-breathe` either way); commit.
