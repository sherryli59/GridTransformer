# Swap-and-Breathe Staged Upgrades (v2 I-MTM + v3 denoiser guidance) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans. Parent spec: docs/superpowers/specs/2026-07-04-swap-and-breathe-design.md (§Staged extensions). Baseline to multiply: v1 = 0.153% equilibrium exchange (G1), τ_s ≈ 26 min (G3).

**Goal:** lift the equilibrium species-exchange rate via (v2) multi-try positions per transposition and (v3) geometry-guided pair selection — both exactly.

**Architecture:** extend `ka_swap_breathe.py` with `sb_mtm_move` (fixed-pair Liu I-MTM: M forward candidates under s′, M−1 reverse + current under s, accept min(1,Σw_f/Σw_r)) and an optional `pair_scorer` hook whose selection probabilities enter the ratio (v3 = scorer from jf_ka100tt's geometry table).

## Global Constraints
- NEVER `git add -A`. fp32. β=2.0. Spline proposal `ka_cluster_flow_full_N100.pt`.
- All Hastings/MTM factors exact; species counts invariant (tripwire); v1 kernel left untouched (new function).
- GPU shared with SMC pilot runs — smoke tests small.

---

### Task 1: v2 `sb_mtm_move` + v3 `pair_scorer` hook + tests

**Files:** Modify `liquid_coupling_flow/ka_swap_breathe.py` (append), extend `liquid_coupling_flow/tests/test_ka_swap_breathe.py`.

**Interfaces:**
- Produces: `sb_mtm_move(P, pos_o, s_o, cl, sc, L, beta=2.0, M=16, gen=None, pair_scorer=None) -> (pos_o', s_o', accepted[B], info)`; `make_jf_pair_scorer(device, ckpt="jf_ka100tt_best.pt") -> callable` returning `scores[B, k, k]` (unnormalized nonneg pair scores over cluster slot pairs (a_slot, b_slot); the kernel masks to unlike pairs and normalizes).
- Consumes: everything v1 consumes; for v3: `liquid_coupling_flow/ipl44/joint_flow.JointSpeciesFlow` + ckpt `liquid_coupling_flow/artifacts/jf_ka100tt_best.pt` (two_time; set `.knn=32`), geometry query at (t_pos=1, t_spec≈0) — see `denoiser_eval` in joint_flow.py for the query pattern (the implementer adapts it; if the import/ckpt is unavailable, `make_jf_pair_scorer` must raise ImportError and tests skip v3).

**v2 core (complete code):**

```python
@torch.no_grad()
def sb_mtm_move(P, pos_o, s_o, cl, sc, L, beta=2.0, M=16, gen=None, pair_scorer=None):
    """Multi-try swap-and-breathe (Liu I-MTM, fixed transposition per move).
    Forward: M position-candidates under the SWAPPED pattern s'; select ∝ w=exp(-beta*U_clu)-logq.
    Reverse: (M-1) fresh candidates under the ORIGINAL pattern s, plus the CURRENT positions.
    alpha = min(1, sum(w_fwd)/sum(w_rev)). At M=1 this reduces exactly to the v1 Hastings ratio.
    pair_scorer (v3): callable(pos_o, s_o, cl) -> scores[B,k,k] (nonneg); selection prob of the chosen pair
    enters the ratio as rho_Y(p)/rho_x(p) with rho evaluated at the CURRENT and SELECTED states."""
    assert getattr(P, "head_mode", "bins") == "spline"
    B, N, _ = pos_o.shape; dev = pos_o.device; k = cl.shape[0]
    ar = torch.arange(B, device=dev)
    s_cl = s_o[:, cl]
    isA = (s_cl == 0); isB = (s_cl == 1)
    ok = isA.any(1) & isB.any(1)
    # --- pair selection (uniform or scored) ---
    if pair_scorer is None:
        pA = torch.where(ok[:, None], isA.float(), torch.ones_like(isA, dtype=torch.float))
        pB = torch.where(ok[:, None], isB.float(), torch.ones_like(isB, dtype=torch.float))
        iA = torch.multinomial(pA, 1, generator=gen).squeeze(1)
        iB = torch.multinomial(pB, 1, generator=gen).squeeze(1)
        log_rho_x = torch.zeros(B, device=dev)                      # uniform: cancels (counts invariant)
        log_rho_y = torch.zeros(B, device=dev)
    else:
        sc_pair = pair_scorer(pos_o, s_o, cl).clamp_min(1e-12)       # [B,k,k]
        mask = (isA[:, :, None] & isB[:, None, :]).float()           # rows=A-slot, cols=B-slot
        w_pair = (sc_pair * mask).reshape(B, k * k)
        w_pair = torch.where(ok[:, None], w_pair, torch.ones_like(w_pair))
        flat = torch.multinomial(w_pair / w_pair.sum(1, keepdim=True), 1, generator=gen).squeeze(1)
        iA = flat // k; iB = flat % k
        log_rho_x = torch.log(w_pair[ar, flat] / w_pair.sum(1))      # forward selection prob (log)
        log_rho_y = None                                              # filled after the selected state is known
    s_new = s_cl.clone(); s_new[ar, iA] = 1; s_new[ar, iB] = 0
    s_prop = s_o.clone(); s_prop[:, cl] = s_new
    # --- forward candidates under s' (one batched call, B*M rows) ---
    posM = pos_o.repeat_interleave(M, 0); spropM = s_prop.repeat_interleave(M, 0)
    yC, logq_y = P.sample(posM, spropM, cl, sc, L)
    yC = yC.view(B, M, k, 2); logq_y = logq_y.view(B, M)
    U_y = cluster_energy(yC, pos_o, cl, s_prop, L)                   # [B,M]
    lw_f = -beta * U_y - logq_y                                       # log importance weights (rest-rest cancels)
    J = torch.multinomial(torch.softmax(lw_f, 1), 1, generator=gen).squeeze(1)
    xC_sel = yC[ar, J]
    # --- reverse candidates under s: (M-1) fresh + the CURRENT positions ---
    if M > 1:
        posR = pos_o.repeat_interleave(M - 1, 0); sR = s_o.repeat_interleave(M - 1, 0)
        xR, logq_r = P.sample(posR, sR, cl, sc, L)
        xR = xR.view(B, M - 1, k, 2); logq_r = logq_r.view(B, M - 1)
        U_r = cluster_energy(xR, pos_o, cl, s_o, L)
        lw_r_fresh = -beta * U_r - logq_r                             # [B,M-1]
    logq_x = P.log_q(pos_o, s_o, cl, pos_o[:, cl], sc, L)
    U_x = cluster_energy(pos_o[:, cl].unsqueeze(1), pos_o, cl, s_o, L).squeeze(1)
    lw_cur = (-beta * U_x - logq_x)[:, None]                          # [B,1]
    lw_r = torch.cat([lw_r_fresh, lw_cur], 1) if M > 1 else lw_cur    # [B,M]
    log_alpha = torch.logsumexp(lw_f, 1) - torch.logsumexp(lw_r, 1)
    if pair_scorer is not None:
        # rho_Y: selection prob of the SAME pair evaluated at the SELECTED state (positions xC_sel, labels s')
        pos_sel = pos_o.clone(); pos_sel[:, cl] = xC_sel
        sc_y = pair_scorer(pos_sel, s_prop, cl).clamp_min(1e-12)
        maskY = ((s_new == 0)[:, :, None] & (s_new == 1)[:, None, :]).float()
        wY = (sc_y * maskY).reshape(B, k * k)
        wY = torch.where(ok[:, None], wY, torch.ones_like(wY))
        flatY = iB * k + iA                                            # reverse pair: A-member=iB slot, B-member=iA slot
        log_rho_y = torch.log(wY[ar, flatY] / wY.sum(1))
        log_alpha = log_alpha + log_rho_y - log_rho_x
    u = torch.rand(B, device=dev, generator=gen)
    accepted = ok & (torch.log(u) < log_alpha)
    pos_out = pos_o.clone(); s_out = s_o.clone()
    pos_out[:, cl] = torch.where(accepted[:, None, None], xC_sel, pos_o[:, cl])
    s_out[:, cl] = torch.where(accepted[:, None], s_new, s_cl)
    assert torch.equal(s_out.sum(1), s_o.sum(1))
    info = {"acceptance": float(accepted.float().mean()), "exchange": float(accepted.float().mean()),
            "abort_frac": float((~ok).float().mean())}
    return pos_out, s_out, accepted, info
```

**v3 scorer (complete code, appended to the module):**

```python
def make_jf_pair_scorer(device, ckpt="jf_ka100tt_best.pt", t_spec=0.05):
    """Geometry-ambiguity pair scorer from the two-time joint flow's species posterior (t_pos=1, t_spec~0).
    score(a_slot, b_slot) = P(slot_a is B | geometry) * P(slot_b is A | geometry): pairs the geometry itself
    considers swappable. Raises ImportError/FileNotFoundError if the model or ckpt is unavailable."""
    import os, torch
    from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow
    from liquid_coupling_flow.ka_cluster_flow import ART
    ck = torch.load(os.path.join(ART, ckpt), map_location=device, weights_only=False)
    # reconstruct with the ckpt's stored hyperparams if present; fall back to campaign defaults 64|4 two_time
    jf = JointSpeciesFlow(n_particles=ck.get("N", 100), L=ck["L"] if "L" in ck else 9.128709291752768,
                          hidden_nf=ck.get("hidden_nf", 64), n_layers=ck.get("n_layers", 4),
                          two_time=True).to(device)
    jf.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    jf.eval(); jf.knn = 32

    @torch.no_grad()
    def scorer(pos_o, s_o, cl):
        B, N, _ = pos_o.shape; k = cl.shape[0]
        t1 = torch.ones(B, device=pos_o.device)
        _, logits = jf.forward(t1, pos_o, s_o, t_spec=torch.full((B,), t_spec, device=pos_o.device))
        p = torch.softmax(logits, -1)                                # [B,N,2] posterior P(species|geometry)
        pB_a = p[:, cl, 1]                                            # P(slot is B) for each cluster slot [B,k]
        pA_b = p[:, cl, 0]
        return pB_a[:, :, None] * pA_b[:, None, :]                    # [B,k,k]: (a-slot gets B) x (b-slot gets A)
    return scorer
```
NOTE for the implementer: verify `JointSpeciesFlow.forward(t, x, s, t_spec=...)` signature and the logits output
against `liquid_coupling_flow/ipl44/joint_flow.py` (the forward returns (velocity, logits) — check order/shape;
adapt the ckpt-key handling to what `jf_ka100tt_best.pt` actually contains; document what you find).

**Tests (append):**

```python
def test_sb_mtm_M1_reduces_to_v1_ratio():
    """At M=1 the I-MTM log-ratio equals v1's Hastings log-ratio on the same draws (same seed)."""
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    cl = KC.cluster_slots(5, sc, 7, L)
    from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move
    torch.manual_seed(123)
    p1, s1, a1, i1 = sb_mtm_move(P, pos, s, cl, sc, L, M=1)
    assert torch.equal(s1.sum(1), s.sum(1)) and torch.isfinite(torch.tensor(i1["acceptance"]))

def test_sb_mtm_counts_and_noncluster():
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=4)
    cl = KC.cluster_slots(5, sc, 7, L)
    from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move
    p2, s2, acc, info = sb_mtm_move(P, pos, s, cl, sc, L, M=4)
    mask = torch.ones(100, dtype=torch.bool, device=DEV); mask[cl] = False
    assert torch.equal(p2[:, mask], pos[:, mask]) and torch.equal(s2[:, mask], s[:, mask])
    assert torch.equal(s2.sum(1), s.sum(1))

def test_jf_pair_scorer_loads_or_skips():
    import pytest
    from liquid_coupling_flow.ka_swap_breathe import make_jf_pair_scorer, sb_mtm_move
    try:
        scorer = make_jf_pair_scorer(DEV)
    except (ImportError, FileNotFoundError, KeyError) as e:
        pytest.skip(f"jf scorer unavailable: {e}")
    torch.manual_seed(0)
    sc, L, geo, pos, s, P = _env(B=2)
    cl = KC.cluster_slots(5, sc, 7, L)
    scores = scorer(pos, s, cl)
    assert scores.shape == (2, 7, 7) and (scores >= 0).all() and torch.isfinite(scores).all()
    p2, s2, acc, info = sb_mtm_move(P, pos, s, cl, sc, L, M=4, pair_scorer=scorer)
    assert torch.equal(s2.sum(1), s.sum(1))
```

Steps: tests first (fail: function missing) → implement → all green (old 4 + new 3, minus possible v3 skip) → guards (`test_ka_local_smc.py` untouched-import check: 4/4) → commit exactly the two files: `feat(swap-breathe): v2 I-MTM multi-try + v3 denoiser-guided pair selection (exact selection factors)`.

### Task 2 (controller): upgrade gates
- **U1 (rate)**: G1-protocol reruns, 1000+ moves each at B=128: v2 M=16 uniform; v2+v3 M=16 scored. Compare exchange rate vs v1's 0.153%; report cost per accepted exchange vs v1.
- **U2 (spot stationarity)**: 15 rounds of [50 disp + 1 sb_mtm sweep] — U/N + BB-contacts hold (short form of G2; the kernel changed, the check must rerun).
- Records: report + memory update (`cluster-move-gate-nogo` SB block or new `swap-and-breathe` memory), commit logs.
