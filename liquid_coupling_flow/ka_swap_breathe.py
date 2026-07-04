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


def make_jf_pair_scorer(device, ckpt="jf_ka100tt_best.pt", t_spec=0.05):
    """Geometry-ambiguity pair scorer from the two-time joint flow's species posterior (t_pos=1, t_spec~0).
    score(a_slot, b_slot) = P(slot_a is B | geometry) * P(slot_b is A | geometry): pairs the geometry itself
    considers swappable. Raises ImportError/FileNotFoundError if the model or ckpt is unavailable.

    NOTE on deviations from the brief's verbatim code (both required — see task-1-report.md for the
    verification writeup):
      1. Ckpt LOCATION: `jf_ka100tt_best.pt` does not exist under `ka_cluster_flow.ART`
         (liquid_coupling_flow/artifacts/); it lives at liquid_coupling_flow/ipl44/data/. We search ART
         first (in case a future run places it there) and fall back to ipl44/data.
      2. Ckpt KEYS: hyperparams are nested under ck["cfg"] (hidden_nf=64, n_layers=4, N=100,
         L=9.128709291752768, two_time=True), not top-level ck["N"]/ck["hidden_nf"]/etc as the brief's
         `ck.get(...)` implied — the brief's literal `ck.get("N", 100)` etc. happen to still resolve to the
         right numbers here only because the fallback defaults match this particular run's cfg by luck; we
         read ck["cfg"] explicitly so this is not accidental. `ck["state_dict"]` is the EMA-style weights
         (co-reported with val_acc=0.9993/val_ece=1.2e-4 in the same ckpt); `ck["raw_state_dict"]` differs
         (61/63 tensors) and is not used.
      3. Tensor SHAPES into `jf.forward`: JointSpeciesFlow.forward expects `t` and `t_spec` of shape
         [B,1,1] (it does `t.expand(-1, N, -1).reshape(B*N, 1)` internally) — the brief's 1-D
         `torch.ones(B)` / `torch.full((B,), t_spec)` raise "expanded size (-1) isn't allowed in a leading,
         non-existing dimension" at that expand call. Fixed by building t1/ts as [B,1,1].
      4. forward() return order/shapes verified by reading joint_flow.py directly: returns (vel, logits)
         (matches the brief's `_, logits = ...` unpacking) and logits has shape [B,N,n_species]=[B,N,2]
         (matches p[:, cl, 1]/p[:, cl, 0] indexing) — no change needed here.
    """
    import os, torch
    from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow
    from liquid_coupling_flow.ka_cluster_flow import ART
    cand_paths = [os.path.join(ART, ckpt), os.path.join(os.path.dirname(__file__), "ipl44", "data", ckpt)]
    path = ckpt if os.path.isabs(ckpt) and os.path.exists(ckpt) else None
    if path is None:
        for cand in cand_paths:
            if os.path.exists(cand):
                path = cand; break
    if path is None:
        raise FileNotFoundError(f"jf ckpt '{ckpt}' not found in any of: {cand_paths}")
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ck.get("cfg", {})
    # reconstruct with the ckpt's stored hyperparams (cfg dict) if present; fall back to campaign defaults
    # 64|4 two_time
    jf = JointSpeciesFlow(n_particles=cfg.get("N", ck.get("N", 100)),
                          L=cfg.get("L", ck.get("L", 9.128709291752768)),
                          hidden_nf=cfg.get("hidden_nf", ck.get("hidden_nf", 64)),
                          n_layers=cfg.get("n_layers", ck.get("n_layers", 4)),
                          two_time=cfg.get("two_time", True)).to(device)
    jf.load_state_dict(ck["state_dict"] if "state_dict" in ck else ck)
    jf.eval(); jf.knn = 32

    @torch.no_grad()
    def scorer(pos_o, s_o, cl):
        B, N, _ = pos_o.shape; k = cl.shape[0]
        t1 = torch.ones(B, 1, 1, device=pos_o.device)
        ts = torch.full((B, 1, 1), t_spec, device=pos_o.device)
        _, logits = jf.forward(t1, pos_o, s_o, t_spec=ts)
        p = torch.softmax(logits, -1)                                # [B,N,2] posterior P(species|geometry)
        pB_a = p[:, cl, 1]                                            # P(slot is B) for each cluster slot [B,k]
        pA_b = p[:, cl, 0]
        return pB_a[:, :, None] * pA_b[:, None, :]                    # [B,k,k]: (a-slot gets B) x (b-slot gets A)
    return scorer
