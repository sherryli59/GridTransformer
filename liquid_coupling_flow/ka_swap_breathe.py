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
