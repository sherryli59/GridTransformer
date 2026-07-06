"""Stage A1: conditional-SMC cluster move around the (frozen) ARM-FULL proposal — LAW 1 made exact.

The proposal ARM-FULL is diffuse-but-calibrated (Hastings +4.26): as a single-shot MH move its per-particle
clash compounds (0.72^7) to ~0.3% acceptance. Conditional SMC / particle Gibbs (Andrieu-Doucet-Holenstein)
fixes this by construction: we draw M candidate clusters sequentially from ARM-FULL's per-step conditional,
weight each partial placement by the exact incremental Boltzmann factor exp(-beta*dU_inc), keep the CURRENT
cluster as one retained reference particle, and select one final trajectory in proportion to its weight. Because
the reference (a good equilibrium cluster) is always in the ensemble, the move is reject-equivalent in the worst
case — it cannot poison the chain the way forced IS-transport did (P2). It is EXACTLY pi(x_C | x_env)-invariant,
so no outer MH is needed.

Core kernel = conditional SIR (no resampling): exact, no reference-lineage bookkeeping, reduces to identity at
M=1. Per-step resampling + PGAS ancestor sampling are efficiency add-ons (resample=/ancestor= flags), gated
empirically by stationarity-from-reference rather than trusted by default.

Energy: ONE code path. `_lj_sum` (candidate vs a partner set) is bound to the canonical `ka_energy` by the
telescope identity (sum of per-step increments == the cluster's energy marginal), tested in test_ka_cluster_csmc.
"""
from __future__ import annotations
import torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR, ka_energy


def _lj_sum(xi, si, xj, sj, L):
    """Sum of shifted/cutoff/min-image KA-LJ pair energies between candidate xi [B,2] (species si [B]) and a
    partner set xj [B,T,2] (species sj [B,T]). Same convention as ka_energy. Returns [B]."""
    t_sig = torch.tensor(SIGMA, device=xi.device, dtype=xi.dtype)
    t_eps = torch.tensor(EPS, device=xi.device, dtype=xi.dtype)
    sig = t_sig[si[:, None].expand_as(sj), sj]                       # [B,T] pair sigma
    eps = t_eps[si[:, None].expand_as(sj), sj]
    rc = RCUT_FACTOR * sig
    diff = xi[:, None, :] - xj                                       # [B,T,2]
    diff = diff - L * torch.round(diff / L)
    r2 = (diff ** 2).sum(-1).clamp_min(1e-12)                        # [B,T]
    inv6 = (sig ** 2 / r2) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    src6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e))
    return e.sum(-1)                                                 # [B]


def _sp_rows(s, idx, B):
    """species at slot indices idx -> [B, len(idx)] whether s is [N] or [B,N]."""
    return s[:, idx] if s.dim() == 2 else s[idx][None].expand(B, -1)


def cluster_energy_total(xC, pos, s, cl, L):
    """Cluster energy MARGINAL (cluster-cluster once + cluster-env once) via canonical ka_energy differencing.
    xC [B,k,2] candidate cluster positions (lab). Returns [B]. Single source of truth for energy."""
    B, N = pos.shape[0], pos.shape[1]
    env_mask = torch.ones(N, dtype=torch.bool, device=pos.device); env_mask[cl] = False
    pos_full = pos.clone(); pos_full[:, cl] = xC
    s_full = s if s.dim() == 1 else s                               # ka_energy accepts [N] or [B,N]
    U_full = ka_energy(pos_full, s_full, L)
    s_env = s[env_mask] if s.dim() == 1 else s[:, env_mask]
    U_env = ka_energy(pos_full[:, env_mask], s_env, L)
    return U_full - U_env


def cluster_energy_incremental(xC, pos, s, cl, L):
    """Per-step incremental energies [B,k]: step i = candidate i vs (env + placed cluster 0..i-1). Sums to
    cluster_energy_total by construction (telescope)."""
    B, N = pos.shape[0], pos.shape[1]; k = cl.shape[0]; dev = pos.device
    env_mask = torch.ones(N, dtype=torch.bool, device=dev); env_mask[cl] = False
    env_x = pos[:, env_mask]; env_s = _sp_rows(s, env_mask.nonzero().squeeze(-1), B)
    s_cl = _sp_rows(s, cl, B)                                        # [B,k]
    inc = torch.zeros(B, k, device=dev)
    for i in range(k):
        e = _lj_sum(xC[:, i], s_cl[:, i], env_x, env_s, L)
        if i > 0:
            e = e + _lj_sum(xC[:, i], s_cl[:, i], xC[:, :i], s_cl[:, :i], L)
        inc[:, i] = e
    return inc


@torch.no_grad()
def csmc_cluster_move(P, pos, s, cl, sc, L, beta=2.0, M=8, resample=False, ancestor=False, gen=None):
    """One conditional-SMC cluster move. pos [B,N,2], s [B,N] or [N] (slot-ordered), cl [k] slot indices.
    Returns (pos_new, s, info). Species are untouched (positional move). EXACT pi(x_C|x_env)-invariant.

    resample: per-step multinomial resampling with fixed reference lineage (efficiency; exact std cSMC).
    ancestor: PGAS-style ancestor resampling for the reference (efficiency; empirically gated, implies resample).
    """
    B, N = pos.shape[0], pos.shape[1]; k = cl.shape[0]; dev = pos.device
    if ancestor:
        resample = True
    origin, R, ctx_tok, q_scaf, ctx_u, ctx_sp = P._ctx_tokens(pos, s, cl, sc, L)
    sp = _sp_rows(s, cl, B)                                          # [B,k] cluster species (fixed)
    xC_ref_lab = pos[:, cl]                                          # [B,k,2] reference cluster (lab, exact)
    u_ref = KC.to_frame(xC_ref_lab, origin, R, L)                   # [B,k,2] reference in-frame
    env_mask = torch.ones(N, dtype=torch.bool, device=dev); env_mask[cl] = False
    env_x = pos[:, env_mask]; env_s = _sp_rows(s, env_mask.nonzero().squeeze(-1), B)

    # expand row-level context to BM = B*M particles (local index 0 within each row = reference)
    BM = B * M
    rep = lambda t: t.repeat_interleave(M, dim=0)
    ctx_tok_bm, q_scaf_bm, ctx_u_bm, ctx_sp_bm, sp_bm = map(rep, (ctx_tok, q_scaf, ctx_u, ctx_sp, sp))
    origin_bm, R_bm = rep(origin), rep(R)
    env_x_bm, env_s_bm = rep(env_x), rep(env_s)
    u_ref_bm, xC_ref_bm = rep(u_ref), rep(xC_ref_lab)
    is_ref = (torch.arange(BM, device=dev) % M) == 0                 # reference slots

    placed_u = torch.zeros(BM, 0, 2, device=dev)
    placed_lab = torch.zeros(BM, 0, 2, device=dev)
    placed_sp = torch.zeros(BM, 0, dtype=torch.long, device=dev)
    logW = torch.zeros(BM, device=dev)
    for i in range(k):
        ctx = P._ctx_at(i, placed_u, placed_sp, ctx_tok_bm, q_scaf_bm, ctx_u_bm, ctx_sp_bm, sp_bm)
        u_samp, _ = P._head_sample(ctx)                             # [BM,2] free proposals
        u_i = torch.where(is_ref[:, None], u_ref_bm[:, i], u_samp)  # force reference coord
        lq_step = P._head_logq(ctx, u_i)                            # [BM] one scoring path
        xi_lab = KC.from_frame(u_i[:, None], origin_bm, R_bm, L)[:, 0]
        xi_lab = torch.where(is_ref[:, None], xC_ref_bm[:, i], xi_lab)   # exact reference lab coords (no roundtrip)
        dU = _lj_sum(xi_lab, sp_bm[:, i], env_x_bm, env_s_bm, L)
        if i > 0:
            dU = dU + _lj_sum(xi_lab, sp_bm[:, i], placed_lab, placed_sp, L)
        logW = logW + (-beta * dU - lq_step)
        placed_u = torch.cat([placed_u, u_i[:, None]], 1)
        placed_lab = torch.cat([placed_lab, xi_lab[:, None]], 1)
        placed_sp = torch.cat([placed_sp, sp_bm[:, i:i + 1]], 1)
        if resample and i < k - 1:
            placed_u, placed_lab, placed_sp, logW = _resample_step(
                placed_u, placed_lab, placed_sp, logW, B, M, is_ref, ancestor, gen)

    # final selection per row in proportion to trajectory weight (reference always in the ensemble)
    logW_r = logW.reshape(B, M)
    probs = torch.softmax(logW_r, dim=1)                            # [B,M]
    sel = torch.multinomial(probs, 1, generator=gen).squeeze(1)     # [B]
    traj_lab = placed_lab.reshape(B, M, k, 2)
    xC_new = traj_lab[torch.arange(B, device=dev), sel]             # [B,k,2]
    pos_new = pos.clone(); pos_new[:, cl] = xC_new
    info = {"ref_survival": (sel == 0).float().mean().item(),
            "mean_logW_spread": (logW_r.max(1).values - logW_r.min(1).values).mean().item()}
    return pos_new, s, info


@torch.no_grad()
def csmc_sweep(P, pos, s, sc, L, geo, k=7, beta=2.0, M=16, n_moves=None, resample=False, ancestor=False, gen=None):
    """A pi_beta-invariant positional cSMC cluster sweep (valid SMC mutation kernel). Mirrors swap_breathe_sweep:
    re-slot-order per move (positions change), apply csmc_cluster_move, scatter cluster positions back. Species
    untouched. Returns (pos, s, info) with info['accept'] = mean fraction of moves that changed the cluster."""
    B, N, _ = pos.shape; dev = pos.device
    n_moves = N if n_moves is None else n_moves
    moved = []
    for _ in range(n_moves):
        seed = int(torch.randint(0, N, (1,), device=dev, generator=gen).item())
        order = geo._curve_order(pos, N)
        pos_o = torch.gather(pos, 1, order[..., None].expand(-1, -1, 2))
        s_o = torch.gather(s, 1, order)
        cl = KC.cluster_slots(seed, sc, k, L)
        pos_n, _, info = csmc_cluster_move(P, pos_o, s_o, cl, sc, L, beta=beta, M=M,
                                           resample=resample, ancestor=ancestor, gen=gen)
        idx = order[:, cl]
        pos = pos.clone()
        pos[torch.arange(B, device=dev)[:, None], idx] = pos_n[:, cl]
        moved.append(1.0 - info["ref_survival"])
    return pos, s, {"accept": sum(moved) / len(moved)}


def _resample_step(placed_u, placed_lab, placed_sp, logW, B, M, is_ref, ancestor, gen):
    """Per-step multinomial resampling within each row's M particles, preserving the reference lineage
    (offspring slot 0 = reference). ancestor=True: PGAS draws the reference's ancestor from the weights too
    (efficiency; empirically gated). Weights reset after resampling (standard). Returns reindexed buffers."""
    dev = logW.device
    logW_r = logW.reshape(B, M)
    probs = torch.softmax(logW_r, dim=1)
    anc = torch.multinomial(probs, M, replacement=True, generator=gen)   # [B,M] ancestors
    if not ancestor:
        anc[:, 0] = 0                                               # fixed reference lineage (offspring 0 = ref)
    idx = (anc + torch.arange(B, device=dev)[:, None] * M).reshape(-1)   # flat gather index
    return placed_u[idx], placed_lab[idx], placed_sp[idx], torch.zeros_like(logW)
