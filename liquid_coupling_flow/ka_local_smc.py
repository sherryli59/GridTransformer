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
from liquid_coupling_flow.ka_cluster_csmc import csmc_sweep
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix


def ess(logw):
    """Effective sample size of log-weights [B]."""
    lw = logw - logw.max()
    w = lw.exp()
    return float((w.sum() ** 2) / (w ** 2).sum())


def next_beta(logw, U, beta, beta1, ess_target):
    """Largest beta' in (beta, beta1] such that ESS(logw - (beta'-beta)*U) >= ess_target * B (ABSOLUTE target;
    a relative target lets an already-degenerate population jump to beta1 in one rung — observed P1 failure)."""
    B = logw.shape[0]
    target = ess_target * B
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
            transport_seeds=None, seed=0, device="cuda", init="warm", n_init=400,
            n_csmc=0, csmc_M=16, csmc_moves=None):
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
    if init == "exact":
        # exact importance weights vs the generator. MEASURED FAILURE MODE (P1 v1): raw generator samples carry
        # hard clashes (U ~ 1e14) -> -beta0*U spans astronomically -> ESS=1 at init (A1's overlap wall at the
        # seed stage). Kept only for experiments.
        U = ka_energy(pos, s, L)
        logw = -beta0 * U - logq_gen
    else:
        # "warm" (default): flow seeds are WARM STARTS for a beta0-equilibration phase (T=1.25 = easy liquid);
        # SMC starts with UNIFORM weights from the ~pi_{beta0} population. Generators are seeds, not proposals
        # (the campaign's standing lesson). Init bias decays with n_init; U/N should plateau before annealing.
        pos = _disp_sweeps(pos, s[0], L, kT=1.0 / beta0, n=n_init)
        pos, s, _ = swap_breathe_sweep(P, pos, s, sc, L, geo, beta=beta0)
        pos, s = _canonicalize(pos, s)
        U = ka_energy(pos, s, L)
        print(f"init(warm): {n_init} disp sweeps @ beta0 -> U/N {float(U.mean())/N:.4f} (T=1.25 liquid)", flush=True)
        logw = torch.zeros(B, device=device)
    beta = beta0; hist = []; t0 = time.time()
    for rung in range(max_rungs):
        s_can = s[0]
        assert (s == s[0:1]).all()
        # --- mutation at current beta (pi_beta-invariant; exactness insurance) ---
        for _ in range(n_mut):
            pos = _disp_sweeps(pos, s_can, L, kT=1.0 / beta, n=n_disp)
            pos, s, sb_info = swap_breathe_sweep(P, pos, s, sc, L, geo, beta=beta)
            pos, s = _canonicalize(pos, s); s_can = s[0]
            # optional cSMC cluster mutation (pi_beta-invariant positional move; the Stage-A1 depth lever)
            for _ in range(n_csmc):
                pos, s, cs_info = csmc_sweep(P, pos, s, sc, L, geo, beta=beta, M=csmc_M, n_moves=csmc_moves)
                pos, s = _canonicalize(pos, s); s_can = s[0]
        # --- ARM-1: stochastic transport toward the next target (positions only) ---
        if arm == "arm1":
            pos, logw = transport_sweep(P, pos, s, logw, sc, L, geo, beta, n_seeds=transport_seeds)
        # --- anneal; resample at the SAME threshold the annealer targets (classic adaptive SMC:
        # anneal to target -> resample -> mutate). A lower resample threshold deadlocks the scheduler:
        # ESS sits just below target, next_beta can only advance by its 1e-4 floor (measured P1 v2 crawl).
        if ess(logw) <= ess_target * B + 1e-6:      # <=: the annealer lands ESS EXACTLY on target;
            pos, s, logw = _resample(pos, s, logw)   # strict < burned an alternating floor-step rung (measured)
        U = ka_energy(pos, s, L)
        new_beta = next_beta(logw, U, beta, beta1, ess_target)
        logw = logw - (new_beta - beta) * U
        beta = new_beta
        e = ess(logw)
        hist.append({"rung": rung, "beta": beta, "ess": e, "U_mean": float(U.mean()) / N,
                     "U_std": float(U.std()) / N, "sb_acc": sb_info["acceptance"], "t": time.time() - t0})
        print(f"rung {rung:3d}: beta {beta:.4f}  ESS {e:7.1f}/{B}  U/N {float(U.mean())/N:.4f}"
              f"  sb {sb_info['acceptance']*100:.2f}%  {time.time()-t0:.0f}s", flush=True)
        if beta >= beta1 - 1e-9:
            break
    if beta < beta1 - 1e-9:
        print(f"WARNING: max_rungs reached at beta={beta:.4f} < beta1={beta1} — ladder incomplete", flush=True)
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
