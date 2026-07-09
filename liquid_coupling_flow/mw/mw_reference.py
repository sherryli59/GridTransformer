"""Displacement-MC reference sampler for mW (Molinero-Moore 2009) — the project's independent ground
truth AND (later) its training-data generator. Batched Metropolis single-site moves over B independent
chains, exactness certified against quadrature in the next task (G1 gate at N=2,3) — kept dead simple,
no cleverness. Convergence proofs follow ka_finite_size.run_unit exactly (budget-halving flatness +
collection-window drift), reusing the same field names.

Run: python -m liquid_coupling_flow.mw.mw_reference N n_equil n_collect [seed]
"""
from __future__ import annotations
import math, os, sys, torch
import numpy as np
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, du_move, T_STAR, RHO_STAR

ART = os.path.join(os.path.dirname(__file__), "artifacts")
DEV = "cuda" if torch.cuda.is_available() else "cpu"


def mc_run(N, L, beta, n_equil, n_collect, every, seed, step0=0.15, B=8, track_every=200, ckpt_path=None):
    """Batched Metropolis single-site displacement MC, B chains in parallel.

    One sweep = N single-site attempts over all B chains: propose x_i + step*randn, wrap, accept
    per-chain with prob min(1, e^{-beta*dU}) via du_move (never a full recompute per attempt). U is
    tracked incrementally (U += dU on accept) so long runs stay affordable; the incremental value is
    refreshed via a full mw_energy_chunked recompute at the equil->collection boundary and drift-checked
    once more at the very end (soft guard: warn + recompute, never crash).
    """
    device = DEV
    gen = torch.Generator(device=device).manual_seed(seed)
    x = torch.remainder(torch.rand(B, N, 3, generator=gen, device=device) * L, L)
    U = mw_energy_chunked(x, L)

    step = step0
    traj = []
    half_equil = n_equil // 2
    adapt_acc = adapt_att = 0

    def do_sweep():
        nonlocal U
        acc_count = 0
        for i in range(N):
            xi_new = torch.remainder(x[:, i] + step * torch.randn(B, 3, generator=gen, device=device), L)
            dU = du_move(x, i, xi_new, L)
            p_acc = torch.exp(torch.clamp(-beta * dU, max=0.0))          # = min(1, e^-b*dU), overflow-safe
            accept = torch.rand(B, generator=gen, device=device) < p_acc
            x[:, i] = torch.where(accept[:, None], xi_new, x[:, i])
            U = torch.where(accept, U + dU, U)
            acc_count += int(accept.sum().item())
        return acc_count

    def save_partial(sweep):
        if ckpt_path is not None:
            torch.save({"traj": traj, "cfgs": x.detach().cpu(), "step": step, "sweep": sweep}, ckpt_path)

    # --- equilibration ---
    for s in range(1, n_equil + 1):
        acc_count = do_sweep()
        if s <= half_equil:
            adapt_acc += acc_count
            adapt_att += B * N
            if s % 100 == 0:
                ratio = adapt_acc / adapt_att
                if ratio > 0.45:
                    step *= 1.15
                elif ratio < 0.35:
                    step *= 0.85
                adapt_acc = adapt_att = 0
        if s % track_every == 0:
            traj.append((s, float((U.mean() / N).item())))
        if s % 2000 == 0:
            save_partial(s)

    # drift guard #1: compare the incrementally-carried U against a fresh recompute at collection start,
    # warn (don't crash) if it drifted, THEN reset to fresh — surfaces equilibration-phase bookkeeping bugs
    # that guard #2 (which only sees collection-phase drift after this reset) would otherwise mask.
    U_fresh_coll = mw_energy_chunked(x, L)
    drift_equil = float((U - U_fresh_coll).abs().max().item())
    if drift_equil >= 1e-2:
        print(f"WARNING mc_run: equilibration drift guard tripped "
              f"(|dU_carried - dU_fresh|={drift_equil:.4f} >= 1e-2); recomputing U fresh.", flush=True)
    U = U_fresh_coll

    # --- collection ---
    n_snap = n_collect // every
    cfgs_list, U_list = [], []
    coll_acc = coll_att = 0
    sweep = n_equil
    for c in range(n_snap):
        for _ in range(every):
            acc_count = do_sweep()
            sweep += 1
            coll_acc += acc_count
            coll_att += B * N
            if sweep % track_every == 0:
                traj.append((sweep, float((U.mean() / N).item())))
            if sweep % 2000 == 0:
                save_partial(sweep)
        cfgs_list.append(x.detach().clone())
        U_list.append(U.detach().clone())

    cfgs = (torch.cat(cfgs_list, dim=0) if cfgs_list else torch.empty(0, N, 3, device=device)).cpu()
    Uall = (torch.cat(U_list, dim=0) if U_list else torch.empty(0, device=device)).cpu()

    # drift guard #2: fresh recompute vs the incrementally-carried U at the very end; soft (warn, don't crash)
    U_fresh_end = mw_energy_chunked(x, L)
    drift = float((U - U_fresh_end).abs().max().item())
    if drift >= 1e-2:
        print(f"WARNING mc_run: drift guard tripped (|dU_carried - dU_fresh|={drift:.4f} >= 1e-2); "
              f"recomputing U fresh.", flush=True)
        U = U_fresh_end

    acc = coll_acc / coll_att if coll_att > 0 else float("nan")

    # convergence proofs, field names/formulas mirror ka_finite_size.run_unit exactly
    sw = np.array([t for t, _ in traj])
    uv = np.array([v for _, v in traj])
    half_mask = (sw > 0.4 * n_equil) & (sw < 0.6 * n_equil)
    u_half = float(uv[half_mask].mean()) if half_mask.any() else float("nan")
    final_mean = float((Uall / N).mean().item()) if Uall.numel() > 0 else float("nan")
    flat_budget = abs(final_mean - u_half)

    # NB: sparse track_every vs short n_collect can yield <4 collection traj points -> NaN by design
    # (mirrors ka_finite_size.run_unit's guard); real reference runs populate this properly.
    coll_vals = uv[sw > n_equil]
    coll_drift = (float(abs(coll_vals[len(coll_vals) // 2:].mean() - coll_vals[:len(coll_vals) // 2].mean()))
                  if len(coll_vals) >= 4 else float("nan"))

    return {"cfgs": cfgs, "U": Uall, "traj": traj, "step": step, "acc": acc,
            "flat_budget": flat_budget, "coll_drift": coll_drift}


def g_r(cfgs, L, nbins=120, rmax=None, chunk=128):
    """(r_centers, g) via min-image pair-distance histogram, ideal-gas normalized (g=1 for uniform gas).
    Chunked over configs (the [n,N,N,3] pair tensor is the documented 10.4 GiB blowup risk for big
    collected stacks — mw_energy_chunked hit the same wall)."""
    n_cfg, N, _ = cfgs.shape
    if rmax is None:
        rmax = L / 2.0
    device = cfgs.device
    iu = torch.triu_indices(N, N, offset=1, device=device)
    edges = torch.linspace(0.0, rmax, nbins + 1)
    counts = torch.zeros(nbins, dtype=torch.float64)
    for s in range(0, n_cfg, chunk):
        xb = cfgs[s:s + chunk]
        d = xb[:, :, None, :] - xb[:, None, :, :]
        d = d - L * torch.round(d / L)
        r = d.norm(dim=-1)
        rij = r[:, iu[0], iu[1]].reshape(-1).cpu()
        c, _ = torch.histogram(rij, bins=nbins, range=(0.0, rmax))
        counts += c.double()
    r_centers = 0.5 * (edges[:-1] + edges[1:])
    dr = rmax / nbins
    rho = N / L ** 3
    norm = n_cfg * (N * (N - 1) / 2.0) * (rho / N) * 4.0 * math.pi * r_centers ** 2 * dr
    g = (counts / norm.clamp_min(1e-12)).float()
    return r_centers, g


if __name__ == "__main__":
    N = int(sys.argv[1])
    n_equil = int(sys.argv[2])
    n_collect = int(sys.argv[3])
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    beta = 1.0 / T_STAR
    L = (N / RHO_STAR) ** (1.0 / 3.0)
    every = 25
    os.makedirs(ART, exist_ok=True)
    partial_path = os.path.join(ART, f"mw_ref_N{N}_s{seed}.partial.pt")
    final_path = os.path.join(ART, f"mw_ref_N{N}_s{seed}.pt")
    print(f"mw_reference: N={N} L={L:.4f} beta={beta:.4f} n_equil={n_equil} n_collect={n_collect} "
          f"every={every} seed={seed}", flush=True)
    out = mc_run(N, L, beta, n_equil, n_collect, every, seed, ckpt_path=partial_path)
    torch.save(out, final_path)
    print(f"mw_reference DONE: step {out['step']:.4f} acc {out['acc']:.3f} "
          f"flat_budget {out['flat_budget']:.4f} coll_drift {out['coll_drift']:.4f} -> {final_path}",
          flush=True)
