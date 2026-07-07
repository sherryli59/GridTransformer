"""Reference protocol for the 2D KA glass: PARALLEL TEMPERING + a convergence gate.

O2 failed because (a) the grid IC is a metastable trap for a glass and (b) cold-T
equilibration from disorder is slow. PT fixes both: a ladder of replicas from cold
(target) to hot (ergodic), with adjacent-temperature config exchanges so hot replicas
feed decorrelated configs down to the cold target. Convergence is judged HONESTLY:
independent PT runs (different seeds) must agree on cold-level <U>/N, and that energy
must plateau vs sweeps. This produces trustworthy references -- the expensive ground
truth the flow must amortize.

Run:  python -m liquid_coupling_flow.ka_reference
"""
from __future__ import annotations
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_mcmc import _pair_energy_one, make_species
from liquid_coupling_flow.ka_energy import ka_energy

ART = os.path.join(os.path.dirname(__file__), "artifacts")
FRACB, RHO = 0.35, 1.2


def _mc_sweep(x, sd, L, kT, step, n_swap, Aidx, Bidx):
    """One sweep on [C,N,2] chains with PER-CHAIN kT [C]: parallel displacement + swaps."""
    C, N = x.shape[0], x.shape[1]
    prop = torch.remainder(x + step * torch.randn_like(x), L)
    dE = (_u_matrix(prop, x, sd, sd, L, True) - _u_matrix(x, x, sd, sd, L, True)).sum(-1)  # [C,N]
    acc = torch.log(torch.rand(C, N, device=x.device)) < (-dE / kT[:, None])
    x = torch.where(acc[:, :, None], prop, x)
    for _ in range(n_swap):
        i = Aidx[torch.randint(len(Aidx), (1,))].item()
        j = Bidx[torch.randint(len(Bidx), (1,))].item()
        xi, xj = x[:, i, :].clone(), x[:, j, :].clone()
        e_old = (_pair_energy_one(xi, sd[i].expand(C), x, sd, i, L)
                 + _pair_energy_one(xj, sd[j].expand(C), x, sd, j, L))
        xp = x.clone(); xp[:, i, :] = xj; xp[:, j, :] = xi
        e_new = (_pair_energy_one(xj, sd[i].expand(C), xp, sd, i, L)
                 + _pair_energy_one(xi, sd[j].expand(C), xp, sd, j, L))
        a = torch.log(torch.rand(C, device=x.device)) < (-(e_new - e_old) / kT)
        x[:, i, :] = torch.where(a[:, None], xj, xi)
        x[:, j, :] = torch.where(a[:, None], xi, xj)
    return x


def parallel_tempering(N, L, sd, T_ladder, device, n_per=8, n_equil=3000, n_collect=80,
                       every=20, step=0.05, n_swap=None, exchange_every=10, track_every=400,
                       seed=0, collect_all_rungs=False, track_all_rungs=False):
    """T_ladder cold->hot [M]. Runs M*n_per replicas. Returns (cold configs, traj, exch_acc, rungs).

    rungs is None unless collect_all_rungs (Stage-0 ladder dataset): then a [n_coll, M, B, N, 2] tensor of
    per-rung snapshots. track_all_rungs additionally records per-rung <U>/N in traj (drift-tail gate)."""
    torch.manual_seed(seed)
    M, B = len(T_ladder), n_per
    n_swap = N // 8 if n_swap is None else n_swap
    Aidx = (sd == 0).nonzero().squeeze(-1); Bidx = (sd == 1).nonzero().squeeze(-1)
    beta = (1.0 / T_ladder).to(device)                              # [M]
    kT = T_ladder[:, None].expand(M, B).reshape(-1).to(device)      # [M*B]
    x = torch.rand(M, B, N, 2, device=device) * L
    traj, exch_acc, snaps, rung_snaps = [], [], [], []
    for sweep in range(n_equil + n_collect):
        x = _mc_sweep(x.reshape(M * B, N, 2), sd, L, kT, step, n_swap, Aidx, Bidx).reshape(M, B, N, 2)
        if (sweep + 1) % exchange_every == 0:
            U = ka_energy(x.reshape(M * B, N, 2), sd, L).reshape(M, B)   # [M,B]
            for parity in (0, 1):
                for l in range(parity, M - 1, 2):
                    p = torch.exp((beta[l] - beta[l + 1]) * (U[l] - U[l + 1])).clamp(max=1.0)
                    a = torch.rand(B, device=device) < p
                    exch_acc.append(a.float().mean().item())
                    xl = x[l].clone(); x[l] = torch.where(a[:, None, None], x[l + 1], x[l]); x[l + 1] = torch.where(a[:, None, None], xl, x[l + 1])
                    Ul = U[l].clone(); U[l] = torch.where(a, U[l + 1], U[l]); U[l + 1] = torch.where(a, Ul, U[l + 1])
        if (sweep + 1) % track_every == 0:
            if track_all_rungs:
                urung = (ka_energy(x.reshape(M * B, N, 2), sd, L).reshape(M, B) / N).mean(1).tolist()
                traj.append((sweep + 1, urung))                          # per-rung <U>/N
            else:
                traj.append((sweep + 1, (ka_energy(x[0], sd, L) / N).mean().item()))   # cold level
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x[0].clone())                                  # cold-level samples
            if collect_all_rungs:
                rung_snaps.append(x.clone())                            # [M,B,N,2] all rungs
    rungs = torch.stack(rung_snaps, 0) if collect_all_rungs else None   # [n_coll,M,B,N,2]
    return torch.cat(snaps, 0), traj, float(np.mean(exch_acc)), rungs


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    N = 256; L = (N / RHO) ** 0.5; T = 0.5
    sd = make_species(N, FRACB).to(device)
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)          # 10-level geometric 0.5->1.25
    reps, trajs, cfgs = [], [], []
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for seed in (0, 1):
        cfg, traj, ex, _ = parallel_tempering(N, L, sd, T_ladder, device, n_per=8,
                                              n_equil=12000, n_collect=150, every=40, seed=seed)
        U = (ka_energy(cfg, sd, L) / N).mean().item()
        se = (ka_energy(cfg, sd, L) / N).std().item() / np.sqrt(cfg.shape[0])
        reps.append((U, se)); trajs.append(traj); cfgs.append(cfg)
        ax[0].plot([t for t, _ in traj], [u for _, u in traj], "-o", label=f"seed {seed} (exch {ex:.2f})")
        print(f"seed {seed}: cold <U>/N = {U:.4f} +/- {se:.4f}  exch_acc={ex:.2f}", flush=True)
    # verdict: independent runs agree (within ~3 sigma) AND the cold energy has plateaued
    d = abs(reps[0][0] - reps[1][0]); tol = max(3 * (reps[0][1] + reps[1][1]), 0.01)

    def plateau_drift(tr):
        u = [x for _, x in tr]
        return abs(np.mean(u[-2:]) - np.mean(u[-4:-2]))             # last-2 vs prev-2 mean
    plat = max(plateau_drift(trajs[0]), plateau_drift(trajs[1]))
    agree, flat = d < tol, plat < 0.01
    ok = agree and flat
    o2_unif = -3.0876
    ax[0].axhline(o2_unif, color="grey", ls=":", label=f"O2 plain-MC uniform {o2_unif:.3f}")
    ax[0].set_xlabel("cold-level sweep"); ax[0].set_ylabel("<U>/N"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"PT cold energy (agree={agree}, plateau={flat}, drift={plat:.3f})")
    ax[1].bar([0, 1], [r[0] for r in reps], yerr=[r[1] for r in reps], capsize=5)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["seed 0", "seed 1"])
    ax[1].set_ylabel("cold <U>/N"); ax[1].set_title(f"agreement |diff|={d:.4f}")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_reference_pt.png"), dpi=120)
    print(f"PT REFERENCE: seed0 {reps[0][0]:.4f} seed1 {reps[1][0]:.4f} |diff| {d:.4f} (tol {tol:.4f}); "
          f"plateau drift {plat:.4f} -> {'CONVERGED' if ok else 'NOT CONVERGED'}", flush=True)
    print(f"  vs O2 plain-MC uniform {o2_unif:.4f}", flush=True)
    # persist the equilibrated cold-level configs as the N=256 reference (both seeds combined)
    ref = torch.cat(cfgs, 0)
    torch.save({"x": ref.cpu(), "s": sd.cpu(), "N": N, "L": L, "T": T,
                "U_per_N": (ka_energy(ref, sd, L) / N).mean().item(),
                "converged": ok, "plateau_drift": plat, "seed_diff": d},
               os.path.join(ART, "ka_reference_N256.pt"))
    print(f"saved {ref.shape[0]} reference configs -> artifacts/ka_reference_N256.pt", flush=True)


def main_ladder(N, n_equil=16000, n_collect=4000, every=8, n_per=8, track_every=500,
                device="cuda" if torch.cuda.is_available() else "cpu"):
    """Stage-0 PT-LADDER dataset: instrument the validated PT protocol to persist ALL M rungs (not just cold),
    with the four honest convergence gates. One run buys (i) trustworthy N reference, (ii) A2 per-rung training
    data, (iii) Stage-B event-mining raw material. Saves artifacts/pt_ladder_N{N}.pt.

    Gates: (1) seed-pair agreement on cold <U>/N; (2) drift-free tail (per-rung 2nd-half vs 1st-half slope ~0);
    (3) exchange acceptance in-band; (4) finite-size direction is a CROSS-N check -> run finite_size_check()."""
    L = (N / RHO) ** 0.5; T = 0.5
    sd = make_species(N, FRACB).to(device)
    T_ladder = 0.5 * (1.25 / 0.5) ** (torch.arange(10) / 9)          # 10-level geometric 0.5->1.25
    beta_ladder = (1.0 / T_ladder).tolist()
    cold, drift_tail, seed_rungs, seed_ex = [], [], [], []
    for seed in (0, 1):
        cfg, traj, ex, rungs = parallel_tempering(N, L, sd, T_ladder, device, n_per=n_per, n_equil=n_equil,
                                                  n_collect=n_collect, every=every, track_every=track_every,
                                                  seed=seed, collect_all_rungs=True, track_all_rungs=True)
        # rungs [n_coll,M,B,N,2] -> per-rung stacks [M, n_coll*B, N, 2]
        nc, M, Bp = rungs.shape[0], rungs.shape[1], rungs.shape[2]
        per_rung = rungs.permute(1, 0, 2, 3, 4).reshape(M, nc * Bp, N, 2).cpu()
        seed_rungs.append(per_rung); seed_ex.append(ex)
        Ucold = (ka_energy(cfg, sd, L) / N).mean().item()
        se = (ka_energy(cfg, sd, L) / N).std().item() / np.sqrt(cfg.shape[0])
        cold.append((Ucold, se))
        # drift-tail gate: COLD-rung <U>/N drift over the COLLECTION window only (2nd-half vs 1st-half mean).
        # Two mislabel modes fixed here (both measured): (a) HOT rungs fluctuate far more than the cold target,
        # so max-over-all-rungs fails a converged cold reference (N=100: seed-diff 0.0005, hot-max 0.058);
        # (b) halving the FULL track includes the equilibration descent from the uniform start, so even the
        # cold drift reads the transient (N=100 regen: 0.044-0.053 on full track). Gate on cold x collection.
        arr = np.array([u for _, u in traj])                        # [T_track, M] per-rung <U>/N track
        sweeps_tr = np.array([t for t, _ in traj])
        coll = arr[sweeps_tr > n_equil]                             # collection-phase entries only
        if coll.shape[0] < 4:                                       # too few tracked points in collection
            coll = arr[-4:]
        half = coll.shape[0] // 2
        cold_drift = float(np.abs(coll[half:, 0].mean() - coll[:half, 0].mean()))
        max_drift = float(np.abs(coll[half:].mean(0) - coll[:half].mean(0)).max())
        drift_tail.append(cold_drift)
        print(f"seed {seed}: cold <U>/N {Ucold:.4f}+/-{se:.4f} exch {ex:.2f} "
              f"cold-drift {cold_drift:.4f} (hot-max {max_drift:.4f}) rung-configs {per_rung.shape[1]}", flush=True)
    d = abs(cold[0][0] - cold[1][0]); tol = max(3 * (cold[0][1] + cold[1][1]), 5e-3)
    agree = d < tol
    flat = max(drift_tail) < 5e-3                                    # cold-rung drift (see above)
    ex_ok = 0.15 < min(seed_ex) and max(seed_ex) < 0.95
    ok = agree and flat and ex_ok
    # combine seeds per rung
    configs_per_rung = [torch.cat([seed_rungs[0][l], seed_rungs[1][l]], 0) for l in range(len(beta_ladder))]
    out = os.path.join(ART, f"pt_ladder_N{N}.pt")
    torch.save({"N": N, "L": L, "T": T, "s": sd.cpu(), "betas": beta_ladder,
                "configs_per_rung": configs_per_rung,           # list[M] of [n, N, 2]
                "cold_U_per_N": cold[0][0], "seed_diff": d, "tol": tol, "tail_drift": max(drift_tail),
                "exch_acc": seed_ex, "gates": {"agree": agree, "flat": flat, "exch_in_band": ex_ok},
                "converged": ok}, out)
    print(f"LADDER N={N}: agree={agree}(|d|={d:.4f}/tol{tol:.4f}) flat={flat}(drift{max(drift_tail):.4f}) "
          f"exch_in_band={ex_ok}({min(seed_ex):.2f}-{max(seed_ex):.2f}) -> {'CONVERGED' if ok else 'NOT CONVERGED'}",
          flush=True)
    print(f"saved ladder ({sum(c.shape[0] for c in configs_per_rung)} configs over {len(beta_ladder)} rungs, "
          f"~{configs_per_rung[0].shape[0]}/rung) -> {out}", flush=True)


def finite_size_check(device="cpu"):
    """Gate 4 (cross-N): intensive <U>/N should be size-consistent with the correct sign. Loads both ladders."""
    res = {}
    for N in (100, 256):
        f = os.path.join(ART, f"pt_ladder_N{N}.pt")
        if os.path.exists(f):
            res[N] = torch.load(f, map_location=device, weights_only=False)["cold_U_per_N"]
    if len(res) == 2:
        print(f"finite-size: N=100 {res[100]:.4f} vs N=256 {res[256]:.4f} "
              f"(intensive |diff| {abs(res[100]-res[256]):.4f})", flush=True)
    else:
        print(f"finite-size: have {list(res.keys())}, need both N=100 and N=256", flush=True)
    return res


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "ladder":
        main_ladder(N=int(sys.argv[2]) if len(sys.argv) > 2 else 100)
    elif len(sys.argv) > 1 and sys.argv[1] == "finite_size":
        finite_size_check()
    else:
        main()
