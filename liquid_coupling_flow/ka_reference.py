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
                       seed=0):
    """T_ladder cold->hot [M]. Runs M*n_per replicas. Returns (cold configs, traj, exch_acc)."""
    torch.manual_seed(seed)
    M, B = len(T_ladder), n_per
    n_swap = N // 8 if n_swap is None else n_swap
    Aidx = (sd == 0).nonzero().squeeze(-1); Bidx = (sd == 1).nonzero().squeeze(-1)
    beta = (1.0 / T_ladder).to(device)                              # [M]
    kT = T_ladder[:, None].expand(M, B).reshape(-1).to(device)      # [M*B]
    x = torch.rand(M, B, N, 2, device=device) * L
    traj, exch_acc, snaps = [], [], []
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
            traj.append((sweep + 1, (ka_energy(x[0], sd, L) / N).mean().item()))   # cold level
        if sweep >= n_equil and (sweep - n_equil) % every == 0:
            snaps.append(x[0].clone())                                  # cold-level samples
    return torch.cat(snaps, 0), traj, float(np.mean(exch_acc))


def main(device="cuda" if torch.cuda.is_available() else "cpu"):
    N = 256; L = (N / RHO) ** 0.5; T = 0.5
    sd = make_species(N, FRACB).to(device)
    T_ladder = torch.tensor([0.50, 0.60, 0.72, 0.87, 1.05, 1.25])    # cold->hot, geometric-ish
    reps = []
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for seed in (0, 1):
        cfg, traj, ex = parallel_tempering(N, L, sd, T_ladder, device, n_per=8,
                                           n_equil=3000, n_collect=80, seed=seed)
        U = (ka_energy(cfg, sd, L) / N).mean().item()
        se = (ka_energy(cfg, sd, L) / N).std().item() / np.sqrt(cfg.shape[0])
        reps.append((U, se))
        ax[0].plot([t for t, _ in traj], [u for _, u in traj], "-o", label=f"seed {seed} (exch {ex:.2f})")
        print(f"seed {seed}: cold <U>/N = {U:.4f} +/- {se:.4f}  exch_acc={ex:.2f}", flush=True)
    # convergence verdict: independent runs agree within ~3 sigma + plateau
    d = abs(reps[0][0] - reps[1][0]); tol = 3 * (reps[0][1] + reps[1][1])
    o2_unif = -3.0876   # the O2 uniform-IC value at N=256 (plain MC, 4000 sweeps)
    ok = d < max(tol, 0.01)
    ax[0].axhline(o2_unif, color="grey", ls=":", label=f"O2 plain-MC uniform {o2_unif:.3f}")
    ax[0].set_xlabel("cold-level sweep"); ax[0].set_ylabel("<U>/N"); ax[0].legend(fontsize=8)
    ax[0].set_title(f"PT cold-level energy (converged={'YES' if ok else 'NO'})")
    ax[1].bar([0, 1], [r[0] for r in reps], yerr=[r[1] for r in reps], capsize=5)
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["seed 0", "seed 1"])
    ax[1].set_ylabel("cold <U>/N"); ax[1].set_title(f"independent-run agreement (|diff|={d:.4f})")
    fig.tight_layout(); fig.savefig(os.path.join(ART, "ka_reference_pt.png"), dpi=120)
    print(f"PT REFERENCE: cold <U>/N seed0 {reps[0][0]:.4f} seed1 {reps[1][0]:.4f} |diff| {d:.4f} "
          f"(tol {max(tol,0.01):.4f}) -> {'CONVERGED' if ok else 'NOT CONVERGED'}", flush=True)
    print(f"  vs O2 plain-MC uniform-IC {o2_unif:.4f} (PT should reach LOWER if it equilibrates better)",
          flush=True)


if __name__ == "__main__":
    main()
