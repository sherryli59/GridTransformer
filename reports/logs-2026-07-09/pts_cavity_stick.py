"""CAVITY G-stick + error-bar gate (discriminator for the hard-wall R=4 marginal stall).

The 2-arm hard-wall run (pts_cavity_physics.py) gave gaps 0.038/0.058/0.032 at R=3/4/5 with a FLAT ~0.37
excess -- ambiguous between:
  H1 (large xi_PTS): physics converges, order genuinely persists past 5 sigma at T=0.5 -> no NN niche, need
                     bigger boxes (transfer).
  H2 (two-arm trap): at R>=4 the interior does NOT equilibrate; both arms coincidentally settle on the same
                     SUB-equilibrated plateau -> the two-arm gate lies -> a real hard interior an NN could attack.

Discriminator = THREE arms (ref + TWO independent melts, different seeds) with per-chain bootstrap CIs, hard
spherical wall (ka_cavity). Verdict per R:
  all three plateaus agree within CI                 -> H1 (trustworthy convergence; physics wins).
  the two independent melts DISAGREE (scrA != scrB)  -> H2 (two-arm gate was false-converged; NICHE real).
Full Q(t) trajectories are saved (record-simulation-data) so slow-drift vs plateaued-apart is visible.
Incremental per-R checkpoint (checkpoint-incrementally). N=256, T=0.5 (PT2 cold ref), hard wall, B=12."""
import time, torch, numpy as np
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR, ka_energy
from liquid_coupling_flow.ka_pin import masked_swap
from liquid_coupling_flow.ipl44.ipl_swap_smc import uniform_weight_fn
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, overlap_Q, overlap_Q_perchain, q_rand
from liquid_coupling_flow.ka_cavity import assert_mobile_inside, cavity_inside

dev = "cuda"; N = 256; T = 0.5; beta = 1.0 / T; B = 12; STEP = 0.08; SHELL = 1.0; QR = q_rand()
R_LIST = (3.0, 4.0, 5.0); OUT = "liquid_coupling_flow/artifacts/pts/cavity_stick_T0.5_N256.pt"
d = torch.load("liquid_coupling_flow/artifacts/pt_ladder_hb_N256.pt", map_location=dev, weights_only=False)
x0 = d["configs_per_rung"][0][:B].to(dev); s0 = d["s"].to(dev).long()[None].expand(B, -1).contiguous(); L = d["L"]
center = torch.tensor([L / 2, L / 2], device=dev)
t_sig = torch.tensor(SIGMA, device=dev); t_eps = torch.tensor(EPS, device=dev)
eye = torch.eye(N, device=dev, dtype=torch.bool)[None]
g = max(1, int(round(L / 0.3)))


def disp(x, s, mob, bet, step, R):
    a = s[:, :, None].expand(-1, -1, N); b = s[:, None, :].expand(-1, N, -1)
    sig = t_sig[a, b]; eps = t_eps[a, b]; rc = RCUT_FACTOR * sig
    prop = torch.remainder(torch.where(mob[..., None], x + step * torch.randn_like(x), x), L)
    inside = cavity_inside(prop, center, R, L)                            # hard spherical wall

    def cross(xa):
        diff = xa[:, :, None, :] - x[:, None, :, :]; diff = diff - L * torch.round(diff / L)
        r2 = (diff ** 2).sum(-1).masked_fill(eye, 1e12); inv6 = (sig ** 2 / r2) ** 3
        e = 4 * eps * (inv6 ** 2 - inv6); src6 = (sig / rc) ** 6
        return torch.where(r2 < rc ** 2, e - 4 * eps * (src6 ** 2 - src6), torch.zeros_like(e)).sum(-1)
    dE = cross(prop) - cross(x)
    acc = (torch.log(torch.rand_like(dE)) < (-bet * dE)) & mob & inside
    return torch.where(acc[..., None], prop, x)


def melt(mob, R, seed, T_hot=1.5, n=400):
    torch.manual_seed(seed); x = x0.clone()
    for _ in range(n):
        x = disp(x, s0, mob, 1.0 / T_hot, 0.12, R)
    assert_mobile_inside(x, mob, center, R, L)
    return x


def cavity_masks(R):
    dp = x0 - center[None, None]; dp = dp - L * torch.round(dp / L)
    mob = dp.norm(dim=-1) < R
    gi = torch.arange(g, device=dev); cc = (gi + 0.5) * L / g
    ii = gi[:, None].expand(g, g).reshape(-1); jj = gi[None, :].expand(g, g).reshape(-1)
    cd = torch.stack([cc[ii], cc[jj]], -1) - center[None]; cd = cd - L * torch.round(cd / L)
    deep = cd.norm(dim=-1) < (R - SHELL)
    excl = (~deep)[None].expand(B, -1).contiguous()
    return mob, excl


def run_arm(x_init, mob, R, occ_ref, excl, n_iter=4000, ndisp=30, nswap=8, rec=400):
    efn = lambda a, bb: ka_energy(a, bb, L)
    x = x_init.clone(); s = s0.clone(); U = efn(x, s)
    ts, Qb, Qc = [], [], []
    for it in range(n_iter + 1):
        if it % rec == 0:
            occ = cell_occupancy(x, L)
            ts.append(it); Qb.append(overlap_Q(occ, occ_ref, excl))
            Qc.append(overlap_Q_perchain(occ, occ_ref, excl).cpu())
        if it == n_iter:
            break
        for _ in range(ndisp):
            x = disp(x, s, mob, beta, STEP, R)
        U = efn(x, s)
        for _ in range(nswap):
            s, U, _ = masked_swap(x, s, U, mob, beta, efn, uniform_weight_fn)
    assert_mobile_inside(x, mob, center, R, L)
    plateau_chain = torch.stack(Qc[-3:]).mean(0)                          # [B] per-chain plateau
    return {"t": ts, "Q": Qb, "Q_chain": torch.stack(Qc), "plateau_chain": plateau_chain}


def boot_gap(pa, pb, nboot=2000):
    a = pa.numpy(); b = pb.numpy(); n = len(a); g = []
    for k in range(nboot):
        idx = np.random.RandomState(k).randint(0, n, n)               # seeded (Date/rand-free reproducible)
        g.append(abs(a[idx].mean() - b[idx].mean()))
    g = np.array(g); return float(np.mean(g)), float(np.percentile(g, 16)), float(np.percentile(g, 84))


torch.manual_seed(0)
occ_ref0 = cell_occupancy(x0, L)
print(f"[stick] N={N} T={T} step={STEP} shell={SHELL} Q_rand={QR:.3f}; 3-arm hard-wall G-stick discriminator", flush=True)
t0 = time.time(); allres = {}
for R in R_LIST:
    mob, excl = cavity_masks(R)
    arms = {"ref": run_arm(x0, mob, R, occ_ref0, excl),
            "scrA": run_arm(melt(mob, R, seed=101), mob, R, occ_ref0, excl),
            "scrB": run_arm(melt(mob, R, seed=202), mob, R, occ_ref0, excl)}
    pr, pa, pb = arms["ref"]["plateau_chain"], arms["scrA"]["plateau_chain"], arms["scrB"]["plateau_chain"]
    g_ra = boot_gap(pr, pa); g_rb = boot_gap(pr, pb); g_ab = boot_gap(pa, pb)
    n_mob = float(mob.float().sum(1).mean())
    verdict = "H2-NICHE (indep melts disagree)" if g_ab[1] > 0.05 else \
              ("H1-CONVERGED" if max(g_ra[0], g_rb[0], g_ab[0]) <= 0.05 else "MARGINAL")
    print(f"[stick R={R:.1f}] n_mob~{n_mob:.0f} | plateaus ref {float(pr.mean()):.3f} scrA {float(pa.mean()):.3f} "
          f"scrB {float(pb.mean()):.3f} | gap ref-scrA {g_ra[0]:.3f}[{g_ra[1]:.3f},{g_ra[2]:.3f}] "
          f"scrA-scrB {g_ab[0]:.3f}[{g_ab[1]:.3f},{g_ab[2]:.3f}] | excess~{float(pr.mean())-QR:.3f} "
          f"-> {verdict}  ({time.time()-t0:.0f}s)", flush=True)
    allres[R] = {"arms": {k: {kk: (vv.tolist() if torch.is_tensor(vv) and vv.ndim <= 1 else vv)
                              for kk, vv in v.items() if kk != "Q_chain"} for k, v in arms.items()},
                 "Q_chain": {k: v["Q_chain"] for k, v in arms.items()},
                 "gaps": {"ref_scrA": g_ra, "ref_scrB": g_rb, "scrA_scrB": g_ab}, "n_mob": n_mob}
    torch.save({"res": allres, "meta": {"N": N, "T": T, "R_list": R_LIST, "B": B, "Q_rand": QR,
                "step": STEP, "shell": SHELL, "note": "3-arm hard-wall cavity G-stick"}}, OUT)  # incremental
    print(f"[stick] saved {OUT} through R={R}", flush=True)
print("[stick] DONE", flush=True)
