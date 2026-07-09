"""Melt-tuning + trajectory probe. Scramble arm starts from a GENTLE melt (T_hot=1.5, liquid but not fully
randomized) so it's decorrelated-but-valid at both strong and weak pinning. Records the full Q(t) trace of
BOTH arms at higher budget -> tells us whether they are CONVERGING (ref decaying + scr rising, meeting -> more
budget wins) or PLATEAUED APART (glassy wall). T=0.8, N=256."""
import time, torch, numpy as np
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q, q_rand
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit

dev = "cuda"; N = 256; T = 0.8; beta = 1.0 / T; B = 12; STEP = 0.10; QR = q_rand()
refs = torch.load("liquid_coupling_flow/artifacts/pts/refs_T0.8_N256.pt", map_location=dev, weights_only=False)
x0 = refs["x"][:B].to(dev); s = refs["s"].to(dev).long(); L = refs["L"]


def disp(x, mob, bet, step):
    prop = torch.remainder(torch.where(mob[..., None], x + step * torch.randn_like(x), x), L)
    dE = (_u_matrix(prop, x, s, s, L, True) - _u_matrix(x, x, s, s, L, True)).sum(-1)
    acc = (torch.log(torch.rand_like(dE)) < (-bet * dE)) & mob
    return torch.where(acc[..., None], prop, x)


def melt(mob, T_hot=1.5, n=300, step=0.12):
    x = x0.clone(); bh = 1.0 / T_hot
    for _ in range(n):
        x = disp(x, mob, bh, step)
    return x


def two_arm(c, n_iter=3000, ndisp=30, rec=300):
    torch.manual_seed(hash(round(c, 3)) % 2**31); n_pin = int(np.ceil(c * N))
    mob = torch.ones(B, N, dtype=torch.bool, device=dev)
    for b in range(B):
        mob[b, torch.randperm(N, device=dev)[:n_pin]] = False
    occ_ref = cell_occupancy(x0, L); excl = pinned_cells(x0, mob, L)
    scr0 = melt(mob)
    traj = {}
    print(f"[traj] --- c={c} lc={(c*1.2)**-0.5:.2f}  (Q_rand={QR:.3f}, scr melt-start Q={overlap_Q(cell_occupancy(scr0,L),occ_ref,excl):.3f}) ---", flush=True)
    for arm, xinit in (("ref", x0.clone()), ("scramble", scr0)):
        x = xinit; t = []; Q = []
        for it in range(n_iter + 1):
            if it % rec == 0:
                q = overlap_Q(cell_occupancy(x, L), occ_ref, excl); t.append(it); Q.append(q)
                print(f"    [{arm:8s}] it {it:4d} Q {q:.3f}", flush=True)
            if it == n_iter:
                break
            for _ in range(ndisp):
                x = disp(x, mob, beta, STEP)
        traj[arm] = (t, Q)
    qi_r = stretched_exp_fit(*traj["ref"])["Qinf"]; qi_s = stretched_exp_fit(*traj["scramble"])["Qinf"]
    # last-third means as a plateau read (avoids fit edge floor)
    lr = np.mean(traj["ref"][1][-3:]); ls = np.mean(traj["scramble"][1][-3:])
    print(f"[traj] c={c}: ref plateau~{lr:.3f} scr plateau~{ls:.3f}  GAP(last3)={abs(lr-ls):.3f}  "
          f"(fit gap {abs(qi_r-qi_s):.3f})  ref-excess~{lr-QR:.3f}", flush=True)
    return {"c": c, "ref": lr, "scr": ls, "gap": abs(lr - ls)}


print(f"[traj] gentle melt T_hot=1.5; budget 3000 it x30 disp; watching for CONVERGING vs PLATEAUED-APART", flush=True)
for c in (0.16, 0.08):
    two_arm(c)
print("[traj] DONE", flush=True)
