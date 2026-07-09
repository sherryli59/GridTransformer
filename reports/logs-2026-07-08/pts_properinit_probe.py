"""Corrected two-arm test: the scramble/independent arm now starts from a VALID melted config (high-T anneal
of the mobile particles among the FIXED pins), NOT uniform-random (which jams at rho=1.2 and traps the arm,
artifactually inflating the gap). If the gap closes with a valid init, the earlier 'broken ergodicity' verdict
was an init bug. Physics displacement, T=0.8, N=256."""
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


def melt_init(mob, n_melt=400, T_hot=5.0):
    """Valid decorrelated init: start mobile at reference, anneal HOT (liquid) among fixed pins, then quench a
    little. Resolves overlaps + decorrelates from the reference WITHOUT jamming. Frozen stay at reference."""
    x = x0.clone(); bet_hot = 1.0 / T_hot
    for _ in range(n_melt):
        x = disp(x, mob, bet_hot, 0.15)                 # hot, larger step -> fast decorrelation, no jam
    return x


def two_arm(c, n_iter=1500, ndisp=30, rec=150):
    torch.manual_seed(hash(round(c, 3)) % 2**31); n_pin = int(np.ceil(c * N))
    mob = torch.ones(B, N, dtype=torch.bool, device=dev)
    for b in range(B):
        mob[b, torch.randperm(N, device=dev)[:n_pin]] = False
    occ_ref = cell_occupancy(x0, L); excl = pinned_cells(x0, mob, L)
    scr0 = melt_init(mob)
    q_scr_start = overlap_Q(cell_occupancy(scr0, L), occ_ref, excl)
    out = {}
    for arm, xinit in (("ref", x0.clone()), ("scramble", scr0)):
        x = xinit; t = []; Q = []
        for it in range(n_iter + 1):
            if it % rec == 0:
                t.append(it); Q.append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            if it == n_iter:
                break
            for _ in range(ndisp):
                x = disp(x, mob, beta, STEP)
        out[arm] = stretched_exp_fit(t, Q)["Qinf"]
    gap = abs(out["ref"] - out["scramble"])
    return {"c": c, "lc": (c * 1.2) ** -0.5, "ref": out["ref"], "scr": out["scramble"],
            "gap": gap, "excess": out["ref"] - QR, "scr_start": q_scr_start}


print(f"[properinit] T={T} N={N} step={STEP} melt=hot(T=5) — scramble starts from a VALID melted config", flush=True)
t0 = time.time()
for c in (0.16, 0.08):
    r = two_arm(c)
    v = "CONVERGED" if r["gap"] <= 0.05 else "NOT-CONV"
    print(f"[properinit] c={c} lc={r['lc']:.2f}: scr_start_Q={r['scr_start']:.3f} -> ref_Qinf={r['ref']:.3f} "
          f"scr_Qinf={r['scr']:.3f} GAP={r['gap']:.3f} (uniform-init gave 0.198/0.070) excess={r['excess']:.3f} "
          f"-> {v}  ({time.time()-t0:.0f}s)", flush=True)
print("[properinit] DONE", flush=True)
