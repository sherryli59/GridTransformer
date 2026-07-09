"""c-scan feasibility map: for each pinning fraction c, does the two-arm gap CLOSE (physics displacement,
the fastest non-OOD mover) and is the overlap excess (Q_inf - Q_rand) measurable? Finds the window where
PTS-by-pinning is feasible: converges AND has signal. T=0.8, N=256."""
import time, torch, numpy as np
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q, q_rand
from liquid_coupling_flow.ka_pin_extract import stretched_exp_fit

dev = "cuda"; N = 256; T = 0.8; beta = 1.0 / T; B = 12; STEP = 0.10; QR = q_rand()
refs = torch.load("liquid_coupling_flow/artifacts/pts/refs_T0.8_N256.pt", map_location=dev, weights_only=False)
x0 = refs["x"][:B].to(dev); s = refs["s"].to(dev).long(); L = refs["L"]


def masked_disp(x, mob):
    prop = torch.remainder(torch.where(mob[..., None], x + STEP * torch.randn_like(x), x), L)
    dE = (_u_matrix(prop, x, s, s, L, True) - _u_matrix(x, x, s, s, L, True)).sum(-1)
    acc = (torch.log(torch.rand_like(dE)) < (-beta * dE)) & mob
    return torch.where(acc[..., None], prop, x)


def two_arm(c, n_iter=1500, ndisp=30, rec=150):
    torch.manual_seed(hash(round(c, 3)) % 2**31); n_pin = int(np.ceil(c * N))
    mob = torch.ones(B, N, dtype=torch.bool, device=dev)
    for b in range(B):
        mob[b, torch.randperm(N, device=dev)[:n_pin]] = False
    occ_ref = cell_occupancy(x0, L); excl = pinned_cells(x0, mob, L)
    out = {}
    for arm in ("ref", "scramble"):
        x = x0.clone() if arm == "ref" else torch.where(mob[..., None], torch.remainder(torch.rand_like(x0) * L, L), x0)
        t = []; Q = []
        for it in range(n_iter + 1):
            if it % rec == 0:
                t.append(it); Q.append(overlap_Q(cell_occupancy(x, L), occ_ref, excl))
            if it == n_iter:
                break
            for _ in range(ndisp):
                x = masked_disp(x, mob)
        out[arm] = stretched_exp_fit(t, Q)["Qinf"]
    lc = (c * 1.2) ** -0.5
    gap = abs(out["ref"] - out["scramble"])
    exc = out["ref"] - QR
    return {"c": c, "lc": lc, "ref": out["ref"], "scr": out["scramble"], "gap": gap, "excess_ref": exc}


t0 = time.time()
print(f"[cscan] T={T} N={N} B={B} step={STEP} Q_rand={QR:.3f}; gap<=0.02 = converged, excess>~0.05 = signal", flush=True)
rows = []
for c in (0.24, 0.16, 0.12, 0.08):
    r = two_arm(c); rows.append(r)
    verdict = ("CONVERGED" if r["gap"] <= 0.05 else "NOT-CONV") + ("/signal" if r["excess_ref"] > 0.05 else "/no-signal")
    print(f"[cscan] c={c} lc={r['lc']:.2f}: ref_Qinf={r['ref']:.3f} scr_Qinf={r['scr']:.3f} "
          f"GAP={r['gap']:.3f} excess(ref-Qrand)={r['excess_ref']:.3f} -> {verdict}  ({time.time()-t0:.0f}s)", flush=True)
torch.save(rows, "liquid_coupling_flow/artifacts/pts/cscan_T0.8_N256.pt")
print("[cscan] DONE", flush=True)
