"""QUICK CHECK: does the REFERENCE-BASIN overlap at R=2.2 match Hocky's published q~(2.2)?

Reproduces the paper's OWN "standard overlap" protocol (arXiv:1201.2888), ML-FREE, on our
rho=1.2 T=0.55 bank: carve cavities, freeze exterior, run cavity-constrained swap-MC inside,
measure the PAPER-EXACT CENTER overlap (5x5x5 cube of side-l boxes at the cavity center,
rho-normalized, bulk-subtracted) vs the frozen reference config. q~(t) decays from ~0.94 (self)
to a plateau = the thermodynamic reference-basin overlap. If plateau ~ 0.41, then w_ref=1 is
self-consistent with the published number; if it plateaus much higher, either residual convention
error OR competing basins should carry weight (w_ref<1). Table-I fit LJ T=0.55: xi=2.055 eta=3.119
=> q~(2.2)=0.41. Runs on CPU (tiny: n~53). Cross-check target for FL reference-family q_c."""
import sys, math, time, statistics as st
import torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
torch.set_grad_enabled(False)

dev = "cpu"
BETA = 1.0 / 0.55; RHO = 1.200; R = 2.2; RCTX = 2.5
L_H = (0.06 / RHO) ** (1.0 / 3.0); BULK = 0.06; CTR = 2.5 * L_H     # 5x5x5 center cube, half-width 2.5l
NCAV = 6; NW = 12; SWEEPS = 5000; MEAS_EVERY = 50; PLATEAU_FRAC = 0.4
DELTA = 0.12; SWAPS_PER_SWEEP = 6
Q_FIT = 0.5 * math.exp(-((R - 1) / 2.055) ** 3.119)                 # paper Table-I fit target
t_sig = torch.tensor(SIGMA); t_eps = torch.tensor(EPS)
OUT = "reports/logs-2026-07-15/ref_plateau_R22.pt"


def q_center(a, b):
    """Paper-exact center overlap, rho-normalized (self->1, independent-bulk->0.06)."""
    def cset(x):
        mm = (x.abs() < CTR).all(-1)
        ijk = torch.floor((x[mm] + CTR) / L_H).long().clamp(0, 4)
        return set((ijk[:, 0] * 25 + ijk[:, 1] * 5 + ijk[:, 2]).tolist())
    return len(cset(a) & cset(b)) / (RHO * L_H ** 3 * 125)


def pair_e(allX, allS, i, xi, si):
    """Energy of mobile particle i at xi (species si) vs all j!=i (isolated cavity, no PBC). [W]"""
    d = allX - xi[:, None]                                          # [W,Nt,3]
    r2 = (d ** 2).sum(-1)                                           # [W,Nt]
    r2[:, i] = 1e12
    sig = t_sig[si[:, None], allS]; eps = t_eps[si[:, None], allS]
    rc = RCUT_FACTOR * sig
    inv6 = (sig ** 2 / r2.clamp_min(1e-12)) ** 3
    e = 4 * eps * (inv6 ** 2 - inv6)
    s6 = (sig / rc) ** 6
    e = torch.where(r2 < rc ** 2, e - 4 * eps * (s6 ** 2 - s6), torch.zeros_like(e))
    return e.sum(1)


BANK = torch.load("reports/logs-2026-07-15/bulk_ka12_T055_N4096.pt", map_location="cpu", weights_only=False)
frames = BANK["bank"]; L = float(BANK["L"])
print(f"bank: {len(frames)} frames x {frames[0]['x'].shape[0]} replicas, rho={BANK['rho']:.4f} T={BANK['T']}; "
      f"paper fit q~(2.2)={Q_FIT:.3f}", flush=True)
gen = torch.Generator().manual_seed(7)
results = {}; plateaus = []
for ci in range(NCAV):
    fr = frames[-1 - ci % len(frames)]; b = ci % frames[0]["x"].shape[0]
    X, S = fr["x"][b], fr["s"][b]
    c = torch.rand(3, generator=gen) * L
    p = carve(X, S, c, R, L)
    if p["n_in"] < 20:
        continue
    x_ref = _mic(p["x_in"], c, L); s_ref = p["s_in"].long(); n = x_ref.shape[0]
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd = xout[bm]; sb = p["s_out"][bm].long()
    m = bnd.shape[0]
    # W walkers, all start at reference; system = [mobile | frozen boundary], only mobile (0..n-1) move
    allX = torch.cat([x_ref[None].expand(NW, n, 3).clone(), bnd[None].expand(NW, m, 3).clone()], 1)
    allS = torch.cat([s_ref[None].expand(NW, n).clone(), sb[None].expand(NW, m).clone()], 1)
    ts, qs = [], []; t0 = time.time()
    for sw in range(SWEEPS):
        for i in range(n):
            xi_old = allX[:, i]; si = allS[:, i]
            prop = xi_old + DELTA * torch.randn(NW, 3, generator=gen)
            ok = prop.norm(dim=-1) < R
            dU = pair_e(allX, allS, i, prop, si) - pair_e(allX, allS, i, xi_old, si)
            acc = ok & (torch.rand(NW, generator=gen).log() < -BETA * dU)
            allX[acc, i] = prop[acc]
        for _ in range(SWAPS_PER_SWEEP):                           # composition-preserving species swaps
            ia = int(torch.randint(n, (), generator=gen)); ib = int(torch.randint(n, (), generator=gen))
            if int(s_ref[ia]) == int(s_ref[ib]):
                continue
            e_old = pair_e(allX, allS, ia, allX[:, ia], allS[:, ia]) + pair_e(allX, allS, ib, allX[:, ib], allS[:, ib])
            Sp = allS.clone(); Sp[:, ia] = allS[:, ib]; Sp[:, ib] = allS[:, ia]
            e_new = pair_e(allX, Sp, ia, allX[:, ia], Sp[:, ia]) + pair_e(allX, Sp, ib, allX[:, ib], Sp[:, ib])
            a2 = torch.rand(NW, generator=gen).log() < -BETA * (e_new - e_old)
            allS = torch.where(a2[:, None], Sp, allS)
        if sw % MEAS_EVERY == 0:
            q = st.mean([q_center(allX[w, :n], x_ref) - BULK for w in range(NW)])
            ts.append(sw); qs.append(q)
    klo = int(len(qs) * (1 - PLATEAU_FRAC))
    plat_samples = qs[klo:]; plat = st.mean(plat_samples); plat_sd = st.pstdev(plat_samples) / max(len(plat_samples) ** .5, 1)
    plateaus.append(plat)
    results[ci] = {"n": n, "m": m, "ts": ts, "qs": qs, "plateau": plat, "plateau_sd": plat_sd}
    torch.save({"results": results, "Q_FIT": Q_FIT, "R": R}, OUT)
    print(f"cav {ci} (n={n}, bnd={m}): q~(t=0) {qs[0]:.3f} -> PLATEAU {plat:+.3f}+-{plat_sd:.3f}  "
          f"[paper {Q_FIT:.3f}]  ({time.time()-t0:.0f}s)", flush=True)
if plateaus:
    print(f"\n=== R={R} REFERENCE-BASIN plateau: mean {st.mean(plateaus):+.3f} +- "
          f"{st.pstdev(plateaus)/max(len(plateaus)**.5,1):.3f} over {len(plateaus)} cavities "
          f"| paper Table-I fit {Q_FIT:.3f} ===", flush=True)
print(f"saved -> {OUT}", flush=True)
