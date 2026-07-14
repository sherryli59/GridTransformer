"""EARLY RL-quality diagnostic (mid-run snapshot). The phase-2 RL run declashes on the R-EXTENDED base;
its whole point is better large-R proposals (xi_PTS~3.8 > trained R<=3.0, now extended to 4.5). The training
log only reports in-dist blob16. Here: full-regen (allmask) clash/p + capped-repulsive energy/p across
R in {2.5,3.0,3.5,4.0,4.5}, comparing the RL WARM-START base (blockcond_knn24_Rext) vs the LIVE RL snapshot
(rl_Rext_knn24.pt). Both instantiated use_demand=True so the demand params (zero-init & inert in base) are a
clean isolation of what RL added. If RL's clash drop HOLDS or GROWS at R>=3.5, the OOD proposal is improving
where it matters. M=6, 6 cav."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS

dev = "cuda"; RCTX = 2.5; M = 6; NCAV = 6; CUT = 0.85; CAP = 5.0
SIG = torch.tensor(SIGMA, device=dev); EP = torch.tensor(EPS, device=dev)
BASE = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_Rext_best.pt"
RL = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24.pt"  # live snapshot
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


@torch.no_grad()
def probe(m, R, gen):
    cl, en, npart = [], [], []
    ncav = 0
    for ci in range(24):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 8:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(), allm, bnd, sb, R, gen=gen)
        for k in range(M):
            xi, si = Xg[k], Sg[k]; d = torch.cdist(xi, xi); sg = SIG[si[:, None], si[None, :]]
            eye = torch.eye(n, dtype=torch.bool, device=dev)
            cl.append(float(((d < CUT * sg) & ~eye).sum(1).float().mean()))
            eps = EP[si[:, None], si[None, :]]; d2 = d.clamp_min(1e-6) ** 2; d2 = d2.masked_fill(eye, 1e12)
            inv6 = (sg ** 2 / d2) ** 3; en.append(float((4 * eps * (inv6 ** 2 - inv6)).clamp(0, CAP).sum(1).mean()))
        npart.append(n); ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cl), st.mean(en) / st.mean(npart), st.mean(npart)


mb, mr = load(BASE), load(RL)
print("=== EARLY RL quality: base(Rext warm-start) vs LIVE RL snapshot, full-regen vs R ===", flush=True)
print(f"{'R':>4} {'n':>4} | {'base cl/p':>9} {'RL cl/p':>8} {'dcl%':>6} | {'base E/p':>8} {'RL E/p':>7} | regime", flush=True)
for R in (2.5, 3.0, 3.5, 4.0, 4.5):
    cb, eb, nb = probe(mb, R, torch.Generator(device=dev).manual_seed(0))
    cr, er, nr = probe(mr, R, torch.Generator(device=dev).manual_seed(0))
    dpct = 100.0 * (cr - cb) / max(cb, 1e-9)
    reg = "in-dist" if R <= 3.0 else ("Rext-range" if R <= 3.8 else ">xi_PTS")
    print(f"{R:>4} {nb:>4.0f} | {cb:>9.3f} {cr:>8.3f} {dpct:>+5.0f}% | {eb:>8.2f} {er:>7.2f} | {reg}", flush=True)
