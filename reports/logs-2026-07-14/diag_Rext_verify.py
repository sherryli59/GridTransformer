"""Did the R-extended retrain fix the OOD proposal at R>3.0? Clash/particle + capped energy vs R, full-regen,
for OLD block-cond (radii<=3.0) vs NEW R-extended block-cond (radii<=4.5). Both use_demand=False (clean
isolation of the R-extension). If NEW clash < OLD at R>3.0 (esp R>=3.5, the OOD regime), the retrain closed
the OOD proposal gap. M=6, 4 cav."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS

dev = "cuda"; RCTX = 2.5; M = 6; NCAV = 4; CUT = 0.85; CAP = 5.0
SIG = torch.tensor(SIGMA, device=dev); EP = torch.tensor(EPS, device=dev)
OLD = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
NEW = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_Rext_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


@torch.no_grad()
def probe(m, R, gen):
    cl, en = [], []
    ncav = 0
    for ci in range(16):
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
            inv6 = (sg ** 2 / d2) ** 3; en.append(float((4 * eps * (inv6 ** 2 - inv6)).clamp(0, CAP).sum(1).mean()) / n)
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cl), st.mean(en)


mo, mn = load(OLD), load(NEW)
print("=== R-extension verify: clash/p (OLD radii<=3.0 vs NEW radii<=4.5 block-cond) ===", flush=True)
print(f"{'R':>4} | {'OLD clash':>9} {'NEW clash':>9} | {'OLD capE/p':>10} {'NEW capE/p':>10} | regime", flush=True)
for R in (2.5, 3.0, 3.5, 4.0, 4.5):
    co, eo = probe(mo, R, torch.Generator(device=dev).manual_seed(0))
    cn, en = probe(mn, R, torch.Generator(device=dev).manual_seed(0))
    reg = "in-dist(both)" if R <= 3.0 else ("NEW in-dist" if R <= 4.5 else "OOD both")
    print(f"{R:>4} | {co:>9.3f} {cn:>9.3f} | {eo:>10.2f} {en:>10.2f} | {reg}", flush=True)
