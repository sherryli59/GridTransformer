"""ROOT-CAUSE test for 'clash gets worse at large R'. Hypothesis: the boundary is a FIXED, fully-visible
constraint (every interior slot attends to all boundary, subject to bnd_cutoff) so it is easy to avoid;
the interior is GROWN autoregressively and mutually constrained, so it accumulates AR-packing clashes that
grow with R (more interior particles, longer sequence, more early-particle over-packing that later
particles cannot undo). PREDICTION: interior<->boundary clash stays LOW and ~flat in R; interior<->interior
clash GROWS with R and dominates. Also split interior<->interior by AR order (partner earlier/later in
morton scaffold order) to expose the packing directionality.

Full-cavity AR generation (allmask, base categorical, NO min_sep) at R in {1.6,2.0,2.5,3.0}, N=4096 model,
several cavities each. Clash = pair distance r_ij < CUT * sigma_ij (deep in the LJ repulsive wall).
Reports clashes per interior particle by partner class, and the mean interior count, vs R."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; BIGL = 100.0; M = 16; NCAV = 10; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@torch.no_grad()
def clash_decomp(R, gen):
    """Returns per-particle clash counts averaged over cavities+samples, split by partner class."""
    cb, ce, cl, npart = [], [], [], []      # clash w/ boundary, earlier-interior, later-interior; n_interior
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)   # morton-ordered interior
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; nb = bnd.shape[0]
        allmask = torch.ones(n, dtype=torch.bool, device=dev)
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     allmask, bnd, sb, R, gen=gen)             # [M,n,3]
        for k in range(M):
            xi = Xg[k]; si = Sg[k]                                            # interior (morton order)
            # interior-interior
            dii = torch.cdist(xi, xi); sii = SIG[si[:, None], si[None, :]]
            eye = torch.eye(n, dtype=torch.bool, device=dev)
            clash_ii = (dii < CUT * sii) & ~eye                              # [n,n]
            tri_earlier = torch.tril(torch.ones(n, n, device=dev, dtype=torch.bool), -1)  # j<i => partner earlier
            ce.append(float((clash_ii & tri_earlier).sum(1).float().mean()))            # per-particle earlier
            cl.append(float((clash_ii & tri_earlier.T).sum(1).float().mean()))          # later
            # interior-boundary
            dib = torch.cdist(xi, bnd); sib = SIG[si[:, None], sb[None, :]]
            cb.append(float((dib < CUT * sib).sum(1).float().mean()))
            npart.append(n)
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cb), st.mean(ce), st.mean(cl), st.mean(npart)


print(f"=== clash-per-interior-particle vs R (full AR regen, CUT={CUT}*sigma, {NCAV} cav x {M} samp) ===", flush=True)
print(f"{'R':>4} {'n_int':>6} | {'clash/bnd':>9} {'clash/int-earlier':>17} {'clash/int-later':>15} {'clash/int-TOT':>13}", flush=True)
for R in (1.6, 2.0, 2.5, 3.0):
    gen = torch.Generator(device=dev).manual_seed(0)
    cb, ce, cl, npart = clash_decomp(R, gen)
    print(f"{R:>4} {npart:>6.1f} | {cb:>9.3f} {ce:>17.3f} {cl:>15.3f} {ce+cl:>13.3f}", flush=True)
print("\nPREDICTION: clash/bnd low+flat; clash/int-TOT grows with R (AR packing). "
      "earlier==later by symmetry of counting (each clash counted from both ends).", flush=True)
