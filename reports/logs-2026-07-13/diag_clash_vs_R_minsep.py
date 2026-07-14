"""FIX-CHECK for the interior-interior AR-packing clash that grows with R (root cause from
diag_clash_vs_R_decomp): does the exact min_sep declash mask suppress it? Compares interior clash/particle
vs R for base sampling vs min_sep=0.9 (hard-exclude any bin within 0.9*sigma of a cage neighbour). The mask
is exactness-preserving (sample==score), so it declashes WITHOUT breaking the tempered density. Plots
boundary + interior(base) + interior(min_sep) vs R. If min_sep flattens the interior curve, the declasher
handles the SYMPTOM at inference; the underlying acceptance floor (structural) is a separate matter."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; M = 16; NCAV = 10; CUT = 0.85; RADII = (1.6, 2.0, 2.5, 3.0)
SIG = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@torch.no_grad()
def clash(R, gen, min_sep):
    cb, cii = [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; allmask = torch.ones(n, dtype=torch.bool, device=dev)
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     allmask, bnd, sb, R, gen=gen, min_sep=min_sep)
        for k in range(M):
            xi, si = Xg[k], Sg[k]
            dii = torch.cdist(xi, xi); sii = SIG[si[:, None], si[None, :]]
            eye = torch.eye(n, dtype=torch.bool, device=dev)
            cii.append(float(((dii < CUT * sii) & ~eye).sum(1).float().mean()))
            dib = torch.cdist(xi, bnd); sib = SIG[si[:, None], sb[None, :]]
            cb.append(float((dib < CUT * sib).sum(1).float().mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(cb), st.mean(cii)


base_b, base_i, ms_b, ms_i = [], [], [], []
print(f"{'R':>4} | {'bnd':>6} {'int-base':>9} {'int-minsep0.9':>13}", flush=True)
for R in RADII:
    b0, i0 = clash(R, torch.Generator(device=dev).manual_seed(0), None)
    b1, i1 = clash(R, torch.Generator(device=dev).manual_seed(0), 0.9)
    base_b.append(b0); base_i.append(i0); ms_b.append(b1); ms_i.append(i1)
    print(f"{R:>4} | {b0:>6.3f} {i0:>9.3f} {i1:>13.3f}", flush=True)

fig, ax = plt.subplots(figsize=(6.2, 4.4))
ax.plot(RADII, base_i, "o-", color="C3", label="interior-interior (base)")
ax.plot(RADII, ms_i, "s-", color="C0", label="interior-interior (min_sep=0.9, cage=8)")
ax.plot(RADII, base_b, "^--", color="C7", label="interior-boundary (base)")
ax.set_xlabel("cavity radius R"); ax.set_ylabel(f"clashes / interior particle (r<{CUT}sigma)")
ax.set_title("Clash source vs R: interior AR-packing GROWS; boundary FALLS;\nmin_sep(cage=8) does NOT help (cage-starved: needs knn>=24)")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
out = "reports/logs-2026-07-13/clash_vs_R_minsep.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)
