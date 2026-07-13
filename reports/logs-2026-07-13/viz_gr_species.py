"""Per-species radial distribution g_AA, g_AB, g_BB for generated cavity blocks: AR vs EGNN-flow vs DATA.

For each held cavity (K=8, R=2.0, M draws), accumulate pair distances from BLOCK particles (centers)
to BLOCK+CAGE particles (neighbors), split by species pair. Normalize as a true g(r):
  g_ab(r) = <hist_ab(r)> / (N_centers_a * rho_b * 4 pi r^2 dr),
rho_b = bulk number density of species b (from the N=4096 box: rho=1.1486, 80:20 -> rho_A, rho_B).
Same normalization for AR/flow/data -> peak positions & heights are directly comparable. The cavity is
finite (R+cutoff), so g(r) is trustworthy only out to ~R (shaded guide); beyond that the shell empties.
"""
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 8; N_CAGE = 48; DUMMY_R = 50.0; NCAV = 12
C_DATA, C_AR, C_FLOW = "#1B9E77", "#D95F02", "#7570B3"

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                              map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_gated_big_best.pt",
                map_location=dev, weights_only=False)
a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"],
                       ode_rtol=1e-5, ode_atol=1e-5, max_steps=1000).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
rho = X.shape[1] / L ** 3
xB = float((S == 1).float().mean()); rho_sp = {0: rho * (1 - xB), 1: rho * xB}
gen = torch.Generator(device=dev).manual_seed(7)

edges = torch.linspace(0.6, R + RCTX, 46); ctr = 0.5 * (edges[1:] + edges[:-1]); dr = float(edges[1] - edges[0])
shell = 4 * torch.pi * ctr ** 2 * dr
PAIRS = [(0, 0, "AA"), (0, 1, "AB"), (1, 1, "BB")]
hist = {tag: {ab: torch.zeros(len(ctr)) for _, _, ab in PAIRS} for tag in ("data", "ar", "flow")}
ncenter = {tag: {ab: 0 for _, _, ab in PAIRS} for tag in ("data", "ar", "flow")}


def fixed_size_cage(cage_x, sp_cage, centroid, n_cage):
    Mn, m = cage_x.shape[0], cage_x.shape[1]
    if m >= n_cage:
        idx = torch.topk((cage_x[0] - centroid).norm(dim=-1), n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad = n_cage - m
    dpos = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cage_x.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cage_x, dpos], 1), torch.cat([sp_cage, torch.zeros(Mn, pad, dtype=sp_cage.dtype, device=dev)], 1)


def accum(tag, blk_x, blk_s, nbr_x, nbr_s):
    """blk_x[M,K,3] centers; nbr_x[M,P,3] neighbors (block+cage). Split by species pair, drop self (r<1e-6)."""
    for sa, sb, ab in PAIRS:
        for mm in range(blk_x.shape[0]):
            ca = blk_x[mm][blk_s[mm] == sa]
            nb = nbr_x[mm][nbr_s[mm] == sb]
            if ca.shape[0] == 0 or nb.shape[0] == 0:
                continue
            d = torch.cdist(ca, nb).flatten()
            d = d[d > 1e-6]
            hist[tag][ab] += torch.histc(d.cpu(), bins=len(ctr), min=float(edges[0]), max=float(edges[-1]))
            ncenter[tag][ab] += ca.shape[0]


ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    xo_ar, so_ar, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), xo_ar[:, ~blk]], 1)
    cage_s = torch.cat([sb[None].expand(M, -1), so_ar[:, ~blk]], 1)
    cfx, cfs = fixed_size_cage(cage_x, cage_s, xo[blk].mean(0), N_CAGE)
    y_blk, _ = flow.flow(xo_ar[:, blk], cfx, so_ar[:, blk], cfs, reverse=False)
    ar_blk, sp_blk = xo_ar[:, blk], so_ar[:, blk]
    data_blk = xo[None, blk].expand(M, K, 3); data_s = so[None, blk].expand(M, K)
    accum("data", data_blk, data_s, torch.cat([data_blk, cage_x], 1), torch.cat([data_s, cage_s], 1))
    accum("ar", ar_blk, sp_blk, torch.cat([ar_blk, cage_x], 1), torch.cat([sp_blk, cage_s], 1))
    accum("flow", y_blk, sp_blk, torch.cat([y_blk, cage_x], 1), torch.cat([sp_blk, cage_s], 1))
    ncav += 1
    if ncav >= NCAV:
        break


def gr(tag, ab, sb):
    nb = ncenter[tag][ab]
    return hist[tag][ab] / (nb * rho_sp[sb] * shell) if nb else torch.zeros(len(ctr))


fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6), sharex=True)
for ax, (sa, sb, ab) in zip(axes, PAIRS):
    for tag, col, lw, z in (("data", C_DATA, 2.4, 3), ("ar", C_AR, 1.8, 2), ("flow", C_FLOW, 1.8, 2)):
        ax.plot(ctr.numpy(), gr(tag, ab, sb).numpy(), color=col, lw=lw, zorder=z,
                label={"data": "DATA (truth)", "ar": "AR proposal", "flow": "EGNN-flow"}[tag])
    ax.axvspan(R, float(edges[-1]), color="#000", alpha=0.045, zorder=0)
    ax.axhline(1.0, color="#BBB", lw=0.8, ls=":", zorder=1)
    ax.set_title(f"$g_{{{ab}}}(r)$", fontsize=13)
    ax.set_xlabel("r / $\\sigma_{AA}$"); ax.set_xlim(float(edges[0]), float(edges[-1]))
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("g(r)")
axes[0].legend(frameon=False, fontsize=10)
axes[2].annotate("shaded r > R:\ncavity shell empties", (R + 0.15, axes[2].get_ylim()[1] * 0.82),
                 fontsize=8, color="#666")
fig.suptitle(f"Per-species g(r) of generated cavity blocks vs data  ({ncav} held cavities x M={M}, "
             f"K={K}, R={R}, pos_temp={PT}; flow=gated_big step {ck['step']})", fontsize=12)
fig.tight_layout(rect=[0, 0, 1, 0.94])
out = "reports/logs-2026-07-13/viz_gr_species.png"
fig.savefig(out, dpi=160)
torch.save({"ctr": ctr, "hist": hist, "ncenter": ncenter, "rho_sp": rho_sp, "ncav": ncav},
           "reports/logs-2026-07-13/viz_gr_species.pt")
print(f"saved -> {out}", flush=True)
for _, sb, ab in PAIRS:
    pk = lambda t: float(ctr[gr(t, ab, sb).argmax()])
    print(f"  g_{ab}: peak r  data {pk('data'):.2f}  ar {pk('ar'):.2f}  flow {pk('flow'):.2f}  "
          f"| height data {gr('data', ab, sb).max():.2f} ar {gr('ar', ab, sb).max():.2f} flow {gr('flow', ab, sb).max():.2f}", flush=True)
