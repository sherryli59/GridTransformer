"""3D cavity-block species-resolved g(r): for each regenerated block particle (species a), distances to
ALL other cavity particles (interior + boundary, species b), 3D ideal-gas shell norm. Ground truth = the
TRUE carved interior block. Compare data vs frozen-phi base vs EBM-potential -> does the learned potential
reproduce first-shell / g_BB structure in 3D? Saves a 3-panel plot + peak/L2 diagnostics.
argv: ebm_ckpt base_ckpt."""
import sys, math
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; R = 2.0; R_CTX = 2.5; K = 8; TRIALS = 16; NCONF = 40; RMAX, NB = 3.0, 60
ebm_p = sys.argv[1] if len(sys.argv) > 1 else "liquid_coupling_flow/artifacts/ka3d_cavity_ebm.pt"
base_p = sys.argv[2] if len(sys.argv) > 2 else "liquid_coupling_flow/artifacts/ka3d_cavity_base.pt"


def load(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


me, mb = load(ebm_p), load(base_p)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(3)
edges = torch.linspace(0, RMAX, NB + 1, device=dev); dr = float(edges[1] - edges[0]); rc = (edges[:-1] + edges[1:]) / 2
vol = (4.0 / 3.0) * math.pi * (R + R_CTX) ** 3


def cavity(ci, c):
    p = carve(X[ci], S[ci], c, R, L)
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + R_CTX)
    return xo, so, xout[bm], p["s_out"][bm], p["n_in"]


def blob(n):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    m = torch.zeros(n, dtype=torch.bool, device=dev); m[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return m


def accum(blk_x, blk_s, all_x, all_s, H, C, nb_sp):
    for a in (0, 1):
        ca = blk_x[blk_s == a]
        if ca.shape[0] == 0:
            continue
        dmat = torch.cdist(ca, all_x)                                     # [|a|, P]
        for b in (0, 1):
            dv = dmat[:, all_s == b]
            dv = dv[(dv > 1e-4) & (dv < RMAX)]                            # drop self (~0) + out of range
            H[a][b] += torch.histc(dv, NB, 0, RMAX); C[a] += ca.shape[0]
    for b in (0, 1):
        nb_sp[b] += int((all_s == b).sum())


def gr(H, C, nb_sp, ncfg):
    out = {}
    for a in (0, 1):
        for b in (0, 1):
            rho_b = (nb_sp[b] / ncfg) / vol
            out[(a, b)] = (H[a][b] / max(C[a], 1)) / (rho_b * 4 * math.pi * rc ** 2 * dr).clamp_min(1e-12)
    return out


def run(model):
    H = {a: {b: torch.zeros(NB, device=dev) for b in (0, 1)} for a in (0, 1)}; C = {0: 0, 1: 0}; nb = {0: 0, 1: 0}
    for ci in range(900, 900 + NCONF):
        c = torch.rand(3, generator=gen, device=dev) * L
        xo, so, bnd, s_bnd, n = cavity(ci, c)
        if n < 10:
            continue
        blk = blob(n)
        allx = torch.cat([xo, bnd], 0); alls = torch.cat([so, s_bnd], 0)
        if model is None:
            bx, bs = xo[blk], so[blk]
        else:
            xn, sn, _ = model.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
            bx, bs = xn[blk], sn[blk]; allx = torch.cat([xn, bnd], 0); alls = torch.cat([sn, s_bnd], 0)
        accum(bx, bs, allx, alls, H, C, nb)
    return gr(H, C, nb, NCONF)


g_data, g_base, g_ebm = run(None), run(mb), run(me)
names = {(0, 0): "g_AA", (0, 1): "g_AB", (1, 1): "g_BB"}; rc_np = rc.cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, key in enumerate([(0, 0), (0, 1), (1, 1)]):
    ax[i].plot(rc_np, g_data[key].cpu(), "k-", lw=2.2, label="data (true block)")
    ax[i].plot(rc_np, g_base[key].cpu(), "C3--", lw=1.6, label="base (no potential)")
    ax[i].plot(rc_np, g_ebm[key].cpu(), "C0-", lw=1.6, label="EBM-potential")
    ax[i].set_title(names[key]); ax[i].set_xlabel("r"); ax[i].axhline(1, color="gray", lw=0.5); ax[i].legend(fontsize=8)
    win = (rc > 0.7) & (rc < 1.8)
    for gg, tag in ((g_base, "base"), (g_ebm, "ebm")):
        l2 = float(((gg[key][win] - g_data[key][win]) ** 2).mean().sqrt())
        print(f"  {names[key]} {tag}: L2(first shell)={l2:.3f}  peak {float(gg[key].max()):.2f} vs data {float(g_data[key].max()):.2f}", flush=True)
ax[0].set_ylabel("g(r)"); plt.tight_layout()
out = "reports/logs-2026-07-11/ebm3d_cavity_gr.png"
plt.savefig(out, dpi=110); print(f"\nsaved {out}", flush=True)
