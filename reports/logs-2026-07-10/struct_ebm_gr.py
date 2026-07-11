"""Structural validation beyond the clash cutoff: block-centered, species-resolved g(r).
For each regenerated block particle (species a), histogram distances to ALL other particles (species b),
normalise to g_ab(r) (2D ideal-gas shell). Ground truth = the TRUE data block in the true config. Compare
data vs factorized vs EBM-potential -> does the learned potential reproduce the first-shell / g_BB structure
the hard cutoff can't see? Saves a 3-panel plot + prints first-peak height and L2 mismatch to data."""
import sys, math
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_localframe_ebm import KALocalFrameEBM
from liquid_coupling_flow.ka_localframe import KALocalFrameModel
from liquid_coupling_flow.ka_gridformer import _wrap_pm

dev = "cuda"; N = 100; K = 8; TRIALS = 24; NCONF = 40
RMAX, NB = 3.5, 70
ebm_path = sys.argv[1] if len(sys.argv) > 1 else "liquid_coupling_flow/artifacts/ka_localframe_ebm_N100.pt"
fac_path = sys.argv[2] if len(sys.argv) > 2 else "liquid_coupling_flow/artifacts/ka_localframe_N100_20k.pt"
cke = torch.load(ebm_path, map_location=dev, weights_only=False)
me = KALocalFrameEBM(rho=cke["rho"], n_bins=cke["n_bins"], knn=cke["knn"]).to(dev); me.load_state_dict(cke["state_dict"], strict=False); me.eval()
ckf = torch.load(fac_path, map_location=dev, weights_only=False)
mf = KALocalFrameModel(rho=ckf["rho"], n_bins=ckf["n_bins"], knn=ckf["knn"]).to(dev); mf.load_state_dict(ckf["state_dict"], strict=False); mf.eval()
ref = torch.load(f"liquid_coupling_flow/artifacts/ka_reference_N{N}.pt", map_location=dev, weights_only=False)
data = ref["x"].to(dev); s0 = ref["s"].to(dev).long(); L = ref["L"]
sc = me.geo._scaffold(N, dev); arc = me._arc_scale(N)
bc = me._bin_center(torch.arange(me.n_bins, device=dev))
edges = torch.linspace(0, RMAX, NB + 1, device=dev); dr = float(edges[1] - edges[0]); rc = (edges[:-1] + edges[1:]) / 2


@torch.no_grad()
def sample_ebm(pos0, sp0, k, B):
    pos = pos0[None].expand(B, N, 2).clone(); sp = sp0[None].expand(B, N).clone()
    for j in range(N - k, N):
        h, origin, nr, nsp, val = me._step_ebm(pos, sp, sc[j], j, L)
        la = F.softmax(me.head_a(h), -1); ba = torch.multinomial(la, 1).squeeze(-1)
        V = me._V_b(ba, nr, nsp, val, sp[:, j], arc, bc)
        lb = F.softmax(me.head_b(h + me.bin_a_emb(ba)) - V, -1); bb = torch.multinomial(lb, 1).squeeze(-1)
        a = me._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * me.bin_w
        b = me._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * me.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
    return pos


@torch.no_grad()
def sample_fac(pos0, sp0, k, B):
    pos = pos0[None].expand(B, N, 2).clone(); sp = sp0[None].expand(B, N).clone()
    for j in range(N - k, N):
        h, origin = mf._step(pos, sp, sc[j], j, L)
        ba = torch.multinomial(F.softmax(mf.head_a(h), -1), 1).squeeze(-1)
        bb = torch.multinomial(F.softmax(mf.head_b(h + mf.bin_a_emb(ba)), -1), 1).squeeze(-1)
        a = mf._bin_center(ba) + (torch.rand(B, device=dev) - 0.5) * mf.bin_w
        b = mf._bin_center(bb) + (torch.rand(B, device=dev) - 0.5) * mf.bin_w
        pos[:, j] = torch.remainder(origin + torch.stack([a, b], -1) * arc, L)
    return pos


def accum(pos, sp, blk_slice, H, C):
    """Add block-centered pair counts. pos [B,N,2], sp [N], blk_slice = indices of block particles.
    H[a][b] histogram counts; C[a] number of block-particle centres of species a."""
    B = pos.shape[0]
    nb_sp = torch.tensor([int((sp == 0).sum()), int((sp == 1).sum())], device=dev)
    for a in (0, 1):
        for b in (0, 1):
            rho_b = nb_sp[b].float() / (L * L)
            centres = [i for i in blk_slice if int(sp[i]) == a]
            if not centres:
                continue
            ci = torch.tensor(centres, device=dev)
            d = pos[:, ci, None, :] - pos[:, None, :, :]                       # [B,|a|,N,2]
            d = d - L * torch.round(d / L); dd = d.norm(dim=-1)               # [B,|a|,N]
            mask_b = (sp[None, None, :] == b)
            for cc, gi in enumerate(ci):
                dd[:, cc, gi] = 1e9                                            # exclude self
            dvals = dd[mask_b.expand(B, len(ci), N)]
            H[a][b] += torch.histc(dvals[dvals < RMAX], NB, 0, RMAX)
            C[a] += B * len(ci)
            # store rho_b weight via C? normalise later using global rho_b (same for all) -> keep separate
    return nb_sp


def gr_from(H, C, nb_sp):
    g = {}
    for a in (0, 1):
        for b in (0, 1):
            rho_b = nb_sp[b].float() / (L * L)
            shell = 2 * math.pi * rc * dr                                     # 2D ideal-gas shell area
            g[(a, b)] = (H[a][b] / max(C[a], 1)) / (rho_b * shell)
    return g


def run(sampler, label):
    H = {a: {b: torch.zeros(NB, device=dev) for b in (0, 1)} for a in (0, 1)}; C = {0: 0, 1: 0}; nb = None
    for ci in range(NCONF):
        order = me.geo._curve_order(data[ci:ci + 1], N)[0]; pos0 = data[ci][order]; sp0 = s0[order]
        if sampler is None:
            pos = pos0[None]                                                  # TRUE data block
        else:
            pos = sampler(pos0, sp0, K, TRIALS)
        nb = accum(pos, sp0, list(range(N - K, N)), H, C)
    print(f"[{label}] centres A={C[0]} B={C[1]}", flush=True)
    return gr_from(H, C, nb)


g_data = run(None, "data")
g_fac = run(sample_fac, "factorized")
g_ebm = run(sample_ebm, "EBM-potential")

names = {(0, 0): "g_AA", (0, 1): "g_AB", (1, 1): "g_BB"}
rc_np = rc.cpu().numpy()
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for i, key in enumerate([(0, 0), (0, 1), (1, 1)]):
    ax[i].plot(rc_np, g_data[key].cpu(), "k-", lw=2.2, label="data (true block)")
    ax[i].plot(rc_np, g_fac[key].cpu(), "C3--", lw=1.6, label="factorized")
    ax[i].plot(rc_np, g_ebm[key].cpu(), "C0-", lw=1.6, label="EBM-potential")
    ax[i].set_title(names[key]); ax[i].set_xlabel("r"); ax[i].axhline(1, color="gray", lw=0.5)
    ax[i].legend(fontsize=8)
    # scalar diagnostics over the first two shells r in [0.6, 2.0]
    win = (rc > 0.6) & (rc < 2.0)
    for gg, tag in ((g_fac, "fac"), (g_ebm, "ebm")):
        l2 = float(((gg[key][win] - g_data[key][win]) ** 2).mean().sqrt())
        pk_d = float(g_data[key].max()); pk = float(gg[key].max())
        print(f"  {names[key]} {tag}: L2(first shells)={l2:.3f}  peak {pk:.2f} vs data {pk_d:.2f}", flush=True)
ax[0].set_ylabel("g(r)")
plt.tight_layout()
out = "reports/logs-2026-07-10/ebm_block_gr.png"
plt.savefig(out, dpi=110); print(f"\nsaved {out}", flush=True)
