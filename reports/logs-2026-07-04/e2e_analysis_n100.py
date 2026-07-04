"""N=100 e2e analysis (matches the N=256 decorrelation + g(r) panels). MH arms N/A here (never run).
full pipeline = flow->MALA+block8 (ka_e2e_traj_mala_block8_N100.pt). PT dashed. Saves plots + arrays."""
import os, math, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; ART = "liquid_coupling_flow/artifacts"; LOG = "reports/logs-2026-07-04"; N = 100
MALA = {
    "flow → MALA+block8 (full pipeline)": ("ka_e2e_traj_mala_block8_N100.pt", "C0"),
    "uniform → MALA+block8":              ("ka_e2e_traj_mala_block8_uniform_N100.pt", "C1"),
    "flow → MALA+random":                 ("ka_e2e_traj_mala_random_N100.pt", "C3"),
}
save = {}
T0 = 4000; NLAG = 1200

def u_acf(UoN, t0, nlag):
    W = UoN[t0:].astype(np.float64); Tw, B = W.shape; t = np.arange(Tw); rhos = []
    for b in range(B):
        v = W[:, b]
        if not np.isfinite(v).all(): continue
        A = np.polyfit(t, v, 1); v = v - (A[0]*t + A[1])
        f = np.fft.rfft(v, n=2*Tw); ac = np.fft.irfft(f*np.conj(f))[:nlag+1]; ac /= (Tw - np.arange(nlag+1))
        if ac[0] <= 0: continue
        rhos.append(ac/ac[0])
    R = np.array(rhos); return R.mean(0), R.shape[0]

def s_acf(s_traj, t0, nlag):
    W = s_traj[t0:].astype(np.float64)*2 - 1; Tw, B, Np = W.shape; num = np.zeros(nlag+1); den = 0.0
    for b in range(B):
        v = W[:, b, :]
        if not np.isfinite(v).all(): continue
        v = v - v.mean(0, keepdims=True)
        f = np.fft.rfft(v, n=2*Tw, axis=0); ac = np.fft.irfft(f*np.conj(f), axis=0)[:nlag+1]; ac /= (Tw - np.arange(nlag+1))[:, None]
        num += ac.sum(1); den += ac[0].sum()
    return num/den

def tau_int(rho, c=6.0):
    for M in range(1, len(rho)):
        tau = 1 + 2*np.sum(rho[1:M+1])
        if M >= c*tau: return tau
    return 1 + 2*np.sum(rho[1:])

uacf, sacf, tauU, tauS = {}, {}, {}, {}
for lab, (f, c) in MALA.items():
    d = torch.load(f"{ART}/{f}", map_location="cpu", weights_only=False)
    ru, nch = u_acf(d["UoN"].numpy(), T0, NLAG); uacf[lab] = ru; tauU[lab] = tau_int(ru)
    rs = s_acf(d["s"].numpy(), T0, min(NLAG, d["s"].shape[0]-T0-1)); sacf[lab] = rs; tauS[lab] = tau_int(rs)
    print(f"[N100 decorr] {lab:40s} nchain={nch} tau_U={tauU[lab]:.1f} tau_s={tauS[lab]:.1f}")
save.update(uacf=uacf, sacf=sacf, tauU=tauU, tauS=tauS, T0=T0)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for lab, (f, c) in MALA.items():
    ax[0].plot(np.arange(len(uacf[lab]))[:400], uacf[lab][:400], color=c, lw=1.8, label=f"{lab.split(' (')[0]}  $\\tau_U$={tauU[lab]:.0f}")
    ax[1].plot(np.arange(len(sacf[lab]))[:1000], sacf[lab][:1000], color=c, lw=1.8, label=f"{lab.split(' (')[0]}  $\\tau_s$={tauS[lab]:.0f}")
ax[0].axhline(0, color="gray", lw=0.6); ax[0].set_xlabel("lag (iterations)"); ax[0].set_ylabel(r"$\rho_U(\ell)$")
ax[0].set_title(f"(a) energy autocorrelation, N={N} (detrended, it>{T0})"); ax[0].legend(fontsize=8)
ax[1].axhline(0, color="gray", lw=0.6); ax[1].set_xlabel("lag (iterations)"); ax[1].set_ylabel(r"$\rho_s(\ell)$ (species)")
ax[1].set_title(f"(b) species-label autocorrelation, N={N}"); ax[1].legend(fontsize=8)
fig.tight_layout(); p2 = f"{ART}/ka_e2e_decorrelation_N100.png"; fig.savefig(p2, dpi=130); plt.close(fig)
print("SAVED", os.path.abspath(p2))

# ---- g(r) overlay ----
def gr_batched(x, s, L, rmax, nbins, pair, bs=256):
    a, b = pair; edges = torch.linspace(0, rmax, nbins+1); centers = 0.5*(edges[:-1]+edges[1:])
    shell = math.pi*(edges[1:]**2 - edges[:-1]**2); counts = torch.zeros(nbins); norm = 0.0; Np = x.shape[1]
    for i in range(0, x.shape[0], bs):
        xb = x[i:i+bs].to(dev); sb = (s if s.dim() == 1 else s[i:i+bs]).to(dev)
        if sb.dim() == 1: sb = sb[None].expand(xb.shape[0], Np)
        diff = xb[:, :, None, :] - xb[:, None, :, :]; diff = diff - L*torch.round(diff/L)
        r = torch.sqrt((diff**2).sum(-1) + 1e-12); ma = (sb == a); mb = (sb == b)
        eye = torch.eye(Np, dtype=torch.bool, device=dev)[None]; pm = ma[:, :, None] & mb[:, None, :] & (~eye)
        dd = r[pm]; dd = dd[dd < rmax].cpu(); counts += torch.histc(dd, bins=nbins, min=0.0, max=rmax)
        na = ma.sum(1).float(); nb_ = mb.sum(1).float(); norm += float(((na*(na-1) if a == b else na*nb_)/(L**2)).sum().cpu())
    return centers.numpy(), (counts/(norm*shell)).numpy()

def finite_finals(f):
    d = torch.load(f"{ART}/{f}", map_location="cpu", weights_only=False); x = d["x_final"]; s = d["s_final"]
    good = torch.isfinite(x).all(dim=(1, 2)); return x[good], s[good]

pt = torch.load(f"{ART}/ka_reference_N100.pt", map_location="cpu", weights_only=False)
xpt, spt, L = pt["x"], pt["s"], pt["L"]
gen = torch.load(f"{ART}/ka_e2e_gen_N100.pt", map_location="cpu", weights_only=False)
xg, sg = gen["x"], gen["s"]
xf, sf = finite_finals("ka_e2e_traj_mala_block8_N100.pt")
xu, su = finite_finals("ka_e2e_traj_mala_block8_uniform_N100.pt")
idx = torch.randperm(xpt.shape[0])[:3000]
arms = [("PT ref (dashed)", xpt[idx], spt, dict(color="k", ls="--", lw=2.2)),
        ("full pipeline (flow→MALA+block8)", xf, sf, dict(color="C0", lw=1.8)),
        ("uniform→MALA+block8", xu, su, dict(color="C1", lw=1.6)),
        ("raw egnn flow (gen)", xg, sg, dict(color="C7", lw=1.4, alpha=0.9))]
gr_data = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for j, (pair, name) in enumerate([((0, 0), "g_AA"), ((0, 1), "g_AB"), ((1, 1), "g_BB")]):
    for lab, xx, ss, st in arms:
        cx, gy = gr_batched(xx, ss, L, 5.0, 150, pair); ax[j].plot(cx, gy, label=lab, **st); gr_data[f"{name}|{lab}"] = (cx, gy)
    ax[j].set_title(f"{name} (N={N})"); ax[j].set_xlabel("r"); ax[j].axhline(1, color="gray", lw=0.5)
ax[0].set_ylabel("g(r)"); ax[0].legend(fontsize=7)
fig.tight_layout(); p4 = f"{ART}/ka_e2e_gr_overlay_N100.png"; fig.savefig(p4, dpi=130); plt.close(fig)
print("SAVED", os.path.abspath(p4)); save["gr"] = gr_data
torch.save(save, f"{LOG}/e2e_analysis_data_N100.pt"); print("SAVED", os.path.abspath(f"{LOG}/e2e_analysis_data_N100.pt"))
