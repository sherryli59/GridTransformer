"""Consolidated e2e analysis (4 tasks):
 1. PT-reference convergence diagnostic (energy-dist shape + temporal drift).
 2. Decorrelation plots (U-ACF + species-ACF, tau_int) for the 3 MALA pipelines. MH arms DROPPED.
 3. (MH arms excluded throughout.)
 4. g(r) overlay: full pipeline, PT (dashed), raw egnn flow, uniform+block8.
Saves plots to artifacts/, computed arrays to reports/logs-2026-07-04/e2e_analysis_data.pt."""
import os, math, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
ART = "liquid_coupling_flow/artifacts"
LOG = "reports/logs-2026-07-04"; os.makedirs(LOG, exist_ok=True)

MALA = {  # label -> (file, color)
    "flow → MALA+block8 (full pipeline)": ("ka_e2e_traj_mala_N256.pt", "C0"),
    "uniform → MALA+block8":              ("ka_e2e_traj_mala_block8_uniform_N256.pt", "C1"),
    "flow → MALA+random":                 ("ka_e2e_traj_mala_random_N256.pt", "C3"),
}
save = {}

# ================= Task 1: PT convergence =================
pt = torch.load(f"{ART}/ka_reference_N256.pt", map_location="cpu", weights_only=False)
xpt, spt, L, N = pt["x"], pt["s"], pt["L"], pt["N"]
Upt = []
for i in range(0, xpt.shape[0], 1024):
    Upt.append((ka_energy(xpt[i:i+1024].to(dev), spt.to(dev), L) / N).cpu())
Upt = torch.cat(Upt).numpy()
nc = len(Upt); K = 24
blk = np.array([Upt[i*nc//K:(i+1)*nc//K].mean() for i in range(K)])
blk_x = np.array([(i+0.5)*nc/K for i in range(K)])
save["pt_UoN"] = Upt; save["pt_block_mean"] = blk
print(f"[PT] mean={Upt.mean():.4f} std={Upt.std():.4f} skew={skew(Upt):.3f} kurt={kurtosis(Upt):.3f}")
print(f"[PT] first-block={blk[0]:.4f} last-block={blk[-1]:.4f} drift={blk[-1]-blk[0]:.4f}")

# final-plateau energy arrays for each MALA arm (finite chains only, last iter)
final_U = {}
for lab, (f, c) in MALA.items():
    d = torch.load(f"{ART}/{f}", map_location="cpu", weights_only=False)
    u = d["UoN"][-1].numpy(); u = u[np.isfinite(u)]
    final_U[lab] = u
save["final_U"] = final_U

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
# (a) temporal drift
ax[0].plot(np.arange(nc), Upt, ".", ms=1, alpha=0.15, color="gray")
ax[0].plot(blk_x, blk, "-o", color="k", lw=2, ms=5, label="block mean (24 blocks)")
ax[0].axhline(Upt.mean(), ls=":", color="C2", label=f"grand mean {Upt.mean():.4f}")
ax[0].set_xlabel("PT config index (save order ≈ replica time)"); ax[0].set_ylabel("U/N")
ax[0].set_title(f"(a) PT ref STILL DRIFTING: {blk[0]:.4f}→{blk[-1]:.4f} ({blk[-1]-blk[0]:+.4f}/particle)")
ax[0].legend(fontsize=8)
# (b) marginal histogram vs MALA plateaus
bins = np.linspace(-3.34, -3.02, 80)
ax[1].hist(Upt, bins=bins, density=True, histtype="step", lw=2, color="k", label=f"PT ref (mean {Upt.mean():.3f})")
for lab, (f, c) in MALA.items():
    ax[1].hist(final_U[lab], bins=bins, density=True, histtype="step", lw=1.6, color=c,
               label=f"{lab.split(' (')[0]} ({np.median(final_U[lab]):.3f})")
ax[1].set_xlabel("U/N"); ax[1].set_ylabel("density")
ax[1].set_title("(b) energy marginal: PT vs MALA endpoints (6000 it)")
ax[1].legend(fontsize=7)
fig.tight_layout(); p1 = f"{ART}/ka_e2e_pt_convergence.png"; fig.savefig(p1, dpi=130); plt.close(fig)
print("SAVED", os.path.abspath(p1))

# ================= Task 2: decorrelation (U-ACF + species-ACF) =================
T0 = 4000; NLAG = 1200  # stationary-ish window (linear-detrended per chain to remove residual drift)

def u_acf(UoN, t0, nlag):
    W = UoN[t0:].astype(np.float64); Tw, B = W.shape
    rhos = []
    t = np.arange(Tw)
    for b in range(B):
        v = W[:, b]
        if not np.isfinite(v).all():
            continue
        A = np.polyfit(t, v, 1); v = v - (A[0]*t + A[1])          # linear detrend
        f = np.fft.rfft(v, n=2*Tw); ac = np.fft.irfft(f*np.conj(f))[:nlag+1]
        ac /= (Tw - np.arange(nlag+1))
        if ac[0] <= 0:
            continue
        rhos.append(ac/ac[0])
    R = np.array(rhos); return R.mean(0), R.shape[0]

def s_acf(s_traj, t0, nlag):
    W = s_traj[t0:].astype(np.float64)*2 - 1; Tw, B, Np = W.shape
    num = np.zeros(nlag+1); den = 0.0
    for b in range(B):
        v = W[:, b, :]
        if not np.isfinite(v).all():
            continue
        v = v - v.mean(0, keepdims=True)                          # connected per particle
        f = np.fft.rfft(v, n=2*Tw, axis=0); ac = np.fft.irfft(f*np.conj(f), axis=0)[:nlag+1]
        ac /= (Tw - np.arange(nlag+1))[:, None]
        num += ac.sum(1); den += ac[0].sum()                     # activity-weighted ensemble ACF
    return num/den

def tau_int(rho, c=6.0):
    for M in range(1, len(rho)):
        tau = 1 + 2*np.sum(rho[1:M+1])
        if M >= c*tau:
            return tau, M
    return 1 + 2*np.sum(rho[1:]), len(rho)-1

uacf, sacf, tauU, tauS = {}, {}, {}, {}
for lab, (f, c) in MALA.items():
    d = torch.load(f"{ART}/{f}", map_location="cpu", weights_only=False)
    ru, nch = u_acf(d["UoN"].numpy(), T0, NLAG); uacf[lab] = ru
    tauU[lab] = tau_int(ru)[0]
    rs = s_acf(d["s"].numpy(), T0, min(NLAG, d["s"].shape[0]-T0-1)); sacf[lab] = rs
    tauS[lab] = tau_int(rs)[0]
    print(f"[decorr] {lab:40s} nchain={nch} tau_U={tauU[lab]:.1f} it  tau_s={tauS[lab]:.1f} it")
save["uacf"] = uacf; save["sacf"] = sacf; save["tauU"] = tauU; save["tauS"] = tauS; save["T0"] = T0

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
for lab, (f, c) in MALA.items():
    lags = np.arange(len(uacf[lab]))
    ax[0].plot(lags[:400], uacf[lab][:400], color=c, lw=1.8,
               label=f"{lab.split(' (')[0]}  $\\tau_U$={tauU[lab]:.0f}")
    ls = np.arange(len(sacf[lab]))
    ax[1].plot(ls[:1000], sacf[lab][:1000], color=c, lw=1.8,
               label=f"{lab.split(' (')[0]}  $\\tau_s$={tauS[lab]:.0f}")
ax[0].axhline(0, color="gray", lw=0.6); ax[0].set_xlabel("lag (iterations, 10 MALA steps each)")
ax[0].set_ylabel(r"$\rho_U(\ell)$"); ax[0].set_title(f"(a) energy autocorrelation (detrended, it>{T0})")
ax[0].legend(fontsize=8)
ax[1].axhline(0, color="gray", lw=0.6); ax[1].set_xlabel("lag (iterations)")
ax[1].set_ylabel(r"$\rho_s(\ell)$ (species)"); ax[1].set_title("(b) species-label autocorrelation")
ax[1].legend(fontsize=8)
fig.tight_layout(); p2 = f"{ART}/ka_e2e_decorrelation.png"; fig.savefig(p2, dpi=130); plt.close(fig)
print("SAVED", os.path.abspath(p2))

# ================= Task 4: g(r) overlay =================
def gr_batched(x, s, L, rmax, nbins, pair, bs=256):
    a, b = pair
    edges = torch.linspace(0, rmax, nbins+1); centers = 0.5*(edges[:-1]+edges[1:])
    shell = math.pi*(edges[1:]**2 - edges[:-1]**2)
    counts = torch.zeros(nbins); norm = 0.0; Np = x.shape[1]
    for i in range(0, x.shape[0], bs):
        xb = x[i:i+bs].to(dev)
        sb = (s if s.dim() == 1 else s[i:i+bs]).to(dev)
        if sb.dim() == 1:
            sb = sb[None].expand(xb.shape[0], Np)
        diff = xb[:, :, None, :] - xb[:, None, :, :]; diff = diff - L*torch.round(diff/L)
        r = torch.sqrt((diff**2).sum(-1) + 1e-12)
        ma = (sb == a); mb = (sb == b)
        eye = torch.eye(Np, dtype=torch.bool, device=dev)[None]
        pm = ma[:, :, None] & mb[:, None, :] & (~eye)
        dd = r[pm]; dd = dd[dd < rmax].cpu()
        counts += torch.histc(dd, bins=nbins, min=0.0, max=rmax)
        na = ma.sum(1).float(); nb_ = mb.sum(1).float()
        norm += float(((na*(na-1) if a == b else na*nb_)/(L**2)).sum().cpu())
    return centers.numpy(), (counts/(norm*shell)).numpy()

def finite_finals(f):
    d = torch.load(f"{ART}/{f}", map_location="cpu", weights_only=False)
    x = d["x_final"]; s = d["s_final"]
    good = torch.isfinite(x).all(dim=(1, 2))
    return x[good], s[good]

gen = torch.load(f"{ART}/ka_e2e_gen_N256.pt", map_location="cpu", weights_only=False)
xg, sg = gen["x"], gen["s"]
xf, sf = finite_finals("ka_e2e_traj_mala_N256.pt")               # full pipeline
xu, su = finite_finals("ka_e2e_traj_mala_block8_uniform_N256.pt")# uniform+block8
idx = torch.randperm(xpt.shape[0])[:3000]                        # subsample PT for memory
arms = [
    ("PT ref (dashed)", xpt[idx], spt, dict(color="k", ls="--", lw=2.2)),
    ("full pipeline (flow→MALA+block8)", xf, sf, dict(color="C0", lw=1.8)),
    ("uniform→MALA+block8", xu, su, dict(color="C1", lw=1.6)),
    ("raw egnn flow (gen)", xg, sg, dict(color="C7", lw=1.4, alpha=0.9)),
]
pairs = [((0, 0), "g_AA"), ((0, 1), "g_AB"), ((1, 1), "g_BB")]
RMAX, NB = 5.0, 150
gr_data = {}
fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
for j, (pair, name) in enumerate(pairs):
    for lab, xx, ss, st in arms:
        cx, gy = gr_batched(xx, ss, L, RMAX, NB, pair)
        ax[j].plot(cx, gy, label=lab, **st)
        gr_data[(name, lab)] = (cx, gy)
    ax[j].set_title(name); ax[j].set_xlabel("r"); ax[j].axhline(1, color="gray", lw=0.5)
    ax[j].set_ylabel("g(r)") if j == 0 else None
ax[0].legend(fontsize=7)
fig.tight_layout(); p4 = f"{ART}/ka_e2e_gr_overlay.png"; fig.savefig(p4, dpi=130); plt.close(fig)
print("SAVED", os.path.abspath(p4))
save["gr"] = {f"{n}|{l}": v for (n, l), v in gr_data.items()}

torch.save(save, f"{LOG}/e2e_analysis_data.pt")
print("SAVED", os.path.abspath(f"{LOG}/e2e_analysis_data.pt"))
