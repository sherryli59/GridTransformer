"""g(r) PER SPECIES for cavity-eRSI samples vs data. The structural test a scalar energy cannot give
([[evaluate-distributions-not-scalars]]).

Cavity g(r) needs care: a sphere of radius R holding n~39 particles has strong finite-size/boundary
geometry, so the ideal-gas normaliser is NOT 4*pi*r^2*rho. We normalise by the SAME geometry, measured:
    g_ab(r) = hist_ab(r) / hist_ideal_ab(r)
with hist_ideal from many uniform-in-sphere configs with the SAME n and SAME species labels. That makes
g -> 1 at large r inside the cavity and cancels the sphere geometry exactly, so generated vs data are
directly comparable.

Curves per species pair (AA, AB, BB):
  DATA      -- carved equilibrium interiors (the truth; should show the excluded-volume hole + first shell)
  eRSI      -- uniform-in-sphere base -> flow (the model)
  UNIFORM   -- the base itself (sanity: must be flat g=1 by construction)
Expect (given GATE1 raw U/n ~ +629): eRSI shows spurious weight at small r = the clash signature = no
excluded-volume hole. That is the visual confirmation of the declash failure.
Usage: gr_cavity_ersi.py [ckpt] [R] [NCAV] [NSAMP]"""
import sys, math, statistics as st
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path("/mnt/ssd/GridTransformer")
sys.path.insert(0, str(REPO))
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka_energy import SIGMA
torch.set_grad_enabled(False)

CKPT = sys.argv[1] if len(sys.argv) > 1 else "ka3d_cavity_ersi_rho12"
R = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
NCAV = int(sys.argv[3]) if len(sys.argv) > 3 else 24
NSAMP = int(sys.argv[4]) if len(sys.argv) > 4 else 8
RCTX = 2.5; DUMMY0, DUMMY_D = 50.0, 5.0
NB = 60; RMAX = 2.6                      # bins / max pair distance probed
dev = "cuda"
ck = torch.load(REPO / f"liquid_coupling_flow/artifacts/{CKPT}.pt", map_location=dev, weights_only=False)
A = ck["args"]; K_MAX = A["K_MAX"]; N_CAGE = A["N_CAGE"]
flow = CavityBlockFlow(n_cage=N_CAGE, k=K_MAX, r_c=2.5, hidden_nf=A["hidden"], n_layers=A["layers"],
                       n_species=2, max_neighbors=16).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
print(f"g(r) eval: {CKPT} step {ck['step']} FM {ck['fm']:.4f} | R={R} NCAV={NCAV} NSAMP={NSAMP}", flush=True)
D = torch.load(REPO / "liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location="cpu", weights_only=False)
Xds, Sds, L = D["x"].float(), D["s"].long(), float(D["L"])
n_train = int(Xds.shape[0] * 0.9)
edges = torch.linspace(0, RMAX, NB + 1); ctr = 0.5 * (edges[1:] + edges[:-1])
PAIRS = {"AA": (0, 0), "AB": (0, 1), "BB": (1, 1)}
acc = {k: {"data": torch.zeros(NB), "ersi": torch.zeros(NB), "ideal": torch.zeros(NB), "unif": torch.zeros(NB)}
       for k in PAIRS}


def hist_pairs(x, sp, key):
    """histogram of pair distances for species pair `key` within one config"""
    a, b = PAIRS[key]
    d = torch.cdist(x, x)
    n = x.shape[0]
    iu = torch.triu_indices(n, n, offset=1)
    dd = d[iu[0], iu[1]]
    sa, sb = sp[iu[0]], sp[iu[1]]
    if a == b:
        msk = (sa == a) & (sb == a)
    else:
        msk = ((sa == a) & (sb == b)) | ((sa == b) & (sb == a))
    return torch.histogram(dd[msk], bins=edges)[0]


def dummies(m):
    d = torch.zeros(m, 3); d[:, 0] = DUMMY0 + DUMMY_D * torch.arange(m, dtype=torch.float32); return d


gen = torch.Generator().manual_seed(99)
# ---- collect ALL cavities first; the fixed-size padding makes every row the same shape, so the
# ---- whole sweep is ONE batched flow() call (B = NCAV*NSAMP) instead of NCAV launch-bound calls.
cavs = []
for _ in range(600):
    ci = int(torch.randint(n_train, Xds.shape[0], (1,), generator=gen))      # HELD-OUT
    c = torch.rand(3, generator=gen) * L
    pr = carve(Xds[ci], Sds[ci], c, R, L); n = pr["n_in"]
    if n < 14 or n > K_MAX:
        continue
    xin = _mic(pr["x_in"], c, L); sin = pr["s_in"].long()
    xout = _mic(pr["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX)
    bx, bs = xout[bm], pr["s_out"][bm].long()
    if bx.shape[0] < 8:
        continue
    idx = bx.norm(dim=-1).argsort()[:N_CAGE]
    cage_x = bx[idx]; sp_cage = bs[idx]; nc = cage_x.shape[0]
    if nc < N_CAGE:
        cage_x = torch.cat([cage_x, dummies(N_CAGE - nc)])
        sp_cage = torch.cat([sp_cage, torch.zeros(N_CAGE - nc, dtype=torch.long)])
    cavs.append({"xin": xin, "sin": sin, "n": n, "cage_x": cage_x, "sp_cage": sp_cage})
    if len(cavs) >= NCAV:
        break
ncav = len(cavs)
# build the full batch
X0, SPB, CG, SPC, NS = [], [], [], [], []
for cv in cavs:
    n = cv["n"]
    dpad = dummies(K_MAX - n) if K_MAX > n else None
    for s in range(NSAMP):
        x0 = torch.zeros(K_MAX, 3); spb = torch.zeros(K_MAX, dtype=torch.long)
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        rr = R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
        x0[:n] = u * rr; spb[:n] = cv["sin"]
        if dpad is not None:
            x0[n:] = dpad
        X0.append(x0); SPB.append(spb); CG.append(cv["cage_x"]); SPC.append(cv["sp_cage"]); NS.append(n)
X0 = torch.stack(X0); SPB = torch.stack(SPB); CG = torch.stack(CG); SPC = torch.stack(SPC)
import time as _t
# FIXED-STEP RK4 for POSITIONS ONLY (viz needs samples, not the exact log-det). flow.flow() hardcodes
# adaptive dopri5, which on this stiff (clashy) velocity field takes ~infinite tiny steps (>45 min, killed).
# RK4 with NSTEPS is bounded, fully batchable, and plenty accurate to SEE g(r) structure.
NSTEPS = int(__import__("os").environ.get("GR_STEPS", 30))
CHUNK = int(__import__("os").environ.get("GR_CHUNK", 48))


def sample_rk4(flow, x0_block, cage_x, sp_block, sp_cage, nsteps):
    k = flow.k; B = x0_block.shape[0]
    cl = x0_block.clone(); cage = cage_x; sp = torch.cat([sp_block, sp_cage], 1)
    dt = 1.0 / nsteps

    def vel(x, tval):
        tt = torch.full((B,), tval, device=x.device, dtype=x.dtype)
        v, _ = flow.ce.vel_div(torch.cat([x, cage], 1), tt, sp, k)
        return v[:, :k]
    for s in range(nsteps):
        t = s * dt
        k1 = vel(cl, t)
        k2 = vel(cl + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = vel(cl + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = vel(cl + dt * k3, t + dt)
        cl = cl + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    return cl


_t0 = _t.time(); outs = []
for i in range(0, X0.shape[0], CHUNK):
    xb = sample_rk4(flow, X0[i:i+CHUNK].to(dev), CG[i:i+CHUNK].to(dev),
                    SPB[i:i+CHUNK].to(dev), SPC[i:i+CHUNK].to(dev), NSTEPS)
    outs.append(xb.cpu()); del xb; torch.cuda.empty_cache()
X1 = torch.cat(outs)
_dt = _t.time() - _t0
print(f"  RK4 sample ({NSTEPS} steps): B={X0.shape[0]} chunk {CHUNK} -> {_dt:.1f}s ({_dt/X0.shape[0]*1000:.0f} ms/sample)", flush=True)
for ic, cv in enumerate(cavs):
    n = cv["n"]; xin = cv["xin"]; sin = cv["sin"]
    for k in PAIRS:
        acc[k]["data"] += hist_pairs(xin, sin, k)
    for s in range(NSAMP):
        row = ic * NSAMP + s
        for k in PAIRS:
            acc[k]["ersi"] += hist_pairs(X1[row, :n], sin, k)
            acc[k]["unif"] += hist_pairs(X0[row, :n], sin, k)
    # ideal-gas normaliser: uniform-in-sphere, SAME n, SAME species labels, same geometry
    for s in range(24):
        u = torch.randn(n, 3, generator=gen); u = u / u.norm(dim=-1, keepdim=True)
        rr = R * torch.rand(n, 1, generator=gen) ** (1.0 / 3.0)
        xu = u * rr
        for k in PAIRS:
            acc[k]["ideal"] += hist_pairs(xu, sin, k)

sig = SIGMA
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
out = {}
for ax, k in zip(axes, PAIRS):
    ideal = acc[k]["ideal"] / 24.0                      # per-config ideal
    ideal = ideal.clamp(min=1e-9)
    g_data = acc[k]["data"] / ncav / ideal
    g_ersi = acc[k]["ersi"] / (ncav * NSAMP) / ideal
    g_unif = acc[k]["unif"] / (ncav * NSAMP) / ideal
    out[k] = {"r": ctr.numpy(), "data": g_data.numpy(), "ersi": g_ersi.numpy(), "unif": g_unif.numpy()}
    ax.plot(ctr, g_data, "k-", lw=2.2, label="DATA (equilibrium)")
    ax.plot(ctr, g_ersi, "-", color="tab:red", lw=2.0, label=f"eRSI (step {ck['step']})")
    ax.plot(ctr, g_unif, "--", color="tab:gray", lw=1.2, label="uniform base (sanity: flat)")
    a, b = PAIRS[k]
    ax.axvline(sig[a][b], color="tab:blue", ls=":", lw=1.5, label=f"$\\sigma_{{{k}}}$={sig[a][b]:.2f}")
    ax.axhline(1.0, color="0.7", lw=0.8)
    ax.set_title(f"$g_{{{k}}}(r)$"); ax.set_xlabel("r"); ax.grid(alpha=0.3); ax.set_xlim(0, RMAX)
axes[0].set_ylabel("g(r)  [normalised by same-geometry ideal]")
axes[0].legend(fontsize=8)
fig.suptitle(f"Cavity eRSI g(r) per species — R={R}, rho=1.2, T=0.5, {ncav} held-out cavities x {NSAMP} samples\n"
             f"excluded-volume hole below sigma = the declash test", fontsize=10)
fig.tight_layout()
png = REPO / "reports/logs-2026-07-15/gr_cavity_ersi.png"
fig.savefig(png, dpi=130)
torch.save({"g": out, "ckpt": CKPT, "step": ck["step"], "fm": ck["fm"], "ncav": ncav, "R": R},
           REPO / "reports/logs-2026-07-15/gr_cavity_ersi.pt")
# quantitative: contact violation = integrated g below 0.9*sigma (should be ~0 for data)
print(f"\n{'pair':>4} | {'g_max data':>10} {'g_max eRSI':>10} | {'contact-violation (int g, r<0.9sig)':>36}")
for k in PAIRS:
    a, b = PAIRS[k]; cut = 0.9 * sig[a][b]
    msk = ctr < cut
    vd = float((torch.tensor(out[k]["data"])[msk]).sum()); ve = float((torch.tensor(out[k]["ersi"])[msk]).sum())
    print(f"{k:>4} | {out[k]['data'].max():10.2f} {out[k]['ersi'].max():10.2f} | data {vd:8.2f}   eRSI {ve:8.2f}")
print(f"\nsaved plot -> {png}", flush=True)
