"""Production-phase statistics for the pipeline (user request): after burn-in from FLOW seeds, run a
stationary production phase per KERNEL (random-pair vs block8) and compare:
  (1) full energy DISTRIBUTIONS (pooled batch x time) vs the PT reference;
  (2) full g_AA/g_AB/g_BB CURVES vs reference;
  (3) acceptance rates — position, swap accepted, and swap EFFECTIVE (labels actually changed; block
      acceptance is inflated by null redraws);
  (4) DECORRELATION: integrated autocorrelation time tau_U of U(t), and species memory time tau_s
      (sweeps for the label-persistence autocorrelation to fall below 1/e).
Decorrelation is a property of the kernel at stationarity — the right equilibrium-efficiency metric.
Usage: python -m liquid_coupling_flow.ka_pipeline_stats REF_N TAG [n_burn] [n_prod] [B]"""
import os, sys, time, torch, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_swap_smc import position_sweep, swap_attempt, block_relabel_attempt, uniform_weight_fn
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

dev = "cuda" if torch.cuda.is_available() else "cpu"
refN = int(sys.argv[1]); tag = sys.argv[2]
n_burn = int(sys.argv[3]) if len(sys.argv) > 3 else 1500
n_prod = int(sys.argv[4]) if len(sys.argv) > 4 else 600
B = int(sys.argv[5]) if len(sys.argv) > 5 else 64
beta, step = 2.0, 0.12
ART = os.path.join(os.path.dirname(__file__), "artifacts")
JFD = os.path.join(os.path.dirname(__file__), "ipl44", "data")

refname = {100: "ka_reference_N100.pt", 256: "ka_reference_N256_thin.pt", 36: "ka_reference_N36_val.pt"}[refN]
ref = torch.load(os.path.join(ART, refname), map_location="cpu", weights_only=False)
N = ref["x"].shape[1]; L = float(ref["L"])
xr = torch.remainder(ref["x"].float(), L); sr = ref["s"].long()
nB = int(sr.sum()); efn = lambda a, b: ka_energy(a, b, L)
UrefAll = (efn(xr[-2048:].to(dev), sr[None].expand(2048, -1).to(dev)) / N).cpu().numpy()

ck = torch.load(f"{JFD}/jf_{tag}_best.pt", map_location=dev, weights_only=False); cfg = ck["cfg"]
jf = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"], two_time=True).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval(); jf.knn = 32
S_CANON = torch.zeros(1, N, dtype=torch.long, device=dev); S_CANON[:, :nB] = 1


def geometry_table(x):
    with torch.no_grad():
        Bx = x.shape[0]
        _, lg = jf(torch.ones(Bx, 1, 1, device=dev), x, S_CANON.expand(Bx, -1),
                   t_spec=torch.zeros(Bx, 1, 1, device=dev))
        return torch.softmax(lg, -1)[..., 1]


def gr_curves(x, s):
    out = {}
    for pair, nm in [((0, 0), "gAA"), ((0, 1), "gAB"), ((1, 1), "gBB")]:
        r, g = partial_gr(x.cpu(), s.cpu(), L, rmax=3.5, nbins=100, pair=pair)
        out[nm] = (r, g)
    return out


REFGR = gr_curves(xr[-2048:], sr[None].expand(2048, -1).contiguous())


def tau_int(series):
    """Integrated autocorrelation time with 5-tau self-consistent window; series [T, B] -> float (sweeps)."""
    ts = series - series.mean(0, keepdims=True)
    T = ts.shape[0]
    var = (ts ** 2).mean()
    if var < 1e-12:
        return float("nan")
    tau = 1.0
    for W in range(1, T // 3):
        rho = (ts[:-W] * ts[W:]).mean() / var
        tau += 2.0 * rho
        if W >= 5 * tau:
            break
    return float(max(tau, 1.0))


torch.manual_seed(7)
x0, s0 = jf.sample(torch.rand(B, N, 2, device=dev) * L, random_22_labeling(B, N, nB, dev), n_steps=250)
print(f"N={N} | burn {n_burn} prod {n_prod} B={B} | flow seeds U/N {float(efn(x0,s0).median())/N:.3f}", flush=True)

results = {}
for kern in ["random", "block8"]:
    x, s = x0.clone(), s0.clone(); U = efn(x, s); t0 = time.time()
    for k in range(n_burn):                                              # burn-in (not recorded)
        x, U, _ = position_sweep(x, s, U, beta, L, step, efn)
        if kern == "random":
            for _ in range(8):
                s, U, _ = swap_attempt(x, s, U, beta, efn, uniform_weight_fn)
        else:
            W = geometry_table(x); tf = lambda _x: W
            for _ in range(8):
                s, U, _ = block_relabel_attempt(x, s, U, beta, efn, tf, 8)
    Us, Ss, pacc_l, sacc_l, seff_l = [], [], [], [], []
    for k in range(n_prod):                                              # production
        x, U, pa = position_sweep(x, s, U, beta, L, step, efn)
        sa = 0.0; ch = 0.0
        if kern == "random":
            for _ in range(8):
                s_old = s; s, U, a = swap_attempt(x, s, U, beta, efn, uniform_weight_fn)
                sa += a.float().mean().item(); ch += (s != s_old).any(1).float().mean().item()
        else:
            W = geometry_table(x); tf = lambda _x: W
            for _ in range(8):
                s_old = s; s, U, a = block_relabel_attempt(x, s, U, beta, efn, tf, 8)
                sa += a.float().mean().item(); ch += (s != s_old).any(1).float().mean().item()
        Us.append((U / N).cpu().clone()); Ss.append(s.cpu().clone())
        pacc_l.append(pa); sacc_l.append(sa / 8); seff_l.append(ch / 8)
    Us = torch.stack(Us).numpy()                                          # [T,B]
    Ss = torch.stack(Ss)                                                  # [T,B,N]
    tU = tau_int(Us)
    # species memory: label-persistence autocorr C(dt) = P(s_i(t+dt)==s_i(t)) rescaled to [0,1]
    base = 2 * (nB / N) * (1 - nB / N)                                    # random-relabel disagreement prob
    taus = None; Cs = []
    for dt in range(1, min(n_prod - 1, 400)):
        dis = (Ss[:-dt] != Ss[dt:]).float().mean().item()
        Cs.append(1.0 - dis / base)                                       # 1 = frozen, 0 = fully decorrelated
        if taus is None and Cs[-1] < np.exp(-1):
            taus = dt
    results[kern] = {"U": Us, "pacc": float(np.mean(pacc_l)), "sacc": float(np.mean(sacc_l)),
                     "seff": float(np.mean(seff_l)), "tauU": tU, "tauS": taus, "Cs": Cs,
                     "gr": gr_curves(x, s), "wall": time.time() - t0}
    print(f"{kern:7s}: pos acc {results[kern]['pacc']:.3f} | swap acc {results[kern]['sacc']:.3f} "
          f"(effective {results[kern]['seff']:.3f}) | tau_U {tU:.1f} sweeps | tau_s {taus} sweeps "
          f"| U/N mean {Us.mean():.4f} (ref {UrefAll.mean():.4f}) | {results[kern]['wall']:.0f}s", flush=True)

torch.save({k: {kk: vv for kk, vv in v.items() if kk != "gr"} for k, v in results.items()},
           f"{ART}/ka_pipeline_stats_N{N}_{tag}.pt")

fig, ax = plt.subplots(2, 3, figsize=(17, 8.5))
ax[0, 0].hist(UrefAll, bins=50, density=True, alpha=0.55, color="k", label="PT reference")
for kern, c in [("random", "tab:blue"), ("block8", "tab:orange")]:
    ax[0, 0].hist(results[kern]["U"].reshape(-1), bins=50, density=True, alpha=0.5, color=c, label=kern)
ax[0, 0].set_title(f"N={N}: production U/N distribution"); ax[0, 0].legend(fontsize=8)
for j, nm in enumerate(["gAA", "gAB", "gBB"]):
    a = ax[0, 1] if j < 1 else (ax[0, 2] if j == 1 else ax[1, 0])
    r, g = REFGR[nm]; a.plot(r, g, "k-", lw=2, label="PT ref")
    for kern, c in [("random", "tab:blue"), ("block8", "tab:orange")]:
        r2, g2 = results[kern]["gr"][nm]; a.plot(r2, g2, color=c, ls="--", label=kern)
    a.set_title(f"{nm}(r)"); a.legend(fontsize=7); a.set_xlabel("r")
for kern, c in [("random", "tab:blue"), ("block8", "tab:orange")]:
    ax[1, 1].plot(results[kern]["Cs"], color=c, label=f"{kern} (tau_s {results[kern]['tauS']})")
ax[1, 1].axhline(np.exp(-1), ls=":", c="gray"); ax[1, 1].set_title("species label-persistence autocorr")
ax[1, 1].set_xlabel("dt (sweeps)"); ax[1, 1].legend(fontsize=8)
labels = ["pos acc", "swap acc", "swap effective"]
Xb = np.arange(3)
for i, (kern, c) in enumerate([("random", "tab:blue"), ("block8", "tab:orange")]):
    vals = [results[kern]["pacc"], results[kern]["sacc"], results[kern]["seff"]]
    ax[1, 2].bar(Xb + 0.35 * i, vals, width=0.3, color=c, label=kern)
ax[1, 2].set_xticks(Xb + 0.17); ax[1, 2].set_xticklabels(labels); ax[1, 2].set_title("acceptance rates"); ax[1, 2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{ART}/ka_pipeline_stats_N{N}_{tag}.png", dpi=110)
print(f"saved {ART}/ka_pipeline_stats_N{N}_{tag}.png", flush=True)
