"""ENTIRE-pipeline transfer audit (user framing): flow generates imperfect (x, s) -> species-flow correction
(position MCMC + block moves off the transferred geometry table) relaxes toward equilibrium. Three chains
{uniform+random, flow+random, flow+block8} per size; tracks U/N AND the species-correction observable
(fraction of labels disagreeing with the geometry oracle's argmax) vs sweeps; ends with the structural battery
(g_AA/g_AB/g_BB peaks + unlike-NN) vs the PT reference. Usage: python -m ...ka_pipeline_audit REF_N TAG [n_sweeps] [B]"""
import os, sys, time, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_swap_smc import position_sweep, swap_attempt, block_relabel_attempt, uniform_weight_fn
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

dev = "cuda" if torch.cuda.is_available() else "cpu"
refN = int(sys.argv[1]); tag = sys.argv[2]
n_sweeps = int(sys.argv[3]) if len(sys.argv) > 3 else 2500
B = int(sys.argv[4]) if len(sys.argv) > 4 else 96
beta, step = 2.0, 0.12
ART = os.path.join(os.path.dirname(__file__), "artifacts")
JFD = os.path.join(os.path.dirname(__file__), "ipl44", "data")

refname = {100: "ka_reference_N100.pt", 256: "ka_reference_N256_thin.pt", 36: "ka_reference_N36_val.pt"}[refN]
ref = torch.load(os.path.join(ART, refname), map_location="cpu", weights_only=False)
N = ref["x"].shape[1]; L = float(ref["L"])
xr = torch.remainder(ref["x"].float(), L); sr = ref["s"].long()
nB = int(sr.sum()); efn = lambda a, b: ka_energy(a, b, L)
UrefN = float(efn(xr[-1024:].to(dev), sr[None].expand(1024, -1).to(dev)).median()) / N

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


def unlike_nn(x, s):
    d = x[:, :, None, :] - x[:, None, :, :]; d = d - L * torch.round(d / L)
    r = d.norm(dim=-1) + torch.eye(N, device=x.device) * 99
    nn = r.argmin(-1)
    return float((s.gather(1, nn) != s).float().mean())


def battery(x, s):
    out = {}
    for pair, nm in [((0, 0), "gAA"), ((0, 1), "gAB"), ((1, 1), "gBB")]:
        _, g = partial_gr(x.cpu(), s.cpu(), L, rmax=3.0, nbins=90, pair=pair)
        out[nm] = float(g.max())
    out["unlikeNN"] = unlike_nn(x, s)
    return out


REFBAT = battery(xr[-1024:].to(dev), sr[None].expand(1024, -1).to(dev).contiguous())
print(f"N={N} PT-ref <U>/N {UrefN:.4f} | battery {REFBAT}", flush=True)

torch.manual_seed(7)
x_uni = torch.rand(B, N, 2, device=dev) * L; s_uni = random_22_labeling(B, N, nB, dev)
x_flo, s_flo = jf.sample(torch.rand(B, N, 2, device=dev) * L, random_22_labeling(B, N, nB, dev), n_steps=250)
W0 = geometry_table(x_flo)
print(f"flow seeds: initial species-oracle disagreement {float(((W0>0.5).long()!=s_flo).float().mean()):.4f} "
      f"| U/N {float(efn(x_flo,s_flo).median())/N:.3f}", flush=True)

chains = {"uniform+random": (x_uni, s_uni, "random"),
          "flow+random": (x_flo.clone(), s_flo.clone(), "random"),
          "flow+block8": (x_flo.clone(), s_flo.clone(), "block8")}
curves = {}
for name, (x0, s0, mover) in chains.items():
    x, s = x0.clone(), s0.clone(); U = efn(x, s)
    rec = {"sweep": [], "UoN": [], "disagree": []}
    t0 = time.time()
    for k in range(n_sweeps + 1):
        if k % 10 == 0:
            W = geometry_table(x)
            rec["sweep"].append(k); rec["UoN"].append(float(U.median()) / N)
            rec["disagree"].append(float(((W > 0.5).long() != s).float().mean()))
        if k == n_sweeps:
            break
        x, U, _ = position_sweep(x, s, U, beta, L, step, efn)
        if mover == "random":
            for _ in range(8):
                s, U, _ = swap_attempt(x, s, U, beta, efn, uniform_weight_fn)
        else:
            W = geometry_table(x); tf = lambda _x: W
            for _ in range(8):
                s, U, _ = block_relabel_attempt(x, s, U, beta, efn, tf, 8)
    bat = battery(x, s)
    curves[name] = {**rec, "battery": bat, "wall": time.time() - t0}
    s30 = next((kk for kk, u in zip(rec["sweep"], rec["UoN"]) if u <= -3.0), None)
    s31 = next((kk for kk, u in zip(rec["sweep"], rec["UoN"]) if u <= -3.1), None)
    print(f"{name:16s}: -3.0 @ {s30} | -3.1 @ {s31} | final U/N {rec['UoN'][-1]:.4f} "
          f"| disagree {rec['disagree'][0]:.3f}->{rec['disagree'][-1]:.3f} | battery {bat} | {curves[name]['wall']:.0f}s", flush=True)

torch.save({"curves": curves, "refbat": REFBAT, "UrefN": UrefN}, f"{ART}/ka_pipeline_audit_N{N}_{tag}.pt")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
for name, r in curves.items():
    ax[0].plot(r["sweep"], r["UoN"], label=name)
    ax[1].plot(r["sweep"], r["disagree"], label=name)
ax[0].axhline(UrefN, ls="--", c="k"); ax[0].axhline(-3.0, ls=":", c="gray"); ax[0].axhline(-3.1, ls=":", c="gray")
ax[0].set_title(f"N={N}: U/N vs sweeps"); ax[0].set_xlabel("sweep"); ax[0].legend(fontsize=8)
ax[1].set_title("species-oracle disagreement (the correction observable)"); ax[1].set_xlabel("sweep"); ax[1].legend(fontsize=8)
names = list(curves); keys = ["gAA", "gAB", "gBB", "unlikeNN"]
Xb = torch.arange(len(keys))
for i, name in enumerate(names):
    ax[2].bar(Xb + 0.2 * i, [curves[name]["battery"][k] for k in keys], width=0.18, label=name)
ax[2].bar(Xb + 0.2 * len(names), [REFBAT[k] for k in keys], width=0.18, color="k", alpha=0.6, label="PT ref")
ax[2].set_xticks(Xb + 0.3); ax[2].set_xticklabels(keys); ax[2].legend(fontsize=7); ax[2].set_title("endpoint structural battery")
plt.tight_layout(); plt.savefig(f"{ART}/ka_pipeline_audit_N{N}_{tag}.png", dpi=110)
print(f"saved {ART}/ka_pipeline_audit_N{N}_{tag}.png", flush=True)
