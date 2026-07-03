"""G2 benchmark: seeds {uniform, tfbank, flow} x proposers {random, learned@t_prop in {0.7,0.9,0.99}}.
B chains at beta=10, up to n_sweeps (record every 10). Learned weight_fn = denoiser pB at t_prop given the
CURRENT (x, s) — paired re-evaluation in swap_attempt keeps MH exact. Wall-clock recorded per config.
Usage: python -m liquid_coupling_flow.ipl44.bench_swap JF_CKPT_TAG [n_sweeps] [B]"""
import os, sys, time, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_swap_smc import run_chain, sweeps_to_band, uniform_weight_fn
from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_gr_partials, ipl_box
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L); beta = 10.0
tag = sys.argv[1]; n_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
B = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ART = os.path.join(os.path.dirname(__file__), "data")

ck = torch.load(f"{ART}/jf_{tag}_best.pt", map_location=dev, weights_only=False)
cfg = ck["cfg"]
jf = JointSpeciesFlow(n_particles=N, L=Lf, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"]).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval()
print(f"loaded jf_{tag}_best (val acc {ck['val_acc']:.4f})", flush=True)

# reference band targets
D = "/mnt/ssd/GridTransformer/datasets"
xr = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sr = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
o = sr.argsort(-1); sr = torch.gather(sr, 1, o); xr = torch.gather(xr, 1, o.unsqueeze(-1).expand(-1, -1, 2))
U_ref = float(ipl_energy(xr[:4096].to(dev), sr[:4096].to(dev)).median())
_, _, _, gbb_r = ipl_gr_partials(xr[:4096], sr[:4096], Lf)
GBB_REF = float(gbb_r.max())
print(f"band targets: U_med {U_ref:.2f} (+-5%), g_BB {GBB_REF:.2f} (+-15%)", flush=True)


def learned_wfn(t_prop):
    def wfn(x, s):
        with torch.no_grad():
            t = torch.full((x.shape[0], 1, 1), t_prop, device=x.device)
            _, logits = jf(t, x, s)
            return torch.softmax(logits, -1)[..., 1].clamp(1e-6, 1 - 1e-6)
    return wfn


def seeds(kind):
    s0 = random_22_labeling(B, N, 22, dev)
    if kind == "uniform":
        return torch.rand(B, N, 2, device=dev) * Lf, s0
    if kind == "flow":
        x0 = torch.rand(B, N, 2, device=dev) * Lf
        return jf.sample(x0, s0, n_steps=250)
    if kind == "tfbank":
        from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model
        tck = torch.load(f"{ART}/ipl44_curveflow.pt", map_location=dev, weights_only=False)
        tm = make_ipl_model(num_bins=tck["num_bins"], tail_bound=tck["tail_bound"], knn=tck["knn"],
                            arc_range=tck["arc_range"], device=dev)
        tm.load_state_dict(tck["state_dict"]); tm.eval()
        with torch.no_grad():
            xs, ss = tm.sample(B, N, n_B=tck["n_B"], device=dev)         # verified signature (ka_curveflow.py:51)
        o2 = ss.argsort(-1)
        return torch.gather(torch.remainder(xs, Lf), 1, o2.unsqueeze(-1).expand(-1, -1, 2)), torch.gather(ss, 1, o2)
    raise ValueError(kind)


results = {}
for seed_kind in ["uniform", "tfbank", "flow"]:
    torch.manual_seed(7)
    x0, s0 = seeds(seed_kind)
    for prop_name, wfn in [("random", uniform_weight_fn)] + [(f"learned{t}", learned_wfn(t)) for t in (0.7, 0.9, 0.99)]:
        t0 = time.time()
        cur = run_chain(x0.clone(), s0.clone(), n_sweeps, beta, Lf, ipl_energy, weight_fn=wfn, n_swap=8,
                        record_every=5)
        stb = sweeps_to_band(cur, U_ref, GBB_REF)
        wall = time.time() - t0
        results[(seed_kind, prop_name)] = {k: cur[k] for k in ("sweep", "U_median", "gbb_peak", "swap_acc", "pos_acc")}
        results[(seed_kind, prop_name)]["sweeps_to_band"] = stb
        results[(seed_kind, prop_name)]["wall_s"] = wall
        print(f"{seed_kind:8s} {prop_name:10s}: sweeps-to-band {stb} | wall {wall:.0f}s "
              f"| final U_med {cur['U_median'][-1]:.2f} g_BB {cur['gbb_peak'][-1]:.2f} "
              f"| swap acc {sum(cur['swap_acc'])/max(len(cur['swap_acc']),1):.3f}", flush=True)
torch.save(results, f"{ART}/bench_swap_curves.pt")

fig, axes = plt.subplots(2, 3, figsize=(17, 8))
for c, seed_kind in enumerate(["uniform", "tfbank", "flow"]):
    for (sk, pn), r in results.items():
        if sk != seed_kind:
            continue
        axes[0, c].plot(r["sweep"], r["U_median"], label=f"{pn} (stb {r['sweeps_to_band']})")
        axes[1, c].plot(r["sweep"], r["gbb_peak"], label=pn)
    axes[0, c].axhline(U_ref, ls="--", c="k"); axes[0, c].set_title(f"{seed_kind}: U median")
    axes[0, c].set_yscale("symlog"); axes[0, c].legend(fontsize=7)
    axes[1, c].axhline(GBB_REF, ls="--", c="k"); axes[1, c].set_title(f"{seed_kind}: g_BB peak"); axes[1, c].set_xlabel("sweep")
    axes[1, c].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{ART}/bench_swap.png", dpi=110)
print(f"saved {ART}/bench_swap.png", flush=True)
