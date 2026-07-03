"""Step-D benchmark: k-site BLOCK RELABEL (frozen geometry table, one denoiser call per sweep) vs pair-swap
(paired re-evaluation, 16 calls per sweep) vs random, on the two-time model. Movers:
  random      : pair swaps, constant weights
  pair-tt     : pair swaps, live-s denoiser at (t_pos=1, t_spec=0.9) — on-manifold query for real positions
  block-k     : k-site relabel from the FROZEN geometry table (t_pos=1, t_spec=0, s_ref fixed) — x-only, exact
All exact-MH. Reports sweeps-to-band AND wall-clock. Usage: python -m ...bench_block JF_TT_TAG [n_sweeps] [B]"""
import os, sys, time, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_swap_smc import (run_chain, sweeps_to_band, uniform_weight_fn,
                                                     swap_attempt, block_relabel_attempt)
from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_gr_partials, ipl_box
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L); beta = 10.0
tag = sys.argv[1]; n_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
B = int(sys.argv[3]) if len(sys.argv) > 3 else 128
ART = os.path.join(os.path.dirname(__file__), "data")

ck = torch.load(f"{ART}/jf_{tag}_best.pt", map_location=dev, weights_only=False)
cfg = ck["cfg"]; assert cfg.get("two_time"), "bench_block needs a two-time checkpoint"
jf = JointSpeciesFlow(n_particles=N, L=Lf, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"],
                      two_time=True).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval()
print(f"loaded jf_{tag}_best (two-time, val acc {ck['val_acc']:.4f})", flush=True)

D = "/mnt/ssd/GridTransformer/datasets"
xr = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sr = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
o = sr.argsort(-1); sr = torch.gather(sr, 1, o); xr = torch.gather(xr, 1, o.unsqueeze(-1).expand(-1, -1, 2))
U_ref = float(ipl_energy(xr[:4096].to(dev), sr[:4096].to(dev)).median())
_, _, _, gbb_r = ipl_gr_partials(xr[:4096], sr[:4096], Lf)
GBB_REF = float(gbb_r.max())
S_REF = torch.zeros(1, N, dtype=torch.long, device=dev); S_REF[:, 22:] = 1   # FIXED canonical labels
print(f"band targets: U_med {U_ref:.2f} (+-5%), g_BB {GBB_REF:.2f} (+-15%)", flush=True)


def geometry_table(x):
    """x-ONLY frozen table: two-time query (t_pos=1, t_spec=0) with a constant s_ref input."""
    with torch.no_grad():
        Bx = x.shape[0]
        one = torch.ones(Bx, 1, 1, device=x.device); zero = torch.zeros(Bx, 1, 1, device=x.device)
        _, logits = jf(one, x, S_REF.expand(Bx, -1), t_spec=zero)
        return torch.softmax(logits, -1)[..., 1]


def live_pair_wfn(x, s):
    """On-manifold pair-proposer weights for real positions: (t_pos=1, t_spec=0.9), live s."""
    with torch.no_grad():
        Bx = x.shape[0]
        one = torch.ones(Bx, 1, 1, device=x.device); ts = torch.full((Bx, 1, 1), 0.9, device=x.device)
        _, logits = jf(one, x, s, t_spec=ts)
        return torch.softmax(logits, -1)[..., 1].clamp(1e-6, 1 - 1e-6)


def block_mover(k, n_att=8):
    def mv(x, s, U):
        W = geometry_table(x)                                             # ONE denoiser call per sweep
        table_fn = lambda _x: W                                           # frozen for all attempts (x fixed)
        sacc = 0.0
        for _ in range(n_att):
            s, U, a = block_relabel_attempt(x, s, U, beta, ipl_energy, table_fn, k)
            sacc += a.float().mean().item()
        return s, U, sacc / n_att
    return mv


def pair_mover(wfn, n_att=8):
    def mv(x, s, U):
        sacc = 0.0
        for _ in range(n_att):
            s, U, a = swap_attempt(x, s, U, beta, ipl_energy, wfn)
            sacc += a.float().mean().item()
        return s, U, sacc / n_att
    return mv


def seeds(kind):
    s0 = random_22_labeling(B, N, 22, dev)
    if kind == "uniform":
        return torch.rand(B, N, 2, device=dev) * Lf, s0
    if kind == "tfbank":
        from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model
        tck = torch.load(f"{ART}/ipl44_curveflow.pt", map_location=dev, weights_only=False)
        tm = make_ipl_model(num_bins=tck["num_bins"], tail_bound=tck["tail_bound"], knn=tck["knn"],
                            arc_range=tck["arc_range"], device=dev)
        tm.load_state_dict(tck["state_dict"]); tm.eval()
        with torch.no_grad():
            xs, ss = tm.sample(B, N, n_B=tck["n_B"], device=dev)
        o2 = ss.argsort(-1)
        return torch.gather(torch.remainder(xs, Lf), 1, o2.unsqueeze(-1).expand(-1, -1, 2)), torch.gather(ss, 1, o2)
    raise ValueError(kind)


movers = [("random", pair_mover(uniform_weight_fn)), ("pair-tt", pair_mover(live_pair_wfn))] + \
         [(f"block{k}", block_mover(k)) for k in (2, 4, 6, 8)]
results = {}
for seed_kind in ["uniform", "tfbank"]:
    torch.manual_seed(7)
    x0, s0 = seeds(seed_kind)
    for name, mv in movers:
        t0 = time.time()
        cur = run_chain(x0.clone(), s0.clone(), n_sweeps, beta, Lf, ipl_energy, move_fn=mv, record_every=5)
        stb = sweeps_to_band(cur, U_ref, GBB_REF)
        wall = time.time() - t0
        tt_band = wall * (stb / n_sweeps) if stb else float("nan")
        results[(seed_kind, name)] = {k: cur[k] for k in ("sweep", "U_median", "gbb_peak", "swap_acc", "pos_acc")}
        results[(seed_kind, name)].update({"sweeps_to_band": stb, "wall_s": wall, "t_to_band": tt_band})
        print(f"{seed_kind:8s} {name:8s}: sweeps-to-band {stb} | wall {wall:.0f}s (t-to-band ~{tt_band:.0f}s) "
              f"| final U_med {cur['U_median'][-1]:.2f} g_BB {cur['gbb_peak'][-1]:.2f} "
              f"| acc {sum(cur['swap_acc'])/max(len(cur['swap_acc']),1):.3f}", flush=True)
torch.save(results, f"{ART}/bench_block_{tag}.pt")

fig, axes = plt.subplots(2, 2, figsize=(13, 8))
for c, seed_kind in enumerate(["uniform", "tfbank"]):
    for (sk, pn), r in results.items():
        if sk != seed_kind:
            continue
        axes[0, c].plot(r["sweep"], r["U_median"], label=f"{pn} (stb {r['sweeps_to_band']})")
        axes[1, c].plot(r["sweep"], r["gbb_peak"], label=pn)
    axes[0, c].axhline(U_ref, ls="--", c="k"); axes[0, c].set_title(f"{seed_kind}: U median")
    axes[0, c].set_yscale("symlog"); axes[0, c].legend(fontsize=7)
    axes[1, c].axhline(GBB_REF, ls="--", c="k"); axes[1, c].set_title(f"{seed_kind}: g_BB peak")
    axes[1, c].set_xlabel("sweep"); axes[1, c].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{ART}/bench_block_{tag}.png", dpi=110)
print(f"saved {ART}/bench_block_{tag}.png", flush=True)
