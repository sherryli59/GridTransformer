"""KA block-mover benchmark driver (spec docs/superpowers/specs/2026-07-03-ka-block-mover-transfer-design.md).
Reuses the system-agnostic exact kernels from ipl44/ipl_swap_smc (pair swap, block relabel, position sweeps)
with ka_energy at T*=0.5 (beta=2). The joint flow trained at N=100 is rebuilt at the TARGET (N, L) — EGNN
weights are N-agnostic; same-density locality is the invariance under test.

Modes:
  e1  <ref>   : benchmark at a reference size (seeds {uniform, flow} x movers {random, block4, block8})
  safety <ref>: 2000-sweep block4 run from REFERENCE seeds — swap-safety check (no drift below PT <U>/N)
Usage: python -m liquid_coupling_flow.ka_block_smc MODE REF_N JF_TAG [n_sweeps] [B]
  e.g.  ... e1 100 ka100tt 1200 128   |   ... e1 256 ka100tt 1500 64   |   ... safety 100 ka100tt"""
import os, sys, time, torch, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_swap_smc import (run_chain, swap_attempt, block_relabel_attempt,
                                                     uniform_weight_fn)
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr

dev = "cuda" if torch.cuda.is_available() else "cpu"
mode = sys.argv[1]; refN = int(sys.argv[2]); tag = sys.argv[3]
n_sweeps = int(sys.argv[4]) if len(sys.argv) > 4 else 1200
B = int(sys.argv[5]) if len(sys.argv) > 5 else 128
beta, step = 2.0, 0.12
ART = os.path.join(os.path.dirname(__file__), "artifacts")
JFD = os.path.join(os.path.dirname(__file__), "ipl44", "data")

refname = {100: "ka_reference_N100.pt", 256: "ka_reference_N256_thin.pt", 36: "ka_reference_N36_val.pt"}[refN]
ref = torch.load(os.path.join(ART, refname), map_location="cpu", weights_only=False)
N = ref["x"].shape[1]; L = float(ref["L"])
xr = torch.remainder(ref["x"].float(), L); sr = ref["s"].long()
nB = int(sr.sum())
efn = lambda a, b: ka_energy(a, b, L)
UrefN = float(efn(xr[-1024:].to(dev), sr[None].expand(1024, -1).to(dev)).median()) / N
print(f"[{mode}] N={N} L={L:.4f} nB={nB} | PT-ref <U>/N {UrefN:.4f} | thresholds -3.0 / -3.1", flush=True)

ck = torch.load(f"{JFD}/jf_{tag}_best.pt", map_location=dev, weights_only=False); cfg = ck["cfg"]
assert cfg.get("two_time") and cfg.get("system") == "ka"
jf = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"],
                      two_time=True).to(dev)                              # rebuilt at TARGET (N, L)
jf.load_state_dict(ck["state_dict"]); jf.eval()
print(f"loaded jf_{tag}_best (trained N={cfg['N']}) -> eval at N={N}", flush=True)
S_CANON = torch.zeros(1, N, dtype=torch.long, device=dev); S_CANON[:, :nB] = 1  # fixed s_ref for the table


def geometry_table(x):
    with torch.no_grad():
        Bx = x.shape[0]
        one = torch.ones(Bx, 1, 1, device=x.device); zero = torch.zeros(Bx, 1, 1, device=x.device)
        _, logits = jf(one, x, S_CANON.expand(Bx, -1), t_spec=zero)
        return torch.softmax(logits, -1)[..., 1]


def block_mover(k, n_att=8):
    def mv(x, s, U):
        W = geometry_table(x); tf = lambda _x: W; sacc = 0.0
        for _ in range(n_att):
            s, U, a = block_relabel_attempt(x, s, U, beta, efn, tf, k)
            sacc += a.float().mean().item()
        return s, U, sacc / n_att
    return mv


def pair_mover(n_att=8):
    def mv(x, s, U):
        sacc = 0.0
        for _ in range(n_att):
            s, U, a = swap_attempt(x, s, U, beta, efn, uniform_weight_fn)
            sacc += a.float().mean().item()
        return s, U, sacc / n_att
    return mv


def obs_fn(x, s):
    """g_BB peak (partial_gr returns (centers, g) numpy arrays; s per-config [B,N])."""
    _, g = partial_gr(x.cpu(), s.cpu(), L, rmax=3.0, nbins=90, pair=(1, 1))
    return float(g.max())


def sweeps_to(curves, thr):
    for k, u in zip(curves["sweep"], curves["U_median"]):
        if u / N <= thr:
            return k
    return None


def seeds(kind):
    s0 = random_22_labeling(B, N, nB, dev)
    if kind == "uniform":
        return torch.rand(B, N, 2, device=dev) * L, s0
    if kind == "flow":
        x0 = torch.rand(B, N, 2, device=dev) * L
        return jf.sample(x0, s0, n_steps=250)
    if kind == "reference":
        return xr[:B].to(dev), sr[None].expand(B, -1).to(dev).contiguous()
    raise ValueError(kind)


if mode == "safety":
    x0, s0 = seeds("reference")
    cur = run_chain(x0, s0, 2000, beta, L, efn, move_fn=block_mover(4), step=step, record_every=100,
                    obs_fn=obs_fn)
    un = [u / N for u in cur["U_median"]]
    print(f"safety: U/N trace {[f'{u:.3f}' for u in un]}", flush=True)
    drift = UrefN - min(un)
    print(f"safety: PT-ref {UrefN:.4f} | min over 2000 sweeps {min(un):.4f} | drift-below {drift:.4f} "
          f"({'OK — swap-safe' if drift < 0.03 else 'DRIFT — investigate'})", flush=True)
    sys.exit(0)

results = {}
for seed_kind in ["uniform", "flow"]:
    torch.manual_seed(7)
    x0, s0 = seeds(seed_kind)
    for name, mv in [("random", pair_mover()), ("block4", block_mover(4)), ("block8", block_mover(8))]:
        t0 = time.time()
        cur = run_chain(x0.clone(), s0.clone(), n_sweeps, beta, L, efn, move_fn=mv, step=step,
                        record_every=5, obs_fn=obs_fn)
        wall = time.time() - t0
        s30, s31 = sweeps_to(cur, -3.0), sweeps_to(cur, -3.1)
        results[(seed_kind, name)] = {**{k: cur[k] for k in ("sweep", "U_median", "gbb_peak", "swap_acc", "pos_acc")},
                                      "s30": s30, "s31": s31, "wall_s": wall}
        t30 = wall * (s30 / n_sweeps) if s30 else float("nan")
        print(f"{seed_kind:8s} {name:7s}: sweeps to -3.0 {s30} (t ~{t30:.0f}s) | to -3.1 {s31} "
              f"| final U/N {cur['U_median'][-1]/N:.4f} g_BB {cur['gbb_peak'][-1]:.2f} "
              f"| acc {sum(cur['swap_acc'])/max(len(cur['swap_acc']),1):.3f} | wall {wall:.0f}s", flush=True)
torch.save(results, f"{ART}/ka_block_bench_N{N}_{tag}.pt")

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
for c, seed_kind in enumerate(["uniform", "flow"]):
    for (sk, pn), r in results.items():
        if sk != seed_kind:
            continue
        axes[c].plot(r["sweep"], [u / N for u in r["U_median"]], label=f"{pn} (-3.0 @ {r['s30']})")
    axes[c].axhline(UrefN, ls="--", c="k"); axes[c].axhline(-3.0, ls=":", c="gray"); axes[c].axhline(-3.1, ls=":", c="gray")
    axes[c].set_title(f"N={N} {seed_kind} seeds: U/N vs sweeps"); axes[c].set_xlabel("sweep"); axes[c].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f"{ART}/ka_block_bench_N{N}_{tag}.png", dpi=110)
print(f"saved {ART}/ka_block_bench_N{N}_{tag}.png", flush=True)
