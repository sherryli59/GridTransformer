"""DOES OUR SMC RECOVER THE CORRECT EQUILIBRIUM DISTRIBUTION? Two-sided convergence bracket at N=64,
beta=10.381, under the exact single-site kernel (pi_beta-invariant, energy-only):

  cold side:  bank configs (mw_ref_N64_ext, alleged equilibrium -1.627) -- must stay FLAT
  hot side 1: certified smc_run final population (mw_smc_g2_s0_N64, resampled by its final logw)
  hot side 2: geometric flow-path SMC final population (resampled by its final logw, post-finisher)
  hot side 3: fresh amortization-protocol endpoint (uniform seeds -> 12-rung x 12-sweep ladder)

All four continued 1200 sweeps at the target; equilibrium recovered iff hot arms converge onto the
cold plateau in U/N, g(r) shells, AND the full U-distribution (max-CDF-distance vs bank at the end).
Distribution-level, not means-only. Configs + traces saved (record-simulation-data)."""
import sys, time, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda"; N, B = 64, 64
L = (N / RHO_STAR) ** (1 / 3); beta = 1.0 / T_STAR
STEP = 0.0783; SWEEPS, REC = 1200, 25
base = UniformBase(N, L)
ART = "liquid_coupling_flow/mw/artifacts/mw_smc_eq_check.pt"


def resample_by(x, logw, gen):
    idx = torch.multinomial(torch.softmax(logw.double(), 0), x.shape[0], replacement=True, generator=gen)
    return x[idx]


gen = torch.Generator(device=DEV).manual_seed(0)
arms = {}
ext = torch.load("liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
arms["bank(cold)"] = ext["cfgs"][-B:].to(DEV).clone()

g2 = torch.load("liquid_coupling_flow/mw/artifacts/mw_smc_g2_s0_N64.pt", map_location=DEV, weights_only=False)
arms["certified-smc"] = resample_by(g2["x"], g2["logw"], gen)[:B].clone()

geo = torch.load("liquid_coupling_flow/mw/artifacts/mw_flow_smc_geo_N64.pt", map_location=DEV, weights_only=False)
arms["flow-geo-smc"] = resample_by(geo["x"].to(DEV), geo["logw"].to(DEV), gen).clone()

gu = torch.Generator(device=DEV).manual_seed(5)
xu = torch.rand(B, N, 3, device=DEV, generator=gu) * L
LAD = (0.5 * (beta / 0.5) ** torch.linspace(0, 1, 12)).tolist()
for bta in LAD:
    xu, _, _, _ = mutation_sweeps(xu, base, 1.0, bta, L, 12, STEP, gu)
arms["amort-ladder"] = xu.clone()

state = {"protocol": {"sweeps": SWEEPS, "rec": REC, "beta": beta, "step": STEP}, "traces": {}, "final": {}}
print(f"{'arm':>14}  start U/N", flush=True)
for k, v in arms.items():
    print(f"{k:>14}  {float(mw_energy(v, L).mean())/N:+.4f}", flush=True)

t0 = time.time()
for k in arms:
    x = arms[k]
    g = torch.Generator(device=DEV).manual_seed(hash(k) % 2**31)
    tr = []
    pool = []
    for sw in range(0, SWEEPS, REC):
        x, U, _, _ = mutation_sweeps(x, base, 1.0, beta, L, REC, STEP, g)
        tr.append((sw + REC, float(U.mean()) / N))
        if sw + REC > SWEEPS - 400:              # pool final 400 sweeps for distribution tests
            pool.append((x.cpu().clone(), U.cpu().clone()))
    state["traces"][k] = tr
    state["final"][k] = {"X": torch.cat([p[0] for p in pool]), "U": torch.cat([p[1] for p in pool])}
    torch.save(state, ART)
    print(f"[{k}] done: U/N {tr[0][1]:+.4f} -> {tr[-1][1]:+.4f}  ({time.time()-t0:.0f}s)", flush=True)

# ---- distribution-level verdicts ----
Ub = state["final"]["bank(cold)"]["U"].double() / N
print("\n=== VERDICT (final 400 sweeps pooled) ===", flush=True)
print(f"{'arm':>14} {'<U/N>':>9} {'std':>7} {'maxCDFdist vs bank':>20} {'shell1':>7} {'shell2':>7}", flush=True)
res = {}
for k in arms:
    Ua = state["final"][k]["U"].double() / N
    us = torch.sort(torch.cat([Ua, Ub])).values
    cdfa = torch.searchsorted(torch.sort(Ua).values, us, right=True).double() / len(Ua)
    cdfb = torch.searchsorted(torch.sort(Ub).values, us, right=True).double() / len(Ub)
    ks = float((cdfa - cdfb).abs().max())
    r, gr = g_r(state["final"][k]["X"][-256:], L)
    i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
    res[k] = (float(Ua.mean()), ks, float(gr[i1]), float(gr[i2]))
    print(f"{k:>14} {float(Ua.mean()):>+9.4f} {float(Ua.std()):>7.4f} {ks:>20.3f} {float(gr[i1]):>7.2f} {float(gr[i2]):>7.2f}", flush=True)
state["verdict"] = res
torch.save(state, ART)

fig, ax = plt.subplots(figsize=(8.8, 5))
for k, tr in state["traces"].items():
    ax.plot([t[0] for t in tr], [t[1] for t in tr], lw=1.7, label=k)
ax.axhline(-1.627, color="k", ls=":", lw=1, label="reference -1.627")
ax.set(xlabel="continued exact-MCMC sweeps at target beta", ylabel="U/N",
       title="Two-sided equilibrium bracket: SMC outputs vs bank under the exact kernel")
ax.legend(fontsize=9); fig.tight_layout()
out = "reports/logs-2026-07-11/mw_smc_eq_check.png"
fig.savefig(out, dpi=150)
print("PLOT:", out, flush=True)
print("DONE", flush=True)
