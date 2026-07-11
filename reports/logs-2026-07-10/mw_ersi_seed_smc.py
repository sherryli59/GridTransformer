"""Seed-SMC amortization (the deliverable): does seeding the annealer from the eRSI flow save energy
evaluations vs uniform seeds? Energy-only tempering ladder (beta 2 -> 10.38) via the certified
mutation_sweeps (UniformBase => density drops => exact MCMC targeting e^{-beta U} at each rung). Two arms,
identical protocol, differ ONLY in the initial population. Metric = du_move/mw_energy calls (each a stand-in
for an expensive MLIP evaluation) to reach equilibrium; the neural flow forwards are FREE in that accounting."""
import math, sys, time, torch
IPL44 = "/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for m in [x for x in sys.modules if x.startswith("learndiffeq")]:
    del sys.modules[m]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, B = 64, 256
L = (N / RHO_STAR) ** (1 / 3); beta_t = 1.0 / T_STAR
REF = -1.627
STEP = 0.0783
LADDER_FULL = 0.5 * (beta_t / 0.5) ** (torch.linspace(0, 1, 12))   # uniform: full hot->cold
LADDER_WARM = 3.5 * (beta_t / 3.5) ** (torch.linspace(0, 1, 6))    # flow: start at its natural temp (structure-preserving)
SW_PER = 12                                                    # relaxation sweeps per rung
base = UniformBase(N, L)


@torch.no_grad()
def flow_samples(B, ckpt, steps=60, seed=0):
    e = EGNN_dynamics(n_particles=N, n_dimension=3, hidden_nf=128, n_layers=4,
                      max_neighbors=12, L=L, n_species=1).to(DEV)
    c = torch.load(ckpt, map_location=DEV, weights_only=False)
    e.load_state_dict({k[2:]: v for k, v in c["state_dict"].items() if k.startswith("b.")}, strict=True)
    e.eval()
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.rand(B, N, 3, device=DEV, generator=g) * L
    a = torch.zeros(B, N, dtype=torch.long, device=DEV)
    ts = torch.linspace(0, 1, steps + 1, device=DEV)
    for i in range(steps):
        dt = float(ts[i + 1] - ts[i])
        v1, _ = e.forward_and_divergence(x, ts[i].expand(B), a)
        xm = torch.remainder(x + 0.5 * dt * v1, L)
        vm, _ = e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(B), a)
        x = torch.remainder(x + dt * vm, L)
    return x


def anneal(seed_x, tag, ladder):
    x = seed_x.clone(); evals = 0; hist = []
    g = torch.Generator(device=DEV).manual_seed(0)
    U0 = float(mw_energy_chunked(x, L).mean()) / N
    print(f"[{tag}] seed U/N {U0:+.4f}", flush=True)
    for bi, bta in enumerate(ladder):
        x, U, _, info = mutation_sweeps(x, base, 1.0, float(bta), L, SW_PER, STEP, g)
        evals += info["evals"]
        um = float(U.mean()) / N
        hist.append((float(bta), um, evals))
        print(f"[{tag}] rung {bi:2d} beta {float(bta):6.3f}  U/N {um:+.4f}  evals {evals:>10,}", flush=True)
    return hist, evals


ck = "liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt"
fx = flow_samples(B, ck)
print(f"flow seed pop: U/N {float(mw_energy_chunked(fx, L).mean())/N:+.4f}\n", flush=True)
g0 = torch.Generator(device=DEV).manual_seed(1)
ux = torch.rand(B, N, 3, device=DEV, generator=g0) * L

hu, eu = anneal(ux, "uniform", LADDER_FULL)
print()
hf, ef = anneal(fx, "flow  ", LADDER_WARM)


def evals_to(hist, thr=-1.60):
    for bta, um, ev in hist:
        if um <= thr:
            return ev
    return None


tu, tf = evals_to(hu), evals_to(hf)
print(f"\nAMORTIZATION (energy-evals to U/N <= -1.60):  uniform {tu}  flow {tf}  "
      f"ratio {tu/tf:.2f}x" if (tu and tf) else f"\nthreshold not reached: uniform {tu} flow {tf}", flush=True)
print(f"final U/N: uniform {hu[-1][1]:+.4f} ({eu:,} evals)  flow {hf[-1][1]:+.4f} ({ef:,} evals)  ref {REF}", flush=True)
torch.save({"hist_uniform": hu, "hist_flow": hf, "evals_uniform": eu, "evals_flow": ef,
            "ladder_full": LADDER_FULL, "ladder_warm": LADDER_WARM, "sw_per": SW_PER}, "liquid_coupling_flow/mw/artifacts/mw_ersi_seed_smc.pt")
