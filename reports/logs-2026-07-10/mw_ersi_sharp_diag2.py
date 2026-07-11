"""Decisive SNIS arbiter with a SELF-CONTAINED composed-likelihood integrator over ipl44's G-a-verified
forward_and_divergence — bypassing both learndiffeq RFM versions (the pip-installed learndiffeq-main has a
species-incompatible compute_div; the trained weights match ipl44's EGNN_dynamics exactly, 60/60 keys).
Forward ODE base->target with instantaneous change-of-variables: d/dt log q = -div v  =>  log q_1 = log q_0 - integral(div).
SNIS <U> vs the reference -1.627 decides whether the flow is usable for one-shot IS."""
import math, sys, torch
# --- force ipl44 (verified, species-aware) over the pip-installed learndiffeq-main ---
IPL44 = "/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for mod in [m for m in sys.modules if m.startswith("learndiffeq")]:
    del sys.modules[mod]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
import os
assert os.path.dirname(os.path.dirname(EGNN_dynamics.__module__ and __import__("learndiffeq").__file__)) == IPL44, \
    "did not load ipl44 learndiffeq"

from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, L = 64, 5.195309753663598
beta = 1.0 / T_STAR
egnn = EGNN_dynamics(n_particles=N, n_dimension=3, hidden_nf=128, n_layers=4,
                     max_neighbors=12, L=L, n_species=1).to(DEV)
lck = torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt",
                 map_location=DEV, weights_only=False)
egnn.load_state_dict({k[2:]: v for k, v in lck["state_dict"].items() if k.startswith("b.")}, strict=True)
egnn.eval()
print(f"ipl44 EGNN loaded, trained step={lck.get('global_step')}", flush=True)


@torch.no_grad()
def sample_and_logq(B, steps=60, seed=0):
    """Integrate uniform-torus base -> target, accumulating -integral(div). RK-less midpoint on the flat torus
    (exp map = add+wrap; flat metric => Euclidean div, which G-a verified exact)."""
    g = torch.Generator(device=DEV).manual_seed(seed)
    x = torch.rand(B, N, 3, device=DEV, generator=g) * L
    a = torch.zeros(B, N, dtype=torch.long, device=DEV)
    logq = torch.full((B,), -N * 3 * math.log(L), device=DEV, dtype=torch.float64)  # base density
    ts = torch.linspace(0, 1, steps + 1, device=DEV)
    for i in range(steps):
        t0 = ts[i].expand(B); dt = float(ts[i + 1] - ts[i])
        # midpoint
        v1, d1 = egnn.forward_and_divergence(x, t0, a)
        xm = torch.remainder(x + 0.5 * dt * v1, L)
        vm, dm = egnn.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(B), a)
        x = torch.remainder(x + dt * vm, L)
        logq = logq - dt * dm.double()
    return x, logq


for steps in (40, 80):
    X, logq = sample_and_logq(192, steps=steps, seed=0)
    U = mw_energy_chunked(X, L).double()
    logw = -beta * U - logq
    w = torch.softmax(logw, 0)
    ess = float(1.0 / (w ** 2).sum())
    u_snis = float((w * U).sum()) / N
    u_raw = float(U.mean()) / N
    r, gr = g_r(X.cpu(), L)
    i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
    print(f"steps={steps:3d}: ESS {ess:6.1f}/192   SNIS <U>/N {u_snis:+.4f} (ref -1.627)  "
          f"raw <U>/N {u_raw:+.3f}  raw g(r) shell1 {float(gr[i1]):.2f} shell2 {float(gr[i2]):.2f} (data 2.12/1.19)",
          flush=True)
torch.save({"X": X.cpu(), "logq": logq.cpu(), "ess": ess, "u_snis": u_snis, "r": r, "gr": gr,
            "step": lck.get("global_step")}, "liquid_coupling_flow/mw/artifacts/mw_ersi_sharp_diag2.pt")
