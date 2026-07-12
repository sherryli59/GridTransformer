"""DESIGN C RACE: transport-conjugated jitter MCMC (NeuTra-style local moves through the eRSI flow)
vs single-site sweeps vs plain global RW, matched energy budget, racing the DEEP TAIL (-1.58 -> bank
plateau -1.625/-1.627) at the target beta.

Exactness: the C-chain state is z (base space); x = T(z) via the FORWARD map only; log alpha =
-beta dU - (lq' - lq) with lq the forward-accumulated exact density (autodiff-proven, +0.10 nats).
The numerical reverse map R is used ONCE to initialize z from the shared start population (any init
is legal); the roundtrip displacement |T(R(x)) - x| is measured and reported (never measured before).
All three arms start from the SAME projected population x0 = T(R(x_pop)).

Budget: 1 round = N site-evals/config for every arm (1 sweep == 1 full-energy proposal). 300 rounds.
Calibration phase picks sigma from acceptance x displacement; if acceptance ~ 0 at every sigma the
soft-core defect dominates Delta-m and C is dead (pre-registered failure mode)."""
import importlib.util, math, sys, time, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/ssd/GridTransformer")
spec = importlib.util.spec_from_file_location(
    "mw_flow_base", "/mnt/ssd/GridTransformer/reports/logs-2026-07-11/mw_flow_base.py")
mfb = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mfb)
FlowBase = mfb.FlowBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda"; N, B = 64, 64
L = (N / RHO_STAR) ** (1 / 3); beta = 1.0 / T_STAR
STEP = 0.0783; ROUNDS = 300; REC = 5
THRESH1, THRESH2 = -1.622, -1.625
fb = FlowBase("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt", N, L, DEV, steps=40)
base = UniformBase(N, L)
ART = "liquid_coupling_flow/mw/artifacts/mw_flow_c_race.pt"
state = {"protocol": {"rounds": ROUNDS, "beta": beta, "thresh": (THRESH1, THRESH2)}}
ts = torch.linspace(0, 1, fb.steps + 1, device=DEV)
A = torch.zeros(B, N, dtype=torch.long, device=DEV)


@torch.no_grad()
def reverse_z(x):
    xi = torch.remainder(x.clone(), L)
    for i in reversed(range(fb.steps)):
        dt = float(ts[i + 1] - ts[i])
        v1, _ = fb.e.forward_and_divergence(xi, ts[i + 1].expand(B), A)
        xm = torch.remainder(xi - 0.5 * dt * v1, L)
        vm, _ = fb.e.forward_and_divergence(xm, (ts[i + 1] - 0.5 * dt).expand(B), A)
        xi = torch.remainder(xi - dt * vm, L)
    return xi


@torch.no_grad()
def fwd_from_z(z):
    x = torch.remainder(z.clone(), L)
    lq = torch.full((B,), -N * 3 * math.log(L), device=DEV, dtype=torch.float64)
    for i in range(fb.steps):
        dt = float(ts[i + 1] - ts[i])
        v1, _ = fb.e.forward_and_divergence(x, ts[i].expand(B), A)
        xm = torch.remainder(x + 0.5 * dt * v1, L)
        vm, dm = fb.e.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(B), A)
        x = torch.remainder(x + dt * vm, L)
        lq = lq - dt * dm.double()
    return x, lq.float()


def mindist(a, b):
    d = a - b
    return (d - L * torch.round(d / L)).norm(dim=-1)


# ---------- shared start: project the -1.58 population through T(R(.)) ----------
ab = torch.load("liquid_coupling_flow/mw/artifacts/mw_flow_ab_speedup.pt", map_location="cpu", weights_only=False)
x_pop = ab["baseline"]["X_final"][:B].to(DEV)
print(f"start pop U/N {float(mw_energy(x_pop, L).mean())/N:+.4f}", flush=True)
z0 = reverse_z(x_pop)
x0, lq0 = fwd_from_z(z0)
U0 = mw_energy(x0, L)
rt = mindist(x0, x_pop)
print(f"roundtrip |T(R(x))-x|: mean {float(rt.mean()):.4f} median {float(rt.median()):.4f} "
      f"max {float(rt.max()):.4f} sigma-units | U/N after projection {float(U0.mean())/N:+.4f}", flush=True)
state["roundtrip"] = {"mean": float(rt.mean()), "max": float(rt.max())}

# ---------- sigma calibration ----------
print("\n=== sigma calibration (acceptance, median logalpha, mean |dx|) ===", flush=True)
gen = torch.Generator(device=DEV).manual_seed(0)
cal = []
for sig in (0.005, 0.01, 0.02, 0.05, 0.1, 0.2):
    zp = torch.remainder(z0 + sig * torch.randn(z0.shape, device=DEV, generator=gen), L)
    xp, lqp = fwd_from_z(zp)
    Up = mw_energy(xp, L)
    la = (-beta * (Up - U0) + (lq0 - lqp)).double()
    acc = float(torch.minimum(torch.ones_like(la), la.exp()).mean())
    dx = float(mindist(xp, x0).mean())
    cal.append((sig, acc, float(la.median()), dx))
    print(f"  sig {sig:5.3f}  acc {acc:6.4f}  med-logalpha {float(la.median()):+8.2f}  |dx| {dx:.4f}", flush=True)
state["calibration"] = cal
# pick sigma maximizing acc * dx^2 (expected squared displacement proxy)
SIG = max(cal, key=lambda c: c[1] * c[3] ** 2)[0]
print(f"picked SIG = {SIG}", flush=True)
torch.save(state, ART)

# ---------- the race (each arm: 300 rounds x N se/cfg) ----------
def race_sweep():
    x = x0.clone(); g = torch.Generator(device=DEV).manual_seed(1); tr = []
    for r in range(0, ROUNDS, REC):
        x, U, _, _ = mutation_sweeps(x, base, 1.0, beta, L, REC, STEP, g)
        tr.append((r + REC, float(U.mean()) / N))
    return x, tr, None


def race_plain_rw(sig_x=0.0098):
    x = x0.clone(); U = U0.clone(); g = torch.Generator(device=DEV).manual_seed(2); tr = []; n_acc = 0
    for r in range(ROUNDS):
        prop = torch.remainder(x + sig_x * torch.randn(x.shape, device=DEV, generator=g), L)
        Up = mw_energy(prop, L)
        a = torch.log(torch.rand(B, device=DEV, generator=g).clamp_min(1e-38)) < -beta * (Up - U)
        x = torch.where(a[:, None, None], prop, x); U = torch.where(a, Up, U); n_acc += int(a.sum())
        if (r + 1) % REC == 0:
            tr.append((r + 1, float(U.mean()) / N))
    return x, tr, n_acc / (ROUNDS * B)


def race_c():
    z = z0.clone(); x = x0.clone(); U = U0.clone(); lq = lq0.clone()
    g = torch.Generator(device=DEV).manual_seed(3); tr = []; n_acc = 0; t0 = time.time()
    for r in range(ROUNDS):
        zp = torch.remainder(z + SIG * torch.randn(z.shape, device=DEV, generator=g), L)
        xp, lqp = fwd_from_z(zp)
        Up = mw_energy(xp, L)
        la = -beta * (Up - U) + (lq - lqp)
        a = torch.log(torch.rand(B, device=DEV, generator=g).clamp_min(1e-38)) < la
        z = torch.where(a[:, None, None], zp, z); x = torch.where(a[:, None, None], xp, x)
        U = torch.where(a, Up, U); lq = torch.where(a, lqp, lq); n_acc += int(a.sum())
        if (r + 1) % REC == 0:
            tr.append((r + 1, float(U.mean()) / N))
        if (r + 1) % 50 == 0:
            print(f"  [C] round {r+1}/{ROUNDS}  U/N {float(U.mean())/N:+.4f}  acc so far "
                  f"{n_acc/((r+1)*B):.3f}  ({time.time()-t0:.0f}s)", flush=True)
            state["c_partial"] = {"tr": tr, "acc": n_acc / ((r + 1) * B)}
            torch.save(state, ART)
    return x, tr, n_acc / (ROUNDS * B)


print("\n=== race: 300 rounds x N se/cfg each ===", flush=True)
res = {}
for tag, fn in (("sweeps", race_sweep), ("plain-RW", race_plain_rw), ("C-move", race_c)):
    xf, tr, acc = fn()
    r, gr = g_r(xf.cpu(), L)
    i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
    cross1 = next((t[0] * N for t in tr if t[1] <= THRESH1), None)
    cross2 = next((t[0] * N for t in tr if t[1] <= THRESH2), None)
    res[tag] = {"tr": tr, "acc": acc, "final_U": tr[-1][1], "shells": (float(gr[i1]), float(gr[i2])),
                "cross1_se": cross1, "cross2_se": cross2, "X_final": xf.cpu()}
    print(f"[{tag:8s}] final U/N {tr[-1][1]:+.4f}  shells {float(gr[i1]):.2f}/{float(gr[i2]):.2f}"
          + (f"  acc {acc:.3f}" if acc is not None else "")
          + f"  cross(-1.622) {cross1 if cross1 else '—'} se/cfg  cross(-1.625) {cross2 if cross2 else '—'}", flush=True)
    state["race"] = res
    torch.save(state, ART)

fig, ax = plt.subplots(figsize=(8.8, 5))
for tag, d in res.items():
    ax.plot([t[0] * N for t in d["tr"]], [t[1] for t in d["tr"]], lw=1.7,
            label=tag + (f" (acc {d['acc']:.2f})" if d["acc"] is not None else ""))
ax.axhline(-1.627, color="k", ls=":", lw=1, label="bank plateau -1.627")
ax.set(xlabel="site-evals per config", ylabel="U/N",
       title=f"Deep-tail race at target beta (matched budget; C: sigma={SIG})")
ax.legend(fontsize=9); fig.tight_layout()
out = "reports/logs-2026-07-11/mw_flow_c_race.png"
fig.savefig(out, dpi=150)
print("PLOT:", out, flush=True)
print("DONE", flush=True)
