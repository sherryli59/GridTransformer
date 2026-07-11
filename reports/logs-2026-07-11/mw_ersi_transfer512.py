"""ZERO-SHOT TRANSFER, RUNG 3: N=64-trained eRSI flow deployed at N=512 (8x particles; box edge
L=10.391 = exactly 2x the training box, so ALL structure beyond r = L64/2 = 2.6 sigma is extrapolated).
Mirrors the N=216 protocol exactly so the size trend {64, 216, 512} is comparable:
  stage 1  flow generation (B=64, chunked) + particle-count/density verification + raw g(r)
  stage 2  REAL N=512 equilibrium, independent of the flow (14-rung anneal x20 sweeps + 40x10 target
           relaxation, B=32) -> the structure arbiter
  stage 3  3-arm seed amortization {uniform, exclvol(G=19: same 0.55-sigma cell as N=216's G=14), flow}
           on the identical 12-rung ladder, natural-rung entry, SW_PER=12, B=64
  stage 4  g(r) comparison plot + trend summary
Every stage/rung checkpoints to the artifact .pt the moment it finishes (CLAUDE.md: incremental)."""
import math, sys, time, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
IPL44 = "/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for m in [x for x in sys.modules if x.startswith("learndiffeq")]:
    del sys.modules[m]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
from liquid_coupling_flow.mw.mw_base import UniformBase, ExcludedVolumeBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda"
N = 512
L = (N / RHO_STAR) ** (1 / 3)
beta_t = 1.0 / T_STAR
STEP = 0.0783
SW_PER = 12
B_ARM, B_EQ = 64, 32
ART = "liquid_coupling_flow/mw/artifacts/mw_ersi_transfer_N512.pt"
CK = torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt",
                map_location=DEV, weights_only=False)
state = {"N": N, "L": L, "protocol": {"STEP": STEP, "SW_PER": SW_PER, "B_ARM": B_ARM, "B_EQ": B_EQ,
                                      "G_exclvol": 19, "ladder_rungs": 12, "eq_rungs": 14}}


def save():
    torch.save(state, ART)


def uN(x):
    return float(mw_energy_chunked(x, L).mean()) / N


def shells(r, gr):
    i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
    return float(gr[i1]), float(gr[i2])


print(f"N={N} L={L:.3f} beta={beta_t:.2f} (train N=64, L=5.195; L/2 here = {L/2:.2f} sigma)", flush=True)

# ---------- stage 1: zero-shot flow generation + verification ----------
egnn = EGNN_dynamics(n_particles=N, n_dimension=3, hidden_nf=128, n_layers=4,
                     max_neighbors=12, L=L, n_species=1).to(DEV)
egnn.load_state_dict({k[2:]: v for k, v in CK["state_dict"].items() if k.startswith("b.")}, strict=True)
egnn.eval()
print("size-agnostic weight load OK (N=64 -> N=512)", flush=True)


@torch.no_grad()
def flow_gen(Btot, chunk=32, steps=60, seed=0):
    outs = []
    for ci in range(0, Btot, chunk):
        Bc = min(chunk, Btot - ci)
        g = torch.Generator(device=DEV).manual_seed(seed + ci)
        x = torch.rand(Bc, N, 3, device=DEV, generator=g) * L
        a = torch.zeros(Bc, N, dtype=torch.long, device=DEV)
        ts = torch.linspace(0, 1, steps + 1, device=DEV)
        for i in range(steps):
            dt = float(ts[i + 1] - ts[i])
            v1, _ = egnn.forward_and_divergence(x, ts[i].expand(Bc), a)
            xm = torch.remainder(x + 0.5 * dt * v1, L)
            vm, _ = egnn.forward_and_divergence(xm, (ts[i] + 0.5 * dt).expand(Bc), a)
            x = torch.remainder(x + dt * vm, L)
        outs.append(x)
    return torch.cat(outs)


t0 = time.time()
fx = flow_gen(B_ARM)
# verification: genuinely N=512 at the right density
d = fx[:, :, None, :] - fx[:, None, :, :]
d = d - L * torch.round(d / L)
rmat = d.norm(dim=-1) + torch.eye(N, device=DEV) * 1e9
nn_med = float(rmat.min(dim=-1).values.median())
print(f"[stage1] flow gen {time.time()-t0:.0f}s | shape {tuple(fx.shape)} density {N/L**3:.4f} "
      f"(rho*={RHO_STAR}) NN-dist median {nn_med:.3f}", flush=True)
del d, rmat
r_f, g_f = g_r(fx.cpu(), L)
s1, s2 = shells(r_f, g_f)
uf = uN(fx)
print(f"[stage1] ZERO-SHOT STRUCTURE @ N=512: U/N {uf:+.4f}  shell1 {s1:.2f} shell2 {s2:.2f}  "
      f"(N=216 was 1.60/1.09; N=64 was 1.72/1.11)", flush=True)
state["flow"] = {"X": fx.cpu(), "U_per_N": uf, "r": r_f, "gr": g_f, "shells": (s1, s2), "nn_med": nn_med}
save()

# ---------- stage 2: REAL N=512 equilibrium (independent anneal, generous relaxation) ----------
base = UniformBase(N, L)
LAD14 = (0.5 * (beta_t / 0.5) ** torch.linspace(0, 1, 14)).tolist()
g = torch.Generator(device=DEV).manual_seed(5)
xq = torch.rand(B_EQ, N, 3, device=DEV, generator=g) * L
state["eq"] = {"anneal_hist": [], "relax_hist": []}
t0 = time.time()
for bi, bta in enumerate(LAD14):
    xq, U, _, _ = mutation_sweeps(xq, base, 1.0, bta, L, 20, STEP, g)
    um = float(U.mean()) / N
    state["eq"]["anneal_hist"].append((bta, um))
    state["eq"]["X_partial"] = xq.cpu()
    save()
    print(f"[stage2] anneal rung {bi:2d}/13 beta {bta:6.2f}  U/N {um:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
for ri in range(40):
    xq, U, _, _ = mutation_sweeps(xq, base, 1.0, beta_t, L, 10, STEP, g)
    um = float(U.mean()) / N
    state["eq"]["relax_hist"].append(um)
    if ri % 5 == 4:
        state["eq"]["X_partial"] = xq.cpu()
        save()
        print(f"[stage2] relax round {ri+1:2d}/40  U/N {um:+.4f}  ({time.time()-t0:.0f}s)", flush=True)
r_eq, g_eq = g_r(xq.cpu(), L)
e1, e2 = shells(r_eq, g_eq)
ueq = uN(xq)
print(f"[stage2] REAL N=512 equilibrium: U/N {ueq:+.4f}  shell1 {e1:.2f} shell2 {e2:.2f}", flush=True)
state["eq"].update({"X": xq.cpu(), "U_per_N": ueq, "r": r_eq, "gr": g_eq, "shells": (e1, e2)})
state["eq"].pop("X_partial", None)
save()

# ---------- stage 3: 3-arm seed amortization on the identical 12-rung ladder ----------
LADDER = (0.5 * (beta_t / 0.5) ** torch.linspace(0, 1, 12)).tolist()
gu = torch.Generator(device=DEV).manual_seed(1)
ux = torch.rand(B_ARM, N, 3, device=DEV, generator=gu) * L
t0 = time.time()
ge = torch.Generator(device=DEV).manual_seed(2)
ex = ExcludedVolumeBase(N, L, G=19).sample(B_ARM, ge).to(DEV)
print(f"[stage3] exclvol seed gen {time.time()-t0:.0f}s", flush=True)
seeds = {"uniform": ux, "exclvol": ex, "flow": fx}
seed_u = {k: uN(v) for k, v in seeds.items()}
for k, v in seed_u.items():
    print(f"[stage3] seed {k:8s}: U/N {v:+.4f}", flush=True)
state["arms"] = {"seed_U": seed_u}
save()


def anneal(x, sr, tag):
    x = x.clone(); ev = 0; h = []
    g = torch.Generator(device=DEV).manual_seed(0)
    t0 = time.time()
    for kk in range(sr, 12):
        x, U, _, info = mutation_sweeps(x, base, 1.0, LADDER[kk], L, SW_PER, STEP, g)
        ev += info["evals"]
        h.append((LADDER[kk], float(U.mean()) / N, ev))
        state["arms"][tag] = {"hist": h, "evals": ev, "start_rung": sr, "X_final": x.cpu()}
        save()
        print(f"[{tag}] rung {kk:2d} beta {LADDER[kk]:6.2f}  U/N {h[-1][1]:+.4f}  evals {ev:>11,}  "
              f"({time.time()-t0:.0f}s)", flush=True)
    return h, ev


hu, eu = anneal(ux, 0, "uniform")
eqmap = [x[1] for x in hu]


def nat(u):
    for kk in range(12):
        if eqmap[kk] <= u:
            return kk
    return 11


for k in ("exclvol", "flow"):
    sr = nat(seed_u[k])
    print(f"[stage3] {k} natural rung {sr} (beta {LADDER[sr]:.2f})", flush=True)
    anneal(seeds[k], sr, k)

print("\n=== N=512 ZERO-SHOT AMORTIZATION (vs uniform) ===", flush=True)
for k in ("uniform", "exclvol", "flow"):
    a = state["arms"][k]
    print(f"  {k:8s}: {a['evals']:>12,} evals  final U/N {a['hist'][-1][1]:+.4f}  ({eu/a['evals']:.2f}x)",
          flush=True)
save()

# ---------- stage 4: g(r) comparison plot + trend ----------
ext = torch.load("liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
L64 = (64 / RHO_STAR) ** (1 / 3)
r64, g64 = g_r(ext["cfgs"][-6400:], L64)
fig, ax = plt.subplots(figsize=(8.6, 5))
ax.plot(r_eq, g_eq, "k", lw=2.4, label=f"REAL N=512 equilibrium (annealed, U/N {ueq:.3f})")
ax.plot(r64, g64, color="gray", lw=1.3, ls=":", label="N=64 reference (g(r) size-invariance)")
ax.plot(r_f, g_f, "royalblue", lw=1.8, label="eRSI flow raw @ N=512 (ZERO-SHOT, 8x train size)")
ax.axvline(5.195 / 2, color="crimson", lw=1.0, ls="--", alpha=0.7)
ax.text(5.195 / 2 + 0.05, 0.25, "train-box L/2\n(beyond = extrapolated)", fontsize=8, color="crimson")
ax.set(xlabel="r (sigma)", ylabel="g(r)", xlim=(0, L / 2), ylim=(0, 2.3),
       title="N=512 zero-shot flow vs ACTUAL N=512 equilibrium g(r)")
ax.legend(fontsize=9)
fig.tight_layout()
out = "reports/logs-2026-07-11/mw_ersi_transfer512_gr.png"
fig.savefig(out, dpi=150)
print("PLOT:", out, flush=True)
print(f"\nshell1(1.19): real-eq {e1:.2f}  flow {s1:.2f}", flush=True)
print(f"shell2(1.85): real-eq {e2:.2f}  flow {s2:.2f}", flush=True)
print("\n=== SIZE TREND (train N=64) ===", flush=True)
print("  N=64 : shells 1.72/1.11 (eq 2.12/1.19)  amort flow 3.00x exclvol 1.33x", flush=True)
print("  N=216: shells 1.60/1.09 (eq 2.13/1.17)  amort flow 2.40x exclvol 1.20x", flush=True)
print(f"  N=512: shells {s1:.2f}/{s2:.2f} (eq {e1:.2f}/{e2:.2f})  amort flow "
      f"{eu/state['arms']['flow']['evals']:.2f}x exclvol {eu/state['arms']['exclvol']['evals']:.2f}x", flush=True)
save()
print("DONE", flush=True)
