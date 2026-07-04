"""E2E stage C: species-flow correction of the SAVED raw generation (ka_e2e_gen_N256.pt), FULL trajectory
retention: per-sweep U[t,B] and s[t,B,N] (int8), x snapshots every 10 sweeps, per-sweep acceptance
(accepted AND effective). Incremental torch.save every 250 sweeps -> analysis can run mid-flight.
Usage: python -m liquid_coupling_flow.ka_e2e_correct KERNEL(block8|random) [n_sweeps] [B]"""
import os, sys, time, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import position_sweep, swap_attempt, block_relabel_attempt, uniform_weight_fn
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
kern = sys.argv[1]; n_sweeps = int(sys.argv[2]) if len(sys.argv) > 2 else 3000
B = int(sys.argv[3]) if len(sys.argv) > 3 else 128
beta, step = 2.0, 0.12
ART = os.path.join(os.path.dirname(__file__), "artifacts")
JFD = os.path.join(os.path.dirname(__file__), "ipl44", "data")

gen = torch.load(f"{ART}/ka_e2e_gen_N256.pt", map_location="cpu", weights_only=False)
N = gen["meta"]["N"]; L = gen["meta"]["L"]
x = gen["x"][:B].to(dev); s = gen["s"][:B].to(dev)
nB = int(s[0].sum()); efn = lambda a, b: ka_energy(a, b, L)
ck = torch.load(f"{JFD}/jf_ka100tt_best.pt", map_location=dev, weights_only=False); cfg = ck["cfg"]
jf = JointSpeciesFlow(n_particles=N, L=L, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"], two_time=True).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval(); jf.knn = 32
S_CANON = torch.zeros(1, N, dtype=torch.long, device=dev); S_CANON[:, :nB] = 1


def geometry_table(xx):
    with torch.no_grad():
        Bx = xx.shape[0]
        _, lg = jf(torch.ones(Bx, 1, 1, device=dev), xx, S_CANON.expand(Bx, -1),
                   t_spec=torch.zeros(Bx, 1, 1, device=dev))
        return torch.softmax(lg, -1)[..., 1]


U = efn(x, s)
OUT = f"{ART}/ka_e2e_traj_{kern}_N{N}.pt"
Us, Ss, Xsnap, pacc, sacc, seff, dis = [], [], [], [], [], [], []
t0 = time.time()
print(f"[{kern}] correcting {B} configs, {n_sweeps} sweeps | start U/N median {float(U.median())/N:.2f}", flush=True)
for k in range(n_sweeps):
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
    Us.append((U / N).cpu().clone()); Ss.append(s.cpu().to(torch.int8))
    pacc.append(pa); sacc.append(sa / 8); seff.append(ch / 8)
    if k % 10 == 0:
        Xsnap.append(x.cpu().clone())
        W = geometry_table(x)
        dis.append(float(((W > 0.5).long() != s).float().mean()))
    if (k + 1) % 250 == 0 or k == n_sweeps - 1:
        torch.save({"UoN": torch.stack(Us), "s": torch.stack(Ss), "x_snap": torch.stack(Xsnap),
                    "snap_every": 10, "pacc": pacc, "sacc": sacc, "seff": seff, "disagree": dis,
                    "kern": kern, "B": B, "N": N, "L": L, "beta": beta, "step": step,
                    "x_final": x.cpu(), "s_final": s.cpu()}, OUT)
        print(f"[{kern}] sweep {k+1:5d} | U/N med {float(U.median())/N:.4f} | disagree {dis[-1]:.4f} "
              f"| pos acc {pa:.3f} sw acc {sa/8:.3f} eff {ch/8:.3f} | {time.time()-t0:.0f}s (saved)", flush=True)
print(f"[{kern}] DONE -> {OUT}", flush=True)
