"""E2E corrector v2 (advanced kernels): MALA positions (all-particle gradient moves, exact MH) + learned
block-species moves off the frozen geometry table. One 'iteration' = 10 MALA steps + 1 species block
(8 x block-8 relabel attempts). Full trajectory retention (per-iter U[B], s[B,N] int8, x snapshots every 20
iters, acceptances incl. effective), incremental saves every 200 iterations.
Usage: python -m liquid_coupling_flow.ka_e2e_mala [n_iter] [B] [dt] [kern=block8|random]"""
import os, sys, time, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import position_mala, block_relabel_attempt, swap_attempt, uniform_weight_fn
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow
from liquid_coupling_flow.ka_energy import ka_energy, ka_forces

dev = "cuda"
n_iter = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
B = int(sys.argv[2]) if len(sys.argv) > 2 else 128
dt = float(sys.argv[3]) if len(sys.argv) > 3 else 0.012
kern = sys.argv[4] if len(sys.argv) > 4 else "block8"
beta = 2.0
ART = os.path.join(os.path.dirname(__file__), "artifacts")
JFD = os.path.join(os.path.dirname(__file__), "ipl44", "data")

gen = torch.load(f"{ART}/ka_e2e_gen_N256.pt", map_location="cpu", weights_only=False)
N = gen["meta"]["N"]; L = gen["meta"]["L"]
x = gen["x"][:B].to(dev).contiguous(); s = gen["s"][:B].to(dev).contiguous()
nB = int(s[0].sum())
efn = lambda a, b: ka_energy(a, b, L); ffn = lambda a, b: ka_forces(a, b, L)
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
OUT = f"{ART}/ka_e2e_traj_mala_{kern}_N{N}.pt"
Us, Ss, Xsnap, macc_l, sacc_l, seff_l, dis = [], [], [], [], [], [], []
t0 = time.time()
print(f"[mala-{kern}] {B} configs, {n_iter} iters (10 MALA dt={dt} + species block each) | start U/N {float(U.median())/N:.2f}", flush=True)
for k in range(n_iter):
    ma = 0.0
    for _ in range(10):
        x, U, a = position_mala(x, s, U, beta, L, dt, efn, ffn); ma += a
    sa = 0.0; ch = 0.0
    if kern == "random":
        for _ in range(8):
            s_old = s; s, U, a2 = swap_attempt(x, s, U, beta, efn, uniform_weight_fn)
            sa += a2.float().mean().item(); ch += (s != s_old).any(1).float().mean().item()
        W = geometry_table(x)                              # oracle still evaluated for the disagreement metric
    else:
        W = geometry_table(x); tf = lambda _x: W
        for _ in range(8):
            s_old = s; s, U, a2 = block_relabel_attempt(x, s, U, beta, efn, tf, 8)
            sa += a2.float().mean().item(); ch += (s != s_old).any(1).float().mean().item()
    Us.append((U / N).cpu().clone()); Ss.append(s.cpu().to(torch.int8))
    macc_l.append(ma / 10); sacc_l.append(sa / 8); seff_l.append(ch / 8)
    if k % 20 == 0:
        Xsnap.append(x.cpu().clone())
        dis.append(float(((W > 0.5).long() != s).float().mean()))
    if (k + 1) % 200 == 0 or k == n_iter - 1:
        torch.save({"UoN": torch.stack(Us), "s": torch.stack(Ss), "x_snap": torch.stack(Xsnap),
                    "snap_every": 20, "macc": macc_l, "sacc": sacc_l, "seff": seff_l, "disagree": dis,
                    "dt": dt, "B": B, "N": N, "L": L, "beta": beta, "mala_per_iter": 10,
                    "x_final": x.cpu(), "s_final": s.cpu()}, OUT)
        print(f"[mala-{kern}] iter {k+1:5d} | U/N med {float(U.median())/N:.4f} | disagree {dis[-1]:.4f} "
              f"| mala acc {ma/10:.2f} sw acc {sa/8:.2f} eff {ch/8:.3f} | {time.time()-t0:.0f}s (saved)", flush=True)
print(f"[mala-{kern}] DONE -> {OUT}", flush=True)
