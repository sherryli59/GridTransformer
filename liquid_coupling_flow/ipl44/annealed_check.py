"""Two-sided convergence check: block4 chains from UNIFORM vs REFERENCE seeds must converge to the SAME
(U, g_BB) if the low-U states are the genuine annealed joint-(x,s) equilibrium (kernel exactness already
enumeration-verified). 2000 sweeps, B=128."""
import os, sys, torch
from liquid_coupling_flow.ipl44.ipl_swap_smc import run_chain, block_relabel_attempt
from liquid_coupling_flow.ipl44.ipl_energy import ipl_energy, ipl_gr_partials, ipl_box
from liquid_coupling_flow.ipl44.joint_flow import JointSpeciesFlow, random_22_labeling
dev = "cuda"; N, L = ipl_box(); Lf = float(L); beta = 10.0; B = 128
ART = os.path.join(os.path.dirname(__file__), "data")
ck = torch.load(f"{ART}/jf_midtt_best.pt", map_location=dev, weights_only=False); cfg = ck["cfg"]
jf = JointSpeciesFlow(n_particles=N, L=Lf, hidden_nf=cfg["hidden_nf"], n_layers=cfg["n_layers"], two_time=True).to(dev)
jf.load_state_dict(ck["state_dict"]); jf.eval()
S_REF = torch.zeros(1, N, dtype=torch.long, device=dev); S_REF[:, 22:] = 1

def geometry_table(x):
    with torch.no_grad():
        Bx = x.shape[0]
        one = torch.ones(Bx, 1, 1, device=x.device); zero = torch.zeros(Bx, 1, 1, device=x.device)
        _, logits = jf(one, x, S_REF.expand(Bx, -1), t_spec=zero)
        return torch.softmax(logits, -1)[..., 1]

def block4_mover(x, s, U):
    W = geometry_table(x); tf = lambda _x: W; sacc = 0.0
    for _ in range(8):
        s, U, a = block_relabel_attempt(x, s, U, beta, ipl_energy, tf, 4)
        sacc += a.float().mean().item()
    return s, U, sacc / 8

D = "/mnt/ssd/GridTransformer/datasets"
xr = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)
sr = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()
o = sr.argsort(-1); sr = torch.gather(sr, 1, o); xr = torch.gather(xr, 1, o.unsqueeze(-1).expand(-1, -1, 2))
seeds = {"uniform": (torch.rand(B, N, 2, device=dev) * Lf, random_22_labeling(B, N, 22, dev)),
         "reference": (xr[:B].to(dev), sr[:B].to(dev))}
for name, (x0, s0) in seeds.items():
    cur = run_chain(x0, s0, 2000, beta, Lf, ipl_energy, move_fn=block4_mover, record_every=100)
    print(f"{name:9s}: U_med trace {[f'{u:.1f}' for u in cur['U_median']]}", flush=True)
    print(f"{name:9s}: g_BB trace  {[f'{g:.2f}' for g in cur['gbb_peak']]}", flush=True)
