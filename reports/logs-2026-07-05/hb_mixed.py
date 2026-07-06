"""Is A2 usable in the SMC stack? Mix heat-bath with displacement (both should be pi-invariant). If the mixed
kernel converges to the true -3.276, the standalone over-cooling is quasi-ergodic and the co-mutation fixes it.
If it settles between -3.276 and -3.30, the heat-bath move is genuinely biased."""
import torch
from liquid_coupling_flow.ka_heatbath_gate import _env
from liquid_coupling_flow.ka_heatbath import single_site_mh_sweep
from liquid_coupling_flow.ka_local_smc import _disp_sweeps, _canonicalize
from liquid_coupling_flow.ka_energy import ka_energy
DEV="cuda" if torch.cuda.is_available() else "cpu"
sc,L,geo,pos,s,HB,N=_env(B=128)
pos,s=_canonicalize(pos,s); gen=torch.Generator(device=DEV).manual_seed(0)
print(f"start {(ka_energy(pos,s,L)/N).mean().item():.4f}",flush=True)
pm=pos.clone()
for k in range(1,11):
    pm,_=single_site_mh_sweep(HB,pm,s,sc,L,geo,beta=2.0,gen=gen)
    pm,s2=_canonicalize(pm,s)
    pm=_disp_sweeps(pm, s2[0], L, kT=0.5, n=40); s=s2
    print(f"  MIXED (hb+40disp) block {k}: <U>/N {(ka_energy(pm,s,L)/N).mean().item():.4f}",flush=True)
print("converge -3.276 => usable (quasi-ergodic, disp fixes); settle <-3.28 => biased move",flush=True)
