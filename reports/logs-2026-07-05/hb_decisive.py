"""DECISIVE: seed EXACT displacement from the heat-bath's -3.31 config. Displacement is known-exact.
If it HOLDS -3.31 -> -3.31 is a true equilibrium (heat-bath exact + faster equilibrator = POSITIVE).
If it RELAXES UP to ~-3.276 -> heat-bath over-cooled below equilibrium (BIASED = bug to fix)."""
import torch
from liquid_coupling_flow.ka_heatbath_gate import _env
from liquid_coupling_flow.ka_heatbath import single_site_mh_sweep
from liquid_coupling_flow.ka_local_smc import _disp_sweeps, _canonicalize
from liquid_coupling_flow.ka_energy import ka_energy

DEV="cuda" if torch.cuda.is_available() else "cpu"
sc,L,geo,pos,s,HB,N=_env(B=128)
gen=torch.Generator(device=DEV).manual_seed(0)
# 1) heat-bath cool to -3.31
ph=pos.clone()
for _ in range(30): ph,_=single_site_mh_sweep(HB,ph,s,sc,L,geo,beta=2.0,gen=gen)
ph,sc_s=_canonicalize(ph,s)
print(f"heat-bath cooled: <U>/N {(ka_energy(ph,sc_s,L)/N).mean().item():.4f}",flush=True)
# 2) EXACT displacement FROM the cooled config -- does it hold or relax up?
pd=ph.clone()
for k in range(1,9):
    pd=_disp_sweeps(pd, sc_s[0], L, kT=0.5, n=200)
    print(f"  EXACT-DISP from cooled, block {k} (+200 sw): <U>/N {(ka_energy(pd,sc_s,L)/N).mean().item():.4f}",flush=True)
print("HOLD ~-3.31 => heat-bath EXACT+faster; RELAX to ~-3.276 => heat-bath BIASED",flush=True)
