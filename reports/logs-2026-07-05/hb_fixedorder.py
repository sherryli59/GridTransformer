"""Localize the sweep bias: run the heat-bath sweep with FIXED slot-ordering (order once, never re-order) vs
the normal re-order-per-move sweep. If fixed-order matches exact-displacement (-3.276) but re-order gives
-3.300, the per-move re-slot-ordering breaks detailed balance."""
import torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_heatbath_gate import _env
from liquid_coupling_flow.ka_heatbath import single_site_mh, single_site_mh_sweep
from liquid_coupling_flow.ka_energy import ka_energy

DEV="cuda" if torch.cuda.is_available() else "cpu"
sc,L,geo,pos,s,HB,N=_env(B=128)
u0=(ka_energy(pos,s,L)/N).mean().item(); print(f"start {u0:.4f}",flush=True)

# FIXED-ORDER sweep: order once (pos already slot-ordered by _env); move each slot, scatter in place, NO re-order
def fixed_sweep(pos,s,gen):
    B=pos.shape[0]
    for _ in range(N):
        site=int(torch.randint(0,N,(1,),device=DEV,generator=gen).item())
        pos,_=single_site_mh(HB,pos,s,site,sc,L,2.0,gen=gen)   # pos stays in the same (fixed) slot frame
    return pos
pf=pos.clone(); gen=torch.Generator(device=DEV).manual_seed(0)
for k in range(1,5):
    for _ in range(10): pf=fixed_sweep(pf,s,gen)
    print(f"  FIXED-ORDER block {k} (+10 sw): <U>/N {(ka_energy(pf,s,L)/N).mean().item():.4f}",flush=True)
print("compare: FIXED-ORDER vs exact-disp -3.276 vs re-order heat-bath -3.300",flush=True)
