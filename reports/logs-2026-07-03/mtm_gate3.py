"""K1 stationarity CONFIRMATION from a RANDOM (representative) reference slice: expect FLAT U/N ~ -3.266."""
import os, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import mtm_sweep
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr
DEV="cuda"; BETA=2.0; B=64; torch.manual_seed(7)
sc,L,geo=_scaffold(100,DEV)
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
s0=ref["s"].to(DEV).long()
idx=torch.randperm(ref["x"].shape[0],device=DEV)[:B]
pos,sso=slot_order(ref["x"][idx].to(DEV),s0,geo,100)
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)
U0=float(ka_energy(pos,sso,L).mean())/100
_,gd=partial_gr(pos,sso,L,4.0,60,(1,1))
print(f"RANDOM-SLICE BASELINE: U/N {U0:.4f}  g_BB {float(np.asarray(gd).max()):.2f}",flush=True)
cur=pos.clone()
for sweep in range(30):
    cur,mp=mtm_sweep(P,cur,sso,sc,L,M=8,beta=BETA)
    if (sweep+1)%5==0:
        U=float(ka_energy(cur,sso,L).mean())/100
        _,gg=partial_gr(cur,sso,L,4.0,60,(1,1))
        print(f"K1 sweep {sweep+1:2d}: U/N {U:.4f} (drift {U-U0:+.4f})  g_BB {float(np.asarray(gg).max()):.2f}  move_prob {mp:.4f}",flush=True)
print("CONFIRMATION COMPLETE",flush=True)
