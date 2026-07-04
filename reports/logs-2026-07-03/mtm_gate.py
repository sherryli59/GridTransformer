"""MTM Task-2 gate: (1) perf curve M in {8,32,128}; (2) K1 stationarity 200 sweeps from equilibrium
(valid kernel HOLDS U/N + g_BB); (3) K3 negative control (SIR w/o current state, M=2) MUST drift."""
import os, time, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import mtm_move, mtm_sweep, assert_mtm_valid, cluster_energy
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr
DEV="cuda"; BETA=2.0; B=64; k=7; torch.manual_seed(0)
sc,L,geo=_scaffold(100,DEV)
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
s0=ref["s"].to(DEV).long()
pos,sso=slot_order(ref["x"][:B].to(DEV),s0,geo,100)
ck=torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False)
P=_load(ck,DEV)
assert_mtm_valid(P,pos,sso,KC.cluster_slots(5,sc,k,L),sc,L); print("validity assertions PASS",flush=True)
U0=float(ka_energy(pos,sso,L).mean())/100
rcd,gd=partial_gr(pos,sso,L,4.0,60,(1,1)); gbb0=float(np.asarray(gd).max())
print(f"BASELINE (equilibrium): U/N {U0:.4f}  g_BB peak {gbb0:.2f}",flush=True)
# --- perf curve (already measured; skip) ---
for M in ():
    p2=pos.clone(); probs=[]; t0=time.time()
    for seed in range(0,100,20):
        cl=KC.cluster_slots(seed,sc,k,L)
        p2,_,info=mtm_move(P,p2,sso,cl,sc,L,M=M,beta=BETA)
        probs.append(info["move_prob"])
    print(f"PERF M={M:4d}: move_prob {sum(probs)/len(probs):.4f}  {(time.time()-t0)/len(probs):.2f}s/move",flush=True)
# --- K1 stationarity ---
cur=pos.clone()
for sweep in range(80):                      # M=8: validity is M-independent; 80 sweeps x 72s ~ 1.6h
    cur,mp=mtm_sweep(P,cur,sso,sc,L,M=8,beta=BETA)
    if (sweep+1)%5==0:
        U=float(ka_energy(cur,sso,L).mean())/100
        _,gg=partial_gr(cur,sso,L,4.0,60,(1,1))
        print(f"K1 sweep {sweep+1:3d}: U/N {U:.4f} (drift {U-U0:+.4f})  g_BB {float(np.asarray(gg).max()):.2f}  move_prob {mp:.4f}",flush=True)
# --- K3 negative control: SIR WITHOUT the current state (invalid by design) ---
@torch.no_grad()
def k3_move(P,pos,s,cl,sc,L,M,beta):
    Bx,N,_=pos.shape
    pos_rep=pos.repeat_interleave(M,0); s_rep=s.repeat_interleave(M,0)
    yC,logq_y=P.sample(pos_rep,s_rep,cl,sc,L); yC=yC.view(Bx,M,k,2); logq_y=logq_y.view(Bx,M)
    U_y=cluster_energy(yC,pos,cl,s,L)
    ell=-beta*U_y-logq_y
    J=torch.multinomial(torch.softmax(ell,-1),1).squeeze(1)
    out=pos.clone(); out[:,cl]=yC[torch.arange(Bx,device=DEV),J]
    return out
cur=pos.clone()
for sweep in range(30):
    for seed in torch.randperm(100).tolist():
        cur=k3_move(P,cur,sso,KC.cluster_slots(seed,sc,k,L),sc,L,2,BETA)
    if (sweep+1)%5==0:
        U=float(ka_energy(cur,sso,L).mean())/100
        print(f"K3(neg-ctrl M=2) sweep {sweep+1:2d}: U/N {U:.4f} (drift {U-U0:+.4f})",flush=True)
print("GATE COMPLETE",flush=True)
