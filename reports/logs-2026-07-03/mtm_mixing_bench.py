"""DECISIVE GATE v2: MTM cluster moves vs swap-MC at MATCHED WALL-CLOCK.
FIX vs v1: ALL arms keep the state in the ORIGINAL particle ordering (single species vector s0 exact for every
row — v1's slot_order gave per-row species maps and the vectorized swap displacement used row-0's = invalid,
U/N went POSITIVE). MTM moves get their slot ordering ON THE FLY per move and scatter the moved cluster back.
Arms from identical shallow-slice starts (U/N -3.2194, equilibrium -3.2600): swap | mtm(M=32) | hybrid(~50/50).
Metrics vs wall-clock: U/N, Q=frac(particles within 0.3 of start), MSD."""
import os, time, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import mtm_move, assert_mtm_valid
from liquid_coupling_flow.ka_cluster_flow import _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_mcmc import _pair_energy_one
from liquid_coupling_flow.ka_gridformer import _wrap_pm

DEV="cuda"; BETA=2.0; KT=0.5; B=64; N=100; K=7; STEP=0.04; NSWAP=N//8; M=32
BUDGET=900.0; REPORT=15.0
torch.manual_seed(0)
sc,L,geo=_scaffold(N,DEV)
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
s0=ref["s"].to(DEV).long()                                   # ONE species vector, exact for every row (original order)
pos0=ref["x"][:B].to(DEV).clone()                            # shallow slice, ORIGINAL ordering
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)
U0=float(ka_energy(pos0,s0,L).mean())/N
print(f"START (shallow slice, original order): U/N {U0:.4f}   equilibrium ~ -3.2600",flush=True)

def metrics(x,x_start):
    U=float(ka_energy(x,s0,L).mean())/N
    d=_wrap_pm(x-x_start,L).norm(dim=-1)
    return U,float((d<0.3).float().mean()),float((d**2).mean())

Aidx=(s0==0).nonzero().squeeze(-1); Bidx=(s0==1).nonzero().squeeze(-1)
def swap_sweep(x):
    prop=torch.remainder(x+STEP*torch.randn_like(x),L)
    Uold=_u_matrix(x,x,s0,s0,L,exclude_diag=True); Unew=_u_matrix(prop,x,s0,s0,L,exclude_diag=True)
    acc=torch.log(torch.rand(B,N,device=DEV))<(-(Unew-Uold).sum(-1)/KT)
    x=torch.where(acc[:,:,None],prop,x)
    for _ in range(NSWAP):
        i=Aidx[torch.randint(len(Aidx),(1,))].item(); j=Bidx[torch.randint(len(Bidx),(1,))].item()
        xi,xj=x[:,i,:].clone(),x[:,j,:].clone()
        e_old=(_pair_energy_one(xi,s0[i].expand(B),x,s0,i,L)+_pair_energy_one(xj,s0[j].expand(B),x,s0,j,L))
        xp=x.clone(); xp[:,i,:]=xj; xp[:,j,:]=xi
        e_new=(_pair_energy_one(xj,s0[i].expand(B),xp,s0,i,L)+_pair_energy_one(xi,s0[j].expand(B),xp,s0,j,L))
        a=torch.log(torch.rand(B,device=DEV))<(-(e_new-e_old)/KT)
        x[:,i,:]=torch.where(a[:,None],xj,xi); x[:,j,:]=torch.where(a[:,None],xi,xj)
    return x

def mtm_step(x,seed):
    """Slot-order on the fly, one MTM move, scatter the cluster back into the ORIGINAL ordering."""
    order=geo._curve_order(x,N)                              # [B,N] slot j <- particle order[b,j]
    xo=torch.gather(x,1,order[...,None].expand(-1,-1,2)); so=s0[order]
    cl=KC.cluster_slots(seed,sc,K,L)
    xo_new,moved,info=mtm_move(P,xo,so,cl,sc,L,M=M,beta=BETA)
    idx=order[:,cl]                                          # original indices of the cluster particles
    x=x.clone(); x[torch.arange(B,device=DEV)[:,None],idx]=xo_new[:,cl]
    return x,info

# validity spot-check once (ordered copy)
_o=geo._curve_order(pos0,N); _xo=torch.gather(pos0,1,_o[...,None].expand(-1,-1,2))
assert_mtm_valid(P,_xo,s0[_o],KC.cluster_slots(5,sc,K,L),sc,L); print("mtm validity assertions PASS",flush=True)

def run_arm(name,do_swap,do_mtm):
    torch.manual_seed(42)
    x=pos0.clone(); t0=time.time(); next_rep=REPORT; probs=[]
    while (time.time()-t0)<BUDGET:
        if do_swap:
            for _ in range(100 if do_mtm else 200): x=swap_sweep(x)
        if do_mtm:
            for _ in range(4 if do_swap else 8):
                x,info=mtm_step(x,int(torch.randint(0,N,(1,)).item())); probs.append(info["move_prob"])
        el=time.time()-t0
        if el>=next_rep:
            U,Q,msd=metrics(x,pos0)
            print(f"{name:6s} t={el:6.0f}s  U/N {U:.4f}  Q {Q:.3f}  MSD {msd:.4f}"+
                  (f"  mtm_p {np.mean(probs[-40:]):.4f}" if probs else ""),flush=True)
            next_rep+=REPORT
    return x
finals={}
for name,dsw,dmtm in (("swap",True,False),("mtm",False,True),("hybrid",True,True)):
    print(f"=== ARM {name} (budget {BUDGET:.0f}s) ===",flush=True)
    finals[name]=run_arm(name,dsw,dmtm).cpu()
import os as _os
torch.save({"finals":finals,"pos0":pos0.cpu(),"s0":s0.cpu(),"L":L},
           _os.path.join(ART,"mtm_mixing_finals.pt"))
print("saved finals",flush=True)
print("BENCH COMPLETE",flush=True)
