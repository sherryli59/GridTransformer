"""U2 short: stationarity spot-check with the upgraded kernel (M=16 scored), 8 rounds."""
import os, time, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move, make_jf_pair_scorer
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_gridformer import _wrap_pm
DEV="cuda"; B=128; N=100; K=7; BETA=2.0; M=16; KT=0.5; STEP=0.04
torch.manual_seed(51)
sc,L,geo=_scaffold(N,DEV)
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
idx=torch.randperm(ref["x"].shape[0],device=DEV)[:B]
pos,s=slot_order(ref["x"][idx].to(DEV),ref["s"].to(DEV).long(),geo,N)
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)
scorer=make_jf_pair_scorer(DEV)
def canonicalize(pos,s):
    perm=torch.argsort(s,dim=1,stable=True)
    return torch.gather(pos,1,perm[...,None].expand(-1,-1,2)).contiguous(),torch.gather(s,1,perm).contiguous()
pos,s=canonicalize(pos,s); s_can=s[0]
U0=float(ka_energy(pos,s,L).mean())/N
d=_wrap_pm(pos[:,:,None]-pos[:,None],L).norm(dim=-1)+torch.eye(N,device=DEV)[None]*1e3
isB=(s==1); bb0=float(((d<1.25)&isB[:,:,None]&isB[:,None,:]).sum((-1,-2)).float().mean()/2)
print(f"U2s BASELINE: U/N {U0:.4f}  BB {bb0:.2f}",flush=True)
t0=time.time()
for rnd in range(8):
    for _ in range(50):
        prop=torch.remainder(pos+STEP*torch.randn_like(pos),L)
        dE=(_u_matrix(prop,pos,s_can,s_can,L,True)-_u_matrix(pos,pos,s_can,s_can,L,True)).sum(-1)
        a=torch.log(torch.rand(B,N,device=DEV))<(-dE/KT)
        pos=torch.where(a[:,:,None],prop,pos)
    for mv in range(50):                                             # half-sweep of upgraded moves
        seed=int(torch.randint(0,N,(1,)).item())
        order=geo._curve_order(pos,N)
        pos_o=torch.gather(pos,1,order[...,None].expand(-1,-1,2)); s_o=torch.gather(s,1,order)
        cl=KC.cluster_slots(seed,sc,K,L)
        pos_o,s_o,acc,info=sb_mtm_move(P,pos_o,s_o,cl,sc,L,beta=BETA,M=M,pair_scorer=scorer)
        ii=order[:,cl]
        pos=pos.clone(); s=s.clone()
        pos[torch.arange(B,device=DEV)[:,None],ii]=pos_o[:,cl]; s[torch.arange(B,device=DEV)[:,None],ii]=s_o[:,cl]
    pos,s=canonicalize(pos,s); s_can=s[0]
    U=float(ka_energy(pos,s,L).mean())/N
    d=_wrap_pm(pos[:,:,None]-pos[:,None],L).norm(dim=-1)+torch.eye(N,device=DEV)[None]*1e3
    isB=(s==1); bb=float(((d<1.25)&isB[:,:,None]&isB[:,None,:]).sum((-1,-2)).float().mean()/2)
    print(f"U2s round {rnd+1}: U/N {U:.4f} (d {U-U0:+.4f})  BB {bb:.2f}  {time.time()-t0:.0f}s",flush=True)
print("U2S COMPLETE",flush=True)
