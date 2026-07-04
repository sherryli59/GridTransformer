"""U1: v2/v3 exchange-rate gate (G1 protocol, B=128): M=16 uniform vs M=16 denoiser-scored, vs v1's 0.153%.
U2: spot stationarity with the upgraded kernel (15 rounds x [50 disp + 1 sb_mtm sweep])."""
import os, time, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move, make_jf_pair_scorer
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_gridformer import _wrap_pm
DEV="cuda"; B=128; N=100; K=7; BETA=2.0; M=16; KT=0.5; STEP=0.04
torch.manual_seed(41)
sc,L,geo=_scaffold(N,DEV)
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
idx=torch.randperm(ref["x"].shape[0],device=DEV)[:B]
pos0,s0=slot_order(ref["x"][idx].to(DEV),ref["s"].to(DEV).long(),geo,N)
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)
scorer=make_jf_pair_scorer(DEV)

def u1_arm(name,pair_scorer,n_sweeps=10):
    pos=pos0.clone(); s=s0.clone(); n_acc=0; n_tot=0; t0=time.time()
    for sw in range(n_sweeps):
        for mv in range(N):
            seed=int(torch.randint(0,N,(1,)).item())
            order=geo._curve_order(pos,N)
            pos_o=torch.gather(pos,1,order[...,None].expand(-1,-1,2)); s_o=torch.gather(s,1,order)
            cl=KC.cluster_slots(seed,sc,K,L)
            pos_o,s_o,acc,info=sb_mtm_move(P,pos_o,s_o,cl,sc,L,beta=BETA,M=M,pair_scorer=pair_scorer)
            ii=order[:,cl]
            pos=pos.clone(); s=s.clone()
            pos[torch.arange(B,device=DEV)[:,None],ii]=pos_o[:,cl]; s[torch.arange(B,device=DEV)[:,None],ii]=s_o[:,cl]
            n_acc+=int(acc.sum()); n_tot+=B
        print(f"U1 {name}: sweep {sw+1:2d}  cum exchange {100*n_acc/n_tot:.4f}%  ({n_acc}/{n_tot})  {time.time()-t0:.0f}s",flush=True)
    rate=n_acc/n_tot; wall=time.time()-t0
    per_ex=wall/max(n_acc/B,1e-9)
    print(f"U1 {name} RESULT: {100*rate:.4f}%  ({n_acc}/{n_tot})  {wall:.0f}s  ~{per_ex:.1f}s per exchange/chain  [v1: 0.153%]",flush=True)
    return rate

r_uni=u1_arm("M16-uniform",None)
r_sco=u1_arm("M16-scored",scorer)
print(f"\nU1 SUMMARY: v1 0.153% -> M16-uniform {100*r_uni:.3f}% -> M16-scored {100*r_sco:.3f}%",flush=True)

# --- U2: spot stationarity with the upgraded kernel (scored arm) ---
def canonicalize(pos,s):
    perm=torch.argsort(s,dim=1,stable=True)
    return torch.gather(pos,1,perm[...,None].expand(-1,-1,2)).contiguous(),torch.gather(s,1,perm).contiguous()
pos,s=canonicalize(pos0.clone(),s0.clone()); s_can=s[0]
U0=float(ka_energy(pos,s,L).mean())/N
d=_wrap_pm(pos[:,:,None]-pos[:,None],L).norm(dim=-1)+torch.eye(N,device=DEV)[None]*1e3
isB=(s==1); bb0=float(((d<1.25)&isB[:,:,None]&isB[:,None,:]).sum((-1,-2)).float().mean()/2)
print(f"\nU2 BASELINE: U/N {U0:.4f}  BB-contacts {bb0:.2f}",flush=True)
t0=time.time()
for rnd in range(15):
    for _ in range(50):
        prop=torch.remainder(pos+STEP*torch.randn_like(pos),L)
        dE=(_u_matrix(prop,pos,s_can,s_can,L,True)-_u_matrix(pos,pos,s_can,s_can,L,True)).sum(-1)
        a=torch.log(torch.rand(B,N,device=DEV))<(-dE/KT)
        pos=torch.where(a[:,:,None],prop,pos)
    for mv in range(N):
        seed=int(torch.randint(0,N,(1,)).item())
        order=geo._curve_order(pos,N)
        pos_o=torch.gather(pos,1,order[...,None].expand(-1,-1,2)); s_o=torch.gather(s,1,order)
        cl=KC.cluster_slots(seed,sc,K,L)
        pos_o,s_o,acc,info=sb_mtm_move(P,pos_o,s_o,cl,sc,L,beta=BETA,M=M,pair_scorer=scorer)
        ii=order[:,cl]
        pos=pos.clone(); s=s.clone()
        pos[torch.arange(B,device=DEV)[:,None],ii]=pos_o[:,cl]; s[torch.arange(B,device=DEV)[:,None],ii]=s_o[:,cl]
    pos,s=canonicalize(pos,s); s_can=s[0]
    if (rnd+1)%3==0:
        U=float(ka_energy(pos,s,L).mean())/N
        d=_wrap_pm(pos[:,:,None]-pos[:,None],L).norm(dim=-1)+torch.eye(N,device=DEV)[None]*1e3
        isB=(s==1); bb=float(((d<1.25)&isB[:,:,None]&isB[:,None,:]).sum((-1,-2)).float().mean()/2)
        print(f"U2 round {rnd+1:2d}: U/N {U:.4f} (d {U-U0:+.4f})  BB-contacts {bb:.2f}  {time.time()-t0:.0f}s",flush=True)
print("U1U2 COMPLETE",flush=True)
