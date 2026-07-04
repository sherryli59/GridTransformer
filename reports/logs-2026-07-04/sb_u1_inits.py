"""U1-inits: v2 (M=16 uniform) exchange rate from FLOW-init and RANDOM-init states (vs equilibrium 1.90%).
Each start gets 200 disp sweeps at beta=2 first (clash removal — raw clashed states would inflate the rate:
w0~0 accepts anything = defect repair, not exchange). 3 sweeps x 100 moves x B=128 per arm."""
import os, time, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_swap_breathe import sb_mtm_move
from liquid_coupling_flow.ka_cluster_flow import _scaffold, _load, ART
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel
DEV="cuda"; B=128; N=100; K=7; BETA=2.0; M=16; KT=0.5; STEP=0.04
torch.manual_seed(61)
sc,L,geo=_scaffold(N,DEV)
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)
ck=torch.load(os.path.join(ART,"ka_flowhead_N100_k8_scratch.pt"),map_location=DEV,weights_only=False)
gen=KAFlowHeadModel(rho=1.2,n_bins=ck["n_bins"],knn=ck["knn"],num_bins=ck["num_bins"],tail_bound=ck["tail_bound"]).to(DEV)
gen.load_state_dict(ck["state_dict"]); gen.eval()
def canonicalize(pos,s):
    perm=torch.argsort(s,dim=1,stable=True)
    return torch.gather(pos,1,perm[...,None].expand(-1,-1,2)).contiguous(),torch.gather(s,1,perm).contiguous()
def relax(pos,s_can,n):
    for _ in range(n):
        prop=torch.remainder(pos+STEP*torch.randn_like(pos),L)
        dE=(_u_matrix(prop,pos,s_can,s_can,L,True)-_u_matrix(pos,pos,s_can,s_can,L,True)).sum(-1)
        a=torch.log(torch.rand(B,N,device=DEV))<(-dE/KT)
        pos=torch.where(a[:,:,None],prop,pos)
    return pos
def rate(name,pos,s):
    n_acc=0;n_tot=0;t0=time.time()
    for sw in range(3):
        for mv in range(100):
            seed=int(torch.randint(0,N,(1,)).item())
            order=geo._curve_order(pos,N)
            pos_o=torch.gather(pos,1,order[...,None].expand(-1,-1,2)); s_o=torch.gather(s,1,order)
            cl=KC.cluster_slots(seed,sc,K,L)
            pos_o,s_o,acc,info=sb_mtm_move(P,pos_o,s_o,cl,sc,L,beta=BETA,M=M)
            ii=order[:,cl]
            pos=pos.clone(); s=s.clone()
            pos[torch.arange(B,device=DEV)[:,None],ii]=pos_o[:,cl]; s[torch.arange(B,device=DEV)[:,None],ii]=s_o[:,cl]
            n_acc+=int(acc.sum()); n_tot+=B
    U=float(ka_energy(pos,s,L).mean())/N
    print(f"{name:12s}: exchange {100*n_acc/n_tot:.3f}% ({n_acc}/{n_tot})  U/N(end) {U:.4f}  {time.time()-t0:.0f}s",flush=True)
# flow init
pos_f,sp_f=gen.sample(B,N,n_B=35,device=DEV)
pos_f,s_f=canonicalize(pos_f,sp_f.long())
pos_f=relax(pos_f,s_f[0],200)
print(f"flow-init after 200 relax sweeps: U/N {float(ka_energy(pos_f,s_f,L).mean())/N:.4f}",flush=True)
rate("flow-init",pos_f,s_f)
# random init
pos_r=torch.rand(B,N,2,device=DEV)*L
s_r=torch.cat([torch.zeros(65,dtype=torch.long),torch.ones(35,dtype=torch.long)]).to(DEV)[None].expand(B,N).clone()
pos_r=relax(pos_r,s_r[0],200)
print(f"rand-init after 200 relax sweeps: U/N {float(ka_energy(pos_r,s_r,L).mean())/N:.4f}",flush=True)
rate("rand-init",pos_r,s_r)
print("U1-INITS COMPLETE  [equilibrium reference: 1.897%]",flush=True)
