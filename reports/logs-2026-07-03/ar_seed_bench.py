"""AR-SEEDED RELAXATION BENCHMARK: start MCMC from AR-transformer full-config samples (28% clash, defect-rich).
Arms at matched wall-clock from IDENTICAL AR seeds: swap | mtm(M=32) | hybrid(~50/50), plus rand+swap baseline
(uniform-random starts). Hypothesis: cluster-MTM is a defect-REPAIR operator -> hybrid edge widens vs swap from
AR starts. Species canonicalized per row (A-block then B-block) so ONE species vector is exact for the vectorized
swap (slot_order pitfall rule). FULL per-step trajectories saved (results-durability rule)."""
import os, time, torch, numpy as np
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import mtm_move
from liquid_coupling_flow.ka_cluster_flow import _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_mcmc import _pair_energy_one
from liquid_coupling_flow.ka_gridformer import _wrap_pm
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel

DEV="cuda"; BETA=2.0; KT=0.5; B=64; N=100; K=7; STEP=0.04; NSWAP=N//8; M=32
BUDGET=900.0; REPORT=15.0
torch.manual_seed(0)
sc,L,geo=_scaffold(N,DEV)
# --- AR seeds ---
ck=torch.load(os.path.join(ART,"ka_flowhead_N100_k8_scratch.pt"),map_location=DEV,weights_only=False)
gen=KAFlowHeadModel(rho=1.2,n_bins=ck["n_bins"],knn=ck["knn"],num_bins=ck["num_bins"],tail_bound=ck["tail_bound"]).to(DEV)
gen.load_state_dict(ck["state_dict"]); gen.eval()
pos_ar,sp_ar=gen.sample(B,N,n_B=35,device=DEV)
perm=torch.argsort(sp_ar,dim=1,stable=True)                          # canonicalize: A-block then B-block
pos_ar=torch.gather(pos_ar,1,perm[...,None].expand(-1,-1,2)).contiguous()
s0=torch.cat([torch.zeros(65,dtype=torch.long),torch.ones(35,dtype=torch.long)]).to(DEV)
pos_rand=torch.rand(B,N,2,device=DEV)*L
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)

def stats(x):
    U=float(ka_energy(x,s0,L).mean())/N
    d=_wrap_pm(x[:,:,None]-x[:,None],L).norm(dim=-1)+torch.eye(N,device=DEV)[None]*1e3
    return U,float((d.min(-1).values<0.7).float().mean()*100)
U_ar,c_ar=stats(pos_ar); U_rd,c_rd=stats(pos_rand)
print(f"AR seeds  : U/N {U_ar:.3e}  clash {c_ar:.1f}%   | RAND seeds: U/N {U_rd:.3e}  clash {c_rd:.1f}%   eq ~ -3.2600",flush=True)

Aidx=(s0==0).nonzero().squeeze(-1); Bidx=(s0==1).nonzero().squeeze(-1)
def swap_sweep(x):
    prop=torch.remainder(x+STEP*torch.randn_like(x),L)
    dE=(_u_matrix(prop,x,s0,s0,L,True)-_u_matrix(x,x,s0,s0,L,True)).sum(-1)
    acc=torch.log(torch.rand(B,N,device=DEV))<(-dE/KT)
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
    order=geo._curve_order(x,N)
    xo=torch.gather(x,1,order[...,None].expand(-1,-1,2)); so=s0[order]
    cl=KC.cluster_slots(seed,sc,K,L)
    xo_new,_,info=mtm_move(P,xo,so,cl,sc,L,M=M,beta=BETA)
    idx=order[:,cl]
    x=x.clone(); x[torch.arange(B,device=DEV)[:,None],idx]=xo_new[:,cl]
    return x,info

def run_arm(name,x_init,do_swap,do_mtm):
    torch.manual_seed(42)
    x=x_init.clone(); t0=time.time(); next_rep=REPORT; probs=[]
    rec_t=[0.0]; rec_x=[x.clone().cpu()]; rec_tag=[-1]; met=[]
    def rec(tag):
        rec_t.append(time.time()-t0); rec_x.append(x.clone().cpu()); rec_tag.append(tag)
    while (time.time()-t0)<BUDGET:
        if do_swap:
            for _ in range(100 if do_mtm else 200):
                x=swap_sweep(x); rec(0)
        if do_mtm:
            for _ in range(4 if do_swap else 8):
                x,info=mtm_step(x,int(torch.randint(0,N,(1,)).item())); probs.append(info["move_prob"]); rec(1)
        el=time.time()-t0
        if el>=next_rep:
            U,cl_=stats(x); met.append((el,U,cl_))
            print(f"{name:9s} t={el:6.0f}s  U/N {U:.4f}  clash {cl_:5.2f}%"+
                  (f"  mtm_p {np.mean(probs[-40:]):.3f}" if probs else ""),flush=True)
            next_rep+=REPORT
    return {"final":x.cpu(),"t":torch.tensor(rec_t),"tag":torch.tensor(rec_tag),
            "x":torch.stack(rec_x,0),"metrics":met,"mtm_probs":probs}

arms=(("ar+swap",pos_ar,True,False),("ar+mtm",pos_ar,False,True),("ar+hybrid",pos_ar,True,True),
      ("rand+swap",pos_rand,True,False))
for name,x0,dsw,dmtm in arms:
    print(f"=== ARM {name} ===",flush=True)
    out=run_arm(name,x0,dsw,dmtm)
    fn=os.path.join(ART,f"arstart_traj_{name.replace('+','_')}.pt")
    torch.save({**out,"x_init":x0.cpu(),"s0":s0.cpu(),"L":L,"seed":42,"M":M,"budget":BUDGET},fn)
    print(f"saved FULL trajectory {fn}  states={out['x'].shape[0]}",flush=True)
print("BENCH COMPLETE",flush=True)
