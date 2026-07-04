"""Add-on arm: rand+hybrid (uniform starts, swap+MTM), same protocol/seeds as ar_seed_bench."""
import runpy, sys, os, time, torch, numpy as np
sys.argv=["x"]
# reuse the harness by importing its module-level definitions via exec of a trimmed copy is fragile;
# simplest: re-declare inline (identical constants) importing the same functions.
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_mtm import mtm_move
from liquid_coupling_flow.ka_cluster_flow import _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_mcmc_fast import _u_matrix
from liquid_coupling_flow.ka_mcmc import _pair_energy_one
from liquid_coupling_flow.ka_gridformer import _wrap_pm
DEV="cuda"; BETA=2.0; KT=0.5; B=64; N=100; K=7; STEP=0.04; NSWAP=N//8; M=32
BUDGET=900.0; REPORT=15.0
torch.manual_seed(0)
sc,L,geo=_scaffold(N,DEV)
_=torch.rand(1)  # keep simple; rand starts reproduced by same manual_seed sequence as bench? bench drew AR first.
# reproduce the bench's rand starts EXACTLY: load them from the saved rand_swap trajectory instead of re-drawing
d=torch.load(os.path.join(ART,"arstart_traj_rand_swap.pt"),map_location=DEV,weights_only=False)
pos_rand=d["x_init"].to(DEV); s0=d["s0"].to(DEV)
P=_load(torch.load(os.path.join(ART,"ka_cluster_flow_full_N100.pt"),map_location=DEV,weights_only=False),DEV)
def stats(x):
    U=float(ka_energy(x,s0,L).mean())/N
    dd=_wrap_pm(x[:,:,None]-x[:,None],L).norm(dim=-1)+torch.eye(N,device=DEV)[None]*1e3
    return U,float((dd.min(-1).values<0.7).float().mean()*100)
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
torch.manual_seed(42)
x=pos_rand.clone(); t0=time.time(); next_rep=REPORT; probs=[]
rec_t=[0.0]; rec_x=[x.clone().cpu()]; rec_tag=[-1]; met=[]
def rec(tag):
    rec_t.append(time.time()-t0); rec_x.append(x.clone().cpu()); rec_tag.append(tag)
while (time.time()-t0)<BUDGET:
    for _ in range(100): x=swap_sweep(x); rec(0)
    for _ in range(4):
        x,info=mtm_step(x,int(torch.randint(0,N,(1,)).item())); probs.append(info["move_prob"]); rec(1)
    el=time.time()-t0
    if el>=next_rep:
        U,cl_=stats(x); met.append((el,U,cl_))
        print(f"rand+hyb t={el:6.0f}s  U/N {U:.4f}  clash {cl_:5.2f}%  mtm_p {np.mean(probs[-40:]):.3f}",flush=True)
        next_rep+=REPORT
out={"final":x.cpu(),"t":torch.tensor(rec_t),"tag":torch.tensor(rec_tag),"x":torch.stack(rec_x,0),
     "metrics":met,"mtm_probs":probs,"x_init":pos_rand.cpu(),"s0":s0.cpu(),"L":L,"seed":42,"M":M,"budget":BUDGET}
fn=os.path.join(ART,"arstart_traj_rand_hybrid.pt"); torch.save(out,fn)
print("saved FULL trajectory",fn,"states",out["x"].shape[0],flush=True)
print("ARM COMPLETE",flush=True)
