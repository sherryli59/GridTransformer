"""Rigor check for the (b)-killing T=0.5 result: does the scramble arm genuinely RISE from a decorrelated melt
start to meet the ref arm (real convergence), or does the melt leave it trivially close to the reference?
Logs melt-start Q + both trajectories at T=0.5, c=0.16, displacement-only."""
import numpy as np, torch
from liquid_coupling_flow.ka_energy import SIGMA, EPS, RCUT_FACTOR
from liquid_coupling_flow.ka_pin_overlap import cell_occupancy, pinned_cells, overlap_Q, q_rand
dev="cuda"; N=256; T=0.5; beta=1/T; B=12; c=0.16; STEP=0.08; QR=q_rand()
d=torch.load("liquid_coupling_flow/artifacts/pt_ladder_hb_N256.pt",map_location=dev,weights_only=False)
x0=d["configs_per_rung"][0][:B].to(dev); s=d["s"].to(dev).long()[None].expand(B,-1).contiguous(); L=d["L"]
tS=torch.tensor(SIGMA,device=dev); tE=torch.tensor(EPS,device=dev); eye=torch.eye(N,device=dev,dtype=torch.bool)[None]
def disp(x,mob,bet,step):
    a=s[:,:,None].expand(-1,-1,N); b=s[:,None,:].expand(-1,N,-1); sig=tS[a,b]; eps=tE[a,b]; rc=RCUT_FACTOR*sig
    prop=torch.remainder(torch.where(mob[...,None],x+step*torch.randn_like(x),x),L)
    def cr(xa):
        df=xa[:,:,None,:]-x[:,None,:,:]; df=df-L*torch.round(df/L); r2=(df**2).sum(-1).masked_fill(eye,1e12)
        i6=(sig**2/r2)**3; e=4*eps*(i6**2-i6); s6=(sig/rc)**6
        return torch.where(r2<rc**2,e-4*eps*(s6**2-s6),torch.zeros_like(e)).sum(-1)
    dE=cr(prop)-cr(x); acc=(torch.log(torch.rand_like(dE))<(-bet*dE))&mob
    return torch.where(acc[...,None],prop,x)
def melt(mob,Th=1.5,n=400):
    x=x0.clone()
    for _ in range(n): x=disp(x,mob,1/Th,0.12)
    return x
torch.manual_seed(0); npn=int(np.ceil(c*N)); mob=torch.ones(B,N,dtype=torch.bool,device=dev)
for bch in range(B): mob[bch,torch.randperm(N,device=dev)[:npn]]=False
occ_ref=cell_occupancy(x0,L); excl=pinned_cells(x0,mob,L)
scr0=melt(mob); qstart=overlap_Q(cell_occupancy(scr0,L),occ_ref,excl)
print(f"[verify] T=0.5 c={c} Q_rand={QR:.3f}  scramble MELT-START Q={qstart:.3f} (low=genuinely decorrelated)",flush=True)
for arm,xi in (("ref",x0.clone()),("scramble",scr0)):
    x=xi
    for it in range(3001):
        if it%500==0: print(f"  [{arm:8s}] it {it:4d} Q {overlap_Q(cell_occupancy(x,L),occ_ref,excl):.3f}",flush=True)
        if it==3000: break
        for _ in range(30): x=disp(x,mob,beta,STEP)
print("[verify] DONE",flush=True)
