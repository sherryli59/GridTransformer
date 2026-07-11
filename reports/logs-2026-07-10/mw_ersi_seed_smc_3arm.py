"""3-arm seed-SMC amortization: uniform vs excluded-volume prior vs eRSI flow. Each seed ENTERS the shared
geometric ladder at its NATURAL temperature (the first rung at least as cold as the seed's <U>), so structure
is preserved not melted. Adjudicates whether the LEARNED flow (shells) beats the cheap non-ML excluded-volume
prior (core only). Metric = energy-evals to equilibrium."""
import math, sys, torch
IPL44 = "/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for m in [x for x in sys.modules if x.startswith("learndiffeq")]: del sys.modules[m]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
from liquid_coupling_flow.mw.mw_base import UniformBase, ExcludedVolumeBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR
DEV="cuda"; N,B=64,256; L=(N/RHO_STAR)**(1/3); beta_t=1.0/T_STAR; STEP=0.0783; SW_PER=12
LADDER=(0.5*(beta_t/0.5)**torch.linspace(0,1,12)).tolist()
base=UniformBase(N,L)
def uN(x): return float(mw_energy_chunked(x,L).mean())/N
@torch.no_grad()
def flow_seed(seed=0,steps=60):
    e=EGNN_dynamics(n_particles=N,n_dimension=3,hidden_nf=128,n_layers=4,max_neighbors=12,L=L,n_species=1).to(DEV)
    c=torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt",map_location=DEV,weights_only=False)
    e.load_state_dict({k[2:]:v for k,v in c["state_dict"].items() if k.startswith("b.")},strict=True); e.eval()
    g=torch.Generator(device=DEV).manual_seed(seed); x=torch.rand(B,N,3,device=DEV,generator=g)*L
    a=torch.zeros(B,N,dtype=torch.long,device=DEV); ts=torch.linspace(0,1,steps+1,device=DEV)
    for i in range(steps):
        dt=float(ts[i+1]-ts[i]); v1,_=e.forward_and_divergence(x,ts[i].expand(B),a)
        xm=torch.remainder(x+0.5*dt*v1,L); vm,_=e.forward_and_divergence(xm,(ts[i]+0.5*dt).expand(B),a)
        x=torch.remainder(x+dt*vm,L)
    return x
seeds={}
gu=torch.Generator(device=DEV).manual_seed(1); seeds["uniform"]=torch.rand(B,N,3,device=DEV,generator=gu)*L
ge=torch.Generator(device=DEV).manual_seed(2); seeds["exclvol"]=ExcludedVolumeBase(N,L,G=16).sample(B,ge).to(DEV)
seeds["flow"]=flow_seed()
for k,v in seeds.items(): print(f"seed {k:8s}: U/N {uN(v):+.4f}",flush=True)
# run uniform first to get the equilibrium map
def anneal(x,start_rung,tag):
    x=x.clone(); evals=0; hist=[]; g=torch.Generator(device=DEV).manual_seed(0)
    for k in range(start_rung,12):
        x,U,_,info=mutation_sweeps(x,base,1.0,LADDER[k],L,SW_PER,STEP,g); evals+=info["evals"]
        hist.append((LADDER[k],float(U.mean())/N,evals))
    print(f"[{tag}] start rung {start_rung} (beta {LADDER[start_rung]:.2f})  final U/N {hist[-1][1]:+.4f}  evals {evals:,}",flush=True)
    return hist,evals
hu,eu=anneal(seeds["uniform"],0,"uniform")
eqmap=[h[1] for h in hu]   # equilibrium <U>/N per rung
def natural_rung(useed):
    for k in range(12):
        if eqmap[k]<=useed: return k
    return 11
res={"uniform":(hu,eu,0)}
for k in ("exclvol","flow"):
    sr=natural_rung(uN(seeds[k])); h,e=anneal(seeds[k],sr,k); res[k]=(h,e,sr)
print("\n=== AMORTIZATION (energy-evals to equilibrium; uniform=baseline) ===",flush=True)
for k in ("uniform","exclvol","flow"):
    h,e,sr=res[k]; print(f"  {k:8s}: {e:>10,} evals  final U/N {h[-1][1]:+.4f}  ({eu/e:.2f}x vs uniform)",flush=True)
torch.save({k:{"hist":res[k][0],"evals":res[k][1],"start_rung":res[k][2],"seed_U":uN(seeds[k])} for k in res},
           "liquid_coupling_flow/mw/artifacts/mw_ersi_seed_3arm.pt")
