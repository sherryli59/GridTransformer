"""ZERO-SHOT TRANSFER: N=64-trained eRSI flow deployed at N=216 (3.4x). (1) does the size-agnostic kNN flow
still make shells? (2) does the seed-SMC amortization survive? 3-arm {uniform, exclvol, flow} at N=216."""
import math, sys, torch
IPL44="/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for m in [x for x in sys.modules if x.startswith("learndiffeq")]: del sys.modules[m]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
from liquid_coupling_flow.mw.mw_base import UniformBase, ExcludedVolumeBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r
DEV="cuda"; N,B=216,96; L=(N/RHO_STAR)**(1/3); beta_t=1.0/T_STAR; STEP=0.0783; SW_PER=12
print(f"N={N} L={L:.3f} beta={beta_t:.2f} (train N=64)",flush=True)
# size-agnostic load: N=64 weights into an N=216 EGNN
egnn=EGNN_dynamics(n_particles=N,n_dimension=3,hidden_nf=128,n_layers=4,max_neighbors=12,L=L,n_species=1).to(DEV)
c=torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt",map_location=DEV,weights_only=False)
egnn.load_state_dict({k[2:]:v for k,v in c["state_dict"].items() if k.startswith("b.")},strict=True); egnn.eval()
print("size-agnostic weight load OK (N=64 -> N=216)",flush=True)
@torch.no_grad()
def flow_seed(steps=60,seed=0):
    g=torch.Generator(device=DEV).manual_seed(seed); x=torch.rand(B,N,3,device=DEV,generator=g)*L
    a=torch.zeros(B,N,dtype=torch.long,device=DEV); ts=torch.linspace(0,1,steps+1,device=DEV)
    for i in range(steps):
        dt=float(ts[i+1]-ts[i]); v1,_=egnn.forward_and_divergence(x,ts[i].expand(B),a)
        xm=torch.remainder(x+0.5*dt*v1,L); vm,_=egnn.forward_and_divergence(xm,(ts[i]+0.5*dt).expand(B),a)
        x=torch.remainder(x+dt*vm,L)
    return x
def uN(x): return float(mw_energy_chunked(x,L).mean())/N
fx=flow_seed()
r,gr=g_r(fx.cpu(),L); i1,i2=int(1.19/(L/2)*len(r)),int(1.85/(L/2)*len(r))
print(f"\n=== ZERO-SHOT STRUCTURE @ N=216 ===",flush=True)
print(f"flow seed U/N {uN(fx):+.4f}  |  g(r) shell1 {float(gr[i1]):.2f} shell2 {float(gr[i2]):.2f}  (N=64 was 1.72/1.11; data ~2.1/1.2)",flush=True)
# 3-arm amortization
LADDER=(0.5*(beta_t/0.5)**torch.linspace(0,1,12)).tolist(); base=UniformBase(N,L)
gu=torch.Generator(device=DEV).manual_seed(1); ux=torch.rand(B,N,3,device=DEV,generator=gu)*L
ge=torch.Generator(device=DEV).manual_seed(2); ex=ExcludedVolumeBase(N,L,G=14).sample(B,ge).to(DEV)
seeds={"uniform":ux,"exclvol":ex,"flow":fx}
for k,v in seeds.items(): print(f"seed {k:8s}: U/N {uN(v):+.4f}",flush=True)
def anneal(x,sr,tag):
    x=x.clone(); ev=0; h=[]; g=torch.Generator(device=DEV).manual_seed(0)
    for kk in range(sr,12):
        x,U,_,info=mutation_sweeps(x,base,1.0,LADDER[kk],L,SW_PER,STEP,g); ev+=info["evals"]; h.append((LADDER[kk],float(U.mean())/N,ev))
    print(f"[{tag}] start rung {sr} (beta {LADDER[sr]:.2f}) final U/N {h[-1][1]:+.4f} evals {ev:,}",flush=True); return h,ev
hu,eu=anneal(ux,0,"uniform"); eqmap=[x[1] for x in hu]
def nat(u):
    for kk in range(12):
        if eqmap[kk]<=u: return kk
    return 11
res={"uniform":(hu,eu,0)}
for k in ("exclvol","flow"): sr=nat(uN(seeds[k])); h,e=anneal(seeds[k],sr,k); res[k]=(h,e,sr)
print("\n=== N=216 ZERO-SHOT AMORTIZATION (vs uniform) ===",flush=True)
for k in ("uniform","exclvol","flow"):
    h,e,sr=res[k]; print(f"  {k:8s}: {e:>11,} evals  final U/N {h[-1][1]:+.4f}  ({eu/e:.2f}x)",flush=True)
torch.save({"N":N,"L":L,"gr":(r,gr),"seed_shells":(float(gr[i1]),float(gr[i2])),
            **{k:{"hist":res[k][0],"evals":res[k][1],"start_rung":res[k][2],"seed_U":uN(seeds[k])} for k in res}},
           "liquid_coupling_flow/mw/artifacts/mw_ersi_transfer_N216.pt")
