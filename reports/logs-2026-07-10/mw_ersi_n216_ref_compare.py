"""Compare the zero-shot N=216 flow g(r) against a REAL N=216 equilibrium (annealed to convergence at N=216,
independent of the flow), not the N=64 stand-in. Also overlays the N=64 reference to show g(r) size-invariance."""
import math, sys, torch, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
IPL44="/mnt/ssd/GridTransformer/liquid_coupling_flow/ipl44/learndiffeq"
for m in [x for x in sys.modules if x.startswith("learndiffeq")]: del sys.modules[m]
sys.path.insert(0, IPL44)
from learndiffeq.particles.velocities.egnn_traceable import EGNN_dynamics
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_smc import mutation_sweeps
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r
DEV="cuda"; N=216; B=64; L=(N/RHO_STAR)**(1/3); beta_t=1.0/T_STAR; STEP=0.0783
def uN(x): return float(mw_energy_chunked(x,L).mean())/N
# --- real N=216 equilibrium: anneal uniform -> target with GENEROUS relaxation (independent of the flow) ---
base=UniformBase(N,L); LAD=(0.5*(beta_t/0.5)**torch.linspace(0,1,14)).tolist()
g=torch.Generator(device=DEV).manual_seed(5); x=torch.rand(B,N,3,device=DEV,generator=g)*L
for bta in LAD:
    x,U,_,_=mutation_sweeps(x,base,1.0,bta,L,20,STEP,g)
for _ in range(40):                       # extra relaxation at target to converge g(r)
    x,U,_,_=mutation_sweeps(x,base,1.0,beta_t,L,10,STEP,g)
print(f"N=216 real equilibrium (independent anneal): U/N {uN(x):+.4f}",flush=True)
r_eq,g_eq=g_r(x.cpu(),L)
# --- zero-shot flow @ N=216 ---
e=EGNN_dynamics(n_particles=N,n_dimension=3,hidden_nf=128,n_layers=4,max_neighbors=12,L=L,n_species=1).to(DEV)
c=torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_N64_checkpoints/last.ckpt",map_location=DEV,weights_only=False)
e.load_state_dict({k[2:]:v for k,v in c["state_dict"].items() if k.startswith("b.")},strict=True); e.eval()
gg=torch.Generator(device=DEV).manual_seed(7); xf=torch.rand(B,N,3,device=DEV,generator=gg)*L
a=torch.zeros(B,N,dtype=torch.long,device=DEV); ts=torch.linspace(0,1,61,device=DEV)
with torch.no_grad():
    for i in range(60):
        dt=float(ts[i+1]-ts[i]); v1,_=e.forward_and_divergence(xf,ts[i].expand(B),a)
        xm=torch.remainder(xf+0.5*dt*v1,L); vm,_=e.forward_and_divergence(xm,(ts[i]+0.5*dt).expand(B),a)
        xf=torch.remainder(xf+dt*vm,L)
print(f"N=216 zero-shot flow raw: U/N {uN(xf):+.4f}",flush=True)
r_f,g_f=g_r(xf.cpu(),L)
# N=64 reference for the size-invariance overlay
ext=torch.load("liquid_coupling_flow/mw/artifacts/mw_ref_N64_ext.pt",map_location="cpu",weights_only=False)
L64=(64/RHO_STAR)**(1/3); r64,g64=g_r(ext["cfgs"][-6400:],L64)
fig,ax=plt.subplots(figsize=(8.6,5))
ax.plot(r_eq,g_eq,"k",lw=2.4,label=f"REAL N=216 equilibrium (annealed, U/N {uN(x):.3f})")
ax.plot(r64,g64,color="gray",lw=1.3,ls=":",label="N=64 reference (shows g(r) size-invariance)")
ax.plot(r_f,g_f,"royalblue",lw=1.8,label="eRSI flow raw @ N=216 (ZERO-SHOT)")
ax.set(xlabel="r (sigma)",ylabel="g(r)",xlim=(0,L/2),ylim=(0,2.3),
       title="N=216 zero-shot flow vs ACTUAL N=216 equilibrium g(r)")
ax.legend(fontsize=9); fig.tight_layout()
out="reports/logs-2026-07-10/mw_ersi_n216_real_compare.png"; fig.savefig(out,dpi=150); print("PLOT:",out)
def sh(r,gr,rr): 
    i=int(rr/(L/2)*len(r)); return float(gr[i])
print(f"shell1(1.19): real-eq {sh(r_eq,g_eq,1.19):.2f}  flow {sh(r_f,g_f,1.19):.2f}")
print(f"shell2(1.85): real-eq {sh(r_eq,g_eq,1.85):.2f}  flow {sh(r_f,g_f,1.85):.2f}")
torch.save({"r_eq":r_eq,"g_eq":g_eq,"r_flow":r_f,"g_flow":g_f,"U_eq":uN(x),"U_flow":uN(xf)},
           "liquid_coupling_flow/mw/artifacts/mw_ersi_n216_real_compare.pt")
