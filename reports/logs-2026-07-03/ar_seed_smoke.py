import os, torch
from liquid_coupling_flow.ka_flowhead import KAFlowHeadModel, ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_gridformer import _wrap_pm
DEV="cuda"; torch.manual_seed(0)
ck=torch.load(os.path.join(ART,"ka_flowhead_N100_k8_scratch.pt"),map_location=DEV,weights_only=False)
m=KAFlowHeadModel(rho=1.2,n_bins=ck["n_bins"],knn=ck["knn"],num_bins=ck["num_bins"],tail_bound=ck["tail_bound"]).to(DEV)
m.load_state_dict(ck["state_dict"]); m.eval()
pos,sp=m.sample(8,100,n_B=35,device=DEV)
L=m._Lof(100)
U=ka_energy(pos,sp,L)/100
d=_wrap_pm(pos[:,:,None]-pos[:,None],L).norm(dim=-1)+torch.eye(100,device=DEV)[None]*1e3
minr=d.min(-1).values
print("AR full-config samples: U/N per config:",[f"{u:.2f}" for u in U.tolist()])
print(f"min-r: mean {minr.mean():.3f}  P(<0.7) {(minr<0.7).float().mean()*100:.1f}%  P(<0.5) {(minr<0.5).float().mean()*100:.1f}%")
print(f"species counts row0: A {(sp[0]==0).sum().item()} B {(sp[0]==1).sum().item()}  L={L:.3f}")
