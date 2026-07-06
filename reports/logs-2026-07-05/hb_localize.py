"""Localized single-site exactness: fixed config, freeze ALL others, move ONLY site i (no re-ordering).
Two kernels target the SAME conditional exp(-beta*U_row(x_i)): (a) heat-bath (learned proposal), (b) exact
Gaussian-displacement MH. If <U_row> agree -> heat-bath move exact. If heat-bath lower -> the MOVE is biased."""
import torch, numpy as np
from liquid_coupling_flow.ka_heatbath_gate import _env
from liquid_coupling_flow.ka_heatbath import single_site_mh
from liquid_coupling_flow.ka_energy import ka_pair_row

DEV="cuda" if torch.cuda.is_available() else "cpu"
sc,L,geo,pos,s,HB,N=_env(B=1)
site=5; beta=2.0
gen=torch.Generator(device=DEV).manual_seed(0)

# (a) heat-bath, fixed site index, cage frozen (others never move)
p=pos.clone(); Uh=[]
for step in range(6000):
    p,_=single_site_mh(HB,p,s,site,sc,L,beta,gen=gen)
    if step>800: Uh.append(ka_pair_row(p,s,site,p[:,site],L).item())

# (b) exact Gaussian single-site displacement MH targeting exp(-beta*U_row(x_i))
p2=pos.clone(); Ud=[]; step_size=0.15
for step in range(20000):
    xi=p2[:,site]
    xin=torch.remainder(xi+step_size*torch.randn(1,2,generator=gen,device=DEV),L)
    dU=ka_pair_row(p2,s,site,xin,L)-ka_pair_row(p2,s,site,xi,L)
    if torch.log(torch.rand(1,generator=gen,device=DEV))< -beta*dU:
        p2=p2.clone(); p2[:,site]=xin
    if step>3000 and step%3==0: Ud.append(ka_pair_row(p2,s,site,p2[:,site],L).item())

print(f"site {site}: <U_row>  heat-bath {np.mean(Uh):.4f} (n={len(Uh)})   exact-disp {np.mean(Ud):.4f} (n={len(Ud)})")
print(f"  diff (HB - exact) = {np.mean(Uh)-np.mean(Ud):+.4f}   [~0 => move exact; <<0 => move biased low]")
