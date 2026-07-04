"""Decorrelation time from the FULL saved trajectories: time-origin-averaged self-overlap
Q(dt) = < frac(|r_i(t+dt)-r_i(t)| < 0.3) > over origins t>=300s (equilibrated window) and chains,
vs WALL-CLOCK lag. tau = interpolated crossing of 1/e (and 0.5). Also origin-averaged MSD(dt)."""
import os, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_cluster_flow import ART
from liquid_coupling_flow.ka_gridformer import _wrap_pm
COL={"swap":"#2a78d6","mtm":"#1baf7a","hybrid":"#e34948"}
LAGS=np.array([10,20,40,80,120,160,240,320,400,480,560])
T0=300.0; TOL=6.0; MAX_ORIG=240
fig,(a1,a2)=plt.subplots(1,2,figsize=(11.5,4.4))
taus={}
for arm in ("swap","mtm","hybrid"):
    d=torch.load(os.path.join(ART,f"mtm_traj_{arm}.pt"),map_location="cpu",weights_only=False)
    t=d["t"].numpy(); x=d["x"]; L=d["L"]
    sel=np.where(t>=T0)[0]
    Q=[]; M=[]
    for lag in LAGS:
        qs=[]; ms=[]
        origins=sel[t[sel]+lag<=t[-1]+TOL]
        if len(origins)>MAX_ORIG:
            origins=origins[np.linspace(0,len(origins)-1,MAX_ORIG).astype(int)]
        for j in origins:
            k=np.argmin(np.abs(t-(t[j]+lag)))
            if abs(t[k]-(t[j]+lag))>TOL: continue
            disp=_wrap_pm(x[k]-x[j],L).norm(dim=-1)                   # [B,N]
            qs.append(float((disp<0.3).float().mean())); ms.append(float((disp**2).mean()))
        Q.append(np.mean(qs)); M.append(np.mean(ms))
    Q=np.array(Q); M=np.array(M)
    def cross(level):
        below=np.where(Q<level)[0]
        if len(below)==0: return np.inf
        i=below[0]
        if i==0: return LAGS[0]
        return float(np.interp(level,[Q[i],Q[i-1]],[LAGS[i],LAGS[i-1]]))
    taus[arm]=(cross(1/np.e),cross(0.5))
    a1.plot(LAGS,Q,"-o",color=COL[arm],lw=1.8,ms=4,label=f"{arm}  τ(1/e)={taus[arm][0]:.0f}s")
    a2.plot(LAGS,M,"-o",color=COL[arm],lw=1.8,ms=4,label=arm)
    print(f"{arm:7s} Q(dt): "+"  ".join(f"{q:.3f}" for q in Q)+f"   tau_1/e {taus[arm][0]:.0f}s  tau_0.5 {taus[arm][1]:.0f}s",flush=True)
a1.axhline(1/np.e,color="#0b0b0b",ls=":",lw=1); a1.text(11,1/np.e+0.01,"1/e",fontsize=8)
a1.set_xlabel("wall-clock lag Δt (s)"); a1.set_ylabel("self-overlap Q(Δt)"); a1.set_xscale("log")
a1.set_title("Equilibrium decorrelation (origins t>300s)"); a1.legend(fontsize=9,frameon=False)
a2.set_xlabel("wall-clock lag Δt (s)"); a2.set_ylabel("MSD(Δt)"); a2.set_xscale("log"); a2.set_yscale("log")
a2.set_title("Origin-averaged MSD vs lag"); a2.legend(fontsize=9,frameon=False); a2.grid(alpha=.15,lw=.5)
out=os.path.join(ART,"mtm_decorrelation.png"); fig.tight_layout(); fig.savefig(out,dpi=130)
print("saved",out,flush=True)
