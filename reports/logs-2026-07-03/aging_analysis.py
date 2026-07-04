"""TWO-TIME AGING ANALYSIS: Q(t_w, t_w+dt) from the full saved trajectories — decorrelation as a function of
sample age, per start (uniform / AR / equilibrium) and kernel (swap / mtm / hybrid).
tau_c(t_w) = lag where Q(t_w,.) crosses 0.7 (interpolated; 1/e unreachable at late ages in-window)."""
import os, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_cluster_flow import ART
from liquid_coupling_flow.ka_gridformer import _wrap_pm

ARMS={ "rand+swap":("arstart_traj_rand_swap.pt","#2a78d6","--"),
       "rand+hybrid":("arstart_traj_rand_hybrid.pt","#e34948","--"),
       "ar+swap":("arstart_traj_ar_swap.pt","#2a78d6","-"),
       "ar+mtm":("arstart_traj_ar_mtm.pt","#1baf7a","-"),
       "ar+hybrid":("arstart_traj_ar_hybrid.pt","#e34948","-"),
       "eq+swap":("mtm_traj_swap.pt","#0b0b0b","-"),
       "eq+hybrid":("mtm_traj_hybrid.pt","#7a7973","-") }
TW=[0,30,100,300,600]; LAGS=np.array([5,10,20,40,80,160,300])
def load(fn):
    d=torch.load(os.path.join(ART,fn),map_location="cpu",weights_only=False)
    return d["t"].numpy(),d["x"],d["L"]
def Qpair(x,L,i,j):
    disp=_wrap_pm(x[j]-x[i],L).norm(dim=-1)
    return float((disp<0.3).float().mean())
res={}
for name,(fn,col,ls) in ARMS.items():
    try: t,x,L=load(fn)
    except FileNotFoundError: print("skip",name); continue
    curves={}
    for tw in TW:
        if tw>t[-1]-LAGS[0]: continue
        i=int(np.argmin(np.abs(t-tw))); q=[]
        for lag in LAGS:
            tgt=t[i]+lag
            if tgt>t[-1]+3: q.append(np.nan); continue
            j=int(np.argmin(np.abs(t-tgt)))
            q.append(Qpair(x,L,i,j) if abs(t[j]-tgt)<=6 else np.nan)
        curves[tw]=np.array(q)
    res[name]=(curves,col,ls)
    taus=[]
    for tw,q in curves.items():
        v=~np.isnan(q)
        if v.sum()<2: taus.append((tw,np.nan)); continue
        qq,ll=q[v],LAGS[v]
        below=np.where(qq<0.7)[0]
        tau=np.inf if len(below)==0 else (ll[0] if below[0]==0 else float(np.interp(0.7,[qq[below[0]],qq[below[0]-1]],[ll[below[0]],ll[below[0]-1]])))
        taus.append((tw,tau))
    res[name]=(curves,col,ls,taus)
    print(f"{name:12s} tau_0.7(t_w): "+"  ".join(f"{tw}s:{'inf' if not np.isfinite(tu) else f'{tu:.0f}s'}" for tw,tu in taus),flush=True)
fig,axes=plt.subplots(1,3,figsize=(16.5,4.6))
# A: aging fan, rand+swap vs ar+swap
for name in ("rand+swap","ar+swap"):
    if name not in res: continue
    curves,col,ls,_=res[name]
    for k,(tw,q) in enumerate(curves.items()):
        axes[0].plot(LAGS,q,color=col,ls=ls,lw=1.2+0.5*k,alpha=0.45+0.11*k,
                     label=f"{name} t_w={tw}s" if tw in (0,600) else None)
axes[0].axhline(0.7,color="#0b0b0b",ls=":",lw=1); axes[0].set_xscale("log")
axes[0].set_xlabel("lag Δt (s)"); axes[0].set_ylabel("Q(t_w, t_w+Δt)")
axes[0].set_title("Aging fan (thicker = older)"); axes[0].legend(fontsize=7,frameon=False)
# B: tau(t_w) all arms
for name,(curves,col,ls,taus) in res.items():
    tws=[tw for tw,tu in taus if np.isfinite(tu)]; ts=[tu for tw,tu in taus if np.isfinite(tu)]
    axes[1].plot(np.array(tws)+1,ts,"-o",color=col,ls=ls,ms=4,lw=1.6,label=name)
axes[1].set_xscale("symlog",linthresh=20); axes[1].set_yscale("log")
axes[1].set_xlabel("waiting time t_w (s)"); axes[1].set_ylabel("τ_0.7(t_w) (s)")
axes[1].set_title("Decorrelation time vs age"); axes[1].legend(fontsize=7,frameon=False); axes[1].grid(alpha=.15)
# C: U/N relaxation overlay (non-eq arms)
for name,(curves,col,ls,_) in res.items():
    if name.startswith("eq"): continue
    d=torch.load(os.path.join(ART,ARMS[name][0]),map_location="cpu",weights_only=False)
    met=np.array([(m[0],m[1]) for m in d["metrics"]])
    axes[2].plot(met[:,0],met[:,1],color=col,ls=ls,lw=1.6,label=name)
axes[2].axhline(-3.26,color="#0b0b0b",ls=":",lw=1); axes[2].text(15,-3.255,"equilibrium",fontsize=8)
axes[2].set_xlabel("wall-clock (s)"); axes[2].set_ylabel("U/N"); axes[2].set_ylim(-3.30,-2.55)
axes[2].set_title("Relaxation from uniform/AR starts"); axes[2].legend(fontsize=7,frameon=False); axes[2].grid(alpha=.15)
out=os.path.join(ART,"aging_two_time.png"); fig.tight_layout(); fig.savefig(out,dpi=130)
print("saved",out,flush=True)
