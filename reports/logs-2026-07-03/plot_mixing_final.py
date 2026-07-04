"""Final-state energy distribution + partial g(r) per benchmark arm vs reference data."""
import os, torch, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka_cluster_flow import ART
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr
DEV="cuda"
COL={"swap":"#2a78d6","mtm":"#1baf7a","hybrid":"#e34948"}
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
s0=ref["s"].to(DEV).long(); L=ref["L"]
torch.manual_seed(3); ridx=torch.randperm(ref["x"].shape[0],device=DEV)[:512]
xref=ref["x"][ridx].to(DEV)
U_ref=(ka_energy(xref,s0,L)/100).cpu().numpy()
arms={}
for a in ("swap","mtm","hybrid"):
    d=torch.load(os.path.join(ART,f"mtm_traj_{a}.pt"),map_location="cpu",weights_only=False)
    arms[a]=d["final"].to(DEV)
fig,axes=plt.subplots(1,4,figsize=(19,4.4))
# --- energy distribution ---
ax=axes[0]
bins=np.linspace(-3.34,-3.16,37)
ax.hist(U_ref,bins=bins,density=True,color="#c8c7c2",alpha=.85,label="reference (512)")
for a in ("swap","mtm","hybrid"):
    U=(ka_energy(arms[a],s0,L)/100).cpu().numpy()
    ax.hist(U,bins=bins,density=True,histtype="step",lw=2,color=COL[a],label=f"{a} (mean {U.mean():.4f})")
ax.axvline(float(ref["U_per_N"]),color="#0b0b0b",ls="--",lw=1.2)
ax.text(float(ref["U_per_N"]),ax.get_ylim()[1]*0.97," ref mean",fontsize=8,va="top")
ax.set_xlabel("U/N (final states, 64 chains)"); ax.set_ylabel("density"); ax.legend(fontsize=8,frameon=False)
ax.set_title("Final energy distribution")
# --- partial g(r) ---
for ax,(pa,pb,nm) in zip(axes[1:],((0,0,"AA"),(0,1,"AB"),(1,1,"BB"))):
    rc,g=partial_gr(xref,s0.expand(512,100),L,4.0,60,(pa,pb))
    ax.plot(np.asarray(rc),np.asarray(g),color="#0b0b0b",ls="--",lw=2.2,label="reference")
    for a in ("swap","mtm","hybrid"):
        rc2,g2=partial_gr(arms[a],s0.expand(64,100),L,4.0,60,(pa,pb))
        ax.plot(np.asarray(rc2),np.asarray(g2),color=COL[a],lw=1.7,label=a)
    ax.set_xlabel("r"); ax.set_ylabel(f"g_{nm}(r)"); ax.set_title(f"g_{nm}(r) final vs reference")
    ax.grid(alpha=.15,lw=.5)
axes[1].legend(fontsize=8,frameon=False)
fig.suptitle("MTM-vs-swap mixing benchmark: final states after 900s matched wall-clock (B=64, from shallow slice)",y=1.02)
out=os.path.join(ART,"mtm_mixing_final_dist.png"); fig.tight_layout(); fig.savefig(out,dpi=130,bbox_inches="tight")
print("saved",out)
