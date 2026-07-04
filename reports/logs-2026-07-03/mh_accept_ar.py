"""MH acceptance for the AR cluster proposal (same protocol as mh_accept.py: equilibrium configs,
log alpha = -beta*dU + logq(x_C|cage(x')) - logq(x'_C|cage(x)); AR logq = one forward, exact)."""
import sys, torch
from liquid_coupling_flow import ka_cluster as KC
from liquid_coupling_flow.ka_cluster_flow import slot_order, _scaffold, _load, ART
from liquid_coupling_flow.ka_energy import ka_energy
import os
DEV="cuda"; BETA=2.0; torch.manual_seed(0)
ck_name = sys.argv[1] if len(sys.argv)>1 else "ka_cluster_flow_full_N100.pt"
B = int(sys.argv[2]) if len(sys.argv)>2 else 64
sc,L,geo=_scaffold(100,DEV)
ref=torch.load(os.path.join(ART,"ka_reference_N100.pt"),map_location=DEV,weights_only=False)
s0=ref["s"].to(DEV).long()
ck=torch.load(os.path.join(ART,ck_name),map_location=DEV,weights_only=False)
P=_load(ck,DEV); k=ck["k"]
print(f"AR model {ck_name} head={ck.get('head')} pair_feats={ck.get('pair_feats')} step {ck['step']} | B={B} beta={BETA}",flush=True)
idx=torch.randperm(ref["x"].shape[0],device=DEV)[:B]
pos,sso=slot_order(ref["x"][idx].to(DEV),s0,geo,100)
acc_all,dU_all,dq_all,clash_all=[],[],[],[]
with torch.no_grad():
    for seed in range(0,100,10):
        cl=KC.cluster_slots(seed,sc,k,L)
        xC_cur=pos[:,cl].clone()
        U_cur=ka_energy(pos,sso,L)
        xC_new,logq_fwd=P.sample(pos,sso,cl,sc,L)
        pos_new=pos.clone(); pos_new[:,cl]=xC_new
        U_new=ka_energy(pos_new,sso,L)
        logq_rev=P.log_q(pos_new,sso,cl,xC_cur,sc,L)
        dU=U_new-U_cur
        log_alpha=-BETA*dU+logq_rev-logq_fwd
        acc=torch.clamp(log_alpha,max=0.0).exp()
        d=xC_new[:,:,None,:]-pos_new[:,None,:,:]; d=d-L*torch.round(d/L)
        r=(d**2).sum(-1).sqrt(); r[:,:,cl]+=torch.eye(k,device=DEV)[None]*1e3
        clash=(r.min(-1).values.min(-1).values<0.7).float()
        acc_all.append(acc); dU_all.append(dU); dq_all.append(logq_rev-logq_fwd); clash_all.append(clash)
acc=torch.cat(acc_all); dU=torch.cat(dU_all); dq=torch.cat(dq_all); clash=torch.cat(clash_all)
print(f"=== AR MH acceptance ({acc.numel()} moves) ===")
print(f"  mean acceptance      : {acc.mean().item()*100:.3f}%   (median {acc.median().item()*100:.3f}%)")
print(f"  acceptance | no-clash : {acc[clash==0].mean().item()*100:.3f}%   (fraction no-clash {(clash==0).float().mean().item()*100:.1f}%)")
print(f"  mean logq_rev-logq_fwd: {dq.mean().item():.3f}   P(dU<0): {(dU<0).float().mean().item()*100:.1f}%")
