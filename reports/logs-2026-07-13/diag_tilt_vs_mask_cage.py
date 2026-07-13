import torch, statistics as st
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA
dev="cpu"; RCTX=2.5; K=8; R=2.0; PT=0.4; M=16; NCAV=8; CUT=0.9
sig=torch.tensor(SIGMA).double()
m=KA3DScaffoldEBMCoarse(cat_bins=128,cat_range=2.5).double()
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_coarse_best.pt",map_location=dev,weights_only=False)["state_dict"],strict=False)
m.eval(); m.use_frame=False
D=torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt",map_location=dev,weights_only=False)
X,S,L=D["x"].double(),D["s"].long(),float(D["L"])
def run(tilt_knn, mask_knn, label):
    m.knn_pot = tilt_knn
    gen=torch.Generator().manual_seed(5); ch=ct=0; ncav=0
    for ci in range(16):
        c=torch.rand(3,generator=gen).double()*L; p=carve(X[ci],S[ci],c,R,L)
        if p["n_in"]<K+6: continue
        xo,so,_=label_to_scaffold(_mic(p["x_in"],c,L),p["s_in"],R)
        xout=_mic(p["x_out"],c,L); bm=xout.norm(dim=-1)<(R+RCTX); bnd,sb=xout[bm],p["s_out"][bm]
        n=xo.shape[0]; anch=fixed_ball_scaffold(n,R,dev).double(); seed=int(torch.randint(n,(),generator=gen))
        blk=torch.zeros(n,dtype=torch.bool); blk[(anch-anch[seed]).norm(dim=-1).topk(K,largest=False).indices]=True
        xs,ss,_=m.sample_block_b(xo[None].expand(M,n,3).contiguous(),so[None].expand(M,n).contiguous(),blk,bnd,sb,R,gen=gen,pos_temp=PT,min_sep=CUT,nested=True,mask_knn=mask_knn)
        bx=xs[:,blk]; bs=ss[:,blk]
        cage_x=torch.cat([bnd[None].expand(M,-1,-1),xs[:,~blk]],1); cage_s=torch.cat([sb[None].expand(M,-1),ss[:,~blk]],1)
        nx=torch.cat([bx,cage_x],1); ns=torch.cat([bs,cage_s],1)
        for mm in range(M):
            d=torch.cdist(bx[mm],nx[mm]); d[torch.arange(K),torch.arange(K)]=float("inf")
            ch+=int(((d/sig[bs[mm][:,None],ns[mm][None,:]]).min(1).values<0.9).sum()); ct+=K
        ncav+=1
        if ncav>=NCAV: break
    print(f"  {label:28s}: clash<0.9 {100*ch/ct:5.1f}%",flush=True)
print("=== ISOLATE tilt-cage vs mask-cage (nested mask, coarse head) ===",flush=True)
run(8, None, "tilt=8  mask=8  (baseline)")
run(8, 32,   "tilt=8  mask=32 (mask only)")
run(32, 32,  "tilt=32 mask=32 (both, ref)")
run(8, 64,   "tilt=8  mask=64 (mask only)")
