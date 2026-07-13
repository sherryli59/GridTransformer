"""Residual decomposition of the NESTED-masked coarse head: is the 27.8% clash mostly FUTURE-BLIND
(mask succeeded, floor is high) or EARLIER/CAGE (mask leaked -> anchor-kNN gap)? For each placed block
particle, split its binding (min sigma-gap) clash partner:
  earlier-block (Morton rank < t)  -> the placing particle's mask SAW it; a leak if it clashes (anchor-kNN gap)
  later-block   (Morton rank > t)  -> future-blind; BUT the LATER particle's mask saw THIS one -> also a leak
                                      unless boxed in. Reported separately.
  retained / boundary              -> always visible -> leak if it clashes
Run 'none' vs 'NESTED' side by side so the DELTA shows what the nested mask removed. CPU/float64 (GPU busy)."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold, ball_unsquash
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cpu"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 16; NCAV = 8; CUT = 0.9; CLASH = 0.9
sig = torch.tensor(SIGMA, device=dev).double()
m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5).to(dev).double()
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_coarse_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).double(), D["s"].to(dev).long(), float(D["L"])


def decompose(kw, label):
    gen = torch.Generator(device=dev).manual_seed(5)
    tot = {"n": 0, "clash": 0, "earlier": 0, "later": 0, "retained": 0, "boundary": 0}
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev).double() * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev).double()
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        xb = xo[None].expand(M, n, 3).contiguous(); sbb = so[None].expand(M, n).contiguous()
        xs, ss, _ = m.sample_block_b(xb, sbb, blk, bnd, sb, R, gen=gen, pos_temp=PT, **kw)
        # Morton generation rank of block cols (block = Morton suffix after retained)
        order = torch.argsort(blk.to(torch.uint8), stable=True); gen_seq = order[n - K:]
        blk_ids = torch.nonzero(blk, as_tuple=True)[0]
        rank = torch.tensor([int((gen_seq == j).nonzero()) for j in blk_ids], device=dev)
        for mm in range(M):
            bx = xs[mm, blk]; bs = ss[mm, blk]; rx = xs[mm, ~blk]; rs = ss[mm, ~blk]
            for q in range(K):
                tot["n"] += 1
                cats = {}
                ex = rank < rank[q]; lt = rank > rank[q]
                if ex.any():
                    cats["earlier"] = (torch.cdist(bx[q:q+1], bx[ex])[0] / sig[bs[q], bs[ex]]).min()
                if lt.any():
                    cats["later"] = (torch.cdist(bx[q:q+1], bx[lt])[0] / sig[bs[q], bs[lt]]).min()
                if rx.shape[0]:
                    cats["retained"] = (torch.cdist(bx[q:q+1], rx)[0] / sig[bs[q], rs]).min()
                cats["boundary"] = (torch.cdist(bx[q:q+1], bnd)[0] / sig[bs[q], sb]).min()
                binder = min(cats, key=lambda k: float(cats[k]))
                if float(cats[binder]) < CLASH:
                    tot["clash"] += 1; tot[binder] += 1
        ncav += 1
        if ncav >= NCAV:
            break
    cl = max(tot["clash"], 1)
    print(f"{label:8s}: clash {100*tot['clash']/tot['n']:5.1f}% | of clashes: earlier {100*tot['earlier']/cl:3.0f}% "
          f"later(future) {100*tot['later']/cl:3.0f}% retained {100*tot['retained']/cl:3.0f}% boundary {100*tot['boundary']/cl:3.0f}%", flush=True)
    return tot


print("=== NESTED-MASK RESIDUAL DECOMPOSITION (coarse head, K=8, cut=0.9) ===", flush=True)
decompose({}, "none")
decompose({"min_sep": CUT, "nested": True}, "NESTED")
print("earlier/retained/boundary = mask SAW it (leak = anchor-kNN gap); later = future-blind for the placer", flush=True)
