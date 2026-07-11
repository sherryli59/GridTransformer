"""3D cavity-EBM validation: (1) block-MTM K x N_trials acceptance, EBM vs frozen-phi base on the SAME
cavities; (2) K=6 clash DECOMPOSITION block<->boundary vs block<->retained vs block<->block -- the
'intra-cluster or w the boundary' split the free-cluster model could not give. argv: ebm_ckpt base_ckpt."""
import sys, statistics as st
import torch
from liquid_coupling_flow.ka3d_scaffold_ebm import KA3DScaffoldEBM
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; beta = 2.0; R_CTX = 2.5; CUT = 0.8
R = float(sys.argv[3]) if len(sys.argv) > 3 else 2.0                # argv[3]: held-out radius for transfer
ebm_p = sys.argv[1] if len(sys.argv) > 1 else "liquid_coupling_flow/artifacts/ka3d_cavity_ebm.pt"
base_p = sys.argv[2] if len(sys.argv) > 2 else "liquid_coupling_flow/artifacts/ka3d_cavity_base.pt"


def load(p):
    ck = torch.load(p, map_location=dev, weights_only=False)
    m = KA3DScaffoldEBM(cat_bins=128, cat_range=2.5).to(dev)
    m.load_state_dict(ck["state_dict"], strict=False); m.eval(); m.use_frame = False
    return m


me, mb = load(ebm_p), load(base_p)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(7)


def cavity(ci, c):
    p = carve(X[ci], S[ci], c, R, L)
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + R_CTX)
    return xo, so, xout[bm], p["s_out"][bm], p["n_in"]


def energy(xo, so, bnd, s_bnd):
    x = torch.cat([xo, bnd], 0); s = torch.cat([so, s_bnd], 0)
    return float(ka_energy(x[None], s.long()[None], 100.0)[0])


def blob(n, K):
    a = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    m = torch.zeros(n, dtype=torch.bool, device=dev); m[(a - a[seed]).norm(dim=-1).topk(min(K, n), largest=False).indices] = True
    return m


def mtm(model, xo, so, bnd, s_bnd, blk, Nt):
    u0 = energy(xo, so, bnd, s_bnd); lqx = float(model.block_log_prob(xo, so, blk, bnd, s_bnd, R))
    lus = [-beta * u0 - lqx]
    for _ in range(Nt):
        xn, sn, lq = model.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
        lus.append(-beta * energy(xn, sn, bnd, s_bnd) - float(lq))
    lu = torch.tensor(lus[1:], device=dev); sf = torch.logsumexp(lu, 0)
    js = int(torch.multinomial(torch.softmax(lu, 0), 1, generator=gen)); lr = lu.clone(); lr[js] = lus[0]
    return float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0)))


print("=== block-MTM acceptance: K x N_trials (EBM vs frozen-phi base) ===", flush=True)
cent = [torch.rand(3, generator=gen, device=dev) * L for _ in range(20)]
cavs = []
for ci, c in zip(range(900, 940), cent):
    xo, so, bnd, s_bnd, n = cavity(ci, c)
    if n >= 10:
        cavs.append((xo, so, bnd, s_bnd, n))
for K in (2, 4, 6):
    line = {"EBM": [], "base": []}
    for Nt in (16, 32, 64):
        acc = {"EBM": [], "base": []}
        for xo, so, bnd, s_bnd, n in cavs:
            blk = blob(n, K)
            acc["EBM"].append(mtm(me, xo, so, bnd, s_bnd, blk, Nt))
            acc["base"].append(mtm(mb, xo, so, bnd, s_bnd, blk, Nt))
        for k in ("EBM", "base"):
            line[k].append(f"N{Nt}:{100*sum(acc[k])/len(acc[k]):.0f}%")
    print(f"  K={K}  base [{' '.join(line['base'])}]   EBM [{' '.join(line['EBM'])}]", flush=True)

print("\n=== K=6 clash decomposition (EBM): boundary vs retained vs block ===", flush=True)
for tag, model in (("base", mb), ("EBM", me)):
    bb, brr, bbnd, tot = 0, 0, 0, 0
    for xo, so, bnd, s_bnd, n in cavs:
        blk = blob(n, 6)
        xn, sn, _ = model.sample_block(xo, so, blk, bnd, s_bnd, R, gen=gen)
        bi = xn[blk]; ret = xn[~blk]
        d_bb = (torch.cdist(bi, bi) + torch.eye(bi.shape[0], device=dev) * 1e9).min(1).values
        d_br = torch.cdist(bi, ret).min(1).values if ret.shape[0] else torch.full((bi.shape[0],), 9.0, device=dev)
        d_bnd = torch.cdist(bi, bnd).min(1).values if bnd.shape[0] else torch.full((bi.shape[0],), 9.0, device=dev)
        bb += int((d_bb < CUT).sum()); brr += int((d_br < CUT).sum()); bbnd += int((d_bnd < CUT).sum()); tot += bi.shape[0]
    print(f"  [{tag}] block particles {tot}:  <->boundary {100*bbnd/tot:.0f}%   "
          f"<->retained {100*brr/tot:.0f}%   <->block {100*bb/tot:.0f}%", flush=True)
