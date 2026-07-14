"""Rank-resolved clash + energy per member for K>8 (12/16/24), base (block-cond) vs RL+demand (step 500).
Shows whether the declash + spike-attenuation transfers at the RANK level to OOD block sizes. 2x3 grid:
top = clash vs rank, bottom = capped-repulsive-E vs rank; columns K=12,16,24. R=2.5, M=12, 10 cav."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 12; NCAV = 10; CUT = 0.85; CAP = 5.0
SIG = torch.tensor(SIGMA, device=dev); EP = torch.tensor(EPS, device=dev)
BASE = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
RLD = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_rl_demand_knn24.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path, use_demand):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=use_demand).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


@torch.no_grad()
def profile(m, K, gen):
    rc = [[] for _ in range(K)]; re = [[] for _ in range(K)]
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        bidx = blk.nonzero().squeeze(1)
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=gen)
        for k in range(M):
            xb, sbk = Xg[k][bidx], Sg[k][bidx]
            fx = torch.cat([Xg[k][(~blk).nonzero().squeeze(1)], bnd]); fs = torch.cat([Sg[k][(~blk).nonzero().squeeze(1)], sb])
            nfix = (torch.cdist(xb, fx) < CUT * SIG[sbk[:, None], fs[None, :]]).sum(1)
            dbb = torch.cdist(xb, xb); clbb = (dbb < CUT * SIG[sbk[:, None], sbk[None, :]]) & ~torch.eye(K, dtype=torch.bool, device=dev)
            nearly = (clbb & torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev), -1)).sum(1)
            allx = torch.cat([xb, fx]); alls = torch.cat([sbk, fs])
            sig = SIG[sbk[:, None], alls[None, :]]; eps = EP[sbk[:, None], alls[None, :]]
            d2 = torch.cdist(xb, allx).clamp_min(1e-6) ** 2
            sm = torch.zeros(K, allx.shape[0], dtype=torch.bool, device=dev); sm[torch.arange(K), torch.arange(K)] = True
            d2 = d2.masked_fill(sm, 1e12); inv6 = (sig ** 2 / d2) ** 3
            e = (4 * eps * (inv6 ** 2 - inv6)).clamp(0, CAP).sum(1)
            for r in range(K):
                rc[r].append(float(nfix[r] + nearly[r])); re[r].append(float(e[r]))
        ncav += 1
        if ncav >= NCAV:
            break
    return [st.mean(v) for v in rc], [st.mean(v) for v in re]


mb = load(BASE, False); mr = load(RLD, True)
KS = (12, 16, 24)
fig, ax = plt.subplots(2, 3, figsize=(14, 7.5))
for j, K in enumerate(KS):
    cb, eb = profile(mb, K, torch.Generator(device=dev).manual_seed(0))
    cr, er = profile(mr, K, torch.Generator(device=dev).manual_seed(0))
    rr = range(1, K + 1)
    ax[0, j].plot(rr, cb, "o-", color="C7", ms=4, label="base")
    ax[0, j].plot(rr, cr, "s-", color="C2", ms=4, label="RL+demand")
    ax[0, j].axhline(0.109, ls="--", color="k", lw=0.8)
    ax[0, j].set_title(f"K={K} (OOD)  clash"); ax[0, j].set_xlabel("rank"); ax[0, j].grid(alpha=0.3)
    ax[1, j].plot(rr, eb, "o-", color="C7", ms=4, label="base")
    ax[1, j].plot(rr, er, "s-", color="C2", ms=4, label="RL+demand")
    ax[1, j].set_title(f"K={K}  energy"); ax[1, j].set_xlabel("rank"); ax[1, j].grid(alpha=0.3)
    print(f"K={K}: rank-8+ spike clash base_last {cb[-1]:.2f} RL_last {cr[-1]:.2f}", flush=True)
ax[0, 0].set_ylabel("clash creation / member"); ax[1, 0].set_ylabel("capped repulsive E / member")
ax[0, 0].legend(fontsize=8); ax[1, 0].legend(fontsize=8)
fig.suptitle("RL+demand (step 500) vs base: rank-resolved clash & energy, OOD block sizes K>8", y=1.0)
fig.tight_layout(); out = "reports/logs-2026-07-13/rl_rank_ktransfer.png"
fig.savefig(out, dpi=120); print(f"saved -> {out}", flush=True)
