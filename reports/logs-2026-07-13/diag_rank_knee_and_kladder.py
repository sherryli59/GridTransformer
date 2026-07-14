"""Two questions on the block-cond-FT model (knn24): (1) WHY does clash creation spike at rank ~7 for
K=8 -- is '7' an absolute capacity (region saturates at a fixed count regardless of K) or positional (the
last members always spike; 7 = K-1)? Measure rank-resolved clash creation for K in {4,6,8,12,16} and look
at the knee vs absolute rank AND vs rank/K. The blob = K nearest scaffold anchors, and anchors sit at
equal-volume (target-density) positions -> a K-blob region holds EXACTLY K at ideal density (zero slack);
the data fills it fine. (2) MTM acceptance for the intermediate K=6 (user asked): did the 50x proposal-
energy drop move acceptance at intermediate K? 5-seed K-ladder {2,4,6,8}, FT vs baseline.
R=2.5, N=4096, knn24."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
BASE = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


def Efn(Xi, Si): return ka_energy(Xi.double(), Si.long(), BIGL).float()


@torch.no_grad()
def rank_curve(m, K, gen):
    per_rank = [[] for _ in range(K)]
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
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=gen)
        bi = blk.nonzero().squeeze(1); ri = (~blk).nonzero().squeeze(1)
        for k in range(M):
            xb_, sb_ = Xg[k][bi], Sg[k][bi]
            fx = torch.cat([Xg[k][ri], bnd]); fs = torch.cat([Sg[k][ri], sb])
            d = torch.cdist(xb_, fx); sg = SIG[sb_[:, None], fs[None, :]]; nfix = (d < CUT * sg).sum(1)
            d = torch.cdist(xb_, xb_); sg = SIG[sb_[:, None], sb_[None, :]]
            cl = (d < CUT * sg) & ~torch.eye(K, dtype=torch.bool, device=dev)
            tri = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev), -1); nearly = (cl & tri).sum(1)
            for r in range(K):
                per_rank[r].append(float(nfix[r] + nearly[r]))
        ncav += 1
        if ncav >= NCAV:
            break
    return [st.mean(v) for v in per_rank]


@torch.no_grad()
def mtm(m, K, gen):
    accs = []
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
        Xr = xo[None].expand(M, n, 3).contiguous(); Sr = so[None].expand(M, n).contiguous()
        Xp, Sp, lqf = m.sample_block_b(Xr, Sr, blk, bnd, sb, R, gen=gen)
        Ep = Efn(torch.cat([Xp, bnd[None].expand(M, -1, 3)], 1), torch.cat([Sp, sb[None].expand(M, -1)], 1))
        E0 = Efn(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        up = -BETA * Ep - lqf
        u0 = -BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R)[0]
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen)); lr = up.clone(); lr[J] = u0
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        ncav += 1
        if ncav >= NCAV:
            break
    return 100 * st.mean(accs)


mft = load(FT)
print("=== (1) rank-resolved clash creation vs K (block-cond FT, knn24) ===", flush=True)
print("    knee = first rank with creation > 0.22 (2x the K=1 floor 0.109)", flush=True)
curves = {}
for K in (4, 6, 8, 12, 16):
    pr = rank_curve(mft, K, torch.Generator(device=dev).manual_seed(0))
    curves[K] = pr
    knee = next((r + 1 for r, v in enumerate(pr) if v > 0.22), None)
    print(f"  K={K:2d}: last-rank {pr[-1]:.2f}  knee@rank {knee} (=K-{K-knee if knee else '?'}, "
          f"frac {knee/K:.2f})  curve {['%.2f' % v for v in pr]}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.4))
for K in curves:
    ax[0].plot(range(1, K + 1), curves[K], "o-", label=f"K={K}")
    ax[1].plot([(r + 1) / K for r in range(K)], curves[K], "o-", label=f"K={K}")
for x in ax:
    x.axhline(0.109, ls="--", color="k", lw=1); x.grid(alpha=0.3); x.legend(fontsize=8)
    x.set_ylabel("clash creations / member")
ax[0].set_xlabel("absolute placement rank"); ax[0].set_title("vs absolute rank (capacity => aligned knee)")
ax[1].set_xlabel("rank / K (fractional)"); ax[1].set_title("vs rank/K (positional => aligned knee)")
fig.tight_layout(); out = "reports/logs-2026-07-13/rank_knee_vs_K.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)

print("\n=== (2) 5-seed MTM acceptance K-ladder (R=2.5, knn24): K=6 asked ===", flush=True)
mbase = load(BASE)
print(f"{'K':>3} | {'baseline knn24':>16} {'block-cond FT':>16}", flush=True)
for K in (2, 4, 6, 8):
    ab = [mtm(mbase, K, torch.Generator(device=dev).manual_seed(s)) for s in range(5)]
    af = [mtm(mft, K, torch.Generator(device=dev).manual_seed(s)) for s in range(5)]
    print(f"{K:>3} | {st.mean(ab):6.1f} +/- {st.pstdev(ab):4.1f}%   {st.mean(af):6.1f} +/- {st.pstdev(af):4.1f}%",
          flush=True)
