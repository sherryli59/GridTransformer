"""Did the block-conditional FT's NLL win (blob8 -2.72->-3.64) translate to the MEASURED clash/energy
lever, and did it cost the base? Compare knn=24 baseline vs knn=24 block-cond-FT on:
  - random-blob K=8: clash vs FIXED, clash INTERNAL, dE/particle, rank-resolved creation curve (the 3x lever)
  - allmask full-regen: interior clash/particle (the SMC island BASE -- did +0.09 NLL degrade the base?)
Both models knn_pot=24, R=2.5, N=4096 data, 12 cav x 16 samp. The NLL is a proxy; this is the metric
that feeds SMC-mutation logZ quality."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12; CUT = 0.85; K = 8
SIG = torch.tensor(SIGMA, device=dev)
CKPTS = {"baseline knn24": "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_knn24_best.pt",
         "block-cond FT ": "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"}
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def load(path):
    m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
    m.load_state_dict(torch.load(path, map_location=dev, weights_only=False)["state_dict"], strict=False)
    m.eval(); m.use_frame = False
    return m


def E(Xi, Si): return ka_energy(Xi.double(), Si.long(), BIGL).float()


@torch.no_grad()
def evalm(m, gen):
    cfix, cint, des, accs, ballmask = [], [], [], [], []
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
        Xg, Sg, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, R, gen=gen)
        E0 = E(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        Eg = E(torch.cat([Xg, bnd[None].expand(M, -1, 3)], 1), torch.cat([Sg, sb[None].expand(M, -1)], 1))
        u0 = -BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R)[0]
        up = -BETA * Eg - lqf; sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen)); lr = up.clone(); lr[J] = u0
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        des.append(float((Eg.median() - E0) / K))
        bi = blk.nonzero().squeeze(1); ri = (~blk).nonzero().squeeze(1)
        # allmask full-regen base check
        allm = torch.ones(n, dtype=torch.bool, device=dev)
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     allm, bnd, sb, R, gen=gen)
        for k in range(M):
            xb_, sb_ = Xg[k][bi], Sg[k][bi]
            fx = torch.cat([Xg[k][ri], bnd]); fs = torch.cat([Sg[k][ri], sb])
            d = torch.cdist(xb_, fx); sg = SIG[sb_[:, None], fs[None, :]]; nfix = (d < CUT * sg).sum(1)
            d = torch.cdist(xb_, xb_); sg = SIG[sb_[:, None], sb_[None, :]]
            cl = (d < CUT * sg) & ~torch.eye(K, dtype=torch.bool, device=dev)
            tri = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev), -1); nearly = (cl & tri).sum(1)
            cfix.append(float(nfix.float().mean())); cint.append(float(cl.sum(1).float().mean()))
            for r in range(K):
                per_rank[r].append(float(nfix[r] + nearly[r]))
            xi, si = Xa[k], Sa[k]
            d = torch.cdist(xi, xi); sg = SIG[si[:, None], si[None, :]]
            ballmask.append(float(((d < CUT * sg) & ~torch.eye(n, dtype=torch.bool, device=dev)).sum(1).float().mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return (st.mean(cfix), st.mean(cint), st.median(des), 100 * st.mean(accs),
            st.mean(ballmask), [st.mean(v) for v in per_rank])


print(f"=== block-cond FT eval (knn24, K={K} blob, R={R}) ===", flush=True)
print(f"{'model':>16} | {'blob clash/FIX':>13} {'blob clash/INT':>13} {'blob dE/p':>10} {'allmask clash':>13}", flush=True)
curves = {}
for name, path in CKPTS.items():
    m = load(path); cf, cn, de, ac, cb, pr = evalm(m, torch.Generator(device=dev).manual_seed(0))
    curves[name] = pr
    print(f"{name:>16} | {cf:>13.3f} {cn:>13.3f} {de:>+10.1f} {cb:>13.3f}", flush=True)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
for name, col in zip(CKPTS, ("C7", "C2")):
    ax.plot(range(1, K + 1), curves[name], "o-", color=col, label=name)
ax.axhline(0.109, ls="--", color="k", lw=1, label="K=1 full-cage level")
ax.set_xlabel("placement rank within block"); ax.set_ylabel(f"clash creations / member (r<{CUT}sigma)")
ax.set_title(f"Block-cond FT vs baseline: rank-resolved clash creation (K={K}, R={R})")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
out = "reports/logs-2026-07-13/blockcond_rank.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)
