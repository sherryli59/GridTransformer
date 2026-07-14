"""VERDICT: rank-resolved clash AND capped-repulsive-energy per block member, base (block-cond FT) vs the
RL+demand checkpoint. Primary question: did the free-run energy objective make ranks 1-4 rise SLIGHTLY
(early precision traded) while the rank-8 spike DROPS (late feasibility gained)? -- the compounding fix --
or did it just globally lower everything (possible under-packing). Also reports totals + block spread
(under-packing guard: RL spread >> base spread = spread-out cheat). K=8 blob, R=2.5, M=16, 12 cav.
Energy per member = sum of its capped repulsive pair energies with ALL others (block+retained+boundary)."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA, EPS

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 16; NCAV = 12; CUT = 0.85; K = 8; CAP = 5.0
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
def profile(m, gen):
    rank_clash = [[] for _ in range(K)]; rank_E = [[] for _ in range(K)]
    tot_clash, tot_E, spread = [], [], []
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
        bidx = blk.nonzero().squeeze(1)                                          # ascending morton = rank
        Xg, Sg, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     blk, bnd, sb, R, gen=gen)
        for k in range(M):
            xb, sbk = Xg[k][bidx], Sg[k][bidx]                                   # block members (rank order)
            fx = torch.cat([Xg[k][(~blk).nonzero().squeeze(1)], bnd])            # fixed context
            fs = torch.cat([Sg[k][(~blk).nonzero().squeeze(1)], sb])
            # clash CREATION (attribute to later member): vs fixed context + earlier block
            dfx = torch.cdist(xb, fx); nfix = (dfx < CUT * SIG[sbk[:, None], fs[None, :]]).sum(1)
            dbb = torch.cdist(xb, xb); clbb = (dbb < CUT * SIG[sbk[:, None], sbk[None, :]]) & ~torch.eye(K, dtype=torch.bool, device=dev)
            earlier = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev), -1)
            nearly = (clbb & earlier).sum(1)
            # capped repulsive energy per member: pairs with ALL others (block + context)
            allx = torch.cat([xb, fx]); alls = torch.cat([sbk, fs])
            sig = SIG[sbk[:, None], alls[None, :]]; eps = EP[sbk[:, None], alls[None, :]]
            d2 = torch.cdist(xb, allx).clamp_min(1e-6) ** 2
            self_mask = torch.zeros(K, allx.shape[0], dtype=torch.bool, device=dev)
            self_mask[torch.arange(K), torch.arange(K)] = True                   # exclude self (block idx 0..K-1)
            d2 = d2.masked_fill(self_mask, 1e12)
            inv6 = (sig ** 2 / d2) ** 3; e = (4 * eps * (inv6 ** 2 - inv6)).clamp(0, CAP).sum(1)   # [K] per member
            for r in range(K):
                rank_clash[r].append(float(nfix[r] + nearly[r])); rank_E[r].append(float(e[r]))
            tot_clash.append(float((nfix + nearly).float().mean())); tot_E.append(float(e.mean()))
            spread.append(float((xb - xb.mean(0)).norm(dim=-1).mean()))
        ncav += 1
        if ncav >= NCAV:
            break
    return ([st.mean(v) for v in rank_clash], [st.mean(v) for v in rank_E],
            st.mean(tot_clash), st.mean(tot_E), st.mean(spread))


mb = load(BASE, False); mr = load(RLD, True)
cb, eb, tcb, teb, spb = profile(mb, torch.Generator(device=dev).manual_seed(0))
cr, er, tcr, ter, spr = profile(mr, torch.Generator(device=dev).manual_seed(0))
print(f"=== RL+demand (step 500) vs base (block-cond FT): K={K} blob, R={R} ===", flush=True)
print(f"  total clash/member  base {tcb:.3f}  RL {tcr:.3f}  ({100*(tcr-tcb)/tcb:+.0f}%)", flush=True)
print(f"  total capE/member   base {teb:.3f}  RL {ter:.3f}  ({100*(ter-teb)/teb:+.0f}%)", flush=True)
print(f"  block spread (under-pack guard) base {spb:.3f}  RL {spr:.3f}  ({100*(spr-spb)/spb:+.1f}%)", flush=True)
print(f"  rank clash base : {['%.2f'%v for v in cb]}", flush=True)
print(f"  rank clash RL   : {['%.2f'%v for v in cr]}", flush=True)

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
rr = range(1, K + 1)
ax[0].plot(rr, cb, "o-", color="C7", label="base (block-cond)")
ax[0].plot(rr, cr, "s-", color="C2", label="RL+demand (step500)")
ax[0].axhline(0.109, ls="--", color="k", lw=1, label="K=1 full-cage")
ax[0].set_xlabel("placement rank"); ax[0].set_ylabel("clash creation / member")
ax[0].set_title(f"Clash vs rank (spread base {spb:.2f} / RL {spr:.2f})"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].plot(rr, eb, "o-", color="C7", label="base")
ax[1].plot(rr, er, "s-", color="C2", label="RL+demand")
ax[1].set_xlabel("placement rank"); ax[1].set_ylabel("capped repulsive E / member")
ax[1].set_title("Energy vs rank"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
fig.tight_layout(); out = "reports/logs-2026-07-13/rl_rank_compare.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)
