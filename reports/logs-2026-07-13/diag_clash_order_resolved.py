"""DISCRIMINATE the K-block clash-growth mechanism (user pushback: 'future-blind clash was NOT dominant'
-- correct: diag_coarse_clash_source measured only 21% future-blind, 79% with visible partners). Two
candidate mechanisms both produce VISIBLE-partner clashes, so the 79% can't separate them:
  (b) SPACE-BUDGETING/compounding: early members don't reserve room -> LATE-placed members face an
      over-packed cage and are forced into clashes they can see coming => creation rate RISES with rank.
  (c) FLAT RESIDUAL: each placement has the same per-step conditional error => creation rate ~flat at the
      K=1 full-cage level (0.109/particle), and total clash is just K x residual.
Measurement: K-block regen from data (R=2.5, production anchor-blob). Attribute every clash PAIR to its
LATER-placed member (the 'creator' -- at creation the partner was visible/fixed by construction unless
both are block and the partner is later). Plot clash-creation per member vs placement rank within the
block, K=8/16. Also re-report the visible-vs-future split for consistency with the 21% number."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; M = 16; NCAV = 12; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@torch.no_grad()
def order_resolved(K, gen):
    per_rank = [[] for _ in range(K)]     # clash creations by the rank-r block member
    vs_fixed_n, vs_earlier_n, vs_future_n = 0, 0, 0
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
        bi = blk.nonzero().squeeze(1)                                     # ascending original index ==
        ri = (~blk).nonzero().squeeze(1)                                  # generation order within block
        for k in range(M):
            xb_, sb_ = Xg[k][bi], Sg[k][bi]
            fx = torch.cat([Xg[k][ri], bnd]); fs = torch.cat([Sg[k][ri], sb])
            # vs fixed (retained + boundary): creator = the block member, its own rank
            d = torch.cdist(xb_, fx); sg = SIG[sb_[:, None], fs[None, :]]
            nfix = (d < CUT * sg).sum(1)                                  # [K]
            # block-internal: creator = later rank
            d = torch.cdist(xb_, xb_); sg = SIG[sb_[:, None], sb_[None, :]]
            cl = (d < CUT * sg) & ~torch.eye(K, dtype=torch.bool, device=dev)
            tri = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev), -1)   # [r, r'<r]
            nearly = (cl & tri).sum(1)                                    # creations vs earlier block
            for r in range(K):
                per_rank[r].append(float(nfix[r] + nearly[r]))
            vs_fixed_n += int(nfix.sum()); vs_earlier_n += int(nearly.sum())
            vs_future_n += int((cl & tri).sum())                          # same pairs, viewed from earlier side
        ncav += 1
        if ncav >= NCAV:
            break
    return [st.mean(v) for v in per_rank], vs_fixed_n, vs_earlier_n


print(f"=== clash CREATION per block member vs placement rank (R={R}, blob, {NCAV} cav x {M} samp) ===", flush=True)
print("    K=1 full-cage reference: 0.109/particle. Rising curve => space-budgeting; flat => residual.", flush=True)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
for K, col in ((8, "C3"), (16, "C0")):
    gen = torch.Generator(device=dev).manual_seed(0)
    pr, nfix, nearly = order_resolved(K, gen)
    tot = nfix + nearly
    print(f"  K={K}: rank-resolved creations {['%.2f' % v for v in pr]}", flush=True)
    print(f"        partner split: vs FIXED(ret+bnd) {100*nfix/tot:.0f}%  vs earlier-BLOCK {100*nearly/tot:.0f}%  "
          f"(creations; every creation has a VISIBLE partner by construction)", flush=True)
    ax.plot(range(1, K + 1), pr, "o-", color=col, label=f"K={K}")
ax.axhline(0.109, ls="--", color="k", lw=1, label="K=1 full-cage level (0.109)")
ax.set_xlabel("placement rank within block"); ax.set_ylabel(f"clash creations / member (r<{CUT}sigma)")
ax.set_title("Clash creation vs placement order: rising = space-budgeting,\nflat = per-step residual")
ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
out = "reports/logs-2026-07-13/clash_order_resolved.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)
