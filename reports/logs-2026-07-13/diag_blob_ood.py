"""GENERALIZABILITY test for the K-blob block conditional (user question: 'how does the morton order for
the K-blob work? could there be a generalizability issue?'). Training = full-sequence morton NLL only, so
the block conditional is ZERO-SHOT from morton-prefix conditionals. The reorder hands block members
frac~1 slot features ('late placement') whose trained analogue is the morton-curve TAIL. Three OOD axes:
(1) frac<->location decorrelation (trained frac~1 slots always sit at the curve tail region; a blob gets
frac~1 anywhere), (2) within-block order = morton-restricted-to-blob (can zigzag across jumps; training
only saw contiguous sweeps), (3) hole-in-full context vs swept prefix.

Arms at K=8, R=2.5 (same harness/seeds):
  A random anchor-blob (production move; all three axes OOD)
  B morton-TAIL block (block = LAST K slots -> reorder is identity; frac, anchors, within-order, context
    ALL exactly a trained suffix conditional; the fully in-distribution control)
  C morton-contiguous segment at random offset (in-curve within-order, random location -> isolates axis 1)
Metrics: clash vs FIXED / INTERNAL per block particle, median dE/particle, exact MTM acceptance, and the
rank-resolved clash-creation curve for A vs B. READ: B==A => no OOD cost, blob is fine & budgeting is
intrinsic (if B also shows the rank rise); B<<A => real generalizability issue in the blob conditional.
CAVEAT: the tail region is a fixed part of the ball (morton +++ corner) -> different cage composition;
noted, not controlled."""
import torch, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12; CUT = 0.85; K = 8
SIG = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def pick_block(arm, n, a, gen):
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    if arm == "A":                                            # random anchor-blob (production)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    elif arm == "B":                                          # morton tail: trained suffix conditional
        blk[n - K:] = True
    else:                                                     # contiguous segment, random offset
        off = int(torch.randint(n - K, (), generator=gen, device=dev))
        blk[off:off + K] = True
    return blk


def energy_set(Xi, Si):
    return ka_energy(Xi.double(), Si.long(), BIGL).float()


@torch.no_grad()
def run(arm):
    cfix, cint, des, accs = [], [], [], []
    per_rank = [[] for _ in range(K)]
    gen = torch.Generator(device=dev).manual_seed(0)
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        blk = pick_block(arm, n, a, gen)
        Xg, Sg, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, R, gen=gen)
        E0 = energy_set(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        Eg = energy_set(torch.cat([Xg, bnd[None].expand(M, -1, 3)], 1), torch.cat([Sg, sb[None].expand(M, -1)], 1))
        u0 = -BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R)[0]
        up = -BETA * Eg - lqf
        sf = torch.logsumexp(up, 0)
        J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        des.append(float((Eg.median() - E0) / K))
        bi = blk.nonzero().squeeze(1); ri = (~blk).nonzero().squeeze(1)
        for k in range(M):
            xb_, sb_ = Xg[k][bi], Sg[k][bi]
            fx = torch.cat([Xg[k][ri], bnd]); fs = torch.cat([Sg[k][ri], sb])
            d = torch.cdist(xb_, fx); sg = SIG[sb_[:, None], fs[None, :]]
            nfix = (d < CUT * sg).sum(1)
            d = torch.cdist(xb_, xb_); sg = SIG[sb_[:, None], sb_[None, :]]
            cl = (d < CUT * sg) & ~torch.eye(K, dtype=torch.bool, device=dev)
            tri = torch.tril(torch.ones(K, K, dtype=torch.bool, device=dev), -1)
            nearly = (cl & tri).sum(1)
            cfix.append(float(nfix.float().mean())); cint.append(float(cl.sum(1).float().mean() / 1))
            for r in range(K):
                per_rank[r].append(float(nfix[r] + nearly[r]))
        ncav += 1
        if ncav >= NCAV:
            break
    return (st.mean(cfix), st.mean(cint), st.median(des), 100 * st.mean(accs),
            [st.mean(v) for v in per_rank])


print(f"=== K-blob OOD test: random blob vs morton-TAIL (trained suffix) vs contiguous segment "
      f"(K={K}, R={R}) ===", flush=True)
print(f"{'arm':>34} | {'clash/FIXED':>11} {'clash/INT':>9} {'dE/p':>9} {'MTM':>6}", flush=True)
curves = {}
for arm, name in (("A", "A random blob (production, OOD)"), ("B", "B morton TAIL (in-distribution)"),
                  ("C", "C contiguous segment (axis-1)")):
    cf, cn, de, ac, pr = run(arm)
    curves[arm] = pr
    print(f"{name:>34} | {cf:>11.3f} {cn:>9.3f} {de:>+9.1f} {ac:>5.1f}%", flush=True)
print(f"\n  rank-resolved creations A: {['%.2f' % v for v in curves['A']]}", flush=True)
print(f"  rank-resolved creations B: {['%.2f' % v for v in curves['B']]}", flush=True)

fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.plot(range(1, K + 1), curves["A"], "o-", color="C3", label="A random blob (OOD axes 1-3)")
ax.plot(range(1, K + 1), curves["B"], "s-", color="C2", label="B morton tail (fully in-distribution)")
ax.plot(range(1, K + 1), curves["C"], "^-", color="C0", label="C contiguous segment (random offset)")
ax.axhline(0.109, ls="--", color="k", lw=1, label="K=1 full-cage level")
ax.set_xlabel("placement rank within block"); ax.set_ylabel(f"clash creations / member (r<{CUT}sigma)")
ax.set_title(f"Is the K-blob conditional OOD? tail (trained pattern) vs blob (K={K}, R={R})")
ax.legend(fontsize=8); ax.grid(alpha=0.3); fig.tight_layout()
out = "reports/logs-2026-07-13/blob_ood_rank.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}", flush=True)
