"""Plot the Hocky box-occupation PTS overlap on our island-SMC populations vs the paper anchor, and the
old species-min-dist metric, and the l-sensitivity. Recomputes (fast) from island_smc_production.pt."""
import sys, statistics as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "reports/logs-2026-07-13")
from box_overlap_vs_hocky import box_overlap, n_boxes_in_sphere, build_cavity, X, S, L, RES

LS = [0.2, 0.3, 0.4, 0.5]
ANCHOR = {2.2: 0.33, 2.4: 0.29}
rows = {}
for (R, c), v in RES.items():
    isl = v["islands"]; xo, so, bnd, sb, xin, sin = build_cavity(X, S, L, R, c)
    xin = xin.float()
    others = [build_cavity(X, S, L, R, cc)[4] for cc in range(2, 8)]
    lz = torch.tensor([i["logZ"] for i in isl]); wj = torch.softmax(lz, 0)
    qtil = {}
    for l in LS:
        qbulk = st.mean([box_overlap(xin, o.float(), R, l) for o in others])
        qg = [st.mean([box_overlap(xin, i["X"].float()[k], R, l) for k in range(i["X"].shape[0])]) for i in isl]
        qtil[l] = float(sum(wj[j] * qg[j] for j in range(len(isl)))) - qbulk
    rows[(R, c)] = {"qtil": qtil, "old": v.get("q_island_reweighted", float("nan"))}

fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
# panel A: q~ vs R at l=0.3, box-occ vs old metric vs paper anchor
labels = list(rows.keys())
xr = [f"R{R}\nc{c}" for (R, c) in labels]
box03 = [rows[k]["qtil"][0.3] for k in labels]
old = [rows[k]["old"] for k in labels]
anch = [ANCHOR.get(R, float("nan")) for (R, c) in labels]
xi = range(len(labels))
ax[0].axhline(0, color="k", lw=0.6)
ax[0].bar([i - 0.22 for i in xi], box03, width=0.2, label="Hocky box-occ q~ (l=0.3)", color="C0")
ax[0].bar([i + 0.00 for i in xi], anch, width=0.2, label="paper LJ T=0.55 (Fig 2c)", color="C2", alpha=0.6)
ax[0].bar([i + 0.22 for i in xi], old, width=0.2, label="old species-min-dist q", color="C3", alpha=0.7)
ax[0].set_xticks(list(xi)); ax[0].set_xticklabels(xr)
ax[0].set_ylabel("overlap"); ax[0].set_title("island-SMC: box-occ q~ vs old metric vs paper anchor")
ax[0].legend(fontsize=8)
# panel B: l-sensitivity of q~ per cavity
for k in labels:
    ax[1].plot(LS, [rows[k]["qtil"][l] for l in LS], "o-", label=f"R{k[0]} c{k[1]}")
ax[1].axhspan(0.29, 0.33, color="C2", alpha=0.2, label="paper anchor band")
ax[1].axhline(0, color="k", lw=0.6)
ax[1].set_xlabel("box side l (sigma)"); ax[1].set_ylabel("q~ (bulk-subtracted)")
ax[1].set_title("l-sensitivity of box-occupation q~"); ax[1].legend(fontsize=8)
fig.tight_layout()
out = "reports/logs-2026-07-13/box_overlap_vs_hocky.png"
fig.savefig(out, dpi=130)
print(f"saved -> {out}", flush=True)
for k in labels:
    print(f"  {k}: box q~(l=.3)={rows[k]['qtil'][0.3]:+.3f}  old={rows[k]['old']:.3f}  anchor={ANCHOR.get(k[0])}", flush=True)
