"""Plot the through-xi_PTS SMC sweep (Hocky box-occupation q~) vs the paper anchors. Shows the FULL
distribution (per-cavity means as points, per-island spread as light scatter), the bulk floor, q_self=rho
ceiling context, and the paper Fig 2c anchors -- so the comparison is honest about convergence + state point.
LEFT: q~_box(R) with per-cavity scatter + island spread. RIGHT: per-island q_box for each cavity (mixing
diagnostic; wide spread => under-converged / basin-trapping)."""
import sys, statistics as st
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PT = sys.argv[1] if len(sys.argv) > 1 else "reports/logs-2026-07-14/smc_pts_hocky_lam05.pt"
OUT = PT.replace(".pt", ".png")
res = torch.load(PT, map_location="cpu", weights_only=False)
ANCHOR = {2.2: 0.33, 2.4: 0.29}

Rs = sorted(res.keys())
fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))

# LEFT: q~_box(R)
qt_mean, qt_se, bulk_mean, R_cav_pts = [], [], [], []
for R in Rs:
    rows = res[R]
    qtils = [r["q_box"] - r["qbulk"] for r in rows]          # per-cavity q~
    qt_mean.append(st.mean(qtils))
    qt_se.append(st.pstdev(qtils) / max(len(qtils) ** 0.5, 1))
    bulk_mean.append(st.mean([r["qbulk"] for r in rows]))
    for r in rows:
        R_cav_pts.append((R, r["q_box"] - r["qbulk"]))
axL.scatter([p[0] for p in R_cav_pts], [p[1] for p in R_cav_pts], s=18, alpha=0.35, color="steelblue", label="per-cavity q~")
axL.errorbar(Rs, qt_mean, yerr=qt_se, marker="o", color="navy", lw=2, capsize=3, label="mean q~_box (ours, rho=1.15 T=0.5)")
axL.plot(Rs, bulk_mean, "--", color="gray", label="bulk floor q_bulk")
axL.scatter(list(ANCHOR.keys()), list(ANCHOR.values()), marker="*", s=260, color="crimson", zorder=5,
            label="paper Fig2c (rho=1.20 T=0.55)")
axL.axhline(0, color="k", lw=0.6)
axL.set_xlabel("cavity radius R"); axL.set_ylabel(r"$\tilde q(R)$ = q - q_bulk (box-occupation, l=0.3)")
axL.set_title("PTS overlap: SMC (ours) vs Hocky PRL 2012\nsame observable; nearby state point")
axL.legend(fontsize=8); axL.grid(alpha=0.3)

# RIGHT: per-island spread (convergence / mixing)
for R in Rs:
    for r in res[R]:
        isl = r["qbox_isl"]
        axR.scatter([R] * len(isl), isl, s=14, alpha=0.4, color="darkorange")
    islmean = [st.mean(r["qbox_isl"]) for r in res[R]]
    axR.scatter([R], [st.mean(islmean)], marker="_", s=400, color="black", zorder=5)
axR.set_xlabel("cavity radius R"); axR.set_ylabel("per-island q_box (raw, not bulk-subtracted)")
axR.set_title("Island spread (mixing / convergence)\nwide => under-converged or basin-trapped")
axR.grid(alpha=0.3)

plt.tight_layout(); plt.savefig(OUT, dpi=130)
print(f"saved -> {OUT}", flush=True)
# also print a compact table
print(f"{'R':>4} {'q~_box':>8} {'+/-SE':>6} {'bulk':>6} {'isl-std/mean':>13} {'paper':>6}")
for i, R in enumerate(Rs):
    islstd = st.mean([st.pstdev(r["qbox_isl"]) for r in res[R]])
    ratio = islstd / max(qt_mean[i] + bulk_mean[i], 1e-6)
    print(f"{R:>4} {qt_mean[i]:>+8.3f} {qt_se[i]:>6.3f} {bulk_mean[i]:>6.3f} {ratio:>13.2f} "
          f"{ANCHOR.get(R, ''):>6}")
