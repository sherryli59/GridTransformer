"""Plot the K-block clash decomposition (numbers from diag_kblock_clash_decomp.out, seed=0 deterministic).
Shows: K=1 = full-cage ideal (clash 0.11); as K grows a clustered block becomes a mini-cavity -- BOTH
vs-block (internal half-cage) AND vs-retained (spill-out) rise toward the full-regen level (0.45)."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import numpy as np

K = [1, 4, 8, 16]
vs_ret = [0.052, 0.242, 0.262, 0.150]
vs_blk = [0.000, 0.042, 0.167, 0.206]
vs_bnd = [0.057, 0.128, 0.138, 0.123]
x = np.arange(len(K))
fig, ax = plt.subplots(figsize=(7, 4.6))
ax.bar(x, vs_ret, 0.6, label="vs RETAINED interior (fully visible)", color="C0")
ax.bar(x, vs_blk, 0.6, bottom=vs_ret, label="vs BLOCK (internal half-cage, residual)", color="C3")
ax.bar(x, vs_bnd, 0.6, bottom=np.array(vs_ret) + np.array(vs_blk), label="vs boundary", color="C7")
ax.axhline(0.453, ls="--", color="k", lw=1, label="full-regen interior clash (all-AR, R=2.5)")
ax.set_xticks(x); ax.set_xticklabels([f"K={k}" for k in K])
ax.set_ylabel("clashes / BLOCK particle (r<0.85 sigma)")
ax.set_title("K-block regen (retained frozen+visible), R=2.5:\nK=1 full-cage ideal; clustered large-K = mini-cavity half-cage")
ax.legend(fontsize=8, loc="upper left")
ax.annotate("full cage\n(your K=1 idea)", (0, 0.109), textcoords="offset points", xytext=(0, 12),
            ha="center", fontsize=8, color="C0")
fig.tight_layout()
out = "reports/logs-2026-07-13/kblock_clash_decomp.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}")
