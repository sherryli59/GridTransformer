"""Plot the Gibbs-polish result (values from diag_gibbs_polish.out) -- rank curve flattening + clash
trajectory. Hardcoded from the run so no GPU needed."""
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ranks = list(range(1, 9))
base = [0.15, 0.18, 0.21, 0.23, 0.23, 0.41, 0.46, 1.01]
tail1 = [0.18, 0.18, 0.18, 0.23, 0.22, 0.45, 0.47, 0.72]
traj_labels = ["base", "tail\nx1", "tail\nx2", "tail\nx3", "all8\nx1", "all8\nx2", "all8\nx3"]
clash = [0.360, 0.328, 0.331, 0.323, 0.301, 0.322, 0.281]
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].plot(ranks, base, "o-", color="C7", label="baseline AR")
ax[0].plot(ranks, tail1, "s-", color="C2", label="after 1 tail polish")
ax[0].axhline(0.109, ls="--", color="k", lw=1, label="K=1 full-cage")
ax[0].set_xlabel("placement rank"); ax[0].set_ylabel("clash creations / member")
ax[0].set_title("Rank-8 spike 1.01->0.72: polish dents but can't remove it"); ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
ax[1].bar(range(len(clash)), clash, color=["C7", "C2", "C2", "C2", "C0", "C0", "C0"])
ax[1].axhline(0.360, ls="--", color="C7", lw=1)
ax[1].set_xticks(range(len(clash))); ax[1].set_xticklabels(traj_labels)
ax[1].set_ylabel("clash / particle (all 8)"); ax[1].set_title("Polish trajectory: modest ~22%, stable (no collapse)")
ax[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); out = "reports/logs-2026-07-13/gibbs_polish.png"
fig.savefig(out, dpi=130); print(f"saved -> {out}")
