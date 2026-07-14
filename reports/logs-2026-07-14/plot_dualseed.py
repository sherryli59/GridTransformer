"""Dual-seed non-ergodicity: alien-seed and data-seed BRACKET the true (PT-reference) overlap. alien = lower
bound (can't REACH the reference basin), data = upper bound (frozen AT reference, mutation can't relax away at
large R). The gap between them = the non-ergodicity / seed-dependence; a converged sampler would have them
coincide on the truth."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = [1.6, 2.0]
alien = [0.126, 0.060]
data = [0.627, 0.983]
PT_TRUE = [0.66, 0.63]          # converged PT-stack reference (memory), approx
x = np.arange(len(R))

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.fill_between(x, alien, data, color="orange", alpha=0.15, label="seed-dependence gap (non-ergodic)")
ax.plot(x, alien, "o-", color="crimson", lw=2, ms=9, label="alien-seed q~ (AR proposal) -- LOWER bound")
ax.plot(x, data, "s-", color="steelblue", lw=2, ms=9, label="data-seed q~ (reference) -- UPPER bound")
ax.plot(x, PT_TRUE, "*", color="black", ms=20, label="PT-reference TRUE overlap")
for i in range(len(R)):
    ax.annotate(f"{data[i]/max(alien[i],1e-3):.0f}x", (x[i], (alien[i]+data[i])/2), fontsize=11,
                ha="center", color="darkorange", fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels([f"R={r}" for r in R])
ax.set_ylabel(r"$\tilde q(R)$ (box-occupation, l=0.368, bulk-subtracted)")
ax.set_ylim(0, 1.1)
ax.set_title("Dual-seed PTS: the estimate depends on the seed => NOT converged\n"
             "reference basin is STABLE (data holds) but UNREACHABLE from alien; truth lies between")
ax.legend(fontsize=9, loc="center left"); ax.grid(alpha=0.3)
plt.tight_layout(); OUT = "reports/logs-2026-07-14/dualseed_bracket.png"
plt.savefig(OUT, dpi=130); print(f"saved -> {OUT}", flush=True)
