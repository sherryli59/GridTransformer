"""Concrete evidence: joint two-blob regen redistributes species, and it SCALES with blob size (refutes the
'no redistribution' claim from the 2-particle case). Left: model_redist vs K against the random-shuffle
ceiling (max possible) and 0 floor. Right: model_redist as a FRACTION of the ceiling (approaches 1 = free
redistribution) + fraction of union slots that flip species. Cold MTM acc=0 annotated (clashy => low-lambda
move, not cold)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

K = [4, 8, 12, 16]
model = [0.191, 0.484, 0.781, 0.801]
ceiling = [0.512, 0.652, 1.152, 0.992]
frac = [0.105, 0.175, 0.220, 0.219]
frac_of_ceil = [m / c for m, c in zip(model, ceiling)]

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
axL.plot(K, ceiling, "s--", color="gray", label="random-shuffle ceiling (max)")
axL.plot(K, model, "o-", color="crimson", lw=2, ms=9, label="model two-blob regen (redistribution)")
axL.axhline(0, color="k", lw=0.8, label="no redistribution (floor)")
axL.fill_between(K, 0, model, color="crimson", alpha=0.1)
axL.set_xlabel("blob size K (union = 2K particles)"); axL.set_ylabel("|#A in blob1 change| after regen")
axL.set_title("Species redistribution GROWS with blob size\n(2-particle case was the low-K artifact)")
axL.legend(fontsize=9); axL.grid(alpha=0.3); axL.set_xticks(K)

axR.plot(K, frac_of_ceil, "o-", color="darkorange", lw=2, ms=9, label="redistribution / random ceiling")
axR.plot(K, frac, "^-", color="steelblue", lw=2, ms=8, label="fraction of union slots flipped")
axR.axhline(1.0, ls=":", color="gray"); axR.text(4.2, 1.02, "free redistribution", color="gray", fontsize=8)
axR.set_xlabel("blob size K"); axR.set_ylabel("fraction")
axR.set_ylim(0, 1.15); axR.set_xticks(K)
axR.set_title("At K=16: 81% of the random ceiling, 22% slots flip\n(cold MTM acc=0 => a LOW-lambda tempered move)")
axR.legend(fontsize=9); axR.grid(alpha=0.3)
plt.tight_layout(); OUT = "reports/logs-2026-07-14/species_redist.png"
plt.savefig(OUT, dpi=130); print(f"saved -> {OUT}", flush=True)
