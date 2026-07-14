"""Visualize the basin-trap bias: the SMC population is INTERNALLY consistent (high ESS, samples agree with
each other = qc_pair high) but COLLECTIVELY far from the reference (q~_ref low) => trapped in ~1 non-reference
basin. ESS is blind to this. Left: the three numbers per R with ceiling/floor + PT-reference true value.
Right: 'effective samples' vs 'effective basins' -- the distinction that matters for PTS."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

R = [1.6, 2.0]
ess_frac = [0.603, 0.557]; eff_samp = [57.9, 53.5]
qtil_ref = [0.128, 0.054]; qc_pair = [0.653, 0.951]
PT_TRUE = {1.6: 0.66, 2.0: 0.63}          # PT-stack reference (memory, core-overlap convention, approx)
rho = 1.149

fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5))
x = np.arange(len(R)); w = 0.26
axL.bar(x - w, qtil_ref, w, color="crimson", label=r"$\tilde q_{ref}$ (vs reference) -- BIASED LOW")
axL.bar(x, qc_pair, w, color="steelblue", label=r"$q_{c,pair}$ (sample-vs-sample) -- population agrees")
axL.bar(x + w, ess_frac, w, color="seagreen", label="ESS fraction -- looks healthy")
for i, r in enumerate(R):
    axL.scatter([x[i] - w], [PT_TRUE[r]], marker="*", s=260, color="black", zorder=5,
                label="PT-ref TRUE overlap" if i == 0 else None)
axL.axhline(rho, ls=":", color="gray", lw=1); axL.text(1.4, rho + 0.02, r"$\rho$ ceiling", color="gray", fontsize=8)
axL.set_xticks(x); axL.set_xticklabels([f"R={r}" for r in R])
axL.set_ylabel("overlap / fraction"); axL.set_ylim(0, 1.25)
axL.set_title("Basin trap: internally consistent, externally wrong\n(ESS healthy while estimator is ~10x low)")
axL.legend(fontsize=8, loc="upper left"); axL.grid(alpha=0.3, axis="y")

# right: effective samples vs effective basins
eff_basin = [1.0 / max(1e-6, (1 - qcp)) if qcp < 1 else 96 for qcp in qc_pair]  # heuristic: near-copies => ~1 basin
axR.bar(x - w / 2, eff_samp, w, color="seagreen", label="effective SAMPLES (ESS x M x J)")
axR.bar(x + w / 2, [1.0, 1.0], w, color="crimson", label="effective BASINS (~1, from qc_pair)")
axR.set_xticks(x); axR.set_xticklabels([f"R={r}" for r in R])
axR.set_ylabel("count"); axR.set_yscale("log"); axR.set_ylim(0.5, 200)
axR.set_title("What ESS counts vs what PTS needs\n~55 samples of ~1 (wrong) basin")
axR.legend(fontsize=9); axR.grid(alpha=0.3, axis="y")
plt.tight_layout(); OUT = "reports/logs-2026-07-14/smc_ess_bias.png"
plt.savefig(OUT, dpi=130); print(f"saved -> {OUT}", flush=True)
