"""Final-state g(r) of all three amortization arms {uniform, exclvol, flow} at N=512, overlaid on the
fully-converged real equilibrium. Question: do the arms reach the SAME distribution (amortization = pure
head-start, identical endpoint) or different endpoints? All three ran the identical final ladder rungs;
they differ only in where they ENTERED (uniform rung 0, exclvol rung 2, flow rung 7) and thus total evals."""
import sys, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.mw.mw_reference import g_r
from liquid_coupling_flow.mw.mw_energy import RHO_STAR

d = torch.load("liquid_coupling_flow/mw/artifacts/mw_ersi_transfer_N512.pt", map_location="cpu", weights_only=False)
N, L = d["N"], d["L"]


def shells(r, gr):
    i1, i2 = int(1.19 / (L / 2) * len(r)), int(1.85 / (L / 2) * len(r))
    return float(gr[i1]), float(gr[i2])


arms = {
    "uniform": ("tab:red", d["arms"]["uniform"]),
    "exclvol": ("tab:green", d["arms"]["exclvol"]),
    "flow": ("royalblue", d["arms"]["flow"]),
}
r_eq, g_eq = d["eq"]["r"], d["eq"]["gr"]
e1, e2 = shells(r_eq, g_eq)

fig, ax = plt.subplots(figsize=(8.8, 5.2))
ax.plot(r_eq, g_eq, "k", lw=2.6, label=f"REAL N=512 equilibrium (converged, U/N {d['eq']['U_per_N']:.3f})", zorder=5)
print(f"{'arm':9s} {'evals':>12s} {'final U/N':>10s} {'shell1':>7s} {'shell2':>7s}  amort")
eu = d["arms"]["uniform"]["evals"]
for name, (col, a) in arms.items():
    r, gr = g_r(a["X_final"], L)
    s1, s2 = shells(r, gr)
    lbl = f"{name} arm (U/N {a['hist'][-1][1]:.3f}, {a['evals']/1e6:.2f}M evals, {eu/a['evals']:.2f}x)"
    ax.plot(r, gr, col, lw=1.7, alpha=0.9, label=lbl)
    print(f"{name:9s} {a['evals']:>12,} {a['hist'][-1][1]:>+10.4f} {s1:>7.2f} {s2:>7.2f}  {eu/a['evals']:.2f}x")
print(f"{'real-eq':9s} {'':>12s} {d['eq']['U_per_N']:>+10.4f} {e1:>7.2f} {e2:>7.2f}")

ax.set(xlabel="r (sigma)", ylabel="g(r)", xlim=(0, L / 2), ylim=(0, 2.3),
       title="N=512: final-state g(r) of all three SMC arms vs converged equilibrium")
ax.legend(fontsize=8.5, loc="upper right")
fig.tight_layout()
out = "reports/logs-2026-07-11/mw_ersi_512_arms_gr.png"
fig.savefig(out, dpi=150)
print("PLOT:", out)
