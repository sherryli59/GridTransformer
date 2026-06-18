"""P3 gate: can each trained arm REPRODUCE THE TRAINING SET (in-distribution, N=100)?
Sample from each N=100-trained arm and compare partial g_AA/g_AB/g_BB(r), the energy
distribution, and overlaps to the N=100 PT reference. This must pass before any transfer
test -- an arm that can't learn the glass at its own size can't be meaningfully transferred."""
from __future__ import annotations
import os, math, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, torch
from liquid_coupling_flow.ka_flow_ar import KAARFlow
from liquid_coupling_flow.ka_flow_coupling import KACouplingFlow
from liquid_coupling_flow.ka_energy import ka_energy
from liquid_coupling_flow.ka_observables import partial_gr
from liquid_coupling_flow.particles import min_image

ART = os.path.join(os.path.dirname(__file__), "artifacts")


def overlap_frac(x, L, N, thresh=0.7):
    _, r = min_image(x, L)
    r = r + torch.eye(N, device=x.device)[None] * 1e3
    return float((r.min(dim=2).values.reshape(-1) < thresh).float().mean())


def build(name, N, L):
    if name == "AR":
        return KAARFlow(N=N, L=L, num_bins=24, hidden=128, cutoff=2.4)
    return KACouplingFlow(N=N, L=L, n_cycles=4, num_bins=24, hidden=128, cutoff=2.4)


def main(N=100, device="cuda" if torch.cuda.is_available() else "cpu"):
    arms = torch.load(os.path.join(ART, f"ka_arms_N{N}.pt"), map_location=device)
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N}.pt"), map_location=device)
    s, L = ref["s"].to(device), ref["L"]
    xref = ref["x"].to(device)
    U_ref = (ka_energy(xref, s, L) / N)
    rmax = L / 2
    pairs = [(0, 0), (0, 1), (1, 1)]
    gref = {p: partial_gr(xref[:2000], s, L, rmax, 80, p) for p in pairs}
    print(f"reference N={N}: <U>/N {U_ref.mean():.3f}, overlaps {overlap_frac(xref[:512], L, N):.3f}", flush=True)

    samp = {}
    for name in ("AR", "coupling"):
        f = build(name, N, L).to(device); f.load_state_dict(arms[name]["state_dict"]); f.eval()
        with torch.no_grad():
            x, _ = f.sample(1500, s, device=device)
        samp[name] = x
        U = (ka_energy(x, s, L) / N)
        print(f"{name}: overlaps {overlap_frac(x, L, N):.3f} (ref {overlap_frac(xref[:512],L,N):.3f}), "
              f"<U>/N median {U[torch.isfinite(U)].median():.3f} (ref {U_ref.mean():.3f})", flush=True)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    for k, p in enumerate(pairs):
        a = ax[k // 2][k % 2]
        rc, g = gref[p]; a.plot(rc, g, "k", lw=2.5, label="reference")
        for name, col in [("AR", "C3"), ("coupling", "C2")]:
            rc2, g2 = partial_gr(samp[name], s, L, rmax, 80, p)
            a.plot(rc2, g2, col, lw=1.6, label=name)
        a.axvline(2 ** (1 / 6), color="grey", ls="--", lw=1)
        a.set_title(f"g_{['A','B'][p[0]]}{['A','B'][p[1]]}(r)"); a.set_xlabel("r"); a.legend(fontsize=8)
    # energy distribution
    ae = ax[1][1]
    Um = U_ref.cpu().numpy(); lo, hi = math.floor(Um.min() - 0.3), math.ceil(Um.max() + 1.0)
    ae.hist(Um, bins=50, range=(lo, hi), density=True, alpha=0.4, color="k", label="reference")
    for name, col in [("AR", "C3"), ("coupling", "C2")]:
        Us = (ka_energy(samp[name], s, L) / N).cpu().numpy()
        ae.hist(Us[(Us > lo) & (Us < hi)], bins=50, range=(lo, hi), density=True,
                histtype="step", color=col, lw=1.6, label=name)
    ae.set_xlabel("U/N"); ae.set_title("energy distribution"); ae.legend(fontsize=8)
    fig.suptitle(f"In-distribution (N={N}): do the trained arms reproduce the training set?", fontsize=13)
    fig.tight_layout(); out = os.path.join(ART, "ka_indist_check.png")
    fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()
