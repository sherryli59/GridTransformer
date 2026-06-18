"""Phase 4 (milestone b): size-transfer benchmark, transformer-AR vs spatial coupling.

Both arms trained at N=100; rebuild each at the transfer size N=256 (conditioner params
are N-independent -> trained weights load directly) and compare the RAW PROPOSAL to the
N=256 PT reference. Central hypothesis: the coupling flow (drift-free, parallel) degrades
LESS with N than the AR (exposure-bias drift over a longer generation sequence). We judge
by overlaps / partial g(r) / energy vs the reference -- NOT plain-IS ESS (saturated)."""
from __future__ import annotations
import os, matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, torch
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


def build(name, N, L, cfg_n):
    if name == "AR":
        return KAARFlow(N=N, L=L, num_bins=24, hidden=128, cutoff=2.4)
    return KACouplingFlow(N=N, L=L, n_cycles=4, num_bins=24, hidden=128, cutoff=2.4)


def main(N_train=100, N_eval=256, device="cuda" if torch.cuda.is_available() else "cpu"):
    arms = torch.load(os.path.join(ART, f"ka_arms_N{N_train}.pt"), map_location=device)
    ref = torch.load(os.path.join(ART, f"ka_reference_N{N_eval}.pt"), map_location=device)
    s, L = ref["s"].to(device), ref["L"]
    U_ref = (ka_energy(ref["x"].to(device), s, L) / N_eval).mean().item()
    ov_ref = overlap_frac(ref["x"][:512].to(device), L, N_eval)
    rmax = L / 2
    rc_ref, g_ref = partial_gr(ref["x"][:2000].to(device), s, L, rmax, 90, (0, 0))
    print(f"N_eval={N_eval} reference: <U>/N {U_ref:.3f}, overlaps {ov_ref:.3f}", flush=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    ax[0].plot(rc_ref, g_ref, color="k", lw=2.5, label="reference (PT)")
    rows = []
    for name, color in [("AR", "C3"), ("coupling", "C2")]:
        flow = build(name, N_eval, L, arms["N"]).to(device)
        flow.load_state_dict(arms[name]["state_dict"])      # N=100-trained weights at N=256
        flow.eval()
        with torch.no_grad():
            x, _ = flow.sample(1000, s, device=device)
        ov = overlap_frac(x, L, N_eval)
        U = (ka_energy(x, s, L) / N_eval)
        Um = U[torch.isfinite(U)].median().item()
        rc, g = partial_gr(x, s, L, rmax, 90, (0, 0))
        ax[0].plot(rc, g, color=color, lw=1.6, label=f"{name} (transfer)")
        # in-distribution (train-size) raw overlaps for reference
        rows.append((name, arms[name]["overlaps"], ov, Um))
        print(f"{name}: train-N100 overlaps {arms[name]['overlaps']:.3f} -> "
              f"transfer-N256 overlaps {ov:.3f} (ref {ov_ref:.3f}); <U>/N median {Um:.3f}", flush=True)
    ax[0].axvline(2 ** (1 / 6), color="grey", ls="--", lw=1); ax[0].set_xlim(0, rmax)
    ax[0].set_xlabel("r"); ax[0].set_ylabel("g_AA(r)"); ax[0].legend(fontsize=9)
    ax[0].set_title(f"raw-proposal g(r) transferred N{N_train}->N{N_eval}")
    names = [r[0] for r in rows]
    ax[1].bar([i - 0.15 for i in range(2)], [r[1] for r in rows], 0.3, label=f"train N{N_train}")
    ax[1].bar([i + 0.15 for i in range(2)], [r[2] for r in rows], 0.3, label=f"transfer N{N_eval}")
    ax[1].axhline(ov_ref, color="k", ls=":", label=f"reference {ov_ref:.3f}")
    ax[1].set_xticks(range(2)); ax[1].set_xticklabels(names); ax[1].set_ylabel("raw overlap frac")
    ax[1].set_title("raw-proposal overlaps (lower=better; does it degrade with N?)"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Size-transfer benchmark: AR (drift) vs coupling (drift-free), N{N_train}->N{N_eval}", fontsize=12)
    fig.tight_layout(); out = os.path.join(ART, "ka_benchmark_transfer.png")
    fig.savefig(out, dpi=120); print(f"saved {out}", flush=True)
    print("\nHYPOTHESIS: coupling's transfer overlaps should rise LESS than AR's (drift-free).", flush=True)


if __name__ == "__main__":
    main()
