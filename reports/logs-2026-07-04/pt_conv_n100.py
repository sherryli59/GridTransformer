"""PT-reference convergence diagnostic for N=100 (same as the N=256 Task-1 panel).
Checks whether the TRAINING reference is itself under-converged (block-mean drift in save order)."""
import os, torch, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
from liquid_coupling_flow.ka_energy import ka_energy
dev = "cuda"; ART = "liquid_coupling_flow/artifacts"; LOG = "reports/logs-2026-07-04"

fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
summ = {}
for col, (f, N, refU) in enumerate([("ka_reference_N100.pt", 100, None), ("ka_reference_N256.pt", 256, None)]):
    pt = torch.load(f"{ART}/{f}", map_location="cpu", weights_only=False)
    x, s, L = pt["x"], pt["s"], pt["N"] and pt["L"]
    U = []
    for i in range(0, x.shape[0], 1024):
        U.append((ka_energy(x[i:i+1024].to(dev), s.to(dev), L) / N).cpu())
    U = torch.cat(U).numpy(); nc = len(U); K = 24
    blk = np.array([U[i*nc//K:(i+1)*nc//K].mean() for i in range(K)])
    bx = np.array([(i+0.5)*nc/K for i in range(K)])
    summ[N] = dict(mean=float(U.mean()), std=float(U.std()), skew=float(skew(U)),
                   drift=float(blk[-1]-blk[0]), first=float(blk[0]), last=float(blk[-1]), nc=nc, stored=pt.get("U_per_N"))
    ax[col].plot(np.arange(nc), U, ".", ms=1, alpha=0.12, color="gray")
    ax[col].plot(bx, blk, "-o", color="k", lw=2, ms=5, label="block mean (24)")
    ax[col].axhline(U.mean(), ls=":", color="C2", label=f"grand mean {U.mean():.4f}")
    ax[col].set_xlabel(f"N={N} PT config index (save order)"); ax[col].set_ylabel("U/N")
    ax[col].set_title(f"N={N}: drift {blk[0]:.4f}→{blk[-1]:.4f} ({blk[-1]-blk[0]:+.4f}/particle)")
    ax[col].legend(fontsize=8)
    print(f"[N={N}] mean={U.mean():.4f} std={U.std():.4f} skew={skew(U):.3f} "
          f"blk {blk[0]:.4f}->{blk[-1]:.4f} drift {blk[-1]-blk[0]:+.4f} nconfigs={nc} stored={pt.get('U_per_N')}")
fig.suptitle("PT reference convergence: block-mean drift in save order (both N)", y=1.02)
fig.tight_layout(); p = f"{ART}/ka_pt_convergence_N100_vs_N256.png"; fig.savefig(p, dpi=130, bbox_inches="tight"); plt.close(fig)
torch.save(summ, f"{LOG}/pt_convergence_summary.pt")
print("SAVED", os.path.abspath(p)); print("SAVED", os.path.abspath(f"{LOG}/pt_convergence_summary.pt"))
