"""RIGOROUS effective-basin count (replacing the qc_pair hand-wave). Pool all J*M SMC samples per cavity,
build the pairwise sample-vs-sample overlap SIMILARITY matrix, and count basins TWO ways:
  (1) CONNECTED COMPONENTS at an overlap threshold tau: two samples share a basin if their normalized overlap
      sim = (q_box - bulk)/(rho - bulk) > tau. Count components (union-find). Interpretable; sweep tau.
  (2) PARTICIPATION RATIO of the similarity (Gram) matrix eigenspectrum: PR = (Σλ)^2 / Σλ^2 ∈ [1, N].
      All-identical samples => rank-1 => PR=1; all-distinct => PR=N. An 'effective number of distinct modes'.
Also report the resampling-collapse alternative: # of DISTINCT samples (unique up to tiny RMSD) in the pooled
set -- if << N, the population collapsed to few ancestors (a resampling artifact, not necessarily basins).
Uses the SAVED populations from smc_pts_hocky_lam05.pt (l=0.368, lam05 Rext model). R in {1.6, 2.0, 2.4}."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-14")
from smc_pts_sweep_hocky import box_overlap

PT = "reports/logs-2026-07-14/smc_pts_hocky_lam05.pt"
res = torch.load(PT, map_location="cpu", weights_only=False)
RHO = 1.149; L_BOX = 0.368; BULK = RHO * L_BOX ** 3


def components(sim, tau):
    """connected components of the graph {edge k-k' iff sim>tau}. union-find. returns #components."""
    n = sim.shape[0]; par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] > tau:
                par[find(i)] = find(j)
    return len({find(i) for i in range(n)})


def part_ratio(sim):
    """participation ratio of the similarity matrix eigenspectrum (symmetrized, clamped PSD)."""
    A = 0.5 * (sim + sim.T)
    ev = torch.linalg.eigvalsh(A).clamp_min(0)
    s1 = ev.sum(); s2 = (ev ** 2).sum()
    return float(s1 ** 2 / s2) if s2 > 0 else 1.0


print(f"=== effective-basin count from pooled J*M SMC samples (lam05, l={L_BOX}) ===", flush=True)
print(f"sim = (q_box - bulk)/(rho - bulk); ceiling rho={RHO}, bulk={BULK:.3f}", flush=True)
print(f"{'R':>4} {'N_pool':>6} | {'CC@0.4':>7} {'CC@0.6':>7} {'CC@0.8':>7} | {'PR':>5} | {'n_distinct(RMSD)':>16} | qc_pair", flush=True)
for R in sorted(res.keys()):
    ncc4, ncc6, ncc8, prs, ndist, qcp = [], [], [], [], [], []
    for row in res[R]:
        # pool all islands' M samples
        allX = torch.cat(row["popX"], 0)                       # [J*M, n, 3]
        N = allX.shape[0]
        # pairwise sample-vs-sample box overlap -> similarity
        sim = torch.zeros(N, N)
        for i in range(N):
            for j in range(i + 1, N):
                q = box_overlap(allX[i], allX[j], R, L_BOX)
                s = (q - BULK) / (RHO - BULK)
                sim[i, j] = sim[j, i] = s
        sim.fill_diagonal_(1.0)
        ncc4.append(components(sim, 0.4)); ncc6.append(components(sim, 0.6)); ncc8.append(components(sim, 0.8))
        prs.append(part_ratio(sim))
        # distinct up to tiny RMSD (resampling-collapse check): cluster by RMSD<0.05
        D = torch.cdist(allX.reshape(N, -1), allX.reshape(N, -1)) / (allX.shape[1] ** 0.5)
        ndist.append(components((D < 0.05).float(), 0.5))
        iu = torch.triu_indices(N, N, 1)
        qcp.append(float(sim[iu[0], iu[1]].mean()))
    print(f"{R:>4} {allX.shape[0]:>6} | {st.mean(ncc4):>7.1f} {st.mean(ncc6):>7.1f} {st.mean(ncc8):>7.1f} | "
          f"{st.mean(prs):>5.1f} | {st.mean(ndist):>16.1f} | {st.mean(qcp):.3f}", flush=True)
print(f"\nCC@tau = # connected components (basins) at overlap threshold tau; PR = eigen participation ratio;", flush=True)
print(f"n_distinct(RMSD) = # samples distinct up to RMSD 0.05 (low => resampling collapse to few ancestors).", flush=True)
