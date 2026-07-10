"""G1 gate driver: bedrock quadrature vs independent displacement-MC at N=2,3 (L=4.0, beta=2.0).
PASS: |<U>_quad - <U>_MC| < halving_err + 3*SE_MC at both N. SE via independent-chain means
(chains never interact -> B chain-means are iid; SE = std(chain_means)/sqrt(B))."""
import torch, numpy as np
from liquid_coupling_flow.mw.mw_bedrock import quad_N2, quad_N3
from liquid_coupling_flow.mw.mw_reference import mc_run

L, BETA, B = 4.0, 2.0, 16
for N, quad in ((2, lambda: quad_N2(L, BETA, 96)), (3, lambda: quad_N3(L, BETA, 20))):
    Uq, err = quad()
    out = mc_run(N=N, L=L, beta=BETA, n_equil=20000, n_collect=20000, every=4, seed=0, B=B)
    U = out["U"].reshape(-1, B)                        # [n_events, B] chains appended per event
    chain_means = U.mean(0)
    Um, se = float(chain_means.mean()), float(chain_means.std() / np.sqrt(B))
    tol = err + 3 * se
    verdict = "PASS" if abs(Uq - Um) < tol else "FAIL"
    print(f"G1 N={N}: quad {Uq:.6f} (halv {err:.2e}) | MC {Um:.6f} +/- {se:.6f} "
          f"(acc {out['acc']:.2f}, step {out['step']:.3f}, flat {out['flat_budget']:.4f}) | "
          f"|diff| {abs(Uq-Um):.6f} < tol {tol:.6f} -> {verdict}", flush=True)
