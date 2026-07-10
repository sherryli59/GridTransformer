"""One-off G2 ladder-protocol experiments (2026-07-09): after G2 N=8 FAIL v2, discriminate
slow-ladder (ess_target) vs finisher as the mutation-budget lever. Results in the .out beside this.
VERDICT: slow ladder ess_target=0.9 reaches ref exactness (gap 3e-3) with NO finisher at 2.8M evals;
the et0.6 fast ladder needs a 10k-sweep lam=1 finisher (42M evals) to converge = 15x more expensive.
Frozen gate protocol <- ess_target 0.9, n_sweeps 10, step=ref frozen, final_sweeps 300 (margin)."""
import torch
from liquid_coupling_flow.mw.mw_smc import smc_run
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import T_STAR, RHO_STAR

N = 8; L = (N/RHO_STAR)**(1/3); beta = 1/T_STAR
base = UniformBase(N, L)
REF = -1.8631
for tag, et, nsw, fin in (("et0.9_ns10_f0", 0.9, 10, 0), ("et0.95_ns10_f0", 0.95, 10, 0),
                          ("et0.9_ns10_f2k", 0.9, 10, 2000), ("et0.6_ns10_f10k", 0.6, 10, 10000)):
    out = smc_run(base, N, L, beta, B=512, ess_target=et, n_sweeps=nsw, step=0.0666,
                  seed=0, save_tag=f"_exp_{tag}", final_sweeps=fin)
    w = torch.softmax(out["logw"], 0)
    um = float((w * out["U"]).sum()) / N
    print(f"{tag:18s} rungs {len(out['history']):4d}  evals {out['evals']:>9d}  wall {out['wall']:6.1f}s  "
          f"U/N {um:+.4f}  (ref {REF})  gap {um-REF:+.4f}", flush=True)
