"""CONTROL: my geometric-SMC loop with UniformBase + single-site mutation (both exact, certified
ingredients) from uniform seeds. Isolates MY loop's weight/schedule bookkeeping from the flow lq:
must land on the certified logZ 1198.1 +/- ~1. High => my loop is buggy; on-target => the +23.5
in the flow runs is flow-lq-specific."""
import sys, time, torch
sys.path.insert(0, "/mnt/ssd/GridTransformer")
from liquid_coupling_flow.mw.mw_smc import next_lambda, ess, _resample, mutation_sweeps, LAM_FLOOR
from liquid_coupling_flow.mw.mw_base import UniformBase
from liquid_coupling_flow.mw.mw_energy import mw_energy, T_STAR, RHO_STAR

DEV = "cuda"; N, B = 64, 64
L = (N / RHO_STAR) ** (1 / 3); beta = 1.0 / T_STAR
base = UniformBase(N, L)
for seed in (1, 2, 3):
    gen = torch.Generator(device=DEV).manual_seed(seed)
    x = base.sample(B, gen); U = mw_energy(x, L); lq = base.log_q(x)
    logw = torch.zeros(B, device=DEV); lam, logZ, rung, floor_streak = 0.0, 0.0, 0, 0
    t0 = time.time()
    while lam < 1.0 and rung < 400:
        phi = -beta * U - lq
        lam_new = next_lambda(logw, phi, lam, 0.6, B); dlam = lam_new - lam
        dlw = dlam * phi
        logZ += float(torch.logsumexp(logw + dlw, 0) - torch.logsumexp(logw, 0))
        logw = logw + dlw; lam = lam_new
        floor_streak = floor_streak + 1 if (dlam <= LAM_FLOOR + 1e-9 and lam < 1.0) else 0
        if floor_streak >= 20:
            print(f"seed {seed}: STALLED at lam {lam:.4f}"); break
        if ess(logw) <= 0.6 * B + 1e-6:
            x, U, lq, logw = _resample(x, U, lq, logw, gen)
        x, U, lq, _ = mutation_sweeps(x, base, lam, beta, L, 3, 0.0783, gen)
        rung += 1
    print(f"seed {seed}: rungs {rung:3d}  logZ {logZ:8.2f}  final U/N {float(U.mean())/N:+.4f}  "
          f"(certified 1198.1 +/- 0.4)  [{time.time()-t0:.0f}s]", flush=True)
