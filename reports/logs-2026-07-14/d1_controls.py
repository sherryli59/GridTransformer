"""D1 controls: validate the forecast harness before believing acc~0 at all K.

1. IDENTITY: proposal == current block positions -> dU must be exactly 0 (fp noise).
   Nonzero => harness/bank bug (wrapping, indexing, dtype); STOP, do not write verdict.
2. JITTER: proposal = current block + 0.05-sigma wrapped Gaussian jitter ->
   expect median beta*dU ~ +K*O(1) (small positive). +100s => upstream breakage.

Same bank / code path as d1_block_acc_forecast.py (dU via mw_energy on a cloned
config with the block rows overwritten, wrapped into [0,L)).
"""
import sys, torch
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR
from liquid_coupling_flow.mw.mw_block_ar import random_block, wrap_pm

BANK = sys.argv[1] if len(sys.argv) > 1 else "liquid_coupling_flow/mw/artifacts/mw_bank_ambient_N64.pt"
TSTAR = float(sys.argv[2]) if len(sys.argv) > 2 else 0.09632
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0); BETA = 1.0 / TSTAR
KS = [1, 4, 8]; NTRIAL = 64; JITTER = 0.05

bank = torch.load(BANK, map_location=DEV, weights_only=False)
cfgs = bank["cfgs"].to(DEV).float()
gen = torch.Generator(device=DEV).manual_seed(0)
cpu_gen = torch.Generator().manual_seed(0)

def knearest_union(x1, c, K):
    return wrap_pm(x1 - c[None], L).norm(dim=-1).topk(K, largest=False).indices

with torch.no_grad():
    for name in ("identity", "jitter"):
        worst = 0.0
        for K in KS:
            dus = []
            for tr in range(NTRIAL):
                x = cfgs[int(torch.randint(len(cfgs), (1,), generator=cpu_gen))][None]
                c = torch.rand(3, device=DEV, generator=gen) * L
                idx = knearest_union(x[0], c, K)
                # same proposal-injection path as the forecast: clone, overwrite block rows, wrap
                new_block = x[0, idx].clone()
                if name == "jitter":
                    new_block = new_block + JITTER * torch.randn(
                        K, 3, device=DEV, generator=gen)
                xp = x.clone()
                xp[0, idx] = torch.remainder(new_block, L)
                dU = float(mw_energy(xp, L) - mw_energy(x, L))
                dus.append(dU)
            dus_s = sorted(dus)
            med = dus_s[len(dus_s) // 2]
            mx = max(abs(d) for d in dus)
            worst = max(worst, mx)
            print(f"{name:8s} K={K:2d} median beta*dU={BETA*med:+.3f} "
                  f"median dU={med:+.4e} max|dU|={mx:.4e}", flush=True)
        if name == "identity":
            verdict = "PASS (fp noise)" if worst < 1e-3 else "FAIL -- harness/bank bug, STOP"
            print(f"IDENTITY control: max|dU| over all K/trials = {worst:.3e} -> {verdict}", flush=True)
