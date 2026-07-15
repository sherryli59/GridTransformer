"""D1: plain-MH block acceptance forecast. acc <= E[min(1, e^{-beta dU} * q_rev/q_fwd)]
computed exactly per proposal -- no chains. Distributions saved, not just means."""
import sys, torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_energy import mw_energy, RHO_STAR
from liquid_coupling_flow.mw.mw_block_train import load_model
from liquid_coupling_flow.mw.mw_block_ar import random_block, wrap_pm

BANK = sys.argv[1]            # e.g. liquid_coupling_flow/mw/artifacts/mw_bank_supercooled_N64.pt
TSTAR = float(sys.argv[2])    # temperature to forecast at
DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0); BETA = 1.0 / TSTAR
KS = [1, 2, 3, 4, 6, 8, 12, 16]; NTRIAL = 256

model, _ = load_model("liquid_coupling_flow/mw/artifacts/mw_block_runs/mw_block_N64_mle.pt", DEV)
model.eval()
bank = torch.load(BANK, map_location=DEV, weights_only=False)
cfgs = bank["cfgs"].to(DEV).float()
gen = torch.Generator(device=DEV).manual_seed(0)
cpu_gen = torch.Generator().manual_seed(0)

def knearest_union(x1, c, K):
    return wrap_pm(x1 - c[None], L).norm(dim=-1).topk(K, largest=False).indices

out = {}
with torch.no_grad():
    for K in KS:
        rows = []
        for tr in range(NTRIAL):
            x = cfgs[int(torch.randint(len(cfgs), (1,), generator=cpu_gen))][None]  # [1,N,3]
            for sel in ("random", "blob"):
                if sel == "random":
                    idx = random_block(N, K, DEV, gen=gen)
                else:
                    c = torch.rand(3, device=DEV, generator=gen) * L
                    idx = knearest_union(x[0], c, K)
                lq_rev = model.block_log_prob(x, idx, L)
                xp, lq_fwd = model.sample_block(x, idx, L, gen=gen)
                dU = (mw_energy(xp, L) - mw_energy(x, L))
                la = (-BETA * dU + lq_rev - lq_fwd).clamp(max=0.0)
                rows.append((sel, K, float(dU), float(la.exp())))
        out[K] = rows
        both = [r for r in rows]
        for sel in ("random", "blob"):
            accs = [a for s, _, _, a in both if s == sel]
            dus = [d for s, _, d, _ in both if s == sel]
            print(f"K={K:3d} sel={sel:6s} mean_acc={sum(accs)/len(accs):.3e} "
                  f"median beta*dU={BETA*sorted(dus)[len(dus)//2]:+.1f}")

torch.save({"rows": out, "tstar": TSTAR, "bank": BANK},
           f"reports/logs-2026-07-14/d1_forecast_T{TSTAR:.4f}.pt")
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
for K in KS:
    dus = [BETA * d for s, _, d, _ in out[K] if s == "blob"]
    ax[0].hist(dus, bins=50, histtype="step", label=f"K={K}")
    ax[1].plot([K], [sum(a for s, _, _, a in out[K] if s == "blob") /
                     max(1, sum(1 for s, _, _, _ in out[K] if s == "blob"))], "o")
ax[0].set(xlabel="beta*dU", title=f"blob proposals, T*={TSTAR}"); ax[0].legend()
ax[1].set(xlabel="K", ylabel="mean acceptance bound", yscale="log")
fig.tight_layout(); p = f"reports/logs-2026-07-14/d1_forecast_T{TSTAR:.4f}.png"
fig.savefig(p, dpi=140); print(f"PLOT: /mnt/ssd/GridTransformer/{p}")
