"""D2: how shift-invariant is v10's canonical-lift log q0? Spread of log_prob under random
torus translations per config; the re-rooted-suffix MH ratio picks up exactly this spread.

Fallback note: the supercooled bank (D0) was not ready when this ran, so this measures the
AMBIENT bank (mw_bank_ambient_N64.pt, T*~0.096, L already matches N=64/RHO_STAR). Re-run
against mw_bank_supercooled_N64.pt once D0 lands and compare.

Also measures, "for free", the spread under 8 sampled O_h group elements (signed coordinate
permutations about the box center), applied AFTER a random torus shift, using the exact
convention in mw_generator._augment_batch.
"""
import torch
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_energy import RHO_STAR
from liquid_coupling_flow.mw.mw_generator_v10 import load_generator_v10

DEV = "cuda"; N = 64; L = (N / RHO_STAR) ** (1.0 / 3.0); M = 32; NCFG = 128
model = load_generator_v10("liquid_coupling_flow/mw/artifacts/mw_gen_N64_v10_best_nll.pt", DEV)
model.eval()

bank = torch.load("liquid_coupling_flow/mw/artifacts/mw_bank_ambient_N64.pt",
                  map_location=DEV, weights_only=False)
assert abs(bank["L"] - L) < 1e-6, f"bank L {bank['L']} != formula L {L}"
x = bank["cfgs"][:NCFG].to(DEV).float()

# --- torus-shift spread ---
gen = torch.Generator(device=DEV).manual_seed(0)
lps = []
with torch.no_grad():
    for m in range(M):
        v = torch.rand(3, device=DEV, generator=gen) * L if m else torch.zeros(3, device=DEV)
        lps.append(model.log_prob(torch.remainder(x + v, L), L).cpu())
lp = torch.stack(lps)                       # [M, NCFG]
spread_std = lp.std(0); spread_rng = lp.max(0).values - lp.min(0).values
print(f"[torus] per-config log q0 spread: std median={spread_std.median():.3f} p90={spread_std.quantile(0.9):.3f}"
      f"  range median={spread_rng.median():.3f} p90={spread_rng.quantile(0.9):.3f} (nats)")

# --- O_h spread (8 sampled group elements: random signed coordinate permutation about box
# center, applied after an independent random torus shift each draw -- same convention as
# mw_generator._augment_batch's rotation half) ---
M_OH = 8
c = L / 2.0
gen_oh = torch.Generator(device=DEV).manual_seed(1)
lps_oh = []
with torch.no_grad():
    for m in range(M_OH):
        v = torch.rand(3, device=DEV, generator=gen_oh) * L
        xs = torch.remainder(x + v, L)
        perm = torch.rand(3, device=DEV, generator=gen_oh).argsort()          # ~U(S_3)
        signs = (torch.randint(0, 2, (3,), device=DEV, generator=gen_oh) * 2 - 1).to(xs.dtype)
        xc = xs - c
        xp = xc[..., perm]
        xoh = torch.remainder(c + signs * xp, L)
        lps_oh.append(model.log_prob(xoh, L).cpu())
lp_oh = torch.stack(lps_oh)                 # [M_OH, NCFG]
spread_std_oh = lp_oh.std(0); spread_rng_oh = lp_oh.max(0).values - lp_oh.min(0).values
print(f"[O_h]   per-config log q0 spread: std median={spread_std_oh.median():.3f} p90={spread_std_oh.quantile(0.9):.3f}"
      f"  range median={spread_rng_oh.median():.3f} p90={spread_rng_oh.quantile(0.9):.3f} (nats)")

torch.save({"lp": lp, "spread_std": spread_std, "spread_rng": spread_rng,
            "lp_oh": lp_oh, "spread_std_oh": spread_std_oh, "spread_rng_oh": spread_rng_oh,
            "bank": "mw_bank_ambient_N64.pt", "tstar": bank["tstar"], "L": L, "M": M, "M_OH": M_OH,
            "NCFG": NCFG},
           "reports/logs-2026-07-14/d2_shift_spread.pt")

fig, ax = plt.subplots(2, 2, figsize=(11, 8))
ax[0, 0].hist(spread_std.numpy(), bins=40); ax[0, 0].set(xlabel="std over shifts (nats)", title="torus-shift: log q0 std")
ax[0, 1].hist(spread_rng.numpy(), bins=40); ax[0, 1].set(xlabel="range over shifts (nats)", title="torus-shift: log q0 range")
ax[1, 0].hist(spread_std_oh.numpy(), bins=40); ax[1, 0].set(xlabel="std over O_h draws (nats)", title="O_h: log q0 std")
ax[1, 1].hist(spread_rng_oh.numpy(), bins=40); ax[1, 1].set(xlabel="range over O_h draws (nats)", title="O_h: log q0 range")
fig.tight_layout(); p = "reports/logs-2026-07-14/d2_shift_spread.png"
fig.savefig(p, dpi=140); print(f"PLOT: /mnt/ssd/GridTransformer/{p}")
