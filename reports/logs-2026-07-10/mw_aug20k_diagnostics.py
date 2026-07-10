"""Early diagnostics on the augmented 20k checkpoint (mw_gen_N64_aug_20k.bak.pt).
(1) sampler counters: n_wrapped, noncanonical fraction; (2) one-shot IS probe: logw = -beta*U - logq0
on B=512 samples -> spread/particle, one-shot ESS, c_hat proxy; (3) g(r) of raw samples vs reference;
(4) sample-energy distribution vs reference. All arrays saved (full-data rule)."""
import torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_generator import load_generator, mw_scaffold, canonical_order
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, T_STAR, RHO_STAR
from liquid_coupling_flow.mw.mw_reference import g_r

DEV = "cuda" if torch.cuda.is_available() else "cpu"
N, B = 64, 512
L = (N / RHO_STAR) ** (1/3); beta = 1.0 / T_STAR
m = load_generator("liquid_coupling_flow/mw/artifacts/mw_gen_N64_aug_20k.bak.pt", DEV)
g = torch.Generator(device=DEV).manual_seed(0)
with torch.no_grad():
    x, logq, nwrap = m.sample(B, N, L, gen=g)
U = mw_energy_chunked(x, L)
t, rank, R = mw_scaffold(N, L, DEV)
perm = canonical_order(x, L, R, rank)
noncanon = float((~(perm == torch.arange(N, device=DEV)[None]).all(-1)).float().mean())
print(f"counters: n_wrapped={nwrap} ({nwrap/(B*N):.2e} per (cfg,step))  noncanonical_frac={noncanon:.3f}")

logw = (-beta * U - logq)                                # unnormalized IS log-weights
lw = logw - logw.max()
ess = float(1.0 / (torch.softmax(lw, 0) ** 2).sum())
print(f"one-shot IS: std(logw)={float(logw.std()):.1f} total ({float(logw.std())/N:.3f}/particle)  "
      f"ESS={ess:.1f}/{B}")
ref = torch.load("liquid_coupling_flow/mw/artifacts/mw_g2_N64.pt", map_location="cpu", weights_only=False)["ref"]
uref = ref["U"].double() / N
print(f"sample U/N: mean {float(U.mean())/N:.4f} std {float((U/N).std()):.4f}  "
      f"| ref: mean {float(uref.mean()):.4f} std {float(uref.std()):.4f}")
# c proxy on data side: E_ref[-logq0]/N vs uniform
logq_ref = []
xr = ref["cfgs"][:512].to(DEV)
with torch.no_grad():
    lp = m.log_prob(torch.remainder(xr, L), L)           # canonical mode on reference configs
print(f"data-side: E_ref[-logq0]/N = {float(-lp.mean())/N:.4f}  (uniform 3*log L = {3*torch.log(torch.tensor(L)):.4f})")

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
r_ref, gr_ref = g_r(ref["cfgs"], L); r_s, gr_s = g_r(x.cpu(), L)
ax[0].plot(r_ref, gr_ref, "k", lw=2, label="reference"); ax[0].plot(r_s, gr_s, "crimson", lw=1.2, label="raw samples (one-shot)")
ax[0].set_xlabel("r"); ax[0].set_ylabel("g(r)"); ax[0].legend(); ax[0].set_title("g(r): raw generator samples vs ref")
ax[1].hist(uref.numpy(), bins=50, density=True, color="k", alpha=0.6, label="ref")
ax[1].hist((U.cpu()/N).numpy(), bins=50, density=True, color="crimson", alpha=0.6, label="samples")
ax[1].set_xlabel("U/N"); ax[1].legend(); ax[1].set_title("P(U/N)")
plt.tight_layout()
out = "reports/logs-2026-07-10/mw_aug20k_diagnostics.png"
plt.savefig(out, dpi=140); print("PLOT:", out)
torch.save({"x": x.cpu(), "logq": logq.cpu(), "U": U.cpu(), "nwrap": nwrap, "noncanon": noncanon,
            "logw": logw.cpu(), "ess": ess, "logq_ref": lp.cpu()},
           "liquid_coupling_flow/mw/artifacts/mw_aug20k_diag.pt")
