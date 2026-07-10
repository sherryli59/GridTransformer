"""Raw-sample structural diagnostic for the completed global-prefix mW v4 model."""
import os
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_reference import g_r
from liquid_coupling_flow.mw.mw_energy import RHO_STAR, mw_energy_chunked

ART = "liquid_coupling_flow/mw/artifacts"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CKPT = os.path.join(ART, "mw_gen_N64_v4.pt")
EXT = os.path.join(ART, "mw_ref_N64_ext.pt")
OUT_PNG = "reports/logs-2026-07-10/mw_v4_gr.png"
OUT_PT = os.path.join(ART, "mw_v4_gr_diag.pt")
B = 512

ck = torch.load(CKPT, map_location=DEV, weights_only=False)
kw = {k: ck[k] for k in ("d_model", "n_layers", "n_heads", "n_mix", "rail_k")}
m = MWGlobalAR(**kw).to(DEV).eval(); m.load_state_dict(ck["state_dict"])
N = ck["train_N"]; L = (N / RHO_STAR) ** (1 / 3)
ext = torch.load(EXT, map_location="cpu", weights_only=False)
ref = ext["cfgs"][-6400:]  # exactly the fresh held-out v3/v4 validation tail
g = torch.Generator(device=DEV).manual_seed(123)
with torch.no_grad():
    x, logq = m.sample(B, N, L, gen=g)
U = mw_energy_chunked(x, L).cpu()
r, gr_ref = g_r(ref, L)
_, gr_gen = g_r(x.cpu(), L)
dr = float((gr_gen - gr_ref).abs().mean())
dmax = float((gr_gen - gr_ref).abs().max())
print(f"v4 raw B={B}: gr_L1={dr:.5f} gr_max={dmax:.5f} U/N={float(U.mean()/N):.5f}", flush=True)
fig, ax = plt.subplots(figsize=(6.4, 4.2))
ax.plot(r, gr_ref, color="black", lw=2, label="fresh extension validation (n=6400)")
ax.plot(r, gr_gen, color="crimson", lw=1.5, label=f"v4 raw samples (n={B})")
ax.set(xlabel="r", ylabel="g(r)", xlim=(0, L / 2), title=f"mW N=64 v4 raw g(r): L1={dr:.3f}, max={dmax:.3f}")
ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(OUT_PNG, dpi=170)
torch.save({"r":r, "g_ref":gr_ref, "g_v4":gr_gen, "x":x.cpu(), "logq":logq.cpu(), "U":U,
            "gr_L1":dr, "gr_max":dmax, "checkpoint":CKPT, "n_ref":len(ref), "n_sample":B}, OUT_PT)
print(f"saved {OUT_PNG} and {OUT_PT}", flush=True)
