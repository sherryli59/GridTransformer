"""Mode-sharpened MDN sampling scan: temper mixture logits (gamma_w) and optionally component
covariances (gamma_c: L -> L/sqrt(gamma_c)) via a _params patch — sample() and log_prob() share
_params, so the tempered model's density stays exact-by-construction (verified per gamma).
Metrics per gamma: free-run g(r) shells, U/N, one-shot logw spread/ESS under the TEMPERED density."""
import math, types, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_reference import g_r
from liquid_coupling_flow.mw.mw_energy import mw_energy_chunked, RHO_STAR, T_STAR

ART = "liquid_coupling_flow/mw/artifacts"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{ART}/mw_gen_N64_v4.pt", map_location=DEV, weights_only=False)
kw = {k: ck[k] for k in ("d_model", "n_layers", "n_heads", "n_mix", "rail_k")}
N = ck["train_N"]; L = (N / RHO_STAR) ** (1/3); beta = 1.0 / T_STAR
ext = torch.load(f"{ART}/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
r_ref, gr_ref = g_r(ext["cfgs"][-6400:], L)

def make(gw, gc):
    m = MWGlobalAR(**kw).to(DEV).eval(); m.load_state_dict(ck["state_dict"])
    orig = m.head._params.__func__
    def tempered(self, h):
        logits, mean, Lm = orig(self, h)
        return gw * logits, mean, Lm / math.sqrt(gc)
    m.head._params = types.MethodType(tempered, m.head)
    return m

rows, curves = [], {}
B = 512
for gw, gc in ((1.0,1.0), (2.0,1.0), (4.0,1.0), (8.0,1.0), (2.0,2.0), (4.0,2.0)):
    m = make(gw, gc)
    g = torch.Generator(device=DEV).manual_seed(123)
    with torch.no_grad():
        x, logq = m.sample(B, N, L, gen=g)
        lp = m.log_prob(x, L, preordered=True)
    cons = float((logq - lp).abs().max())
    U = mw_energy_chunked(x, L).double()
    _, gr = g_r(x.cpu(), L)
    logw = (-beta * U.to(DEV) - logq.double()).cpu()
    ess = float(1.0 / (torch.softmax(logw - logw.max(), 0) ** 2).sum())
    i1, i2 = int(1.19/(L/2)*len(r_ref)), int(1.85/(L/2)*len(r_ref))
    rows.append((gw, gc, float(gr[i1]), float(gr[i2]), float((U/N).median()), float((U/N).mean()), ess, cons))
    curves[(gw, gc)] = gr
    print(f"gw={gw:.0f} gc={gc:.0f}: shell1 {gr[i1]:.2f} shell2 {gr[i2]:.2f} "
          f"U/N med {float((U/N).median()):+.3f} mean {float((U/N).mean()):+.2f} "
          f"ESS {ess:.1f}/{B} consistency {cons:.1e}", flush=True)
print(f"reference: shell1 {float(gr_ref[i1]):.2f} shell2 {float(gr_ref[i2]):.2f} U/N -1.627")
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.plot(r_ref, gr_ref, "k", lw=2.2, label="data")
for (gwc, gr) in curves.items():
    ax.plot(r_ref, gr, lw=1.2, label=f"gw={gwc[0]:.0f},gc={gwc[1]:.0f}")
ax.set(xlabel="r", ylabel="g(r)", xlim=(0, L/2), title="v4 free-run g(r) vs mixture temperature")
ax.legend(fontsize=7); fig.tight_layout()
out = "reports/logs-2026-07-10/mw_v4_gamma_scan.png"
fig.savefig(out, dpi=170); print("PLOT:", out)
torch.save({"rows": rows, "curves": curves, "r": r_ref, "g_ref": gr_ref}, f"{ART}/mw_v4_gamma_scan.pt")
