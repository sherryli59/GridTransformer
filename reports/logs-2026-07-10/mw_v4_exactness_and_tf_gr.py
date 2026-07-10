"""v4 gate: (1) sample<->log_prob(preordered) exactness, as-trained + perturbed-weights;
(2) TF-vs-FR g(r) discriminator: teacher-forced placement (condition on TRUE prefix, sample x_j,
pool pair distances vs true predecessors) vs free-run g(r) vs data. If TF has the 2nd shell and
FR doesn't -> exposure bias/drift; if TF lacks it too -> conditional under-resolution."""
import math, torch, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from liquid_coupling_flow.mw.mw_generator_v4 import MWGlobalAR
from liquid_coupling_flow.mw.mw_generator import mw_scaffold, canonical_order, wrap_pm
from liquid_coupling_flow.mw.mw_energy import RHO_STAR

ART = "liquid_coupling_flow/mw/artifacts"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
ck = torch.load(f"{ART}/mw_gen_N64_v4.pt", map_location=DEV, weights_only=False)
kw = {k: ck[k] for k in ("d_model", "n_layers", "n_heads", "n_mix", "rail_k")}
N = ck["train_N"]; L = (N / RHO_STAR) ** (1/3)
m = MWGlobalAR(**kw).to(DEV).eval(); m.load_state_dict(ck["state_dict"])

# ---- (1) exactness gate ----
for tag, model in (("as-trained", m),):
    g = torch.Generator(device=DEV).manual_seed(11)
    with torch.no_grad():
        x, logq = model.sample(16, N, L, gen=g)
        lp = model.log_prob(x, L, preordered=True)
    print(f"exactness [{tag}]: max|logq - log_prob| = {float((logq - lp).abs().max()):.2e} "
          f"(per-particle {float((logq - lp).abs().max())/N:.2e})")
mp = MWGlobalAR(**kw).to(DEV).eval(); mp.load_state_dict(ck["state_dict"])
torch.manual_seed(4)
with torch.no_grad():
    for p in mp.parameters(): p.add_(0.05 * torch.randn_like(p))
    g = torch.Generator(device=DEV).manual_seed(11)
    xp, logqp = mp.sample(16, N, L, gen=g)
    lpp = mp.log_prob(xp, L, preordered=True)
print(f"exactness [perturbed] : max|logq - log_prob| = {float((logqp - lpp).abs().max()):.2e} "
      f"(per-particle {float((logqp - lpp).abs().max())/N:.2e})")

# ---- (2) TF placement vs FR vs data ----
ext = torch.load(f"{ART}/mw_ref_N64_ext.pt", map_location="cpu", weights_only=False)
xd = torch.remainder(ext["cfgs"][-512:].to(DEV), L)                    # fresh val-tail configs
t, rank, R = mw_scaffold(N, L, DEV); s = (L / R) / 2.0; bound = float(R)
perm = canonical_order(xd, L, R, rank)
xo = torch.gather(xd, 1, perm[..., None].expand(-1, -1, 3))
u_true = wrap_pm(xo - t[None], L) / s
with torch.no_grad():
    h = m._hidden(u_true, xo, t, L)                                     # [B,N,d] causal TF embeddings
    g2 = torch.Generator(device=DEV).manual_seed(5)
    u_tf, _ = m.head.sample(h.reshape(-1, h.shape[-1]), bound, g2)
x_tf = torch.remainder(t[None] + s * u_tf.reshape(-1, N, 3), L)         # model's TF placements

def pooled_gr(x_place, x_true, L, nbins=120):
    """pair distances between placed particle j and TRUE predecessors <j, pooled; ideal-gas norm."""
    B, N, _ = x_true.shape
    d = wrap_pm(x_place[:, :, None, :] - x_true[:, None, :, :], L).norm(dim=-1)   # [B, j_place, i_true]
    jj = torch.arange(N, device=x_true.device)
    mask = jj[None, :, None] > jj[None, None, :]                        # i < j
    r = d[mask.expand(B, -1, -1)].cpu()
    edges = torch.linspace(0, L/2, nbins+1)
    cnt = torch.histogram(r.float(), bins=nbins, range=(0.0, L/2))[0].double()
    ctr = 0.5*(edges[1:]+edges[:-1]); dr = edges[1]-edges[0]
    rho = N / L**3
    norm = B * (N*(N-1)/2) * (4*math.pi*ctr**2*dr) * rho / N   # = 1/V; same normalizer as g_r
    return ctr, cnt / norm

r1, g_tf = pooled_gr(x_tf, xo, L)
r2, g_data = pooled_gr(xo, xo, L)                                       # true placements, same pooling
diag = torch.load(f"{ART}/mw_v4_gr_diag.pt", map_location="cpu", weights_only=False)
fig, ax = plt.subplots(figsize=(7.2, 4.4))
ax.plot(r2, g_data, "k", lw=2, label="data (same pooling)")
ax.plot(r1, g_tf, "royalblue", lw=1.6, label="TF: model placement | true prefix")
ax.plot(diag["r"], diag["g_v4"], "crimson", lw=1.3, label="FR: free-run samples")
ax.set(xlabel="r", ylabel="g(r)", xlim=(0, L/2), title="v4: TF vs FR vs data — exposure-bias discriminator")
ax.legend(fontsize=8); fig.tight_layout()
out = "reports/logs-2026-07-10/mw_v4_tf_vs_fr_gr.png"
fig.savefig(out, dpi=170); print("PLOT:", out)
torch.save({"r": r1, "g_tf": g_tf, "g_data": g_data}, f"{ART}/mw_v4_tf_gr.pt")
for rr in (1.19, 1.85):
    i = int(rr/(L/2)*120)
    print(f"g({rr}): data {float(g_data[i]):.2f}  TF {float(g_tf[i]):.2f}  FR {float(diag['g_v4'][int(rr/(L/2)*len(diag['r']))]):.2f}")
