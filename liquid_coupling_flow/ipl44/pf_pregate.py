"""G0-PF pre-gate: TF-vs-FR per-index clash decomposition for the IPL44 curve-flow transformer.
Professor forcing can only attack the DRIFT component (FR clash growing above TF along the AR rollout).
If TF ~= FR (residual-dominated, as on KA), PF cannot help -> kill the arm.
TF: for each AR index j, condition on the TRUE curve-ordered prefix 0..j-1 and sample particle j; clash vs the
true prefix. FR: the model's own rollout; clash vs its own prefix. clash = min-image NN dist < 0.9 (=0.9*sigma_AA).
Usage: python -m liquid_coupling_flow.ipl44.pf_pregate [B]"""
import os, sys, torch, torch.nn.functional as F, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from liquid_coupling_flow.ipl44.ipl_model import make_ipl_model
from liquid_coupling_flow.ipl44.ipl_energy import ipl_box

dev = "cuda" if torch.cuda.is_available() else "cpu"
N, L = ipl_box(); Lf = float(L)
B = int(sys.argv[1]) if len(sys.argv) > 1 else 512
ART = os.path.join(os.path.dirname(__file__), "data")
THR = 0.9  # 0.9 * sigma_AA


def clash_by_index(cand, prefix, L, thr=THR):
    """cand [B,N,2] (the sampled particle j lives at cand[:, j]), prefix [B,N,2]. Returns clash[N] (j>=1)."""
    n = cand.shape[1]; out = np.zeros(n)
    for j in range(1, n):
        df = cand[:, j:j + 1] - prefix[:, :j]
        df = df - L * torch.round(df / L)
        out[j] = float(((df ** 2).sum(-1).min(1).values.sqrt() < thr).float().mean())
    return out


ck = torch.load(f"{ART}/ipl44_curveflow.pt", map_location=dev, weights_only=False)
m = make_ipl_model(num_bins=ck["num_bins"], tail_bound=ck["tail_bound"], knn=ck["knn"],
                   arc_range=ck["arc_range"], device=dev)
m.load_state_dict(ck["state_dict"]); m.eval()
nB = ck["n_B"]
print(f"loaded ipl44_curveflow ({ck['n_params']/1e6:.2f}M params)", flush=True)

# reference configs, curve-ordered exactly as the model orders its own samples
D = "/mnt/ssd/GridTransformer/datasets"
xr = torch.remainder(torch.load(f"{D}/ipl44_T0.1_positions.pt", weights_only=False).float(), Lf)[:B].to(dev)
sr = torch.load(f"{D}/ipl44_T0.1_species.pt", weights_only=False).long()[:B].to(dev)
perm = m.geo._curve_order(xr, N)
xr = torch.gather(xr, 1, perm[..., None].expand(-1, -1, 2)); sr = torch.gather(sr, 1, perm)

with torch.no_grad():
    # ---- FR: model's own rollout ----
    x_fr, s_fr = m.sample(B, N, n_B=nB, device=dev)
    clash_fr = clash_by_index(x_fr, x_fr, Lf)

    # ---- TF: place each j from the TRUE prefix (causal _step reads only 0..j-1) ----
    sc = m.geo._scaffold(N, dev); arc = m._arc_scale(N); cf = m._curve_feat(N, dev)
    cand = torch.zeros(B, N, 2, device=dev)
    rem = torch.zeros(B, m.n_species, device=dev); rem[:, 0] = N - nB; rem[:, 1] = nB
    for j in range(N):
        h, origin = m._step(xr, sr, sc[j], j, Lf)
        h = h + cf[j]
        ab, _ = m.flow.sample(h)
        cand[:, j] = torch.remainder(origin + ab * arc, Lf)
        # keep rem bookkeeping consistent with the true prefix
        rem[torch.arange(B, device=dev), sr[:, j]] -= 1
    clash_tf = clash_by_index(cand, xr, Lf)

q = N - N // 4
drift_share = float((clash_fr[q:] - clash_tf[q:]).mean() / max(clash_fr[q:].mean(), 1e-9))
print(f"clash (last quartile j>={q}): TF {clash_tf[q:].mean():.4f}  FR {clash_fr[q:].mean():.4f}"
      f"  -> drift share {drift_share:.2f}", flush=True)
print(f"G0-PF: {'PASS (drift >30% -> PF can act)' if drift_share > 0.30 else 'KILL (residual-dominated, PF cannot help)'}", flush=True)

plt.figure(figsize=(8, 4.5))
plt.plot(clash_tf, label=f"TF (true prefix), mean {clash_tf[1:].mean():.3f}")
plt.plot(clash_fr, label=f"FR (own prefix), mean {clash_fr[1:].mean():.3f}")
plt.xlabel("AR index j (curve order)"); plt.ylabel(f"P(NN dist < {THR})"); plt.legend()
plt.title(f"IPL44 curveflow TF-vs-FR clash — drift share (last quartile) = {drift_share:.2f}")
plt.tight_layout(); plt.savefig(f"{ART}/pf_pregate.png", dpi=110)
print(f"saved {ART}/pf_pregate.png", flush=True)
