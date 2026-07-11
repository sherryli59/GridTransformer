"""Exactness gate for the 3D cavity local-frame AR model (random-init, pre-training):
  (1) sample()'s accumulated logq == log_prob_pair(emitted, preordered=True)  [head-math consistency];
  (2) generation slot order vs canonical Morton order coincidence rate  [IS-exactness of using
      log_prob as the proposal weight];
  (3) log_prob of DATA interiors is finite and sane.
Run on GPU."""
import torch
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, cavity_order, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
torch.manual_seed(0)
m = KA3DCavityAR().to(dev); m.eval()   # eval -> dropout off, required for logq==log_prob exactness
print(f"KA3DCavityAR params={sum(p.numel() for p in m.parameters())/1e6:.3f}M  device={dev}", flush=True)

r_ctx = 2.5
for R in (1.6, 2.0, 2.4):
    xb, sb = X[-1], S[-1]
    center = xb[(31 * int(R * 10)) % xb.shape[0]].clone()
    pair = carve(xb, sb, center, R, L)
    interior_rel = _mic(pair["x_in"], center, L)
    bnd_all = _mic(pair["x_out"], center, L)
    shell = bnd_all.norm(dim=-1) < R + r_ctx
    bnd_rel, s_bnd = bnd_all[shell], pair["s_out"][shell]
    n_B = int((pair["s_in"] == 1).sum()); n_A = pair["n_in"] - n_B

    # data log_prob finite
    lp_data = m.log_prob_pair(interior_rel, pair["s_in"], bnd_rel, s_bnd, R)

    # exactness gate: sample logq == log_prob(emitted, preordered)
    xo, so, logq = m.sample_pair(bnd_rel, s_bnd, n_A, n_B, R, return_logq=True)
    lp_reeval = m.log_prob_pair(xo, so, bnd_rel, s_bnd, R, preordered=True)
    exact_err = (logq - lp_reeval).abs().item()

    # ordering coincidence: does canonical Morton order of the emitted sample reproduce slot order?
    order = cavity_order(xo, R)
    coincide = (order == torch.arange(xo.shape[0], device=dev)).float().mean().item()

    print(f"R={R:.1f} n={pair['n_in']:3d} (A{n_A}/B{n_B}) shell={int(shell.sum())} | "
          f"logp_data/n={lp_data.item()/pair['n_in']:+.3f} | logq==logprob err={exact_err:.2e} "
          f"{'EXACT' if exact_err < 1e-3 else 'MISMATCH'} | slot==morton {coincide*100:.0f}%", flush=True)
print("done", flush=True)
