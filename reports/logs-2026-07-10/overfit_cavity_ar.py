"""OVERFIT-FIRST test of the 3D local-frame AR cavity generator (the decisive architecture gate;
the uniform-base flow arm died here). Memorize a handful of carved cavities (NO augmentation) via
-log_prob/n. Success = loss drops far below random-init AND generated interiors are PHYSICAL
(energy/particle near the true carved interior, few clashes) AND overlap is sane.
"""
import time
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_pts_observables import core_overlap
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
r_ctx, RADII = 2.5, [1.6, 2.4]
torch.manual_seed(0)

# --- carve a fixed overfit set ---
pairs = []
for i in range(12):
    R = RADII[i % 2]
    xb, sb = X[i], S[i]
    center = xb[(37 * i + 5) % xb.shape[0]].clone()
    p = carve(xb, sb, center, R, L)
    irel = _mic(p["x_in"], center, L)
    ball = _mic(p["x_out"], center, L)
    shell = ball.norm(dim=-1) < R + r_ctx
    n_B = int((p["s_in"] == 1).sum())
    pairs.append(dict(irel=irel, s_in=p["s_in"], bnd=ball[shell], s_bnd=p["s_out"][shell], R=R,
                      n_A=p["n_in"] - n_B, n_B=n_B, n=p["n_in"], center=center,
                      x_out=_mic(p["x_out"], center, L), s_out=p["s_out"]))
print(f"overfit set: {len(pairs)} cavities, n_in={[p['n'] for p in pairs]}", flush=True)

m = KA3DCavityAR().to(dev)
opt = torch.optim.Adam(m.parameters(), lr=3e-4)
t0 = time.time()
for step in range(2000):
    idx = torch.randint(len(pairs), (4,))
    loss = 0.0
    for i in idx:
        p = pairs[i]
        loss = loss - m.log_prob_pair(p["irel"], p["s_in"], p["bnd"], p["s_bnd"], p["R"]) / p["n"]
    loss = loss / len(idx)
    opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    if step % 200 == 0 or step == 1999:
        with torch.no_grad():
            full = sum(-m.log_prob_pair(p["irel"], p["s_in"], p["bnd"], p["s_bnd"], p["R"]).item() / p["n"]
                       for p in pairs) / len(pairs)
        print(f"  step {step:4d}  -logp/n(all)={full:+.3f}  ({time.time()-t0:.0f}s)", flush=True)

# --- physical check: generated vs true interior energy & overlap ---
print("\n[physical check] generated interior vs true carved interior:", flush=True)
m.eval()
def full_energy(center, interior_rel, s_in, x_out_rel, s_out):
    N = interior_rel.shape[0] + x_out_rel.shape[0]
    x = torch.empty(1, N, 3, device=dev); s = torch.empty(1, N, device=dev, dtype=torch.long)
    x[0, :interior_rel.shape[0]] = torch.remainder(center + interior_rel, L)
    x[0, interior_rel.shape[0]:] = torch.remainder(center + x_out_rel, L)
    s[0, :interior_rel.shape[0]] = s_in; s[0, interior_rel.shape[0]:] = s_out
    return (ka_energy(x, s, L) / N).item()
with torch.no_grad():
    for p in pairs[:6]:
        u_true = full_energy(p["center"], p["irel"], p["s_in"], p["x_out"], p["s_out"])
        gi, gs, _ = m.sample_pair(p["bnd"], p["s_bnd"], p["n_A"], p["n_B"], p["R"], return_logq=True)
        u_gen = full_energy(p["center"], gi, gs, p["x_out"], p["s_out"])
        # overlap of generated vs true interior (same-species NN core), both center-relative -> use center=0
        z = torch.zeros(3, device=dev)
        q = core_overlap(gi[None], gs[None], p["irel"][None], p["s_in"][None], z, L)[0].item()
        print(f"  R={p['R']} n={p['n']:2d}: U/N true={u_true:+.3f} gen={u_gen:+.3f}  overlap(gen,true)={q:.3f}", flush=True)
print("done", flush=True)
