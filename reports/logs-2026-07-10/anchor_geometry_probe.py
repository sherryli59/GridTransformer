"""Pure-geometry probe (no model): WHERE does each candidate anchor sit, relative to (a) the placed
particles and (b) the TRUE particle it should place? Decides the anchor choice for the AR head.

For each interior slot j (Morton order), compute, under three anchor choices:
  CENTROID-interior : soft centroid of placed INTERIOR prefix (v2 current)
  CENTROID-all      : soft centroid of placed boundary+interior (v1 buggy)
  SCAFFOLD          : the fixed fibonacci-Morton slot t_j (mW-style empty target)
Report medians of:
  d_nbr  = min distance from the anchor to ANY placed particle (small => anchor ON a neighbour =>
           argmax placement clashes)
  d_true = distance from the anchor to the TRUE particle j (large => big residual to model)
A safe anchor has d_nbr > ~0.8 sigma (empty) AND modest d_true.
"""
import torch, statistics as st
from liquid_coupling_flow.ka3d_cavity_ar import fibonacci_ball_scaffold, cavity_order, soft_origin, _mic, SIGMA_LOC, W0_PRIOR
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
r_ctx = 2.5; g = torch.Generator(device=dev).manual_seed(7)

acc = {k: {"d_nbr": [], "d_true": []} for k in ("centroid_int", "centroid_all", "scaffold")}
for ci in range(900, 940):
    xb, sb = X[ci], S[ci]
    center = xb[int(torch.randint(xb.shape[0], (), generator=g, device=dev))].clone()
    R = float([1.6, 2.0, 2.4][ci % 3])
    p = carve(xb, sb, center, R, L)
    if p["n_in"] < 6:
        continue
    irel = _mic(p["x_in"], center, L); ball = _mic(p["x_out"], center, L)
    shell = ball.norm(dim=-1) < R + r_ctx
    bnd = ball[shell]
    n, mm = p["n_in"], bnd.shape[0]
    scaffold = fibonacci_ball_scaffold(n, R, device=dev, dtype=irel.dtype)
    order = cavity_order(irel, R); xo = irel[order]
    combined = torch.cat([bnd, xo]); idx = torch.arange(mm + n, device=dev)
    for j in range(n):
        placed_all = idx < (mm + j)                            # boundary + interior prefix
        placed_int = (idx >= mm) & (idx < (mm + j))            # interior prefix only
        o_all = soft_origin(combined, placed_all, scaffold[j], SIGMA_LOC, W0_PRIOR)
        o_int = soft_origin(combined, placed_int, scaffold[j], SIGMA_LOC, W0_PRIOR)
        o_sc = scaffold[j]
        placed_pts = combined[placed_all]                      # all already-placed real particles
        if placed_pts.shape[0] == 0:
            continue
        for name, o in (("centroid_int", o_int), ("centroid_all", o_all), ("scaffold", o_sc)):
            acc[name]["d_nbr"].append((placed_pts - o).norm(dim=-1).min().item())
            acc[name]["d_true"].append((xo[j] - o).norm().item())

print("anchor           d_nbr(median)  frac<0.8(on-neighbour)   d_true(median)", flush=True)
for k in ("centroid_int", "centroid_all", "scaffold"):
    dn = acc[k]["d_nbr"]; dt = acc[k]["d_true"]
    frac = sum(x < 0.8 for x in dn) / len(dn)
    print(f"  {k:13s}  {st.median(dn):6.3f}         {frac*100:5.1f}%                  {st.median(dt):6.3f}", flush=True)
print("\nRead: high d_nbr + low frac<0.8 = SAFE anchor (empty target). Low d_nbr = argmax lands on a neighbour.", flush=True)
