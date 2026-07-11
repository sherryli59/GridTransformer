"""Confirm the free-cluster verdict: in cavity generation, are the CATASTROPHIC clashes interior<->
interior or interior<->BOUNDARY? For each generated interior particle, min-distance to (a) other
generated interior, (b) frozen boundary. Count clashes (<0.8 sigma) by type. If interior<->boundary
dominates, the fix is a cheap EXACT boundary-exclusion mask during sampling (boundary is fixed+known).
"""
import torch
from liquid_coupling_flow.ka3d_cavity_ar import KA3DCavityAR, _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda" if torch.cuda.is_available() else "cpu"
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ar.pt", map_location=dev, weights_only=False)
m = KA3DCavityAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"cavity ckpt step={ck['step']}", flush=True)
r_ctx, CLASH = 2.5, 0.8; g = torch.Generator(device=dev).manual_seed(3)


def mindist(a, b):
    if b.shape[0] == 0:
        return a.new_full((a.shape[0],), 1e9)
    return torch.cdist(a, b).min(1).values


int_clash = bnd_clash = tot = 0
gen_min_bnd = []
with torch.no_grad():
    for ci in range(900, 940):
        xb, sb = X[ci], S[ci]
        center = xb[int(torch.randint(xb.shape[0], (), generator=g, device=dev))].clone()
        R = float([1.6, 2.0, 2.4][ci % 3])
        p = carve(xb, sb, center, R, L)
        if p["n_in"] < 6:
            continue
        ball = _mic(p["x_out"], center, L)
        shell = ball.norm(dim=-1) < R + r_ctx
        bnd, s_bnd = ball[shell], p["s_out"][shell]
        n_B = int((p["s_in"] == 1).sum())
        gi, gs = m.sample_pair(bnd, s_bnd, p["n_in"] - n_B, n_B, R)          # generated interior (center-rel)
        d_int = torch.cdist(gi, gi) + torch.eye(gi.shape[0], device=dev) * 1e9
        d_int = d_int.min(1).values                                          # nearest OTHER interior
        d_bnd = mindist(gi, bnd)                                             # nearest boundary atom
        int_clash += int((d_int < CLASH).sum()); bnd_clash += int((d_bnd < CLASH).sum())
        tot += gi.shape[0]; gen_min_bnd.append(float(d_bnd.min()))

print(f"generated interior particles: {tot}", flush=True)
print(f"  clashing (<{CLASH}sig) with another INTERIOR : {int_clash} ({100*int_clash/tot:.1f}%)", flush=True)
print(f"  clashing (<{CLASH}sig) with a frozen BOUNDARY : {bnd_clash} ({100*bnd_clash/tot:.1f}%)", flush=True)
print(f"  median per-config nearest interior->boundary dist: {torch.tensor(gen_min_bnd).median():.3f}", flush=True)
print(f"VERDICT: {'BOUNDARY overlaps dominate -> exact boundary-exclusion mask fixes the catastrophe' if bnd_clash >= int_clash else 'interior overlaps dominate -> intra-cluster issue'}", flush=True)
