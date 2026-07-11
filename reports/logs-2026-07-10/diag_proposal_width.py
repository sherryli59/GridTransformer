"""Is the proposal conditional stuck at the spline resolution floor (mw precedent: num_bins 8->32)?
For held single-slot contexts: draw 64 regenerations of the SAME slot, measure the std of the
placed position (x-space) and the distance from the placement mean to the true position.
Cage scale ~0.1-0.2 sigma; knot scale = 2*tail_bound/num_bins = 10/12 = 0.83 (y-space).
If measured width ~ knot scale >> cage scale -> capacity floor -> flow_bins 12->32 justified."""
import shutil, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_block import sample_block
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "diag_w_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR().to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}  flow_bins={m.flow_bins} tail={m.flow_tail} "
      f"-> knot scale ~{2*m.flow_tail/m.flow_bins:.2f} (y-space)", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R = 2.0; gen = torch.Generator(device=dev).manual_seed(4)
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)

widths, dist_to_true, nn_of_true = [], [], []
with torch.no_grad():
    for ci in range(900, 925):
        center = torch.rand(3, generator=gen, device=dev) * L
        p = carve(X[ci], S[ci], center, R, L)
        if p["n_in"] < 6:
            continue
        x, s, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
        n = x.shape[0]
        j = int(torch.randint(n, (), generator=gen, device=dev))       # one slot
        blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[j] = True
        draws = []
        for _ in range(64):
            xn, _, _ = sample_block(m, x, s, blk, E3, E1, R, gen=gen)
            draws.append(xn[j])
        D = torch.stack(draws)
        mean = D.mean(0)
        widths.append(float((D - mean).norm(dim=-1).pow(2).mean().sqrt()))   # rms spread
        dist_to_true.append(float((mean - x[j]).norm()))
        others = torch.cat([x[:j], x[j + 1:]])
        nn_of_true.append(float(torch.cdist(x[j:j + 1], others).min()))

print(f"sites={len(widths)}", flush=True)
print(f"proposal rms WIDTH     : median={st.median(widths):.3f}  (cage ~0.1-0.2; knot ~0.83)", flush=True)
print(f"|mean - true|          : median={st.median(dist_to_true):.3f}", flush=True)
print(f"true NN distance       : median={st.median(nn_of_true):.3f}  (context scale check)", flush=True)
print("width >> cage & ~knot -> RESOLUTION FLOOR (mw precedent) -> flow_bins 12->32 retrain.", flush=True)
