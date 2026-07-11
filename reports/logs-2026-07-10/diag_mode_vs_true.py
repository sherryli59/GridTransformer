"""BUG PROBE: if block_nll is good, the true position has high density -> the conditional's MODE
should sit NEAR the true position (which is clash-free by construction). Measure, per block slot,
the distance from the sampled MODE to the TRUE position, and compare the log_prob the model assigns
to (a) the true position vs (b) its own mode. If mode is FAR from true / has HIGHER logp than true
-> the sampled mode is not where the density peak is = coordinate/context bug, not capacity."""
import shutil
from pathlib import Path
import torch
import torch.nn.functional as F
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold,
                                                   ball_squash, ball_unsquash)
from liquid_coupling_flow.ka3d_block import _order

dev = "cuda"
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "mvt_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar32.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR(flow_bins=32).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R = 2.0; torch.manual_seed(0)

mode_true_dist, u_true_norm, u_mode_norm, lp_true_gt_mode = [], [], [], []
with torch.no_grad():
    for ci in range(900, 930):
        center = torch.rand(3, device=dev) * L
        p = carve(X[ci], S[ci], center, R, L)
        if p["n_in"] < 6: continue
        x, s, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
        n = x.shape[0]; anchors = fixed_ball_scaffold(n, R, dev)
        blk = torch.ones(n, dtype=torch.bool, device=dev)     # full teacher-forced
        anch_y, _ = ball_unsquash(anchors, R)
        combined = x.clone(); scomb = s.clone(); kind = torch.zeros(n, dtype=torch.long, device=dev)
        idx = torch.arange(n, device=dev)
        for jj in range(2, n):                                 # skip first 2 (frame degenerate)
            valid = (idx < jj)[None]
            sf = m._slot_features(anchors[jj:jj+1], jj, n, R)
            h, frame = m._frame_context(combined, scomb, kind, valid, anchors[jj:jj+1], anchors[jj:jj+1], slot_feat=sf)
            e = m.sp_out_emb(s[jj:jj+1])
            # TRUE u for this slot
            y_true, _ = ball_unsquash(x[jj:jj+1], R)
            u_true = torch.einsum("naj,nj->na", frame, y_true - anch_y[jj:jj+1])   # [1,3]
            # MODE u: base z=0 through spline forward
            ctx = h + e; vals = []
            for a in range(3):
                ui, _ = m.flow.spline.forward(torch.zeros(1, 1, device=dev), m.flow.heads[a](ctx))
                vals.append(ui); ctx = torch.cat([ctx, ui], -1)
            u_mode = torch.cat(vals, -1)                                          # [1,3]
            # positions
            y_mode = anch_y[jj] + torch.einsum("naj,na->nj", frame, u_mode)[0]
            x_mode, _ = ball_squash(y_mode, R)
            mode_true_dist.append(float((x_mode - x[jj]).norm()))
            u_true_norm.append(float(u_true.norm())); u_mode_norm.append(float(u_mode.norm()))
            lp_true = m.flow.log_prob(h + e, u_true)
            lp_mode = m.flow.log_prob(h + e, u_mode)
            lp_true_gt_mode.append(float(lp_true) >= float(lp_mode))

import statistics as st
N = len(mode_true_dist)
print(f"slots={N}", flush=True)
print(f"  |mode - true| position   : median={st.median(mode_true_dist):.3f}  (cage NN ~0.9; small=good)", flush=True)
print(f"  |u_true| (frame coords)   : median={st.median(u_true_norm):.3f}", flush=True)
print(f"  |u_mode| (should ~ u_true): median={st.median(u_mode_norm):.3f}", flush=True)
print(f"  logp(true) >= logp(mode)  : {100*sum(lp_true_gt_mode)/N:.0f}%  (mode is argmax => should be ~0%; "
      f"if HIGH, true has higher density than the 'mode' => mode extraction wrong)", flush=True)
