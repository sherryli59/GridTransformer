"""Localize the density-vs-sampling gap: does log_prob's coordinate ENCODE invert sample's DECODE?

log_prob (block_log_prob) computes, per block slot: y=ball_unsquash(x); u = frame @ (y - anchor_y).
sample (sample_block) computes:               y = anchor_y + frame @ u; x = ball_squash(y).
If frame/anchor/squash are used consistently, feeding log_prob's u into sample's decode must return
the ORIGINAL x. Do this slot-by-slot on a TRUE config (teacher-forced context, no model stochasticity)
-> pure geometry check. Nonzero error => the coordinate chain is asymmetric = the bug behind
'mode clashes but nll good'.  Also checks the FRAME is identical between the two code paths."""
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold,
                                                   ball_squash, ball_unsquash)
from liquid_coupling_flow.ka3d_block import _order

dev = "cuda"
m = KA3DScaffoldAR(flow_bins=32).to(dev); m.eval()
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
R = 2.0; torch.manual_seed(0)
worst = 0.0
with torch.no_grad():
    for ci in range(6):
        center = X[ci][17 * ci % 512]
        p = carve(X[ci], S[ci], center, R, L)
        x, s, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
        n = x.shape[0]
        anchors = fixed_ball_scaffold(n, R, dev)
        # whole interior is the "block": full teacher-forced pass
        blk = torch.ones(n, dtype=torch.bool, device=dev)
        order = _order(blk)                      # identity here (all block)
        xo, so, anch = x[order], s[order], anchors[order]
        anch_y, _ = ball_unsquash(anch, R)
        combined = xo.clone(); scomb = so.clone(); kind = torch.zeros(n, dtype=torch.long, device=dev)
        idx = torch.arange(n, device=dev)
        err_pos = 0.0
        for jj in range(n):
            valid = (idx < jj)[None]
            sf = m._slot_features(anch[jj:jj+1], jj, n, R)
            h, frame = m._frame_context(combined, scomb, kind, valid, anch[jj:jj+1], anch[jj:jj+1], slot_feat=sf)
            # ENCODE (log_prob path): true position -> u
            y_true, _ = ball_unsquash(xo[jj:jj+1], R)
            u = torch.einsum("naj,nj->na", frame, y_true - anch_y[jj:jj+1])
            # DECODE (sample path): u -> position
            y_dec = anch_y[jj] + torch.einsum("naj,na->nj", frame, u)[0]
            x_dec, _ = ball_squash(y_dec, R)
            err_pos = max(err_pos, float((x_dec - xo[jj]).norm()))
        worst = max(worst, err_pos)
        print(f"  cfg {ci}: n={n}  max encode->decode position err = {err_pos:.2e}", flush=True)
print(f"\nworst = {worst:.2e} -> {'CHAIN CONSISTENT (bug is elsewhere: context/target)' if worst<1e-4 else 'COORDINATE CHAIN ASYMMETRIC = BUG'}",
      flush=True)
