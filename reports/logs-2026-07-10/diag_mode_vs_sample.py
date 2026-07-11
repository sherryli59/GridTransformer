"""Decouple density-fit from sampling: for a k=4 block into the true surround, compare clashes when
placing at the spline MODE (z=0 base -> deterministic) vs random SAMPLES (z~N), vs a LOW-TEMP sample
(z*0.5). If MODE is ~clash-free but samples clash -> tail-mass problem (sharpen/anneal proposal, or
low-T MTM). If MODE also clashes -> conditional can't carve the core (architecture/context, not res).
Also reports the block-particle nearest-neighbour distance distribution to see WHERE mass sits."""
import shutil, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import (KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold,
                                                   ball_squash, ball_unsquash)
from liquid_coupling_flow.ka3d_block import _order

dev = "cuda"; CUT = 0.8
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "mvs_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar32.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR(flow_bins=32).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']}", flush=True)
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(1); R = 2.0


@torch.no_grad()
def sample_block_z(x, s, blk, zscale):
    """Regenerate block; base draws scaled by zscale (0=mode). Returns clashes among block particles."""
    n, m_ = x.shape[0], 0
    anchors = fixed_ball_scaffold(n, R, dev)
    order = _order(blk); n_ret = int((~blk).sum())
    xo = x[order].clone(); so = s[order].clone(); anch = anchors[order]
    anch_y, _ = ball_unsquash(anch, R)
    combined = xo.clone(); scomb = so.clone(); kind = torch.zeros(n, dtype=torch.long, device=dev)
    idx = torch.arange(n, device=dev)
    for jj in range(n_ret, n):
        valid = (idx < jj)[None]
        sf = m._slot_features(anch[jj:jj + 1], jj, n, R)
        h, frame = m._frame_context(combined, scomb, kind, valid, anch[jj:jj + 1], anch[jj:jj + 1], slot_feat=sf)
        e = m.sp_out_emb(so[jj:jj + 1])
        z = torch.randn(1, 3, device=dev, generator=gen) * zscale                 # zscale=0 -> mode
        # 3-axis AR spline forward manually so we can scale the base draw z (temperature):
        ctx = h + e; vals = []
        for a in range(3):
            za = z[:, a:a + 1]
            ui, _ = m.flow.spline.forward(za, m.flow.heads[a](ctx))
            vals.append(ui); ctx = torch.cat([ctx, ui], -1)
        u = torch.cat(vals, -1)
        y = anch_y[jj] + torch.einsum("naj,na->nj", frame, u)[0]
        pos, _ = ball_squash(y, R)
        xo[jj] = pos; combined[jj] = pos
    bi = xo[n_ret:]
    dd = torch.cdist(bi, xo);
    for a in range(bi.shape[0]): dd[a, n_ret + a] = 1e9
    nn = dd.min(1).values
    return int((nn < CUT).sum()), nn.min().item()


res = {0.0: [], 0.5: [], 1.0: []}
with torch.no_grad():
    for ci in range(900, 940):
        center = torch.rand(3, generator=gen, device=dev) * L
        p = carve(X[ci], S[ci], center, R, L)
        if p["n_in"] < 6: continue
        x, s, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
        n = x.shape[0]; anchors = fixed_ball_scaffold(n, R, dev)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        idx = (anchors - anchors[seed]).norm(dim=-1).topk(min(4, n), largest=False).indices
        blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[idx] = True
        for zs in res:
            nc, _ = sample_block_z(x, s, blk, zs)
            res[zs].append(nc)

for zs, lab in [(0.0, "MODE (z=0)"), (0.5, "low-T (z*0.5)"), (1.0, "full sample")]:
    v = res[zs]; N = len(v)
    print(f"  {lab:16s}: clash-free {100*sum(c==0 for c in v)/N:.0f}%  mean #clashes/block {st.mean(v):.2f}", flush=True)
print("\nMODE clash-free>>sample -> tail-mass (anneal/low-T MTM). MODE also clashes -> conditional/core problem.",
      flush=True)
