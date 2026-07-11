"""Interpretable clash metrics on the current 32-bin checkpoint, for BOTH generation modes.

For each held cavity (free cluster, R in {1.6,2.0,2.4}):
  * BLOCK mode (what MTM uses): regenerate a k-blob into the TRUE surround; count clashes among
    (block particle, any particle) pairs where the block particle is involved.
  * FULL mode: one-shot generate the entire cluster; count ALL pairs < 0.8 sigma.
Reports, per mode: % clash-FREE configs, mean #clashes per config, and the old per-particle rate
(for continuity). A "clash" = a pair at distance < clash_cut (0.8 sigma)."""
import shutil, statistics as st
from pathlib import Path
import torch
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_scaffold_ar import KA3DScaffoldAR, label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_block import sample_block

dev = "cuda"; CUT = 0.8
SCR = Path("/tmp/claude-1002/-mnt-ssd-GridTransformer/271167b9-117a-400e-b11f-c2c1bfa0b6df/scratchpad")
ckp = SCR / "clash_ckpt.pt"; shutil.copy("liquid_coupling_flow/artifacts/ka3d_blob_ar32.pt", ckp)
ck = torch.load(ckp, map_location=dev, weights_only=False)
m = KA3DScaffoldAR(flow_bins=32).to(dev); m.load_state_dict(ck["state_dict"]); m.eval()
print(f"ckpt step={ck['step']} (flow_bins=32)", flush=True)

d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N512_T0.5.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(2)
E3, E1 = torch.zeros(0, 3, device=dev), torch.zeros(0, dtype=torch.long, device=dev)


def block_clashes(xn, blk):
    """# pairs < CUT that involve a block particle (each such pair counted once)."""
    bi = xn[blk]; d = torch.cdist(bi, xn)
    d[torch.arange(bi.shape[0]), torch.nonzero(blk).squeeze(1)] = 1e9   # exclude self
    close = d < CUT                                                     # [k, n]
    # block-block pairs would be double counted (both endpoints in bi): halve those
    bb = torch.cdist(bi, bi); bb.fill_diagonal_(1e9)
    return int(close.sum()) - int((bb < CUT).sum()) // 2, int(bi.shape[0])


def full_clashes(xn):
    n = xn.shape[0]; dd = torch.cdist(xn, xn); dd.fill_diagonal_(1e9)
    iu = torch.triu_indices(n, n, 1, device=dev)
    return int((dd[iu[0], iu[1]] < CUT).sum())


blk_free, blk_nclash, blk_prate = [], [], []
full_free, full_nclash = [], []
with torch.no_grad():
    for ci in range(900, 960):
        center = torch.rand(3, generator=gen, device=dev) * L
        p = carve(X[ci], S[ci], center, 2.0, L)
        if p["n_in"] < 6:
            continue
        R = 2.0
        x, s, _ = label_to_scaffold(_mic(p["x_in"], center, L), p["s_in"], R)
        n = x.shape[0]; n_B = int((s == 1).sum())
        # BLOCK
        blk = fixed_ball_scaffold(n, R, dev)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        idx = (blk - blk[seed]).norm(dim=-1).topk(min(4, n), largest=False).indices
        bm = torch.zeros(n, dtype=torch.bool, device=dev); bm[idx] = True
        xn, sn, _ = sample_block(m, x, s, bm, E3, E1, R, gen=gen)
        nc, k = block_clashes(xn, bm)
        blk_nclash.append(nc); blk_free.append(nc == 0); blk_prate.append(nc / k)
        # FULL
        fx, fs = m.sample_pair(E3, E1, n - n_B, n_B, R)
        fc = full_clashes(fx)
        full_nclash.append(fc); full_free.append(fc == 0)

N = len(blk_free)
print(f"\nheld configs = {N}   (clash = pair < {CUT} sigma)", flush=True)
print(f"BLOCK (k=4 blob into true surround; what MTM uses):", flush=True)
print(f"   clash-FREE blocks   : {100*sum(blk_free)/N:.0f}%   ({sum(blk_free)}/{N})", flush=True)
print(f"   mean #clashes/block : {st.mean(blk_nclash):.2f}   (median {st.median(blk_nclash)})", flush=True)
print(f"FULL (one-shot whole cluster, ~40 particles):", flush=True)
print(f"   clash-FREE clusters : {100*sum(full_free)/N:.0f}%   ({sum(full_free)}/{N})", flush=True)
print(f"   mean #clashes/cluster: {st.mean(full_nclash):.2f}   (median {st.median(full_nclash)})", flush=True)
