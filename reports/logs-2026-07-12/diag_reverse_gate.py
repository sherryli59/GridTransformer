"""Reverse-image gate diagnostic for the gated AR base (decides the gate cut BEFORE the retrain).

Two questions, per candidate cut:
  (1) REVERSE pass rate: reverse-flow (t 1->0) the DATA block to its base image x0_rev; what fraction
      pass the sigma-scaled worst-pair screen? This bounds how often the current-state evaluation
      q_gated(x) = 0 would (validly but wastefully) reject a move at equilibrium. Low -> soften cut
      or select-after-flow instead. NOTE: measured with the CURRENT (ungated-trained) flow -> a
      conservative lower bound; the gated retrain pulls base images toward the gated manifold.
  (2) FORWARD pass rate of raw AR@0.4 draws (per-draw): sets the expected redraw count (1/rate) and
      the training-side fraction of garbage the gate removes.
Held reference chains only, K=8, same frames as gate_structure_v3 (_mic gets the BOX L).
"""
import torch
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka3d_gated_base import sigma_min_ratio

dev = "cuda"; RCTX = 2.5; K = 8; M = 8; POS_TEMP = 0.4; N_CAGE = 48; DUMMY_R = 50.0
CUTS = (0.70, 0.75, 0.80, 0.85)
FLOW_CKPT = "liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_best.pt"

ck = torch.load(FLOW_CKPT, map_location=dev, weights_only=False)
a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"]).to(dev).double()
flow.load_state_dict(ck["state_dict"]); flow.eval()
ar_ck = torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)
ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(ar_ck["state_dict"], strict=False); ar.eval(); ar.use_frame = False
d = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = d["x"].to(dev).float(), d["s"].to(dev).long(), float(d["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def fixed_size_cage(cage_x, sp_cage, centroid, n_cage):
    Mn, m = cage_x.shape[0], cage_x.shape[1]
    if m >= n_cage:
        idx = torch.topk((cage_x[0] - centroid).norm(dim=-1), n_cage, largest=False).indices
        return cage_x[:, idx], sp_cage[:, idx]
    pad = n_cage - m
    dpos = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cage_x.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cage_x, dpos], 1), torch.cat([sp_cage, torch.zeros(Mn, pad, dtype=sp_cage.dtype, device=dev)], 1)


rev_ratio, fwd_ratio = [], []
ncav = 0
for ci in range(16):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, 2.0, L)
    if p["n_in"] < K + 6:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], 2.0)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (2.0 + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    assert bnd.shape[0] < 600, f"frame sanity: n_bnd={bnd.shape[0]}"
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, 2.0, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    block_mask = torch.zeros(n, dtype=torch.bool, device=dev)
    block_mask[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    # forward: raw AR draws -> per-draw min ratio (cage = boundary + retained, matching deployment)
    xo_ar, so_ar, _ = ar.sample_block_b(xo_b, so_b, block_mask, bnd, sb, 2.0, gen=gen, pos_temp=POS_TEMP)
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), xo_ar[:, ~block_mask]], 1)
    cage_s = torch.cat([sb[None].expand(M, -1), so_ar[:, ~block_mask]], 1)
    fwd_ratio += sigma_min_ratio(xo_ar[:, block_mask], so_ar[:, block_mask], cage_x, cage_s).tolist()
    # reverse: DATA block -> base image under the current flow (48-nearest cage for message passing)
    data_blk = xo[None, block_mask].expand(M, K, 3).contiguous()
    data_s = so[None, block_mask].expand(M, K)
    cage_data = torch.cat([bnd[None].expand(M, -1, -1), xo[None, ~block_mask].expand(M, -1, -1)], 1)
    cage_s_data = torch.cat([sb[None].expand(M, -1), so[None, ~block_mask].expand(M, -1)], 1)
    centroid = xo[block_mask].mean(0)
    cage_fix, cage_s_fix = fixed_size_cage(cage_data, cage_s_data, centroid, N_CAGE)
    try:
        x0_rev, _ = flow.flow(data_blk.double(), cage_fix.double(), data_s, cage_s_fix, reverse=True)
    except AssertionError as e:
        print(f"  cav {ncav+1}: reverse solve FAILED ({e}) -> counts as pass-rate 0 rows", flush=True)
        rev_ratio += [0.0] * M
        ncav += 1
        continue
    rev_ratio += sigma_min_ratio(x0_rev.float(), data_s, cage_data, cage_s_data).tolist()
    ncav += 1
    print(f"  cav {ncav}: fwd min-ratio med {torch.tensor(fwd_ratio[-M:]).median():.3f} | "
          f"rev min-ratio med {torch.tensor(rev_ratio[-M:]).median():.3f}", flush=True)

fwd = torch.tensor(fwd_ratio); rev = torch.tensor(rev_ratio)
print(f"\n=== REVERSE-IMAGE GATE DIAGNOSTIC ({ncav} held cavities x M={M}, K={K}, pos_temp={POS_TEMP}) ===")
print(f"  {'cut':>5} | {'fwd pass (AR draws)':>20} | {'expected redraws':>17} | {'rev pass (data images)':>23}")
for cut in CUTS:
    fp = (fwd >= cut).float().mean().item(); rp = (rev >= cut).float().mean().item()
    exp_rd = (1.0 / fp) if fp > 0 else float("inf")
    print(f"  {cut:5.2f} | {100*fp:19.1f}% | {exp_rd:17.1f} | {100*rp:22.1f}%")
torch.save({"fwd_ratio": fwd, "rev_ratio": rev, "cuts": CUTS, "ncav": ncav},
           "reports/logs-2026-07-12/diag_reverse_gate.pt")
print("saved -> reports/logs-2026-07-12/diag_reverse_gate.pt", flush=True)
