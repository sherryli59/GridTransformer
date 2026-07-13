"""Nested within-cell min-sep mask on the coarse head: EXACTNESS (sample==score) + CLASH floor.
The geometrically-correct declasher (mask fine-a/b/c by nf^3-cube column availability) -- does it finally
reach the ~5% future-blind floor where the last-axis masks stalled at ~23%? float64 for the exact bar."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cpu"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 16; NCAV = 8; CUT = 0.9
sig = torch.tensor(SIGMA, device=dev).double()
m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5).to(dev).double()
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_coarse_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).double(), D["s"].to(dev).long(), float(D["L"])

print("=== NESTED within-cell min-sep mask (coarse head, float64, K=8, cut=0.9) ===", flush=True)
for mode in ("none", "last-axis (min_sep)", "NESTED"):
    gen = torch.Generator(device=dev).manual_seed(5)
    kw = {} if mode == "none" else {"min_sep": CUT} if mode.startswith("last") else {"min_sep": CUT, "nested": True}
    diffs, ch, ct = [], 0, 0
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev).double() * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev).double()
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        xb = xo[None].expand(M, n, 3).contiguous(); sbb = so[None].expand(M, n).contiguous()
        xs, ss, lq_s = m.sample_block_b(xb, sbb, blk, bnd, sb, R, gen=gen, pos_temp=PT, **kw)
        lq_c = m.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=PT, **kw)
        diffs += (lq_c - lq_s).abs().tolist()
        blk_x, blk_s = xs[:, blk], ss[:, blk]
        cage_x = torch.cat([bnd[None].expand(M, -1, -1), xs[:, ~blk]], 1)
        cage_s = torch.cat([sb[None].expand(M, -1), ss[:, ~blk]], 1)
        nbr_x = torch.cat([blk_x, cage_x], 1); nbr_s = torch.cat([blk_s, cage_s], 1)
        for mm in range(M):
            d = torch.cdist(blk_x[mm], nbr_x[mm]); d[torch.arange(K), torch.arange(K)] = float("inf")
            ch += int(((d / sig[blk_s[mm][:, None], nbr_s[mm][None, :]]).min(1).values < 0.9).sum()); ct += K
        ncav += 1
        if ncav >= NCAV:
            break
    dd = torch.tensor(diffs)
    print(f"  {mode:22s}: roundtrip median {dd.median():.1e} match(<1e-3) {100*float((dd<1e-3).float().mean()):5.1f}%"
          f"  | clash<0.9 {100*ch/ct:5.1f}%", flush=True)
