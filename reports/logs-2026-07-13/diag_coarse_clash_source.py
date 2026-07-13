"""WHERE do the coarse head's clashes come from, and why did min_sep barely help (25.3->23.2%)?
Decompose each placed block particle's binding (min sigma-gap) clash partner:
  MASKABLE  = already-placed when this particle was drawn: boundary + retained + earlier-Morton block
  FUTURE    = later-Morton block particle (causal AR blind to it -> NO mask can forbid)
Then two sub-questions on the maskable part:
  (a) THRESHOLD gap: clash counts <0.9 but min_sep=0.85 -> particles at 0.85-0.9 obey the mask yet count
      as clash. How much of the 'clash' is just this mismatch? (fix: raise min_sep toward 0.9)
  (b) STAGE: for a maskable clash, is the COARSE cell centre already within min_sep of the partner
      (coarse mask should forbid it -- leaked via the conservative margin), or is the cell fine but the
      FINE 8-bin placement dips in (fine stage is UNMASKED by design)? (fix: add a fine-stage mask)
Plus a min_sep sweep {0.85,0.90,0.95} with the exact round-trip re-checked, to see the achievable floor.
"""
import torch, statistics as st
from liquid_coupling_flow.ka3d_coarse_head import KA3DScaffoldEBMCoarse
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold, ball_unsquash, ball_squash
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 8; NCAV = 12; CLASH = 0.9
sig = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMCoarse(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_coarse_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
co = m.coarse
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def setup(ci, g):
    while True:
        c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=g, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return xo, so, blk, bnd, sb, n, anch


def decompose(cut):
    stats = {"total": 0, "clash": 0, "maskable": 0, "future": 0, "thresh_only": 0, "coarse_leak": 0, "fine_leak": 0}
    gsel = torch.Generator(device=dev).manual_seed(123)   # fixed blocks across cuts
    gsmp = torch.Generator(device=dev).manual_seed(7)
    ncav = 0
    for ci in range(16):
        xo, so, blk, bnd, sb, n, anch = setup(ci, gsel)
        xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
        xa, sa, _ = m.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gsmp, pos_temp=PT, min_sep=cut)
        # Morton generation rank of block slots (block = Morton-ordered suffix after retained)
        order = torch.argsort(blk.to(torch.uint8), stable=True); gen_seq = order[n - K:]
        blk_ids = torch.nonzero(blk, as_tuple=True)[0]
        rank = torch.tensor([int((gen_seq == j).nonzero()) for j in blk_ids], device=dev)  # [K] per block col
        anchor_y, _ = ball_unsquash(fixed_ball_scaffold(n, R, dev), R)
        for mm in range(M):
            blk_x = xa[mm, blk]; blk_s = sa[mm, blk]
            ret_x = xa[mm, ~blk]; ret_s = sa[mm, ~blk]
            for q in range(K):
                stats["total"] += 1
                placed_x = [bnd, ret_x] + ([blk_x[rank < rank[q]]] if (rank < rank[q]).any() else [])
                placed_s = [sb, ret_s] + ([blk_s[rank < rank[q]]] if (rank < rank[q]).any() else [])
                px = torch.cat(placed_x, 0); ps = torch.cat(placed_s, 0)
                fx = blk_x[rank > rank[q]]; fs = blk_s[rank > rank[q]]           # future block
                dpl = (torch.cdist(blk_x[q:q + 1], px)[0] / sig[blk_s[q], ps]).min() if px.shape[0] else torch.tensor(9., device=dev)
                dfu = (torch.cdist(blk_x[q:q + 1], fx)[0] / sig[blk_s[q], fs]).min() if fx.shape[0] else torch.tensor(9., device=dev)
                gmin = min(float(dpl), float(dfu))
                if gmin >= CLASH:
                    continue
                stats["clash"] += 1
                if float(dfu) < float(dpl):
                    stats["future"] += 1                                        # bound by a later particle
                    continue
                stats["maskable"] += 1                                          # bound by already-placed
                if float(dpl) >= cut:
                    stats["thresh_only"] += 1                                   # obeys min_sep, just <0.9
                    continue
                # real leak below min_sep: coarse-cell vs fine-within-cell
                # cell centre position of this particle:
                y = ball_unsquash(blk_x[q:q + 1], R)[0]; u = y - anchor_y[gen_seq[q]][None]
                ci_cell = co.cell_index(u); cctr = co.cell_center(ci_cell)
                cell_pos, _ = ball_squash(anchor_y[gen_seq[q]][None] + cctr, R)
                dcell = (torch.cdist(cell_pos, px)[0] / sig[blk_s[q], ps]).min()
                if float(dcell) < cut:
                    stats["coarse_leak"] += 1                                   # cell centre itself clashes (conservative-margin miss)
                else:
                    stats["fine_leak"] += 1                                     # cell ok, fine 8-bin placement dipped in
        ncav += 1
        if ncav >= NCAV:
            break
    return stats


print("=== COARSE-HEAD CLASH SOURCE DECOMPOSITION (12 held cages, M=8, K=8) ===", flush=True)
for cut in (0.85, 0.90, 0.95):
    s = decompose(cut)
    t = s["total"]; c = s["clash"]
    print(f"min_sep={cut}: clash {100*c/t:5.1f}%  | of clashes: future-blind {100*s['future']/max(c,1):3.0f}%  "
          f"maskable {100*s['maskable']/max(c,1):3.0f}% (= thresh-only {100*s['thresh_only']/max(c,1):3.0f}% "
          f"+ coarse-leak {100*s['coarse_leak']/max(c,1):3.0f}% + fine-leak {100*s['fine_leak']/max(c,1):3.0f}%)", flush=True)
torch.save({"note": "coarse clash source"}, "reports/logs-2026-07-13/diag_coarse_clash_source.pt")
print("saved", flush=True)
