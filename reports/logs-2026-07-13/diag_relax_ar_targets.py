"""Does frozen-cage T=0.5 MC relaxation of AR blocks give GOOD structure -> usable as multiple flow
targets per cage? (User idea: fix CFM's one-target collapse without solving cavity-PT.)

Per held cage: AR-sample M blocks (frozen cage), then relax EACH with single-site displacement + swap
MC at T=0.5 (cage frozen), N_SWEEP sweeps. Measure raw-AR vs relaxed-AR vs DATA:
  (A) GOOD STRUCTURE: energy/particle above data; clash%<0.9; g_BB peak location (raw AR misses the
      B-B contact shell at ~1.34, smears to 2.55 -- does relaxation recover it?).
  (B) MULTIPLE VALID TARGETS: per-cage RMS spread across the M relaxed blocks (breadth for CFM) and the
      fraction of relaxed blocks that are low-energy+clash-free (valid target yield).
Contrast with the deterministic-corrector result (T=0 L-BFGS stalled at +6.9/particle): T=0.5 MC is
thermal, so it can declash over sweeps rather than freezing into a strained inherent structure.
"""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA
from liquid_coupling_flow.ka_cavity_3d import local_displacement, local_identity_swap

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; BETA = 2.0; BIGL = 100.0; M = 8; NCAV = 10
N_SWEEP = 80; STEP = 0.06; SWAP_EVERY = 4
sig = torch.tensor(SIGMA, device=dev)

ar = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt", map_location=dev, weights_only=False)["state_dict"], strict=False)
ar.eval(); ar.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)
edges = torch.linspace(0.6, 2.6, 41); ctr = 0.5 * (edges[1:] + edges[:-1])


def clash_and_gbb(blk_x, blk_s, cage_x, cage_s):
    """clash frac (min sigma-gap<0.9, index self-excl) + B-B distance hist (block-B vs all-B)."""
    Mn, Kk = blk_x.shape[0], blk_x.shape[1]
    nbr_x = torch.cat([blk_x, cage_x], 1); nbr_s = torch.cat([blk_s, cage_s], 1)
    hits = 0; gbb = torch.zeros(len(ctr))
    for mm in range(Mn):
        d = torch.cdist(blk_x[mm], nbr_x[mm]); d[torch.arange(Kk), torch.arange(Kk)] = float("inf")
        hits += int(((d / sig[blk_s[mm][:, None], nbr_s[mm][None, :]]).min(1).values < 0.9).sum())
        bb = blk_s[mm] == 1
        if bb.any():
            dd = torch.cdist(blk_x[mm][bb], nbr_x[mm][nbr_s[mm] == 1]).flatten(); dd = dd[dd > 0.05]
            gbb += torch.histc(dd.cpu(), bins=len(ctr), min=0.6, max=2.6)
    return hits / (Mn * Kk), gbb


def relax(blk_x, blk_s, cage_x, cage_s):
    """T=0.5 frozen-cage MC on the block. Positions are CENTER-RELATIVE (cavity centre at origin, coords
    can be negative), so shift the whole config to the big-box centre (+BIGL/2, all positive) before MC to
    avoid remainder() wrapping negatives, and confine the mobile block to |x-centre|<R_WALL (hard wall the
    move supports) so declashed particles can't evaporate into the vacuum. Energy min-image (extent ~5 <<
    BIGL/2=50) is unaffected by the shift. Returns the relaxed block, shifted back to centre-relative."""
    Mn, Kk = blk_x.shape[0], blk_x.shape[1]; m = cage_x.shape[1]
    off = BIGL / 2
    x = (torch.cat([blk_x, cage_x], 1).float() + off).clone(); s = torch.cat([blk_s, cage_s], 1).long().clone()
    mobile = torch.zeros(Mn, Kk + m, dtype=torch.bool, device=dev); mobile[:, :Kk] = True
    ctr_w = torch.full((Mn, 3), off, device=dev, dtype=x.dtype); R_WALL = R + 0.3
    U = ka_energy(x, s, BIGL)
    for sw in range(N_SWEEP):
        for _ in range(Kk):
            x, U, _ = local_displacement(x, s, U, mobile, BETA, BIGL, STEP, center=ctr_w, R=R_WALL)
        if sw % SWAP_EVERY == 0:
            s, U, _ = local_identity_swap(x, s, U, mobile, BETA, BIGL)   # returns (species, U, acc); x unchanged
    return x[:, :Kk] - off, s[:, :Kk]


e_raw, e_rel, e_dat, cl_raw, cl_rel, spread_rel, spread_raw, valid_yield = [], [], [], [], [], [], [], []
gbb = {"raw": torch.zeros(len(ctr)), "rel": torch.zeros(len(ctr)), "dat": torch.zeros(len(ctr))}
ncav = 0
for ci in range(16):
    while True:
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev); seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    xo_b = xo[None].expand(M, n, 3).contiguous(); so_b = so[None].expand(M, n).contiguous()
    xa, sa, _ = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    ar_blk, sp_blk = xa[:, blk], sa[:, blk]
    cage_x = torch.cat([bnd[None].expand(M, -1, -1), xa[:, ~blk]], 1); cage_s = torch.cat([sb[None].expand(M, -1), sa[:, ~blk]], 1)
    data_blk = xo[None, blk].expand(M, K, 3).contiguous(); data_s = so[None, blk].expand(M, K).contiguous()
    Ed = float(ka_energy(torch.cat([data_blk[:1], cage_x[:1]], 1).double(), torch.cat([data_s[:1], cage_s[:1]], 1).long(), BIGL)[0]) / K
    rel_blk, rel_s = relax(ar_blk, sp_blk, cage_x, cage_s)
    def epp(b, s_):
        return (ka_energy(torch.cat([b, cage_x], 1).double(), torch.cat([s_, cage_s], 1).long(), BIGL) / K - Ed)
    e_raw += epp(ar_blk, sp_blk).clamp(max=1e4).tolist(); e_rel += epp(rel_blk, rel_s).tolist(); e_dat.append(0.0)
    cr, gr = clash_and_gbb(ar_blk, sp_blk, cage_x, cage_s); cl_raw.append(cr); gbb["raw"] += gr
    ce, ge = clash_and_gbb(rel_blk, rel_s, cage_x, cage_s); cl_rel.append(ce); gbb["rel"] += ge
    _, gd = clash_and_gbb(data_blk, data_s, cage_x, cage_s); gbb["dat"] += gd
    spread_raw.append(float((ar_blk - ar_blk.mean(0)).norm(dim=-1).mean()))
    spread_rel.append(float((rel_blk - rel_blk.mean(0)).norm(dim=-1).mean()))
    er = epp(rel_blk, rel_s)
    valid_yield.append(float(((er < 0.5) & (torch.tensor(ce, device=dev) < 1)).float().mean()) if False else float((er < 0.5).float().mean()))
    ncav += 1
    if ncav >= NCAV:
        break

shellv = ctr ** 2                                                   # 4*pi*r^2 dr up to a constant
pk = lambda h: float(ctr[(h / shellv).argmax()])                   # volume-NORMALIZED g_BB peak (structural)
print(f"=== FROZEN-CAGE T=0.5 RELAXATION OF AR BLOCKS ({ncav} held cages, M={M}, {N_SWEEP} sweeps) ===", flush=True)
print(f"(A) energy/particle above data:  raw-AR median {st.median(e_raw):+8.1f}   relaxed {st.median(e_rel):+6.2f}   data 0.0", flush=True)
print(f"    clash%<0.9:                   raw-AR {100*st.mean(cl_raw):.1f}%   relaxed {100*st.mean(cl_rel):.1f}%", flush=True)
print(f"    g_BB peak r:                  raw-AR {pk(gbb['raw']):.2f}   relaxed {pk(gbb['rel']):.2f}   data {pk(gbb['dat']):.2f}", flush=True)
print(f"(B) per-cage RMS spread across M: raw-AR {st.median(spread_raw):.3f}   relaxed {st.median(spread_rel):.3f}   (breadth for CFM)", flush=True)
print(f"    valid-target yield (<0.5/part above data): {100*st.mean(valid_yield):.0f}% of relaxed blocks", flush=True)
torch.save({"e_raw": e_raw, "e_rel": e_rel, "cl_raw": cl_raw, "cl_rel": cl_rel, "gbb": gbb, "ctr": ctr,
            "spread_raw": spread_raw, "spread_rel": spread_rel, "valid_yield": valid_yield}, "reports/logs-2026-07-13/diag_relax_ar_targets.pt")
print("saved -> reports/logs-2026-07-13/diag_relax_ar_targets.pt", flush=True)
