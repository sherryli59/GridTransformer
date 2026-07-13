"""Is the flow-MTM 0/96 acceptance a LIKELIHOOD BUG or the overconfidence wall? Four concrete checks
(never conclude 'no bug' without them). Uses the repprior flow + rho115 AR on held cavities, K=8 R=2.0.

C1 APPLICATION-LEVEL ROUND-TRIP (the untested exactness): forward-sample y=flow(x0) with composed
   logq_fwd = logq_ar(x0)+logdet_fwd; then score y AS IF a current MTM state via the ladder's reverse
   formula lq_rev = block_log_prob_b(reverse_flow(y)) - ld_rev. Same point y -> lq_rev must == logq_fwd.
   Mismatch => the ladder's reverse likelihood (sign / AR re-score / integrator) is buggy.
C2 GUARD-REJECT BREAKDOWN on real DATA states: reverse-flow each data block, categorize
   {ODE-fail, outside-ball, outside-gate(0.55), valid}. This is the 36/96 the ladder reported.
C3 REVERSE-IMAGE ROUND-TRIP on data: x0_rev=reverse(x_data); x_back=forward(x0_rev); |x_back-x_data|.
   Large => the ODE inverse is inaccurate AT the data states we must score (would corrupt ld_rev).
C4 OVERCONFIDENCE MAGNITUDE: for valid moves, logq(flow trial y) vs logq(data x). If logq(y) >> logq(x)
   systematically, tiny trial weights are EXPECTED (concentration wall, not a bug).
"""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_cavity_egnn import CavityBlockFlow
from liquid_coupling_flow.ka3d_gated_base import GatedARBase, gate_pass
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; N_CAGE = 48; DUMMY_R = 50.0; GATE = 0.55; NCAV = 8
CKPT = "liquid_coupling_flow/artifacts/ka3d_cavity_egnn_flow_repprior_best.pt"

ar_base = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
ar_base.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                                   map_location=dev, weights_only=False)["state_dict"], strict=False)
ar_base.eval(); ar_base.use_frame = False
ar = GatedARBase(ar_base, cut=GATE)
ck = torch.load(CKPT, map_location=dev, weights_only=False); a = ck["args"]
flow = CavityBlockFlow(k=a["k"], n_cage=a["n_cage"], r_c=a.get("r_c", 2.5), hidden_nf=a["hidden_nf"],
                       n_layers=a["n_layers"], n_species=2, max_neighbors=a["max_neighbors"],
                       rep_prior=a.get("rep_prior", False), ode_rtol=1e-5, ode_atol=1e-5, max_steps=1000).to(dev)
flow.load_state_dict(ck["state_dict"]); flow.eval()
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
gen = torch.Generator(device=dev).manual_seed(0)


def fixed_size_cage(cx, cs, cen, ncg):
    Mn, m = cx.shape[0], cx.shape[1]
    if m >= ncg:
        idx = torch.topk((cx[0] - cen).norm(dim=-1), ncg, largest=False).indices
        return cx[:, idx], cs[:, idx]
    pad = ncg - m
    dp = torch.tensor([DUMMY_R, 0., 0.], device=dev, dtype=cx.dtype)[None, None].expand(Mn, pad, 3)
    return torch.cat([cx, dp], 1), torch.cat([cs, torch.zeros(Mn, pad, dtype=cs.dtype, device=dev)], 1)


def setup(ci):
    while True:
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] >= K + 6:
            break
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
    seed = int(torch.randint(n, (), generator=gen, device=dev))
    blk = torch.zeros(n, dtype=torch.bool, device=dev)
    blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
    return xo, so, blk, bnd, sb, n


c1, c3, c4_qy, c4_qx = [], [], [], []
cats = {"valid": 0, "ode_fail": 0, "outside_ball": 0, "outside_gate": 0}
for ci in range(16):
    xo, so, blk, bnd, sb, n = setup(ci)
    Mn = 8
    xo_b = xo[None].expand(Mn, n, 3).contiguous(); so_b = so[None].expand(Mn, n).contiguous()
    xo_ar, so_ar, lqf = ar.sample_block_b(xo_b, so_b, blk, bnd, sb, R, gen=gen, pos_temp=PT)
    ar_blk, sp_blk = xo_ar[:, blk], so_ar[:, blk]
    cage_x = torch.cat([bnd[None].expand(Mn, -1, -1), xo_ar[:, ~blk]], 1)
    cage_s = torch.cat([sb[None].expand(Mn, -1), so_ar[:, ~blk]], 1)
    cfx, cfs = fixed_size_cage(cage_x, cage_s, xo[blk].mean(0), N_CAGE)
    # C1: forward-sample then reverse-score the SAME point
    y, ld_fwd = flow.flow(ar_blk, cfx, sp_blk, cfs, reverse=False)
    logq_fwd = lqf + ld_fwd                                       # composed logq of y
    x0_rev, ld_rev = flow.flow(y, cfx, sp_blk, cfs, reverse=True)
    # re-score x0_rev under the tempered AR (write into a full config, score the block)
    Xrev = xo_ar.clone(); Xrev[:, blk] = x0_rev
    valid_rev = x0_rev.norm(dim=-1).max(-1).values < R * (1 - 1e-4)
    if valid_rev.any():
        idx = valid_rev.nonzero(as_tuple=True)[0]
        lq_ar_rev = ar.block_log_prob_b(Xrev[idx], so_ar[idx], blk, bnd, sb, R, pos_temp=PT)
        lq_rev = lq_ar_rev - ld_rev[idx]
        c1 += (lq_rev - logq_fwd[idx]).abs().tolist()

    # C2/C3/C4 on the DATA state (the MTM current-state role)
    data_blk = xo[None, blk].contiguous(); data_s = so[None, blk].contiguous()
    cage0 = torch.cat([bnd[None], xo[None, ~blk]], 1); cage0s = torch.cat([sb[None], so[None, ~blk]], 1)
    cfx0, cfs0 = fixed_size_cage(cage0, cage0s, xo[blk].mean(0), N_CAGE)
    try:
        x0d, ld_revd = flow.flow(data_blk, cfx0, data_s, cfs0, reverse=True)
    except AssertionError:
        cats["ode_fail"] += 1; continue
    if x0d.norm(dim=-1).max() >= R * (1 - 1e-4):
        cats["outside_ball"] += 1; continue
    if not gate_pass(x0d, data_s, cage0, cage0s, GATE)[0]:
        cats["outside_gate"] += 1; continue
    cats["valid"] += 1
    # C3: reverse-image round-trip
    xback, _ = flow.flow(x0d, cfx0, data_s, cfs0, reverse=False)
    c3.append(float((xback - data_blk).norm(dim=-1).max()))
    # C4: logq(data) vs logq(flow trial). logq(data) via reverse formula:
    Xrevd = xo[None].clone(); Xrevd[:, blk] = x0d
    lq_data = float((ar.block_log_prob_b(Xrevd, so[None], blk, bnd, sb, R, pos_temp=PT) - ld_revd)[0])
    c4_qx.append(lq_data / K); c4_qy.append(float(logq_fwd.median()) / K)

print(f"=== FLOW-LIKELIHOOD DIAGNOSTIC (repprior, {sum(cats.values())} data states, K={K}) ===", flush=True)
print(f"C1 application round-trip |lq_rev - logq_fwd|: median {st.median(c1):.2e}  max {max(c1):.2e}  "
      f"(n={len(c1)})  -> BUG if >>1e-2", flush=True)
print(f"C2 data-state reverse-image categories: {cats}", flush=True)
print(f"C3 reverse->forward round-trip max|dx| on data: median {st.median(c3) if c3 else float('nan'):.2e}  "
      f"max {max(c3) if c3 else float('nan'):.2e}  -> integrator inaccurate if >>1e-3", flush=True)
if c4_qx:
    print(f"C4 logq/particle: data-state median {st.median(c4_qx):+.2f}  flow-trial median {st.median(c4_qy):+.2f}  "
          f"gap {st.median(c4_qy)-st.median(c4_qx):+.2f}  -> overconfidence wall if flow>>data", flush=True)
torch.save({"c1": c1, "c3": c3, "c4_qx": c4_qx, "c4_qy": c4_qy, "cats": cats}, "reports/logs-2026-07-13/diag_flow_likelihood.pt")
print("saved -> reports/logs-2026-07-13/diag_flow_likelihood.pt", flush=True)
