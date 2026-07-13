"""Nucleus (top-p) truncation of the AR position categoricals: exactness round-trip + clash effect.
Hypothesis: clashes live in the hedged low-probability tail; top-p removes it exactly (deterministic
mask from logits, identical at sample & score). Sweep top_p, report round-trip + clash%<0.9 + logq shift.
"""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import SIGMA

dev = "cuda"; RCTX = 2.5; K = 8; R = 2.0; PT = 0.4; M = 16; NCAV = 8
sig = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])

print("=== TOP-P NUCLEUS: exactness + clash (K=8, pos_temp=0.4, 8 held cavities x M=16) ===", flush=True)
for top_p in (None, 0.99, 0.95, 0.90, 0.80):
    gen = torch.Generator(device=dev).manual_seed(5)
    diffs, clash_h, clash_t, dlogq = [], 0, 0, []
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; anch = fixed_ball_scaffold(n, R, dev)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(anch - anch[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        xb = xo[None].expand(M, n, 3).contiguous(); sbb = so[None].expand(M, n).contiguous()
        xs, ss, lq_s = m.sample_block_b(xb, sbb, blk, bnd, sb, R, gen=gen, pos_temp=PT, top_p=top_p)
        lq_c = m.block_log_prob_b(xs, ss, blk, bnd, sb, R, pos_temp=PT, top_p=top_p)
        diffs += (lq_c - lq_s).abs().tolist(); dlogq += (lq_s / K).tolist()
        blk_x, blk_s = xs[:, blk], ss[:, blk]
        cage_x = torch.cat([bnd[None].expand(M, -1, -1), xs[:, ~blk]], 1)
        cage_s = torch.cat([sb[None].expand(M, -1), ss[:, ~blk]], 1)
        nbr_x = torch.cat([blk_x, cage_x], 1); nbr_s = torch.cat([blk_s, cage_s], 1)
        for mm in range(M):
            d = torch.cdist(blk_x[mm], nbr_x[mm]); d[torch.arange(K), torch.arange(K)] = float("inf")
            mn = (d / sig[blk_s[mm][:, None], nbr_s[mm][None, :]]).min(1).values
            clash_h += int((mn < 0.9).sum()); clash_t += K
        if len(diffs) >= NCAV * M:
            break
    dd = torch.tensor(diffs)
    print(f"top_p={str(top_p):5s}: roundtrip median {dd.median():.2e} match(<1e-3) {100*float((dd<1e-3).float().mean()):5.1f}%  "
          f"| clash<0.9 {100*clash_h/clash_t:5.1f}%  | logq/K med {st.median(dlogq):+.2f}", flush=True)
