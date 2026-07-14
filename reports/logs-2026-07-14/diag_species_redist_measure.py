"""CONCRETE EVIDENCE: does regenerating two well-separated blobs redistribute species BETWEEN them? (user
challenge: the 2% 'no-swap' result was the 2-particle minimal case; large blobs may differ.) sample_block_b
on the UNION preserves the union's total A:B budget but freely assigns species to slots -> it CAN move A's to
one blob and B's to the other. Measure, per blob size K in {4,8,12,16}, two well-separated K-blobs:
  redist = |new(#A in blob1) - orig(#A in blob1)|   averaged over M regens x cavities  (species moved between blobs)
  frac_changed = fraction of union slots whose species flipped after regen
compared against:
  RANDOM ceiling = same metric if the union species were shuffled uniformly (max redistribution given budget)
  ZERO floor     = no redistribution
Also report exact MTM acceptance of the regen (is the redistributed config energetically viable?). If model
redist ~ random ceiling => strong redistribution; if ~0 => geometry-pinned. lam05, l match not needed. R=2.5."""
import sys, statistics as st
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve

dev = "cuda"; RCTX = 2.5; R = 2.5; BETA = 2.0; ART = "liquid_coupling_flow/artifacts"
CK = f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt"
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(CK, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
M = 32; NCAV = 8


@torch.no_grad()
def measure(K, gen):
    redist, fracch, sep, accs, randred = [], [], [], [], []
    ncav = 0
    for ci in range(40):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        n = p["n_in"]
        if n < 2 * K + 4:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        n = xo.shape[0]
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        s1 = int(torch.randint(n, (), generator=gen, device=dev))
        far = (a - a[s1]).norm(dim=-1); s2 = int(far.topk(min(8, n), largest=True).indices[0])
        b1 = (a - a[s1]).norm(dim=-1).topk(K, largest=False).indices
        b2 = (a - a[s2]).norm(dim=-1).topk(K, largest=False).indices
        b1s = set(b1.tolist()); b2s = set(b2.tolist())
        if b1s & b2s:
            continue                                                  # require disjoint blobs
        U = torch.zeros(n, dtype=torch.bool, device=dev); U[b1] = True; U[b2] = True
        sep.append(float((a[b1].mean(0) - a[b2].mean(0)).norm()))
        orig_b1A = float((so[b1] == 0).sum())
        # regenerate the union (normal: model assigns species with the union budget)
        Xn, Sn, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       U, bnd, sb, R, gen=gen)
        new_b1A = (Sn[:, b1] == 0).sum(1).float()                     # [M]
        redist.append(float((new_b1A - orig_b1A).abs().mean()))
        fracch.append(float((Sn[:, U] != so[U][None]).float().mean()))
        # random-shuffle ceiling: permute the union species uniformly
        rr = []
        for _ in range(M):
            perm = torch.randperm(int(U.sum()), generator=gen, device=dev)
            sh = so[U][perm]
            # map shuffled back: first |b1| entries of U-order aren't b1; do it by index
            su = so.clone(); su[U] = sh
            rr.append(abs(float((su[b1] == 0).sum()) - orig_b1A))
        randred.append(st.mean(rr))
        # exact MTM acceptance of the union regen (vs current)
        En = energy_b(Xn, Sn, bnd, sb); E0 = energy_b(xo[None], so[None], bnd, sb)[0]
        lq0 = m.block_log_prob_b(xo[None], so[None], U, bnd, sb, R)[0]
        up = -BETA * En - lqf; u0 = -BETA * E0 - lq0
        sf = torch.logsumexp(up, 0); Jsel = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[Jsel] = u0
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        ncav += 1
        if ncav >= NCAV:
            break
    return (st.mean(redist), st.mean(randred), st.mean(fracch), st.mean(sep), st.mean(accs))


print(f"=== species redistribution when regenerating TWO separated K-blobs (concrete evidence) ===", flush=True)
print(f"redist = |#A in blob1 change| after union regen; RANDOM = shuffle ceiling; 0 = none. M={M} ncav={NCAV}", flush=True)
print(f"{'K':>4} {'2K':>4} {'sep':>5} | {'model_redist':>12} {'random_ceiling':>14} {'frac_changed':>12} | {'MTM_acc':>7}", flush=True)
for K in (4, 8, 12, 16):
    rd, rr, fc, sp, ac = measure(K, torch.Generator(device=dev).manual_seed(0))
    print(f"{K:>4} {2*K:>4} {sp:>5.1f} | {rd:>12.3f} {rr:>14.3f} {fc:>12.3f} | {ac:>7.3f}", flush=True)
print(f"\nmodel_redist ~ random_ceiling => STRONG redistribution; ~0 => geometry-pinned (no swap).", flush=True)
