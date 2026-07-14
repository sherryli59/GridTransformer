"""Does a larger K-block redistribute species MORE? Regenerate K-blobs from data, compare regenerated species
(per morton slot) to the original data species. Two metrics:
  RAW  = fraction of block slots whose species changed vs data (the shuffle of the fixed multiset).
  NORM = RAW / max_possible, max = 2*min(nA,nB)/K (most you can change given the block's own A:B counts).
NORM controls for the TRIVIAL reason bigger blocks show more redistribution (they contain more mixed
multisets). If NORM rises with K => genuinely MORE redistribution per available swap; flat => the extra is
just composition, per-slot the AR stays geometry-slaved. Also report block mixedness min(nA,nB)/K and MTM
acceptance. block-cond FT knn24, R=2.5, M=24."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 24; NCAV = 12
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


def energy_b(Xb, Sb, bnd, sb):
    Mn, mm = Xb.shape[0], bnd.shape[0]
    x = torch.cat([Xb, bnd[None].expand(Mn, mm, 3)], 1); s = torch.cat([Sb, sb[None].expand(Mn, mm)], 1)
    return ka_energy(x.double(), s.long(), BIGL).float()


@torch.no_grad()
def measure(K, gen):
    raw, norm, mixed, acc = [], [], [], []
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < K + 6:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=gen, device=dev))
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(K, largest=False).indices] = True
        bidx = blk.nonzero().squeeze(1); s0 = so[bidx]                          # data species (morton order)
        nA = int((s0 == 0).sum()); nB = K - nA; mx = min(nA, nB) / K
        if mx == 0:                                                             # single-species block: no redist possible
            continue
        Xp, Sp, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, R, gen=gen)
        changed = (Sp[:, bidx] != s0[None]).float().mean(1)                     # [M] per-slot change frac
        raw.append(float(changed.mean())); mixed.append(mx)
        norm.append(float(changed.mean()) / (2 * mx))                          # 2*min(nA,nB)/K = max changeable
        Ep = energy_b(Xp, Sp, bnd, sb); E0 = energy_b(xo[None], so[None], bnd, sb)[0]
        up = -BETA * Ep - lqf; u0 = (-BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R))[0]
        sf = torch.logsumexp(up, 0); J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0
        acc.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(raw), st.mean(norm), st.mean(mixed), 100 * st.mean(acc)


print(f"=== species redistribution vs K (block-cond FT knn24, R={R}, M={M}) ===", flush=True)
print(f"{'K':>3} | {'mixedness':>9} {'RAW redist':>11} {'NORM redist':>12} {'MTM acc':>8}", flush=True)
for K in (2, 4, 8, 16, 24):
    raw, norm, mixed, acc = measure(K, torch.Generator(device=dev).manual_seed(0))
    print(f"{K:>3} | {mixed:>9.2f} {100*raw:>10.0f}% {100*norm:>11.0f}% {acc:>7.0f}%", flush=True)
print("\n  RAW = % of block slots whose species changed. NORM = RAW / (2*min(nA,nB)/K) controls for multiset."
      "\n  NORM rising with K => genuinely more per-swap redistribution; flat => just more mixed multisets.", flush=True)
