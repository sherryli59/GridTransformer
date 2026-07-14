"""USER IDEA: replace TWO well-separated blobs at once so the count-preserving AR can redistribute species
NON-LOCALLY (species from region 1 <-> region 2). Test the minimal case: a 2-particle block = one A + one B
that are well-separated (>2.5 sigma). block_mask supports arbitrary subsets, so this is a valid exact move.
The COMBINED budget is {1A,1B}; the AR assigns it to the two slots conditioned on each local pocket -> it
CAN swap (slot that was A gets B and vice versa) if the geometry prefers it. Measure: P(species assignment
SWAPPED) over samples, and exact MTM acceptance -- vs the single-particle move (budget=1 -> species FORCED,
swap impossible). If P(swap)>0 with nonzero acceptance, the two-blob move realizes a non-local learned swap.
FT block-cond knn24, R=2.5, N=4096."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 32; NCAV = 12
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
def run():
    pswap, accs, sep = [], [], []
    gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 12:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]
        Aidx = (so == 0).nonzero().squeeze(1); Bidx = (so == 1).nonzero().squeeze(1)
        if len(Aidx) == 0 or len(Bidx) == 0:
            continue
        # pick a well-separated A-B pair
        found = None
        for _ in range(40):
            i = int(Aidx[torch.randint(len(Aidx), (), generator=gen, device=dev)])
            j = int(Bidx[torch.randint(len(Bidx), (), generator=gen, device=dev)])
            if (xo[i] - xo[j]).norm() > 2.5:
                found = (i, j); break
        if found is None:
            continue
        i, j = found; sep.append(float((xo[i] - xo[j]).norm()))
        blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[i] = True; blk[j] = True
        bidx = blk.nonzero().squeeze(1)                                          # morton-sorted 2 slots
        s_orig = so[bidx]                                                        # original (A,B) or (B,A) in morton order
        Xp, Sp, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, R, gen=gen)
        s_new = Sp[:, bidx]                                                      # [M,2] regenerated species
        swapped = (s_new[:, 0] != s_orig[0]).float().mean()                     # slot-0 species flipped => swap
        pswap.append(float(swapped))
        # exact MTM acceptance of the 2-block move
        Ep = energy_b(Xp, Sp, bnd, sb); E0 = energy_b(xo[None], so[None], bnd, sb)[0]
        up = -BETA * Ep - lqf; u0 = (-BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R))[0]
        sf = torch.logsumexp(up, 0); J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0
        accs.append(float(torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))))
        ncav += 1
        if ncav >= NCAV:
            break
    return st.mean(pswap), st.mean(accs), st.mean(sep)


ps, acc, sep = run()
print(f"=== two-blob (A+B, well-separated) non-local species swap test (FT knn24, {NCAV} pairs) ===", flush=True)
print(f"  mean pair separation: {sep:.2f} sigma (well-separated => species exchange is NON-LOCAL)", flush=True)
print(f"  P(species assignment SWAPPED across the two slots): {100*ps:.0f}%", flush=True)
print(f"  exact 2-block MTM acceptance: {100*acc:.0f}%", flush=True)
print(f"\n  single-particle move: budget=1 => species FORCED => P(swap)=0 by construction.", flush=True)
print(f"  => two-blob move {'REALIZES' if ps > 0.02 else 'does NOT realize'} non-local species redistribution.", flush=True)
