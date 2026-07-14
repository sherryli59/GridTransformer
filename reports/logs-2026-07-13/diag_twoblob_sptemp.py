"""SPECIES-TEMPERATURE variant of the two-blob swap: flatten the species categorical (sp_temp>1) at proposal
time to propose more swaps; MTM filters. Kept EXACT: proposal drawn from AND scored under the same tempered
species density (sp_temp identical in sample_block_b + block_log_prob_b). Sweep sp_temp on a well-separated
A+B 2-block. Metrics: P(swap PROPOSED), MTM acceptance, and P(ACCEPTED swap) = the useful non-local species
mixing rate (accepted move whose selected trial is a swap). Higher sp_temp proposes more swaps but they may
be against-geometry (rejected) -> the question is whether ACCEPTED-swap rate rises. First: exactness check."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 32; NCAV = 16
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


def pairs(gen):
    """Yield (xo,so,bnd,sb,n,blk,bidx,s_orig) for a well-separated A-B 2-block per cavity."""
    out = []; ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 12:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; Aidx = (so == 0).nonzero().squeeze(1); Bidx = (so == 1).nonzero().squeeze(1)
        if len(Aidx) == 0 or len(Bidx) == 0:
            continue
        found = None
        for _ in range(40):
            i = int(Aidx[torch.randint(len(Aidx), (), generator=gen, device=dev)])
            j = int(Bidx[torch.randint(len(Bidx), (), generator=gen, device=dev)])
            if (xo[i] - xo[j]).norm() > 2.5:
                found = (i, j); break
        if found is None:
            continue
        i, j = found; blk = torch.zeros(n, dtype=torch.bool, device=dev); blk[i] = True; blk[j] = True
        bidx = blk.nonzero().squeeze(1)
        out.append((xo, so, bnd, sb, n, blk, bidx, so[bidx].clone()))
        ncav += 1
        if ncav >= NCAV:
            break
    return out


# exactness check (sp_temp=4)
g = torch.Generator(device=dev).manual_seed(0); P = pairs(g)
xo, so, bnd, sb, n, blk, bidx, s0 = P[0]
Xp, Sp, lqf = m.sample_block_b(xo[None].expand(4, n, 3).clone(), so[None].expand(4, n).clone(), blk, bnd, sb, R, gen=g, sp_temp=4.0)
lqs = m.block_log_prob_b(Xp, Sp, blk, bnd, sb, R, sp_temp=4.0)
print(f"exactness sp_temp=4 roundtrip |lqf-lqs| max: {float((lqf-lqs).abs().max()):.2e} (want <5e-2)", flush=True)


@torch.no_grad()
def sweep(sp_temp, gen):
    prop, acc, acc_swap = [], [], []
    for (xo, so, bnd, sb, n, blk, bidx, s0) in pairs(gen):
        Xp, Sp, lqf = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, R, gen=gen, sp_temp=sp_temp)
        swapped = (Sp[:, bidx][:, 0] != s0[0])                                   # [M] which trials are swaps
        prop.append(float(swapped.float().mean()))
        Ep = energy_b(Xp, Sp, bnd, sb); E0 = energy_b(xo[None], so[None], bnd, sb)[0]
        up = -BETA * Ep - lqf; u0 = (-BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, R, sp_temp=sp_temp))[0]
        sf = torch.logsumexp(up, 0); J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
        lr = up.clone(); lr[J] = u0
        accepted = torch.rand((), device=dev, generator=gen).log() < (sf - torch.logsumexp(lr, 0))
        acc.append(float(accepted)); acc_swap.append(float(accepted and bool(swapped[J])))
    return st.mean(prop), st.mean(acc), st.mean(acc_swap)


print(f"\n=== species-temperature two-blob swap sweep (FT knn24, {NCAV} well-sep A-B pairs) ===", flush=True)
print(f"{'sp_temp':>8} | {'P(swap proposed)':>16} {'MTM accept':>11} {'P(ACCEPTED swap)':>16}", flush=True)
for spt in (1.0, 2.0, 4.0, 8.0, 20.0):
    pr, ac, asw = sweep(spt, torch.Generator(device=dev).manual_seed(0))
    print(f"{spt:>8.1f} | {100*pr:>15.0f}% {100*ac:>10.0f}% {100*asw:>15.0f}%", flush=True)
print("\n  P(ACCEPTED swap) = the useful non-local species-mixing rate. Rising with sp_temp => the variant"
      "\n  turns the geometry-timid proposer into a working learned swap kernel.", flush=True)
