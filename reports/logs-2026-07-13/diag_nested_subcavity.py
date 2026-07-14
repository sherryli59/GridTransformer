"""USER PROPOSAL test: 'maybe the K particles shouldn't be scattered -- resample within a really small
cavity instead'. Fact 1: the production block already IS a blob (K nearest scaffold anchors to a seed).
Fact 2: the sharper variant -- regenerate a NESTED SUB-CAVITY as a fresh cavity task -- differs in real
ways: in-distribution (training = 'fill ball radius r given frozen shell', radii 1.6-3.0), fixed
surroundings enter through the model's BOUNDARY stream, and regeneration is confined to the sub-ball
(ball_squash at r_sub) which makes position-based selection exactly reversible (classic cavity-regen MC).

Head-to-head at matched particle count, same cavities (R_big=2.5, N=4096 data, isolated-cavity energies):
  ARM A (production): anchor-blob block of K=n_sub via sample_block_b on the big problem (reordered
         block-conditional; regen confined to the BIG ball).
  ARM B (nested):     the same region as a fresh sub-cavity problem: interior = particles within r_sub of
         a random sub-center (sub-ball fully inside the big cavity), boundary = ALL other fixed particles
         (retained big-interior + big-boundary) within r_sub+2.5; full regen (allmask) at R=r_sub=1.6.
Metrics per regenerated particle: clash vs FIXED context, clash INTERNAL (among regenerated), median
dE/particle, exact tempered MTM acceptance (16 trials). Both arms exact; species multisets preserved.
PREDICTION (honest): the internal half-cage is framing-independent, so clash/dE similar and MTM 0% at
n~20 (the +1.4/particle clash-free floor => ~60 nats); the test is whether in-distribution conditioning
buys anything measurable."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; RBIG = 2.5; RSUB = 1.6; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5).to(dev)
m.load_state_dict(torch.load("liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


def energy_set(Xi, Si):
    """Isolated-cavity energy of [Mn, ntot, 3] sets (fixed-fixed terms identical across arms -> cancel in dE)."""
    return ka_energy(Xi.double(), Si.long(), BIGL).float()


def clash_counts(regen_x, regen_s, fix_x, fix_s):
    d = torch.cdist(regen_x, fix_x); sg = SIG[regen_s[:, None], fix_s[None, :]]
    cfix = float((d < CUT * sg).sum(1).float().mean())
    n = regen_x.shape[0]
    d = torch.cdist(regen_x, regen_x); sg = SIG[regen_s[:, None], regen_s[None, :]]
    eye = torch.eye(n, dtype=torch.bool, device=dev)
    cint = float(((d < CUT * sg) & ~eye).sum(1).float().mean())
    return cfix, cint


def mtm_accept(up, u0, gen):
    sf = torch.logsumexp(up, 0)
    J = int(torch.multinomial(torch.softmax(up, 0), 1, generator=gen))
    lr = up.clone(); lr[J] = u0; sr = torch.logsumexp(lr, 0)
    return float(torch.rand((), device=dev, generator=gen).log() < (sf - sr))


@torch.no_grad()
def run():
    A = {"cfix": [], "cint": [], "de": [], "acc": []}
    B = {"cfix": [], "cint": [], "de": [], "acc": []}
    nsubs = []
    gen = torch.Generator(device=dev).manual_seed(0)
    ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, RBIG, L)
        if p["n_in"] < 30:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], RBIG)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (RBIG + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]
        # sub-center fully inside the big cavity (|c_sub| <= RBIG - RSUB), retry for a sane count
        for _ in range(30):
            cs = (torch.rand(3, generator=gen, device=dev) * 2 - 1) * (RBIG - RSUB)
            if cs.norm() > (RBIG - RSUB):
                continue
            mob = (xo - cs).norm(dim=-1) < RSUB
            if 12 <= int(mob.sum()) <= 28:
                break
        nsub = int(mob.sum()); nsubs.append(nsub)
        E0 = energy_set(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]

        # ---- ARM A: production anchor-blob, K = nsub, seeded at the anchor nearest c_sub ----
        a = fixed_ball_scaffold(n, RBIG, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int((a - cs).norm(dim=-1).argmin())
        blk = torch.zeros(n, dtype=torch.bool, device=dev)
        blk[(a - a[seed]).norm(dim=-1).topk(nsub, largest=False).indices] = True
        XA, SA, lqA = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                       blk, bnd, sb, RBIG, gen=gen)
        EA = energy_set(torch.cat([XA, bnd[None].expand(M, -1, 3)], 1), torch.cat([SA, sb[None].expand(M, -1)], 1))
        u0A = -BETA * E0 - m.block_log_prob_b(xo[None], so[None], blk, bnd, sb, RBIG)[0]
        A["acc"].append(mtm_accept(-BETA * EA - lqA, u0A, gen))
        A["de"].append(float((EA.median() - E0) / nsub))
        bi = blk.nonzero().squeeze(1); ri = (~blk).nonzero().squeeze(1)
        fixA_x = torch.cat([xo[ri], bnd]); fixA_s = torch.cat([so[ri], sb])
        cf, cn = zip(*[clash_counts(XA[k][bi], SA[k][bi], fixA_x, fixA_s) for k in range(M)])
        A["cfix"].append(st.mean(cf)); A["cint"].append(st.mean(cn))

        # ---- ARM B: nested sub-cavity regen at RSUB (user proposal) ----
        xin_s = xo[mob] - cs; s_in_s = so[mob]
        oth_x = torch.cat([xo[~mob], bnd]) - cs; oth_s = torch.cat([so[~mob], sb])
        shell = oth_x.norm(dim=-1) < (RSUB + RCTX)
        bnd_s, sb_s = oth_x[shell], oth_s[shell]
        xos, sos, _ = label_to_scaffold(xin_s, s_in_s, RSUB)
        allm = torch.ones(nsub, dtype=torch.bool, device=dev)
        XBs, SBs, lqB = m.sample_block_b(xos[None].expand(M, nsub, 3).clone(), sos[None].expand(M, nsub).clone(),
                                         allm, bnd_s, sb_s, RSUB, gen=gen)
        XB = XBs + cs                                                        # back to big-cavity frame
        fixB_x = torch.cat([xo[~mob], bnd]); fixB_s = torch.cat([so[~mob], sb])
        EB = energy_set(torch.cat([XB, fixB_x[None].expand(M, -1, 3)], 1),
                        torch.cat([SBs, fixB_s[None].expand(M, -1)], 1))
        u0B = -BETA * E0 - m.block_log_prob_b(xos[None], sos[None], allm, bnd_s, sb_s, RSUB)[0]
        B["acc"].append(mtm_accept(-BETA * EB - lqB, u0B, gen))
        B["de"].append(float((EB.median() - E0) / nsub))
        cf, cn = zip(*[clash_counts(XB[k], SBs[k], fixB_x, fixB_s) for k in range(M)])
        B["cfix"].append(st.mean(cf)); B["cint"].append(st.mean(cn))

        ncav += 1
        if ncav >= NCAV:
            break
    return A, B, nsubs


A, B, nsubs = run()
print(f"=== anchor-blob (production) vs NESTED SUB-CAVITY regen (user proposal), matched sets ===", flush=True)
print(f"    R_big={RBIG}, r_sub={RSUB} (min trained radius), n_sub mean {st.mean(nsubs):.1f}, {NCAV} cav x {M} samp", flush=True)
print(f"{'arm':>26} | {'clash/FIXED':>11} {'clash/INTERNAL':>14} {'dE/particle':>11} {'MTM accept':>10}", flush=True)
for name, d in (("A blob-block (big frame)", A), ("B nested sub-cavity", B)):
    print(f"{name:>26} | {st.mean(d['cfix']):>11.3f} {st.mean(d['cint']):>14.3f} "
          f"{st.median(d['de']):>+11.2f} {100*st.mean(d['acc']):>9.1f}%", flush=True)
