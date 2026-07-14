"""USER IDEA: to move 8 particles cleanly, 'pretend it's K=12, keep the first 8'. Since the clash spike is
POSITIONAL (last members inherit the compounded deficit; TF proved the conditional itself is fine), padding
pushes the spike onto discarded ranks 9-12, leaving the morton-first-8 clean. CATCH: the discarded 4 revert
to DATA, and the kept 8 were generated NOT seeing them (they were future) -> the 8 may clash with the
reset-4. Measure clash/particle (r<0.85 sigma) and dE/particle for the MOVED 8 under:
  (a) standalone K=8   : block = 8 nearest anchors, generate 8. (the baseline to beat)
  (b) pad12 first8 IDEAL: block = 12 nearest, generate 12, look at first-8's clash with ALL-FRESH-12
                          (best case -- if the 4 didn't need resetting)
  (c) pad12 first8 RESET: generate 12, keep first-8 fresh, revert last-4 to DATA; first-8 clash vs
                          retained + boundary + reset-4-data + each other (the ACTUAL usable move)
Exactness note: (c) is a valid exact proposal on the 8 -- their density q(x1..8 | retained_n-12, boundary)
is the first-8 causal prefix of a 12-block (independent of the hidden 4); forward+reverse both hide the 4.
FT model knn24, R=2.5, 12 cav x 16 samp."""
import torch, statistics as st
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold, fixed_ball_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy, SIGMA

dev = "cuda"; RCTX = 2.5; R = 2.5; BIGL = 100.0; BETA = 2.0; M = 16; NCAV = 12; CUT = 0.85
SIG = torch.tensor(SIGMA, device=dev)
FT = "liquid_coupling_flow/artifacts/ka3d_cavity_ebm3ax_rho115_blockcond_knn24_best.pt"
D = torch.load("liquid_coupling_flow/artifacts/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24).to(dev)
m.load_state_dict(torch.load(FT, map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False


def Efn(Xi, Si): return ka_energy(Xi.double(), Si.long(), BIGL).float()


def clash_of(idx, Xfull, Sfull, bnd, sb):
    """mean clashes per particle in `idx` vs ALL other interior + boundary."""
    xi, si = Xfull[idx], Sfull[idx]
    allx = torch.cat([Xfull, bnd]); alls = torch.cat([Sfull, sb])
    d = torch.cdist(xi, allx); sg = SIG[si[:, None], alls[None, :]]
    cl = d < CUT * sg
    # remove self (each idx particle matches itself at distance 0)
    for j, ii in enumerate(idx.tolist()):
        cl[j, ii] = False
    return float(cl.sum(1).float().mean())


@torch.no_grad()
def run():
    res = {"a": {"cl": [], "de": []}, "b": {"cl": []}, "c": {"cl": [], "de": []}}
    g = torch.Generator(device=dev).manual_seed(0); ncav = 0
    for ci in range(16):
        c = torch.rand(3, generator=g, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
        if p["n_in"] < 18:
            continue
        xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
        xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
        n = xo.shape[0]; a = fixed_ball_scaffold(n, R, dev, xo.dtype, ordering=m.scaffold_order)
        seed = int(torch.randint(n, (), generator=g, device=dev))
        near = (a - a[seed]).norm(dim=-1).topk(12, largest=False).indices
        b12 = torch.zeros(n, dtype=torch.bool, device=dev); b12[near] = True
        b12idx = b12.nonzero().squeeze(1)                       # ascending morton = rank order
        first8, last4 = b12idx[:8], b12idx[8:]
        b8 = torch.zeros(n, dtype=torch.bool, device=dev); b8[first8] = True   # standalone on SAME first-8 anchors
        E0 = Efn(torch.cat([xo, bnd])[None], torch.cat([so, sb])[None])[0]
        # (a) standalone K=8 on first8
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     b8, bnd, sb, R, gen=g)
        # (b,c) pad-12
        Xp, Sp, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     b12, bnd, sb, R, gen=g)
        for k in range(M):
            res["a"]["cl"].append(clash_of(first8, Xa[k], Sa[k], bnd, sb))
            Ea = Efn(torch.cat([Xa[k], bnd])[None], torch.cat([Sa[k], sb])[None])[0]
            res["a"]["de"].append(float((Ea - E0) / 8))
            res["b"]["cl"].append(clash_of(first8, Xp[k], Sp[k], bnd, sb))     # all-12-fresh
            Xc = Xp[k].clone(); Sc = Sp[k].clone(); Xc[last4] = xo[last4]; Sc[last4] = so[last4]  # reset tail
            res["c"]["cl"].append(clash_of(first8, Xc, Sc, bnd, sb))
            Ec = Efn(torch.cat([Xc, bnd])[None], torch.cat([Sc, sb])[None])[0]
            res["c"]["de"].append(float((Ec - E0) / 8))
        ncav += 1
        if ncav >= NCAV:
            break
    return res


r = run()
print(f"=== pad-and-discard: MOVED-8 clash/energy (FT knn24, R={R}, {NCAV} cav) ===", flush=True)
print(f"  (a) standalone K=8          : clash/p {st.mean(r['a']['cl']):.3f}   dE/p {st.median(r['a']['de']):+8.1f}", flush=True)
print(f"  (b) pad12 first-8, ALL fresh: clash/p {st.mean(r['b']['cl']):.3f}   (ideal, no reset)", flush=True)
print(f"  (c) pad12 first-8, tail RESET: clash/p {st.mean(r['c']['cl']):.3f}   dE/p {st.median(r['c']['de']):+8.1f}  <- usable move", flush=True)
print(f"\n  reset cost (c-b) = {st.mean(r['c']['cl'])-st.mean(r['b']['cl']):+.3f} clash/p; "
      f"padding win (a-c) = {st.mean(r['a']['cl'])-st.mean(r['c']['cl']):+.3f} clash/p, "
      f"{st.median(r['a']['de'])-st.median(r['c']['de']):+.1f} dE/p", flush=True)
