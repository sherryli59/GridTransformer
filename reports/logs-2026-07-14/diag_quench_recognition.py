"""T=0 QUENCH RECOGNITION: is the INHERENT-STRUCTURE energy the basin label thermal energy failed to be?
Deep-relax-by-MC failed to reveal basins (corr(q~,U)~0 after 12 sweeps) because thermal noise swamps the
depth signal. The classical recognizer: gradient-quench each walker to its inherent structure (frozen
boundary; interior coords minimized by Adam on the differentiable KA energy) -- deterministic, no barriers
to cross, U_IS IS the basin identity. Test on data-seeded (ref basin) vs alien-seeded walkers + the alien
HIT subset (q~>0.3): does U_IS separate ref from wrong basins, and is corr(q~, U_IS) strongly negative
across alien walkers? If yes -> recognition is ~free (no NN needed): cluster/select by U_IS.
M=48/arm, 200 Adam steps, R=2.0, 2 cavities."""
import sys, statistics as st, functools
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from ka3d_smc_sweep import energy_b
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.0; ART = "liquid_coupling_flow/artifacts"; BIGL = 100.0
L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3; M = 48; QSTEPS = 400
m = KA3DScaffoldEBMBatched(cat_bins=128, cat_range=2.5, knn_pot=24, use_demand=True).to(dev)
m.load_state_dict(torch.load(f"{ART}/ka3d_cavity_ebm3ax_rho115_rl_Rext_knn24_lam05_best.pt",
                             map_location=dev, weights_only=False)["state_dict"], strict=False)
m.eval(); m.use_frame = False
D = torch.load(f"{ART}/ka3d_dataset_N4096_T0.5_rho1.15.pt", map_location=dev, weights_only=False)
X, S, L = D["x"].to(dev).float(), D["s"].to(dev).long(), float(D["L"])


@functools.lru_cache(maxsize=16)
def n_boxes(l, R):
    g = torch.arange(-int(R / l) - 1, int(R / l) + 2) * l + l / 2
    gx, gy, gz = torch.meshgrid(g, g, g, indexing="ij")
    return int((gx ** 2 + gy ** 2 + gz ** 2 < R * R).sum())


def box_id_set(x, l, R):
    mm = x.norm(dim=-1) < R; ijk = torch.floor(x[mm] / l).long(); off = ijk - ijk.min(0).values
    span = off.max(0).values + 1
    return set((off[:, 0] * span[1] * span[2] + off[:, 1] * span[2] + off[:, 2]).tolist())


def qtil(Xb, xin, R):
    ref = box_id_set(xin, L_BOX, R); Nb = n_boxes(L_BOX, R); Xc = Xb.detach().cpu()
    return torch.tensor([len(ref & box_id_set(Xc[k], L_BOX, R)) / (L_BOX ** 3 * Nb) - BULK for k in range(Xb.shape[0])])


def quench(Xb, Sb, bnd, sb, steps=QSTEPS, dmax=0.05):
    """DISPLACEMENT-CAPPED gradient quench (FIRE-lite): per step each particle moves along -grad by at most
    dmax*sigma (no runaway on the r^-12 core; plain Adam diverged 1e15). float64, lr-decayed cap."""
    Mn, n, _ = Xb.shape; mm = bnd.shape[0]
    xi = Xb.double().clone()
    bndd = bnd.double(); sfull = torch.cat([Sb, sb[None].expand(Mn, mm)], 1).long()
    for t in range(steps):
        xi = xi.detach().requires_grad_(True)
        xfull = torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1)
        U = ka_energy(xfull, sfull, BIGL)
        (gr,) = torch.autograd.grad(U.sum(), xi)
        step = dmax * (0.3 + 0.7 * (1 - t / steps))                       # decaying cap
        gn = gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        disp = -gr / gn * torch.minimum(gn * 1e-3, torch.full_like(gn, step))  # capped steepest descent
        xi = xi + disp
    with torch.no_grad():
        xfull = torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1)
        return ka_energy(xfull, sfull, BIGL).float(), xi.detach().float()


gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    g = torch.Generator(device=dev).manual_seed(100 + ci)
    with torch.no_grad():
        Xa, Sa, _ = m.sample_block_b(xo[None].expand(M, n, 3).clone(), so[None].expand(M, n).clone(),
                                     allm, bnd, sb, R, gen=g)
    Xd = xo[None].expand(M, n, 3).clone(); Sd = so[None].expand(M, n).clone()
    qa0 = qtil(Xa, xin, R)
    Uis_d, _ = quench(Xd, Sd, bnd, sb)
    Uis_a, Xa_q = quench(Xa, Sa, bnd, sb)
    qa_q = qtil(Xa_q, xin, R)                                    # q~ AFTER quench (basins held?)
    ud, ua = Uis_d / n, Uis_a / n
    hit = qa0 > 0.30
    corr = float(torch.corrcoef(torch.stack([qa0.to(dev), Uis_a]))[0, 1])
    sep = float(ua.mean() - ud.mean()); noise = float((ud.std()**2/M + ua.std()**2/M)**.5)
    print(f"=== cav {ci} (n={n})  U_IS/n after {QSTEPS}-step quench ===", flush=True)
    print(f"  data-seed (ref basin) : {float(ud.mean()):8.3f} +- {float(ud.std()):.3f}", flush=True)
    print(f"  alien all             : {float(ua.mean()):8.3f} +- {float(ua.std()):.3f}   sep/noise {sep/max(noise,1e-9):.1f}", flush=True)
    if hit.any():
        print(f"  alien HITS (q~>{0.3}) n={int(hit.sum()):2d}: {float(ua[hit].mean()):8.3f} +- {float(ua[hit].std()):.3f}", flush=True)
        print(f"  alien non-hits        : {float(ua[~hit].mean()):8.3f} +- {float(ua[~hit].std()):.3f}", flush=True)
    print(f"  corr(q~_seed, U_IS) over alien walkers: {corr:+.2f}   (strongly negative => U_IS recognizes)", flush=True)
    print(f"  q~ of alien hits after quench: {float(qa_q[hit].mean()) if hit.any() else float('nan'):.3f} (held?)", flush=True)
    ncav += 1
    if ncav >= 2:
        break
