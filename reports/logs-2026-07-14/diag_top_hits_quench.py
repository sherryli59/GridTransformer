"""THE DECISIVE MEASUREMENT: does q0 EVER enter the reference attraction basin? Draw a large pool per cavity,
keep the TOP-q~ draws (the strongest geometric look-alikes, q~ up to ~0.5), quench them + the reference, and
test membership by the RIGHT criteria: (a) U_IS within 3 sigma_alien of the reference IS depth, (b) IS-to-IS
box overlap with the QUENCHED reference (fixes the thermal-vs-IS metric flaw). Also quench a random non-hit
control. Outcomes: any true member => draw+quench+select basin-finder WORKS (energy-free draws, cheap
gradient quenches, U_IS selection); none => attraction-basin coverage wall is real -> dual-seed PT.
lam05 Rext, R=2.0, cavities 0/1 (same as before), pool 4096 draws, quench top 16 + 8 controls."""
import sys, functools
import torch
sys.path.insert(0, "reports/logs-2026-07-13")
from liquid_coupling_flow.ka3d_ebm_batched import KA3DScaffoldEBMBatched
from liquid_coupling_flow.ka3d_scaffold_ar import label_to_scaffold
from liquid_coupling_flow.ka3d_cavity_ar import _mic
from liquid_coupling_flow.ka3d_cavity_carve import carve
from liquid_coupling_flow.ka_energy import ka_energy

dev = "cuda"; RCTX = 2.5; R = 2.0; ART = "liquid_coupling_flow/artifacts"; BIGL = 100.0
L_BOX = 0.368; RHO = 1.149; BULK = RHO * L_BOX ** 3
POOL = 4096; BATCH = 512; NTOP = 16; NCTL = 8; QSTEPS = 400
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


def q_vs(a, b, R):
    return len(box_id_set(a, L_BOX, R) & box_id_set(b, L_BOX, R)) / (L_BOX ** 3 * n_boxes(L_BOX, R)) - BULK


def quench(Xb, Sb, bnd, sb, steps=QSTEPS, dmax=0.05):
    Mn, n, _ = Xb.shape; mm = bnd.shape[0]
    xi = Xb.double().clone(); bndd = bnd.double()
    sfull = torch.cat([Sb, sb[None].expand(Mn, mm)], 1).long()
    for t in range(steps):
        xi = xi.detach().requires_grad_(True)
        U = ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL)
        (gr,) = torch.autograd.grad(U.sum(), xi)
        step = dmax * (0.3 + 0.7 * (1 - t / steps))
        gn = gr.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        xi = xi + (-gr / gn * torch.minimum(gn * 1e-3, torch.full_like(gn, step)))
    with torch.no_grad():
        return ka_energy(torch.cat([xi, bndd[None].expand(Mn, mm, 3)], 1), sfull, BIGL).float(), xi.detach().float()


gen = torch.Generator(device=dev).manual_seed(0); ncav = 0
for ci in range(24):
    c = torch.rand(3, generator=gen, device=dev) * L; p = carve(X[ci], S[ci], c, R, L)
    if p["n_in"] < 14:
        continue
    xo, so, _ = label_to_scaffold(_mic(p["x_in"], c, L), p["s_in"], R)
    xin = _mic(p["x_in"], c, L)
    xout = _mic(p["x_out"], c, L); bm = xout.norm(dim=-1) < (R + RCTX); bnd, sb = xout[bm], p["s_out"][bm]
    n = xo.shape[0]; allm = torch.ones(n, dtype=torch.bool, device=dev)
    # pool draws, keep top-q~ and controls
    g = torch.Generator(device=dev).manual_seed(500 + ci)
    allX, allS, allq = [], [], []
    with torch.no_grad():
        for b0 in range(0, POOL, BATCH):
            nb = min(BATCH, POOL - b0)
            Xa, Sa, _ = m.sample_block_b(xo[None].expand(nb, n, 3).clone(), so[None].expand(nb, n).clone(),
                                         allm, bnd, sb, R, gen=g)
            Xc = Xa.cpu()
            for k in range(nb):
                allq.append(q_vs(Xc[k], xin.cpu(), R))
            allX.append(Xc); allS.append(Sa.cpu())
    allX = torch.cat(allX); allS = torch.cat(allS); allq = torch.tensor(allq)
    top = allq.topk(NTOP).indices; ctl = torch.randperm(POOL)[:NCTL]
    sel = torch.cat([top, ctl])
    Xsel = allX[sel].to(dev); Ssel = allS[sel].to(dev)
    # quench reference + selection
    Uref, Xref_q = quench(xo[None], so[None], bnd, sb)
    Usel, Xsel_q = quench(Xsel, Ssel, bnd, sb)
    uref = float(Uref[0]) / n; usel = Usel / n
    # alien IS noise scale from controls
    sig = float(usel[NTOP:].std())
    q_is = torch.tensor([q_vs(Xsel_q[k].cpu(), Xref_q[0].cpu(), R) for k in range(len(sel))])
    print(f"=== cav {ci} (n={n})  ref U_IS/n={uref:.3f}  alien-IS sigma~{sig:.3f} ===", flush=True)
    print(f"{'rank':>4} {'q~_raw':>7} | {'U_IS/n':>8} {'depth_z':>8} | {'q~_IS_vs_refIS':>14} | member?", flush=True)
    for k in range(len(sel)):
        tag = "top" if k < NTOP else "ctl"
        z = (float(usel[k]) - uref) / max(sig, 1e-6)
        member = (z < 3.0) and (float(q_is[k]) > 0.4)
        print(f"{tag}{k if k < NTOP else k - NTOP:>3} {float(allq[sel[k]]):>7.3f} | {float(usel[k]):>8.3f} {z:>8.1f} | "
              f"{float(q_is[k]):>14.3f} | {'YES' if member else 'no'}", flush=True)
    ncav += 1
    if ncav >= 2:
        break
print("\nANY 'YES' => q0 does enter the attraction basin; draw+quench+select basin-finder is live.", flush=True)
print("NONE => attraction-basin coverage wall REAL => dual-seed PT is the path.", flush=True)
